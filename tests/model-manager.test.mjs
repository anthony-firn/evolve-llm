import test from 'node:test';
import assert from 'node:assert/strict';
import { LEVELS, QUALITY_FLOOR, frontier, pickModel, rungFor, toRemove, shouldFetch, isMetered } from '../web/model-manager.mjs';
import { MODELS } from '../web/models.js';

const GB = 1e9;
const DEVICES = {
  laptop: { gpuBytes: 5 * GB, storageFree: 100 * GB },
  'laptop, little storage': { gpuBytes: 5 * GB, storageFree: 8 * GB },
  'small GPU': { gpuBytes: 2 * GB, storageFree: 50 * GB },
  phone: { gpuBytes: 3 * GB, storageFree: 20 * GB, mobile: true },
  'tiny GPU': { gpuBytes: 1 * GB, storageFree: 50 * GB },
};
const name = (r) => (r ? r.id + (r.thinking ? '+thinking' : '') : null);
const levels = (device) => { const l = frontier(MODELS, device); return Array.from({ length: LEVELS }, (_, v) => name(rungFor(l, v))); };

test('every catalog model has what the frontier needs', () => {
  for (const m of MODELS) {
    assert.match(m.sha256, /^[0-9a-f]{64}$/, m.id);
    assert.ok(m.speed > 0, m.id);
    if (m.vision) continue;
    assert.ok(m.modes?.length, m.id);
    for (const mode of m.modes) assert.ok(mode.coding >= 0 && mode.tokens > 0 && typeof mode.thinking === 'boolean', m.id);
  }
});

test('the catalog holds only frontier models: each one is used at some level on some device', () => {
  const used = new Set(MODELS.filter((m) => m.vision).map((m) => m.id));
  for (const device of Object.values(DEVICES)) {
    const l = frontier(MODELS, device);
    for (let v = 0; v < LEVELS; v++) used.add(rungFor(l, v)?.id);
    used.add(pickModel(MODELS, device)?.id);
  }
  for (const m of MODELS) assert.ok(used.has(m.id), `${m.id} is never used: it is not on any device's frontier`);
});

test('each device climbs its own frontier, fastest for everyday requests and best for the hardest', () => {
  assert.deepEqual(levels(DEVICES.laptop), ['qwen3-4b-2507', 'qwen3.5-4b', 'qwen3-coder-30b-a3b', 'qwen3.6-35b-a3b+thinking']);
  // No room for the 13 and 17 GB experts models: Qwen3.5 4B thinking is the top.
  assert.deepEqual(levels(DEVICES['laptop, little storage']), ['qwen3-4b-2507', 'qwen3.5-4b', 'qwen3.5-4b', 'qwen3.5-4b+thinking']);
  // Already saved, an experts model counts even when storage is short.
  assert.equal(levels({ ...DEVICES['laptop, little storage'], saved: new Set(['qwen3-coder-30b-a3b']) })[3], 'qwen3-coder-30b-a3b');
  // Nothing clears the quality floor on a phone: its best model does everything.
  assert.deepEqual(levels(DEVICES.phone), Array(LEVELS).fill('qwen3.5-2b'));
  assert.deepEqual(levels(DEVICES['tiny GPU']), Array(LEVELS).fill('qwen3-0.6b'));
});

test('on the frontier, a faster rung never scores higher than a slower one', () => {
  for (const device of Object.values(DEVICES)) {
    const l = frontier(MODELS, device);
    for (let i = 1; i < l.length; i++) assert.ok(l[i].time >= l[i - 1].time && l[i].coding > l[i - 1].coding);
  }
});

test('rungs spread over the levels, and the floor keeps weak rungs for when nothing else fits', () => {
  const r = (id, coding, time) => ({ id, coding, time, thinking: false });
  assert.deepEqual([0, 1, 2, 3].map((v) => rungFor([r('a', 9, 1), r('b', 14, 2)], v).id), ['a', 'a', 'b', 'b']);
  assert.deepEqual([0, 1, 2, 3].map((v) => rungFor([r('tiny', 1, 0.1), r('a', 9, 1)], v).id), ['a', 'a', 'a', 'a']);
  assert.equal(rungFor([r('tiny', 1, 0.1), r('small', QUALITY_FLOOR - 1, 1)], 0).id, 'small');
  assert.equal(rungFor([], 2), null);
});

test('the model kept ready runs wholly on the GPU', () => {
  assert.equal(pickModel(MODELS, DEVICES.laptop).id, 'qwen3.5-4b');
  assert.equal(pickModel(MODELS, DEVICES.phone).id, 'qwen3.5-2b');
  assert.equal(pickModel(MODELS, { gpuBytes: 0.2 * GB, storageFree: 50 * GB }).id, 'qwen3-0.6b');
  assert.equal(pickModel(MODELS, { storageFree: 50 * GB }).id, 'qwen3.5-4b');
});

test('removal frees what a download needs, least recently used first, and drops stale models', () => {
  const now = 100 * 864e5;
  const saved = [
    { key: 'a', stored: 2 * GB, complete: true },
    { key: 'b', stored: 1 * GB, complete: true },
    { key: 'c', stored: 3 * GB, complete: true },
    { key: 'p', stored: 0.5 * GB, complete: false },
  ];
  const lastUsed = { a: now - 2 * 864e5, b: now - 1 * 864e5, c: now - 10 * 864e5, p: now - 1 * 864e5 };
  const keys = (opts) => toRemove(saved, { now, lastUsed, ...opts }).map((m) => m.key);
  // Room for 2 GB with 0.5 free: the least recently used (c) goes.
  assert.deepEqual(keys({ need: 2 * GB, free: 0.5 * GB }), ['c']);
  // c is kept (running or chosen): a, then b.
  assert.deepEqual(keys({ need: 3 * GB, free: 0.5 * GB, keep: new Set(['c']) }), ['a', 'b']);
  // Nothing needed and nothing stale: nothing goes.
  assert.deepEqual(keys({}), []);
  // Unused for 60 days: removed; a partial download after a week.
  const later = now + 55 * 864e5; // c: 65 days unused, the partial p: 56; a and b under 60
  assert.deepEqual(toRemove(saved, { now: later, lastUsed }).map((m) => m.key), ['c', 'p']);
});

test('the pick is fetched only when idle, on an unmetered connection, and not saved or running', () => {
  const pick = { id: 'qwen3.5-4b' };
  const base = { pick, saved: new Set(), running: 'qwen3.5-2b', metered: false, idle: true };
  assert.equal(shouldFetch(base), true);
  assert.equal(shouldFetch({ ...base, idle: false }), false);
  assert.equal(shouldFetch({ ...base, metered: true }), false);
  assert.equal(shouldFetch({ ...base, saved: new Set(['qwen3.5-4b']) }), false);
  assert.equal(shouldFetch({ ...base, running: 'qwen3.5-4b' }), false);
  assert.equal(isMetered({ saveData: true }), true);
  assert.equal(isMetered({ effectiveType: '3g' }), true);
  assert.equal(isMetered({ effectiveType: '4g' }), false);
  assert.equal(isMetered(undefined), false);
});
