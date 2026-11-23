import test from 'node:test';
import assert from 'node:assert/strict';
import { pickModel, toRemove, shouldFetch, isMetered, LADDER } from '../web/model-manager.mjs';
import { MODELS } from '../web/models.js';

const GB = 1e9;
const catalog = [
  { id: 'qwen3.5-4b', bytes: 2.74 * GB },
  { id: 'qwen3.5-2b', bytes: 1.28 * GB },
  { id: 'qwen3.5-0.8b', bytes: 0.53 * GB },
];

test('the ladder is in the catalog', () => {
  for (const id of LADDER) assert.ok(MODELS.some((m) => m.id === id), id);
});

test('each device gets the best model that fits on its GPU, has room to save, and is quick enough', () => {
  const pick = (device) => pickModel(catalog, device).id;
  assert.equal(pick({ gpuBytes: 6 * GB, storageFree: 50 * GB }), 'qwen3.5-4b');
  assert.equal(pick({ gpuBytes: 2.5 * GB, storageFree: 50 * GB }), 'qwen3.5-2b');
  assert.equal(pick({ gpuBytes: 1 * GB, storageFree: 50 * GB }), 'qwen3.5-0.8b');
  // Too little storage for the 4B, unless it is already saved.
  assert.equal(pick({ gpuBytes: 6 * GB, storageFree: 2 * GB }), 'qwen3.5-2b');
  assert.equal(pick({ gpuBytes: 6 * GB, storageFree: 2 * GB, saved: new Set(['qwen3.5-4b']) }), 'qwen3.5-4b');
  // Phones: small models even with a large shared GPU.
  assert.equal(pick({ gpuBytes: 8 * GB, storageFree: 50 * GB, mobile: true }), 'qwen3.5-2b');
  // Nothing fits: the smallest (it streams what does not fit).
  assert.equal(pick({ gpuBytes: 0.2 * GB, storageFree: 50 * GB }), 'qwen3.5-0.8b');
  // Unknown GPU memory: the best.
  assert.equal(pick({ storageFree: 50 * GB }), 'qwen3.5-4b');
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
