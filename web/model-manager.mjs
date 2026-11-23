/**
 * Models that manage themselves: the device's frontier of speed against quality, which rung of
 * it a request needs, when to fetch a model in the background, and which saved models to remove. Plain data in, decisions out, so the app's
 * own code stays small and these rules are tested (tests/model-manager.test.mjs).
 */

/** A model needs its file on the GPU and about a fifth more for its context and working buffers. */
export const GPU_OVERHEAD = 1.2;

/** On phones, models past this size answer too slowly even when they fit. */
export const MOBILE_MAX = 1.5e9;

/** Rungs scoring below this on the Coding Index are used only when nothing better fits. */
export const QUALITY_FLOOR = 5;

/** How hard a request is: 0 everyday, 1 real work, 2 hard, 3 the hardest (after failures). */
export const LEVELS = 4;

const DAY = 864e5;

/**
 * Whether a catalog model runs on this device: wholly on its GPU, or (a mixture of experts) with
 * its attention on the GPU and its experts read from storage. Either way it must be saved, or have
 * room to be. device: { gpuBytes, storageFree, mobile, saved: Set of catalog ids already saved }.
 */
export function fits(m, device) {
  const stored = device.saved?.has(m.id) || m.bytes * 1.05 <= (device.storageFree ?? Infinity);
  if (m.experts) return stored && !device.mobile && (device.gpuBytes ?? Infinity) >= m.experts.minGpu;
  return stored && m.bytes * GPU_OVERHEAD <= (device.gpuBytes ?? Infinity) && (!device.mobile || m.bytes <= MOBILE_MAX);
}

/**
 * The device's frontier, fastest first: every (model, thinking or not) that fits, where no other
 * that fits answers at least as fast and scores at least as well (and better at one). Time per
 * answer is the tokens the model writes (Artificial Analysis's count) over its speed.
 * -> [{ id, model, thinking, coding, time }]
 */
export function frontier(catalog, device) {
  const points = [];
  for (const m of catalog) {
    if (m.vision || !m.speed || !fits(m, device)) continue;
    for (const mode of m.modes || []) points.push({ id: m.id, model: m, thinking: mode.thinking, coding: mode.coding, time: mode.tokens / m.speed });
  }
  const beaten = (p) => points.some((o) => o !== p && o.time <= p.time && o.coding >= p.coding && (o.time < p.time || o.coding > p.coding));
  return points.filter((p) => !beaten(p)).sort((a, b) => a.time - b.time);
}

/**
 * The rung for a request of difficulty `level` (0 .. LEVELS - 1): the frontier's rungs that clear
 * the quality floor spread over the levels, everyday requests on the fastest and the hardest on
 * the best. When none clears it (a phone), the best that fits does everything. Null for an empty
 * ladder.
 */
export function rungFor(ladder, level) {
  const useful = ladder.filter((r) => r.coding >= QUALITY_FLOOR);
  const rungs = useful.length ? useful : ladder.slice(-1);
  if (!rungs.length) return null;
  const at = Math.max(0, Math.min(LEVELS - 1, Math.round(level)));
  // With fewer rungs than levels, the hardest level gets the best rung and the rest share.
  return rungs[Math.min(rungs.length - 1, Math.round((at * (rungs.length - 1)) / (LEVELS - 1)))];
}

/**
 * The model this device should keep ready: the rung for real work (level 1), counting only models
 * that fit wholly on its GPU, since that is what most requests use; the smallest when none fits.
 */
export function pickModel(catalog, device) {
  const pick = rungFor(frontier(catalog.filter((m) => !m.experts), device), 1)?.model;
  if (pick) return pick;
  // Nothing fits on the GPU: the smallest, which streams what does not fit.
  return catalog.filter((m) => !m.experts && !m.vision && m.speed).sort((a, b) => a.bytes - b.bytes)[0] ?? null;
}

/**
 * Saved models to remove, least recently used first: enough to make `need` bytes fit in `free`,
 * plus any not used for `staleDays` (partial downloads after a week). Never those in `keep`
 * (the running model and the device's pick).
 * saved: [{ key, stored, complete }]; lastUsed: { key: time }.
 */
export function toRemove(saved, { need = 0, free = Infinity, keep = new Set(), lastUsed = {}, now = Date.now(), staleDays = 60 } = {}) {
  const used = (m) => lastUsed[m.key] || 0;
  const candidates = saved.filter((m) => !keep.has(m.key)).sort((a, b) => used(a) - used(b));
  const out = [];
  let room = free;
  for (const m of candidates) {
    const stale = now - used(m) > (m.complete ? staleDays : 7) * DAY;
    if (room < need || stale) {
      out.push(m);
      room += m.stored || 0;
    }
  }
  return out;
}

/**
 * Whether to download the pick now, while the app is idle: not saved yet, not already running,
 * and not on a connection the person pays for by the byte.
 */
export function shouldFetch({ pick, saved, running, metered, idle }) {
  return Boolean(pick && idle && !metered && !saved.has(pick.id) && running !== pick.id);
}

/** Whether a connection is likely metered or slow (Save-Data, or 2G/3G). */
export function isMetered(connection) {
  if (!connection) return false;
  return Boolean(connection.saveData) || /(^|-)(2g|3g)$/.test(String(connection.effectiveType || ''));
}
