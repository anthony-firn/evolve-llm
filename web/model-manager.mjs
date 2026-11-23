/**
 * Models that manage themselves: which catalog model this device should run, when to fetch it in
 * the background, and which saved models to remove. Plain data in, decisions out, so the app's
 * own code stays small and these rules are tested (tests/model-manager.test.mjs).
 */

/** Best first: the Qwen3.5 sizes, from laptops to phones (benchmark rounds 9 and 10 for 4B). */
export const LADDER = ['qwen3.5-4b', 'qwen3.5-2b', 'qwen3.5-0.8b'];

/** A model needs its file on the GPU and about a fifth more for its context and working buffers. */
export const GPU_OVERHEAD = 1.2;

/** On phones, models past this size answer too slowly even when they fit. */
export const MOBILE_MAX = 1.5e9;

const DAY = 864e5;

/**
 * The model this device should run: the first on the ladder that fits on its GPU (all of it, so
 * nothing is streamed), that it has room to save (or has saved), and that is quick enough on a
 * phone. The smallest when none fits (it streams what does not).
 * device: { gpuBytes, storageFree, mobile, saved: Set of catalog ids already saved }.
 */
export function pickModel(catalog, device, ladder = LADDER) {
  const rungs = ladder.map((id) => catalog.find((m) => m.id === id)).filter(Boolean);
  const fits = (m) =>
    m.bytes * GPU_OVERHEAD <= (device.gpuBytes ?? Infinity) &&
    (device.saved?.has(m.id) || m.bytes * 1.05 <= (device.storageFree ?? Infinity)) &&
    (!device.mobile || m.bytes <= MOBILE_MAX);
  return rungs.find(fits) || rungs.at(-1) || null;
}

/**
 * Saved models to remove, least recently used first: enough to make `need` bytes fit in `free`,
 * plus any not used for `staleDays` (partial downloads after a week). Never those in `keep`
 * (the running model, the device's pick, the ones the person chose themselves).
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
