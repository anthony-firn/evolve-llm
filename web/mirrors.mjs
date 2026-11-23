// Model downloads from several sources at once: Hugging Face plus mirrors that host the same
// files (ModelScope keeps copies of most open GGUF repositories). The file is fetched in pieces
// over several connections per source and written in place, so one slow or failing source only
// slows its own pieces. A mirror is used only for a file with a published chunk manifest
// (swarm-protocol.mjs): every piece from it is checked against the manifest before it is written,
// and a mirror that sends a wrong piece is dropped. Hugging Face itself is trusted as before.

import { CHUNK } from './swarm-protocol.mjs';

/** Bytes per piece: 32 manifest chunks, so each piece is checked as a whole. */
export const PIECE = 32 * CHUNK;

/** Hugging Face resolve URL -> {repo, revision, file}, or null. */
export function parseResolveUrl(url) {
  const m = /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/.exec(url || '');
  return m ? { repo: m[1], revision: m[2], file: m[3] } : null;
}

/**
 * Where a file can come from, best first. `verified`: a manifest exists for it, so mirrors may
 * serve it. Hugging Face gets three connections (each is limited to a few MB/s); a mirror three.
 */
export function sourcesFor(url, { verified = false, mirrors = true } = {}) {
  const sources = [{ name: 'Hugging Face', url, connections: 3, trusted: true }];
  const hf = parseResolveUrl(url);
  if (hf && verified && mirrors) {
    sources.push({ name: 'ModelScope', url: `https://www.modelscope.cn/models/${hf.repo}/resolve/master/${hf.file}`, connections: 3, trusted: false });
  }
  return sources;
}

/** Done pieces as a compact string (one bit each, base64) for the download's record, and back. */
export function packPieces(done, count) {
  const bytes = new Uint8Array(Math.ceil(count / 8));
  for (const i of done) if (i < count) bytes[i >> 3] |= 1 << (i & 7);
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

export function unpackPieces(packed, count) {
  const done = new Set();
  if (typeof packed !== 'string') return done;
  let s;
  try { s = atob(packed); } catch { return done; }
  for (let i = 0; i < count; i++) if ((s.charCodeAt(i >> 3) || 0) & (1 << (i & 7))) done.add(i);
  return done;
}

async function readRange(fetchFn, url, start, end, signal) {
  const res = await fetchFn(url, { headers: { Range: `bytes=${start}-${end - 1}` }, signal });
  if (res.status !== 206 && !(res.status === 200 && start === 0)) {
    const e = new Error(`HTTP ${res.status}`);
    e.status = res.status;
    throw e;
  }
  const bytes = new Uint8Array(await res.arrayBuffer());
  if (res.status === 200 ? bytes.byteLength < end : bytes.byteLength !== end - start) throw new Error('short piece');
  return res.status === 200 ? bytes.subarray(0, end) : bytes;
}

/**
 * Fetch every piece of a `size`-byte file not in `done`, from `sources`, calling `write(start,
 * bytes)` for each (in any order) and `onPiece(index, bytes, sourceName)` after it is written.
 * `verify(start, bytes)` -> true, false, or null (no manifest); untrusted sources need true.
 * Resolves when all pieces are written; rejects with the last error once every source has failed
 * or with an AbortError when `signal` aborts.
 */
export async function fetchPieces({ size, sources, done = new Set(), write, verify = async () => null, onPiece = () => {},
  signal, fetchFn = (...a) => fetch(...a), wait = (ms) => new Promise((r) => setTimeout(r, ms)), maxFailures = 4 }) {
  const count = Math.ceil(size / PIECE);
  const queue = [];
  for (let i = 0; i < count; i++) if (!done.has(i)) queue.push(i);
  let remaining = queue.length;
  if (!remaining) return;
  const state = sources.map((s) => ({ ...s, failures: 0, dropped: false, error: null }));
  let lastError = null;
  let fatal = null;
  const live = () => state.some((s) => !s.dropped);

  async function worker(src) {
    while (!fatal && !src.dropped && remaining > 0) {
      if (signal?.aborted) return;
      const i = queue.shift();
      if (i === undefined) {
        // Others hold the last pieces; wait in case one of them fails and puts its piece back.
        await wait(200);
        continue;
      }
      const start = i * PIECE;
      const end = Math.min(size, start + PIECE);
      let bytes;
      try {
        bytes = await readRange(fetchFn, src.url, start, end, signal);
      } catch (e) {
        queue.unshift(i);
        if (e?.name === 'AbortError' || signal?.aborted) return;
        lastError = e;
        src.failures += 1;
        // A source that keeps failing (a 404 at once) stops; others carry on.
        if (src.failures >= maxFailures || e.status === 404 || e.status === 401 || e.status === 403) {
          src.dropped = true;
          src.error = e;
        } else {
          await wait(1000 * 2 ** src.failures);
        }
        continue;
      }
      const ok = await verify(start, bytes);
      if (ok === false || (!src.trusted && ok !== true)) {
        queue.unshift(i);
        lastError = new Error(`${src.name} sent a piece that does not match the published hashes`);
        if (src.trusted) { fatal = lastError; return; }
        src.dropped = true;
        continue;
      }
      src.failures = 0;
      try {
        await write(start, bytes);
      } catch (e) {
        fatal = e;
        return;
      }
      remaining -= 1;
      onPiece(i, bytes, src.name);
    }
  }

  const workers = [];
  for (const src of state) for (let c = 0; c < (src.connections || 1); c++) workers.push(worker(src));
  // Every worker of every source may stop (all dropped) while pieces remain: then it failed.
  await Promise.all(workers);
  if (signal?.aborted) {
    const e = new Error('aborted');
    e.name = 'AbortError';
    throw e;
  }
  if (fatal) throw fatal;
  if (remaining > 0) throw lastError || new Error(live() ? 'download stopped' : 'every source failed');
}
