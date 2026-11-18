// The page's side of the weight swarm for one loaded model: it starts the swarm for the model's
// files that have published manifests, serves chunks from this device's cache (the worker writes
// it; the page reads it), and answers the worker's requests for chunks.

import { CHUNK } from './swarm-protocol.mjs';
import { Swarm } from './swarm.mjs';

const ENTRY = /^(\d+)-(\d+)-(\d+)\.([te])$/;

/**
 * `parts`: [{sha256, bytes}] of the model, in file order; `cacheKey`: its cache directory under
 * `stream/`. Returns null when no part has a manifest (nothing to share yet).
 */
export async function startSwarm({ parts, cacheKey, worker, uploadBytesPerSec, onStats, swarmOptions = {} }) {
  const files = parts.filter((p) => /^[0-9a-f]{64}$/.test(p.sha256 || '')).map((p) => ({ sha: p.sha256, size: p.bytes }));
  if (!files.length) return null;
  const shaOf = (file) => parts[file]?.sha256;
  const entries = new Map(); // sha -> [{offset, len, name}]
  let dir = null;
  try {
    dir = await (await (await navigator.storage.getDirectory()).getDirectoryHandle('stream')).getDirectoryHandle(cacheKey);
  } catch { /* no cache yet */ }
  const add = (file, offset, len, every) => {
    const sha = shaOf(file);
    if (!sha) return;
    if (!entries.has(sha)) entries.set(sha, []);
    entries.get(sha).push({ offset, len, name: `${file}-${offset}-${len}.${every ? 't' : 'e'}` });
    // Only whole chunks can be shared (the last chunk of a file may be short).
    const size = parts[file].bytes;
    const s = Math.ceil(offset / CHUNK);
    const e = offset + len >= size ? Math.ceil(size / CHUNK) : Math.floor((offset + len) / CHUNK);
    if (e > s) swarm.addHave(sha, s, e);
  };
  const read = async (sha, s, e) => {
    if (!dir) {
      try { dir = await (await (await navigator.storage.getDirectory()).getDirectoryHandle('stream')).getDirectoryHandle(cacheKey); } catch { return null; }
    }
    const file = parts.findIndex((p) => p.sha256 === sha);
    const from = s * CHUNK;
    const to = Math.min(e * CHUNK, parts[file].bytes);
    const hit = (entries.get(sha) || []).find((x) => x.offset <= from && x.offset + x.len >= to);
    if (!hit) return null;
    try {
      const blob = await (await dir.getFileHandle(hit.name)).getFile();
      return new Uint8Array(await blob.slice(from - hit.offset, to - hit.offset).arrayBuffer());
    } catch {
      return null; // evicted since
    }
  };
  const swarm = new Swarm({ files, read, uploadBytesPerSec, onStats, ...swarmOptions });
  const usable = await swarm.start();
  if (!usable.length) {
    swarm.stop();
    return null;
  }
  // What earlier visits cached can be shared too.
  if (dir) {
    for await (const [name] of dir.entries()) {
      const m = ENTRY.exec(name);
      if (m) add(Number(m[1]), Number(m[2]), Number(m[3]), m[4] === 't');
    }
  }
  const onMessage = async ({ data }) => {
    if (data?.type === 'swarm-have') add(data.file, data.offset, data.len, data.every);
    if (data?.type === 'swarm-get') {
      const bytes = await swarm.get(data.sha, data.s, data.e).catch(() => null);
      worker.postMessage({ type: 'swarm-data', id: data.id, bytes: bytes?.buffer ?? null }, bytes ? [bytes.buffer] : []);
    }
  };
  worker.addEventListener('message', onMessage);
  worker.postMessage({ type: 'swarm', on: true });
  return {
    files: usable,
    stop() {
      worker.removeEventListener('message', onMessage);
      worker.postMessage({ type: 'swarm', on: false });
      swarm.stop();
    },
  };
}
