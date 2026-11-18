// The page's side of the weight swarm for one loaded model: it starts the swarm for the model's
// files that have published manifests, serves chunks from this device's cache (the worker writes
// it; the page reads it), and answers the worker's requests for chunks.

import { CHUNK } from './swarm-protocol.mjs';
import { Swarm } from './swarm.mjs';

const ENTRY = /^(\d+)-(\d+)-(\d+)\.([te])$/;

async function cacheDir(cacheKey) {
  try {
    return await (await (await navigator.storage.getDirectory()).getDirectoryHandle('stream')).getDirectoryHandle(cacheKey);
  } catch {
    return null; // nothing cached yet
  }
}

/**
 * `parts`: [{sha256, bytes}] of the model, in file order; `cacheKey`: its cache directory under
 * `stream/`. Starts at once, so the worker's first reads (the headers) can wait for peers: it
 * listens to the worker straight away and answers its requests once the swarm is up. `ready`
 * resolves to the shared files' SHA-256s, or null when no part has a published manifest.
 */
export function startSwarm({ parts, cacheKey, worker, uploadBytesPerSec, onStats, swarmOptions = {} }) {
  const files = parts.filter((p) => /^[0-9a-f]{64}$/.test(p.sha256 || '')).map((p) => ({ sha: p.sha256, size: p.bytes }));
  const shaOf = (file) => parts[file]?.sha256;
  const entries = new Map(); // sha -> [{offset, len, name}]
  let dir = null;
  let swarm = null;
  let stopped = false;
  const early = [];

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
    dir ||= await cacheDir(cacheKey);
    if (!dir) return null;
    const file = parts.findIndex((p) => p.sha256 === sha);
    const from = s * CHUNK;
    const to = Math.min(e * CHUNK, parts[file].bytes);
    // A request can cross the cache entries it was stored in: put it together from them.
    const out = new Uint8Array(to - from);
    let at = from;
    const list = (entries.get(sha) || []).filter((x) => x.offset < to && x.offset + x.len > from).sort((a, b) => a.offset - b.offset);
    for (const x of list) {
      if (x.offset > at) return null; // a gap
      if (x.offset + x.len <= at) continue;
      const end = Math.min(to, x.offset + x.len);
      try {
        const blob = await (await dir.getFileHandle(x.name)).getFile();
        out.set(new Uint8Array(await blob.slice(at - x.offset, end - x.offset).arrayBuffer()), at - from);
      } catch {
        return null; // evicted since
      }
      at = end;
      if (at >= to) return out;
    }
    return null;
  };

  const ready = (async () => {
    if (!files.length) return null;
    const s = new Swarm({ files, read, uploadBytesPerSec, onStats, ...swarmOptions });
    const usable = await s.start();
    if (!usable.length || stopped) {
      s.stop();
      return null;
    }
    swarm = s;
    // What earlier visits cached can be shared too.
    dir = await cacheDir(cacheKey);
    if (dir) {
      for await (const [name] of dir.entries()) {
        const m = ENTRY.exec(name);
        if (m) add(Number(m[1]), Number(m[2]), Number(m[3]), m[4] === 't');
      }
    }
    for (const d of early.splice(0)) add(d.file, d.offset, d.len, d.every);
    return swarm;
  })().catch(() => null);
  ready.then((s) => { if (!s && !stopped) worker.postMessage({ type: 'swarm', on: false }); });

  const onMessage = async ({ data }) => {
    if (data?.type === 'swarm-have') {
      if (swarm) add(data.file, data.offset, data.len, data.every);
      else early.push(data);
    }
    if (data?.type === 'swarm-get') {
      const s = await ready;
      const bytes = s && !stopped ? await s.get(data.sha, data.s, data.e).catch(() => null) : null;
      worker.postMessage({ type: 'swarm-data', id: data.id, bytes: bytes?.buffer ?? null }, bytes ? [bytes.buffer] : []);
    }
  };
  worker.addEventListener('message', onMessage);

  return {
    ready: ready.then((s) => (s ? [...s.files.keys()] : null)),
    stop() {
      stopped = true;
      worker.removeEventListener('message', onMessage);
      worker.postMessage({ type: 'swarm', on: false });
      ready.then((s) => s?.stop());
    },
  };
}
