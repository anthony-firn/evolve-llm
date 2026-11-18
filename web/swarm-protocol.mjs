// The weight swarm's pure parts, shared by the page and tests: chunk ranges, the messages peers
// exchange over a WebRTC data channel, and checking chunks against the published manifests.
//
// Model files are cut into 256 KiB chunks and named by the SHA-256 of the whole file (Hugging
// Face lists it for every GGUF), so the same file in any repository is one swarm. Every chunk a
// peer sends is checked against the manifest (tools/chunk_manifest.py) before it is used.

export const CHUNK = 256 * 1024;
export const PAGE_CHUNKS = 4096;
/** Largest request one peer serves at once, in chunks (4 MiB). */
export const MAX_REQUEST_CHUNKS = 16;
/** Data frames on the channel, under every browser's message size limit. */
export const FRAME = 64 * 1024;
export const PROTOCOL = 'evolve-swarm/1';

/** Chunk indices [start, end) covering the byte range [offset, offset + len) of a file. */
export function chunkSpan(offset, len) {
  return [Math.floor(offset / CHUNK), Math.ceil((offset + len) / CHUNK)];
}

/** Sorted, merged chunk ranges [[start, end), ...]. */
export class RangeSet {
  constructor(ranges = []) {
    this.r = [];
    for (const [s, e] of ranges) this.add(s, e);
  }

  add(s, e) {
    if (!(e > s)) return;
    const out = [];
    let placed = false;
    for (const [a, b] of this.r) {
      if (b < s) out.push([a, b]);
      else if (a > e) {
        if (!placed) { out.push([s, e]); placed = true; }
        out.push([a, b]);
      } else {
        s = Math.min(s, a);
        e = Math.max(e, b);
      }
    }
    if (!placed) out.push([s, e]);
    this.r = out.sort((x, y) => x[0] - y[0]);
  }

  has(s, e) {
    return this.r.some(([a, b]) => a <= s && e <= b);
  }

  get size() {
    return this.r.reduce((n, [a, b]) => n + b - a, 0);
  }

  toJSON() {
    return this.r.map(([a, b]) => [a, b]);
  }
}

/** A data frame: [request id u32][byte offset in the response u32][bytes]. */
export function encodeFrame(id, at, bytes) {
  const out = new Uint8Array(8 + bytes.byteLength);
  const v = new DataView(out.buffer);
  v.setUint32(0, id);
  v.setUint32(4, at);
  out.set(bytes, 8);
  return out;
}

export function decodeFrame(buf) {
  const u = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const v = new DataView(u.buffer, u.byteOffset, u.byteLength);
  return { id: v.getUint32(0), at: v.getUint32(4), bytes: u.subarray(8) };
}

/**
 * Control messages are JSON text: hello {files: {sha: ranges}}, have {sha, ranges},
 * req {id, sha, s, e}, nope {id}. Anything else, or anything malformed, is null.
 */
export function parseControl(text) {
  let m;
  try { m = JSON.parse(text); } catch { return null; }
  if (!m || typeof m !== 'object') return null;
  const sha = (x) => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x);
  const ranges = (x) => Array.isArray(x) && x.length <= 100000 && x.every((r) => Array.isArray(r) && r.length === 2 && Number.isInteger(r[0]) && Number.isInteger(r[1]) && r[0] >= 0 && r[1] > r[0]);
  const uint = (x) => Number.isInteger(x) && x >= 0 && x < 2 ** 32;
  switch (m.t) {
    case 'hello':
      if (m.v !== PROTOCOL || !m.files || typeof m.files !== 'object') return null;
      if (!Object.entries(m.files).every(([k, v]) => sha(k) && ranges(v))) return null;
      return { t: 'hello', files: m.files };
    case 'have':
      return sha(m.sha) && ranges(m.ranges) ? { t: 'have', sha: m.sha, ranges: m.ranges } : null;
    case 'req':
      return uint(m.id) && sha(m.sha) && uint(m.s) && uint(m.e) && m.e > m.s && m.e - m.s <= MAX_REQUEST_CHUNKS
        ? { t: 'req', id: m.id, sha: m.sha, s: m.s, e: m.e } : null;
    case 'nope':
      return uint(m.id) ? { t: 'nope', id: m.id } : null;
    default:
      return null;
  }
}

const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');

export async function sha256(bytes) {
  return new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
}

/**
 * The published chunk hashes: index.json per file (trusted, from the manifests repository over
 * TLS) and pages of 4096 chunk hashes, each checked against the index before use.
 */
export class Manifests {
  constructor(base, { fetchFn = (...a) => fetch(...a) } = {}) {
    this.base = base.endsWith('/') ? base : base + '/';
    this.fetchFn = fetchFn;
    this.indexes = new Map();
    this.pages = new Map();
  }

  /** The index of a file, or null when no manifest is published for it. */
  async index(sha) {
    if (!this.indexes.has(sha)) {
      this.indexes.set(sha, (async () => {
        const res = await this.fetchFn(`${this.base}${sha}/index.json`);
        if (res.status === 404) return null;
        if (!res.ok) throw new Error(`manifest index: HTTP ${res.status}`);
        const ix = await res.json();
        if (ix.format !== 'evolve-llm-chunks/1' || ix.sha256 !== sha || ix.chunk !== CHUNK || ix.page_chunks !== PAGE_CHUNKS) return null;
        return ix;
      })().catch((e) => { this.indexes.delete(sha); throw e; }));
    }
    return this.indexes.get(sha);
  }

  /** The 32-byte hashes of page `n` of a file, checked against its index. */
  async page(sha, n) {
    const key = `${sha}/${n}`;
    if (!this.pages.has(key)) {
      this.pages.set(key, (async () => {
        const ix = await this.index(sha);
        if (!ix || !ix.pages[n]) throw new Error('no manifest for this part of the file');
        const res = await this.fetchFn(`${this.base}${sha}/${n}.bin`);
        if (!res.ok) throw new Error(`manifest page: HTTP ${res.status}`);
        const bytes = new Uint8Array(await res.arrayBuffer());
        if (hex(await sha256(bytes)) !== ix.pages[n]) throw new Error('a manifest page does not match its index');
        return bytes;
      })().catch((e) => { this.pages.delete(key); throw e; }));
    }
    return this.pages.get(key);
  }

  /** True when `bytes` are exactly chunks [s, e) of the file. */
  async verify(sha, s, e, bytes) {
    const ix = await this.index(sha);
    if (!ix) return false;
    const want = Math.min(e * CHUNK, ix.size) - s * CHUNK;
    if (bytes.byteLength !== want) return false;
    for (let c = s; c < e; c++) {
      const page = await this.page(sha, Math.floor(c / PAGE_CHUNKS));
      const i = (c % PAGE_CHUNKS) * 32;
      const got = await sha256(bytes.subarray((c - s) * CHUNK, Math.min((c - s + 1) * CHUNK, bytes.byteLength)));
      for (let k = 0; k < 32; k++) if (got[k] !== page[i + k]) return false;
    }
    return true;
  }
}
