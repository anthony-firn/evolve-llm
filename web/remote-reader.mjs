// Models that stay on Hugging Face instead of being downloaded first. The engine says which
// byte ranges it is about to read (prefetch); they are fetched many at once and handed to its
// synchronous reads. A size-capped cache in the origin-private file system keeps what it can:
// weights read for every token first, because a cache smaller than one token's reads would get
// no hits otherwise (every token reads them in the same order), then experts, least recently
// used out first. Anything read without a prefetch is fetched with a synchronous request,
// which a worker may make.

const PIECE = 16 << 20; // largest single request, so big tensors download in parallel
const TRIES = 8;

export const spanKey = (file, offset, len) => `${file}-${offset}-${len}`;

/** Milliseconds until a signed URL's `Expires=` timestamp, or half an hour without one. */
export function expiresIn(url, now = Date.now()) {
  const m = /[?&]Expires=(\d+)/.exec(url);
  return m ? Math.max(0, Number(m[1]) * 1000 - now) : 30 * 60 * 1000;
}

/** `{file}-{offset}-{len}.{t|e}`: a cached range and whether every token reads it. */
export function parseEntry(name) {
  const m = /^(\d+)-(\d+)-(\d+)\.([te])$/.exec(name);
  return m ? { key: `${m[1]}-${m[2]}-${m[3]}`, len: Number(m[3]), every: m[4] === 't' } : null;
}

const entryName = (key, every) => `${key}.${every ? 't' : 'e'}`;

/** Byte ranges kept as files in an OPFS directory, up to `cap` bytes. */
export class RangeCache {
  constructor(dir, cap) {
    this.dir = dir;
    this.cap = cap;
    this.used = 0;
    this.clock = 0;
    this.entries = new Map(); // key -> { every, len, used }
  }

  /** `align`: drop entries that do not start on that many bytes (made before chunk alignment). */
  static async open(dir, cap, { align = 0 } = {}) {
    const c = new RangeCache(dir, cap);
    const found = [];
    for await (const [name] of dir.entries()) {
      const e = parseEntry(name);
      if (e && align && Number(e.key.split('-')[1]) % align !== 0) await dir.removeEntry(name).catch(() => {});
      else if (e) found.push(e);
      else if (name.endsWith('.tmp')) await dir.removeEntry(name).catch(() => {});
    }
    for (const e of found) {
      c.entries.set(e.key, { every: e.every, len: e.len, used: ++c.clock });
      c.used += e.len;
    }
    // A smaller cap than last time: shed experts first.
    while (c.used > c.cap) {
      const victim = [...c.entries].sort((a, b) => (a[1].every - b[1].every) || (a[1].used - b[1].used))[0];
      await c.remove(victim[0]);
    }
    return c;
  }

  has(key) { return this.entries.has(key); }

  async get(key) {
    const e = this.entries.get(key);
    if (!e) return null;
    try {
      const file = await (await this.dir.getFileHandle(entryName(key, e.every))).getFile();
      if (file.size !== e.len) throw new Error('short');
      const bytes = new Uint8Array(await file.arrayBuffer());
      e.used = ++this.clock;
      return bytes;
    } catch {
      await this.remove(key);
      return null;
    }
  }

  async remove(key) {
    const e = this.entries.get(key);
    if (!e) return;
    this.entries.delete(key);
    this.used -= e.len;
    await this.dir.removeEntry(entryName(key, e.every)).catch(() => {});
  }

  /** Keep a range if there is room, evicting experts to make it. Returns whether it is kept. */
  async put(key, bytes, every) {
    if (this.entries.has(key)) return true;
    const len = bytes.byteLength;
    if (len > this.cap) return false;
    let experts = 0;
    for (const e of this.entries.values()) if (!e.every) experts += e.len;
    if (this.used - experts + len > this.cap) return false;
    while (this.used + len > this.cap) {
      let victim = null;
      for (const [k, e] of this.entries) if (!e.every && (!victim || e.used < victim[1].used)) victim = [k, e];
      await this.remove(victim[0]);
    }
    const name = entryName(key, every);
    try {
      const tmp = await this.dir.getFileHandle(`${name}.tmp`, { create: true });
      const access = await tmp.createSyncAccessHandle();
      try {
        access.truncate(0);
        access.write(bytes, { at: 0 });
        access.flush();
      } finally {
        access.close();
      }
      await tmp.move(name);
    } catch {
      // Out of space, or no move(): give up on this range rather than risk a torn entry.
      await this.dir.removeEntry(`${name}.tmp`).catch(() => {});
      return false;
    }
    this.entries.set(key, { every, len, used: ++this.clock });
    this.used += len;
    return true;
  }
}

function httpMessage(status) {
  if (status === 401 || status === 403) return 'This model needs a Hugging Face sign-in or license approval, so it cannot be read from here.';
  if (status === 404) return 'Hugging Face does not have this file any more.';
  return `Hugging Face answered with HTTP ${status}.`;
}

/**
 * The parts of one model on the network. `urls` are their download (resolve) addresses.
 * Options: `cache` (a RangeCache or null), `parallel` requests in flight, and for tests
 * `fetchFn`, `XHR` and `sleep`.
 */
export class RemoteModel {
  /**
   * Options also take `align` (fetch and cache whole chunks of that many bytes, so the cache
   * holds pieces peers can serve and check), `shas` (each part's SHA-256), `peers(sha, s, e)`
   * (chunks [s, e) from the swarm, or null) and `onCached(file, offset, len)`.
   */
  constructor(urls, { cache = null, parallel = 6, fetchFn = (...a) => fetch(...a), XHR = globalThis.XMLHttpRequest, sleep = (ms) => new Promise((r) => setTimeout(r, ms)), align = 0, shas = [], peers = null, onCached = null } = {}) {
    this.files = urls.map((url, i) => ({ url, direct: null, until: 0, size: 0, sha: shas[i] || null }));
    this.align = align;
    this.peers = peers;
    this.onCached = onCached;
    this.cache = cache;
    this.parallel = parallel;
    this.fetchFn = fetchFn;
    this.XHR = XHR;
    this.sleep = sleep;
    this.staged = new Map();
    this.stats = { fetched: 0, requests: 0, retries: 0, fetchMs: 0, cacheHit: 0, fromPeers: 0 };
    this.aborted = false;
    this.controller = new AbortController();
  }

  /** The whole-chunk range around a span (the span itself without alignment). */
  aligned(s) {
    if (!this.align) return { file: s.file, offset: s.offset, len: s.len };
    const size = this.files[s.file].size || Infinity;
    const start = Math.floor(s.offset / this.align) * this.align;
    const end = Math.min(Math.ceil((s.offset + s.len) / this.align) * this.align, size);
    return { file: s.file, offset: start, len: end - start };
  }

  /** End reads in flight and refuse new ones until `resume` (Stop during a slow token). */
  cancel() {
    this.aborted = true;
    this.controller.abort();
  }

  resume() {
    if (!this.aborted) return;
    this.aborted = false;
    this.controller = new AbortController();
  }

  /** Resolve every part: its size and the CDN address it redirects to. */
  async open() {
    await Promise.all(this.files.map((_, i) => this.resolve(i)));
    return this.files.map((f) => f.size);
  }

  async resolve(i) {
    const f = this.files[i];
    const res = await this.fetchFn(f.url, { headers: { Range: 'bytes=0-0' } });
    if (res.status !== 206 && !res.ok) throw new Error(httpMessage(res.status));
    const total = /\/(\d+)\s*$/.exec(res.headers.get('content-range') || '');
    if (res.status !== 206 || !total) throw new Error('Hugging Face did not answer with a byte range, so this model cannot be read in pieces.');
    await res.arrayBuffer().catch(() => {});
    f.size = Number(total[1]);
    // Reuse the signed CDN address until shortly before it expires: one API request per file
    // and hour instead of one per read.
    f.direct = res.url || f.url;
    f.until = Date.now() + expiresIn(f.direct) - 120_000;
  }

  async directUrl(i) {
    const f = this.files[i];
    if (!f.direct || Date.now() > f.until) await this.resolve(i);
    return f.direct;
  }

  async fetchRange(file, offset, len) {
    let last = '';
    for (let attempt = 0; attempt < TRIES; attempt++) {
      if (this.aborted) throw new Error('Reading the model was cancelled.');
      if (attempt) {
        this.stats.retries++;
        await this.sleep(500 * 2 ** Math.min(attempt, 5));
      }
      let res;
      try {
        const url = await this.directUrl(file);
        res = await this.fetchFn(url, { headers: { Range: `bytes=${offset}-${offset + len - 1}` }, signal: this.controller.signal });
      } catch (e) {
        if (this.aborted) throw new Error('Reading the model was cancelled.');
        last = String(e?.message || e);
        continue;
      }
      if (res.status === 200) throw new Error('The server ignored the byte range, so this model cannot be read in pieces.');
      if (res.status === 401 || res.status === 403 || res.status === 410) {
        this.files[file].direct = null; // an expired signature: resolve again
        last = `HTTP ${res.status}`;
        continue;
      }
      if (res.status !== 206) {
        last = `HTTP ${res.status}`;
        continue;
      }
      try {
        const bytes = new Uint8Array(await res.arrayBuffer());
        if (this.aborted) throw new Error('Reading the model was cancelled.');
        if (bytes.byteLength !== len) {
          last = `got ${bytes.byteLength} of ${len} bytes`;
          continue;
        }
        this.stats.fetched += len;
        this.stats.requests++;
        return bytes;
      } catch (e) {
        last = String(e?.message || e);
      }
    }
    throw new Error(`Could not read the model from Hugging Face (${last}). Check the connection and try again.`);
  }

  /**
   * Fetch spans in pieces, `parallel` at a time. With a swarm, aligned pieces of up to 4 MiB are
   * asked of peers first (checked against the manifest there) and fetched from Hugging Face when
   * no peer delivers them.
   */
  async fetchAll(spans) {
    const out = spans.map((s) => new Uint8Array(s.len));
    const pieces = [];
    spans.forEach((s, i) => {
      const swarm = this.peers && this.align && this.files[s.file].sha;
      const step = swarm ? 16 * this.align : PIECE;
      for (let at = 0; at < s.len; at += step) pieces.push([i, at, Math.min(step, s.len - at), swarm]);
    });
    let next = 0;
    const t0 = performance.now();
    const worker = async () => {
      while (next < pieces.length) {
        const [i, at, n, swarm] = pieces[next++];
        const s = spans[i];
        let bytes = null;
        if (swarm) {
          const first = (s.offset + at) / this.align;
          bytes = await this.peers(this.files[s.file].sha, first, first + Math.ceil(n / this.align)).catch(() => null);
          if (bytes && bytes.byteLength !== n) bytes = null;
          if (bytes) this.stats.fromPeers += n;
        }
        if (this.aborted) throw new Error('Reading the model was cancelled.');
        out[i].set(bytes || await this.fetchRange(s.file, s.offset + at, n), at);
        this.onPiece?.();
      }
    };
    await Promise.all(Array.from({ length: Math.min(this.parallel, pieces.length) }, worker));
    this.stats.fetchMs += performance.now() - t0;
    return out;
  }

  /**
   * Make spans readable: from the cache, or fetched. `spans` is an array of
   * `{file, offset, len, every}` or a Float64Array of (file, offset, len, every) quads.
   */
  async prefetch(spans) {
    if (ArrayBuffer.isView(spans)) {
      const quads = spans;
      spans = [];
      for (let i = 0; i < quads.length; i += 4) spans.push({ file: quads[i], offset: quads[i + 1], len: quads[i + 2], every: quads[i + 3] !== 0 });
    }
    // Anything still staged was never read (an earlier pass failed).
    this.staged.clear();
    // Each span is read as the whole-chunk range around it (when aligning), shared by spans
    // that fall in the same chunks.
    const groups = new Map();
    for (const s of spans) {
      const a = this.aligned(s);
      const key = spanKey(a.file, a.offset, a.len);
      if (!groups.has(key)) groups.set(key, { ...a, key, every: false, wants: [] });
      const g = groups.get(key);
      g.every ||= s.every;
      g.wants.push(s);
    }
    const unique = [...groups.values()];
    // Cached ranges are read from the device together (a layer can need dozens of experts).
    const hits = await Promise.all(unique.map((g) => (this.cache?.has(g.key) ? this.cache.get(g.key) : null)));
    const want = [];
    unique.forEach((g, i) => {
      if (hits[i]) {
        this.stats.cacheHit += hits[i].byteLength;
        g.bytes = hits[i];
      } else {
        want.push(g);
      }
    });
    if (want.length) {
      const data = await this.fetchAll(want);
      for (let i = 0; i < want.length; i++) {
        want[i].bytes = data[i];
        if (this.cache && await this.cache.put(want[i].key, data[i], want[i].every)) this.onCached?.(want[i].file, want[i].offset, want[i].len);
      }
    }
    for (const g of unique) {
      for (const s of g.wants) this.staged.set(spanKey(s.file, s.offset, s.len), g.bytes.subarray(s.offset - g.offset, s.offset - g.offset + s.len));
    }
  }

  /** A synchronous read of one range, for reads nothing prefetched (rare). */
  readNow(file, offset, target) {
    const key = spanKey(file, offset, target.length);
    const staged = this.staged.get(key);
    if (staged) {
      target.set(staged);
      this.staged.delete(key);
      return;
    }
    // A synchronous request in a worker. It goes through the download address, which the
    // browser follows to the CDN.
    const xhr = new this.XHR();
    xhr.open('GET', this.files[file].url, false);
    xhr.responseType = 'arraybuffer';
    xhr.setRequestHeader('Range', `bytes=${offset}-${offset + target.length - 1}`);
    xhr.send();
    if (xhr.status !== 206 || xhr.response?.byteLength !== target.length) {
      throw new Error(`Could not read the model from Hugging Face (HTTP ${xhr.status}).`);
    }
    this.stats.fetched += target.length;
    this.stats.requests++;
    target.set(new Uint8Array(xhr.response));
  }

  /** Readers with the interface of the local file readers. */
  readers() {
    return this.files.map((f, i) => ({
      size: f.size,
      read: (offset, target) => this.readNow(i, offset, target),
      close: () => {},
    }));
  }

  takeStats() {
    const s = { ...this.stats, cacheBytes: this.cache?.used ?? 0 };
    this.stats = { fetched: 0, requests: 0, retries: 0, fetchMs: 0, cacheHit: 0, fromPeers: 0 };
    return s;
  }
}
