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

  static async open(dir, cap) {
    const c = new RangeCache(dir, cap);
    const found = [];
    for await (const [name] of dir.entries()) {
      const e = parseEntry(name);
      if (e) found.push(e);
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
  constructor(urls, { cache = null, parallel = 6, fetchFn = (...a) => fetch(...a), XHR = globalThis.XMLHttpRequest, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) } = {}) {
    this.files = urls.map((url) => ({ url, direct: null, until: 0, size: 0 }));
    this.cache = cache;
    this.parallel = parallel;
    this.fetchFn = fetchFn;
    this.XHR = XHR;
    this.sleep = sleep;
    this.staged = new Map();
    this.stats = { fetched: 0, requests: 0, retries: 0, fetchMs: 0, cacheHit: 0 };
    this.aborted = false;
    this.controller = new AbortController();
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

  /** Fetch spans in pieces, `parallel` at a time. */
  async fetchAll(spans) {
    const out = spans.map((s) => new Uint8Array(s.len));
    const pieces = [];
    spans.forEach((s, i) => {
      for (let at = 0; at < s.len; at += PIECE) pieces.push([i, at, Math.min(PIECE, s.len - at)]);
    });
    let next = 0;
    const t0 = performance.now();
    const worker = async () => {
      while (next < pieces.length) {
        const [i, at, n] = pieces[next++];
        const s = spans[i];
        out[i].set(await this.fetchRange(s.file, s.offset + at, n), at);
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
    const want = [];
    const seen = new Set();
    const unique = [];
    for (const s of spans) {
      const key = spanKey(s.file, s.offset, s.len);
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push({ ...s, key });
    }
    // Cached ranges are read from the device together (a layer can need dozens of experts).
    const hits = await Promise.all(unique.map((s) => (this.cache?.has(s.key) ? this.cache.get(s.key) : null)));
    unique.forEach((s, i) => {
      if (hits[i]) {
        this.stats.cacheHit += hits[i].byteLength;
        this.staged.set(s.key, hits[i]);
      } else {
        want.push(s);
      }
    });
    if (!want.length) return;
    const data = await this.fetchAll(want);
    for (let i = 0; i < want.length; i++) {
      this.staged.set(want[i].key, data[i]);
      if (this.cache) await this.cache.put(want[i].key, data[i], want[i].every);
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
    this.stats = { fetched: 0, requests: 0, retries: 0, fetchMs: 0, cacheHit: 0 };
    return s;
  }
}
