import test from 'node:test';
import assert from 'node:assert/strict';
import { RangeCache, RemoteModel, expiresIn, parseEntry, spanKey } from '../web/remote-reader.mjs';

// Two model parts on a fake Hugging Face: `resolve` URLs redirect to signed CDN URLs.
const parts = [new Uint8Array(40000).map((_, i) => i % 251), new Uint8Array(3000).map((_, i) => (i * 7) % 253)];
const resolveUrl = (i) => `https://huggingface.co/o/r/resolve/main/m-0000${i + 1}-of-00002.gguf`;

function fakeHub({ expireAfter = Infinity } = {}) {
  const log = [];
  let signed = 0;
  const fetchFn = async (url, { headers }) => {
    const m = /bytes=(\d+)-(\d+)/.exec(headers.Range);
    const [a, b] = [Number(m[1]), Number(m[2])];
    let i;
    let finalUrl = url;
    const r = /m-0000(\d)-of/.exec(url);
    if (url.startsWith('https://huggingface.co/')) {
      i = Number(r[1]) - 1;
      signed++;
      finalUrl = `https://cdn.example/${i}?Expires=${Math.floor(Date.now() / 1000) + 3600}&sig=${signed}`;
      log.push({ kind: 'resolve', i });
    } else {
      i = Number(/cdn\.example\/(\d)/.exec(url)[1]);
      const sig = Number(/sig=(\d+)/.exec(url)[1]);
      log.push({ kind: 'range', i, a, b, sig });
      if (log.filter((e) => e.kind === 'range').length > expireAfter && sig === 1) {
        return { status: 403, ok: false, url: finalUrl, headers: new Map(), arrayBuffer: async () => new ArrayBuffer(0) };
      }
    }
    const body = parts[i].slice(a, b + 1);
    return {
      status: 206, ok: true, url: finalUrl,
      headers: new Map([['content-range', `bytes ${a}-${b}/${parts[i].length}`]]),
      arrayBuffer: async () => body.buffer,
    };
  };
  return { fetchFn, log };
}

// The parts of the origin-private file system the cache uses, in memory.
function fakeDir() {
  const files = new Map();
  const handle = (name) => ({
    getFile: async () => {
      const b = files.get(name);
      if (!b) throw new DOMException('missing', 'NotFoundError');
      return { size: b.length, arrayBuffer: async () => b.slice().buffer };
    },
    createSyncAccessHandle: async () => ({
      truncate: () => files.set(name, new Uint8Array(0)),
      write: (bytes, { at }) => {
        const old = files.get(name) || new Uint8Array(0);
        const next = new Uint8Array(Math.max(old.length, at + bytes.length));
        next.set(old);
        next.set(bytes, at);
        files.set(name, next);
      },
      flush() {},
      close() {},
    }),
    move: async (to) => { files.set(to, files.get(name)); files.delete(name); },
  });
  return {
    files,
    async *entries() { for (const name of [...files.keys()]) yield [name, handle(name)]; },
    getFileHandle: async (name, opts = {}) => {
      if (!files.has(name) && !opts.create) throw new DOMException('missing', 'NotFoundError');
      if (!files.has(name)) files.set(name, new Uint8Array(0));
      return handle(name);
    },
    removeEntry: async (name) => { files.delete(name); },
  };
}

const read = (model, file, offset, len) => {
  const out = new Uint8Array(len);
  model.readers()[file].read(offset, out);
  return [...out];
};

test('parts are resolved once to their signed addresses and read by range', async () => {
  const hub = fakeHub();
  const model = new RemoteModel([resolveUrl(0), resolveUrl(1)], { fetchFn: hub.fetchFn });
  assert.deepEqual(await model.open(), [40000, 3000]);
  await model.prefetch([{ file: 0, offset: 100, len: 50, every: true }, { file: 1, offset: 2990, len: 10, every: false }]);
  assert.deepEqual(read(model, 0, 100, 50), [...parts[0].slice(100, 150)]);
  assert.deepEqual(read(model, 1, 2990, 10), [...parts[1].slice(2990)]);
  // Two resolves at open, then only CDN requests.
  assert.equal(hub.log.filter((e) => e.kind === 'resolve').length, 2);
  assert.ok(hub.log.filter((e) => e.kind === 'range').every((e) => e.sig >= 1));
});

test('the engine passes spans as Float64Array quads', async () => {
  const hub = fakeHub();
  const model = new RemoteModel([resolveUrl(0)], { fetchFn: hub.fetchFn });
  await model.open();
  await model.prefetch(new Float64Array([0, 7, 5, 1, 0, 30000, 9, 0]));
  assert.deepEqual(read(model, 0, 30000, 9), [...parts[0].slice(30000, 30009)]);
  assert.deepEqual(read(model, 0, 7, 5), [...parts[0].slice(7, 12)]);
});

test('an expired signature is resolved again and the read retried', async () => {
  const hub = fakeHub({ expireAfter: 1 });
  const model = new RemoteModel([resolveUrl(0)], { fetchFn: hub.fetchFn, sleep: async () => {} });
  await model.open();
  await model.prefetch([{ file: 0, offset: 0, len: 10, every: false }]);
  await model.prefetch([{ file: 0, offset: 10, len: 10, every: false }]);
  assert.deepEqual(read(model, 0, 10, 10), [...parts[0].slice(10, 20)]);
  assert.equal(hub.log.filter((e) => e.kind === 'resolve').length, 2);
  assert.equal(model.takeStats().retries, 1);
});

test('cached ranges are read from the device, and every-token ranges outrank experts', async () => {
  const hub = fakeHub();
  const dir = fakeDir();
  const cache = await RangeCache.open(dir, 100);
  const model = new RemoteModel([resolveUrl(0)], { fetchFn: hub.fetchFn, cache });
  await model.open();
  await model.prefetch([{ file: 0, offset: 0, len: 60, every: true }, { file: 0, offset: 1000, len: 30, every: false }]);
  read(model, 0, 0, 60);
  read(model, 0, 1000, 30);
  // Another expert evicts the first; the every-token range stays; one that does not fit is skipped.
  await model.prefetch([{ file: 0, offset: 2000, len: 30, every: false }, { file: 0, offset: 5000, len: 50, every: true }]);
  assert.ok(cache.has(spanKey(0, 0, 60)) && cache.has(spanKey(0, 2000, 30)));
  assert.ok(!cache.has(spanKey(0, 1000, 30)) && !cache.has(spanKey(0, 5000, 50)));
  const before = hub.log.length;
  await model.prefetch([{ file: 0, offset: 0, len: 60, every: true }]);
  assert.equal(hub.log.length, before, 'served from the cache');
  assert.deepEqual(read(model, 0, 0, 60), [...parts[0].slice(0, 60)]);
  // Reopening finds the entries again; a smaller cap sheds experts first.
  const again = await RangeCache.open(dir, 70);
  assert.ok(again.has(spanKey(0, 0, 60)) && !again.has(spanKey(0, 2000, 30)));
  assert.ok([...dir.files.keys()].every((n) => parseEntry(n)));
});

test('a read nothing prefetched falls back to a synchronous request', async () => {
  const hub = fakeHub();
  class XHR {
    open(method, url, async) { this.url = url; assert.equal(async, false); }
    setRequestHeader(k, v) { this.range = v; }
    send() {
      const [, a, b] = /bytes=(\d+)-(\d+)/.exec(this.range).map(Number);
      this.status = 206;
      this.response = parts[0].slice(a, b + 1).buffer;
    }
  }
  const model = new RemoteModel([resolveUrl(0)], { fetchFn: hub.fetchFn, XHR });
  await model.open();
  assert.deepEqual(read(model, 0, 500, 4), [...parts[0].slice(500, 504)]);
});

test('signed URL expiry', () => {
  const now = Date.now();
  assert.equal(expiresIn(`https://x/y?a=1&Expires=${Math.floor(now / 1000) + 60}`, now) > 58000, true);
  assert.equal(expiresIn('https://x/y', now), 30 * 60 * 1000);
});

test('Stop cancels a read in flight; the next reply can read again', async () => {
  const hub = fakeHub();
  let hang = true;
  const fetchFn = (url, init) => {
    if (!hang || url.startsWith('https://huggingface.co/')) return hub.fetchFn(url, init);
    // A range that would take minutes: settles only when aborted.
    return new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))));
  };
  const model = new RemoteModel([resolveUrl(0)], { fetchFn, sleep: async () => {} });
  await model.open();
  const pending = model.prefetch([{ file: 0, offset: 0, len: 10, every: false }]);
  setTimeout(() => model.cancel(), 10);
  await assert.rejects(pending, /cancelled/);
  hang = false;
  model.resume();
  await model.prefetch([{ file: 0, offset: 0, len: 10, every: false }]);
  assert.deepEqual(read(model, 0, 0, 10), [...parts[0].slice(0, 10)]);
});
