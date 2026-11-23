import test from 'node:test';
import assert from 'node:assert/strict';
import { PIECE, fetchPieces, packPieces, parseResolveUrl, sourcesFor, unpackPieces } from '../web/mirrors.mjs';

const HF = 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf';

// A file of `size` bytes whose byte at offset o is o % 251, served by URL with optional faults.
function server(size, faults = {}) {
  const calls = [];
  const fetchFn = async (url, { headers, signal }) => {
    if (signal?.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.Range);
    const start = +a, end = Math.min(+b + 1, size);
    calls.push({ url, start });
    const f = faults[url];
    if (f?.status) return { status: f.status, arrayBuffer: async () => new ArrayBuffer(0) };
    if (f?.down) throw new TypeError('Failed to fetch');
    const bytes = new Uint8Array(end - start);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (start + i) % 251;
    if (f?.corrupt) bytes[0] ^= 1;
    return { status: 206, arrayBuffer: async () => bytes.buffer };
  };
  return { fetchFn, calls };
}
const goodBytes = (start, bytes) => bytes.every((v, i) => v === (start + i) % 251);
const noWait = async () => {};

function sink(size) {
  const out = new Uint8Array(size);
  const written = [];
  return { out, written, write: async (start, bytes) => { out.set(bytes, start); written.push(start); } };
}

test('Hugging Face URLs map to their ModelScope copy, and mirrors need a manifest', () => {
  assert.deepEqual(parseResolveUrl(HF), { repo: 'unsloth/Qwen3.5-4B-GGUF', revision: 'main', file: 'Qwen3.5-4B-Q4_K_M.gguf' });
  assert.equal(sourcesFor(HF).length, 1);
  assert.equal(sourcesFor(HF, { verified: true, mirrors: false }).length, 1);
  const both = sourcesFor(HF, { verified: true });
  assert.deepEqual(both.map((s) => [s.name, s.trusted]), [['Hugging Face', true], ['ModelScope', false]]);
  assert.equal(both[1].url, 'https://www.modelscope.cn/models/unsloth/Qwen3.5-4B-GGUF/resolve/master/Qwen3.5-4B-Q4_K_M.gguf');
  assert.equal(sourcesFor('https://example.com/m.gguf', { verified: true }).length, 1);
});

test('done pieces survive a round trip through the download record', () => {
  const done = new Set([0, 3, 8, 9, 64]);
  assert.deepEqual([...unpackPieces(packPieces(done, 70), 70)].sort((a, b) => a - b), [0, 3, 8, 9, 64]);
  assert.equal(unpackPieces(undefined, 10).size, 0);
  assert.equal(unpackPieces('not base64!', 10).size, 0);
});

test('pieces come from both sources and land in place', async () => {
  const size = PIECE * 5 + 1234;
  const { fetchFn, calls } = server(size);
  const s = sink(size);
  const from = new Set();
  await fetchPieces({ size, sources: sourcesFor(HF, { verified: true }), write: s.write, fetchFn, wait: noWait,
    verify: async (start, bytes) => goodBytes(start, bytes), onPiece: (i, b, name) => from.add(name) });
  assert.ok(goodBytes(0, s.out));
  assert.equal(s.written.length, 6);
  assert.deepEqual([...from].sort(), ['Hugging Face', 'ModelScope']);
  assert.equal(new Set(calls.map((c) => c.start)).size, 6);
});

test('a mirror that sends a wrong piece is dropped and the piece comes from Hugging Face', async () => {
  const size = PIECE * 6;
  const [hf, ms] = sourcesFor(HF, { verified: true });
  const { fetchFn, calls } = server(size, { [ms.url]: { corrupt: true } });
  const s = sink(size);
  await fetchPieces({ size, sources: [hf, ms], write: s.write, fetchFn, wait: noWait, verify: async (start, bytes) => goodBytes(start, bytes) });
  assert.ok(goodBytes(0, s.out));
  // ModelScope's three connections each got one wrong piece at most, then stopped.
  assert.ok(calls.filter((c) => c.url === ms.url).length <= 3);
});

test('without a manifest a mirror is never written from', async () => {
  const size = PIECE * 2;
  const mirror = { name: 'Mirror', url: 'https://mirror.example/m.gguf', connections: 2, trusted: false };
  const { fetchFn } = server(size);
  const s = sink(size);
  const from = [];
  await fetchPieces({ size, sources: [{ ...sourcesFor(HF)[0] }, mirror], write: s.write, fetchFn, wait: noWait, onPiece: (i, b, n) => from.push(n) });
  assert.deepEqual([...new Set(from)], ['Hugging Face']);
});

test('a resumed download fetches only the missing pieces', async () => {
  const size = PIECE * 4;
  const { fetchFn, calls } = server(size);
  const s = sink(size);
  await fetchPieces({ size, sources: sourcesFor(HF), done: new Set([0, 2]), write: s.write, fetchFn, wait: noWait });
  assert.deepEqual(calls.map((c) => c.start / PIECE).sort(), [1, 3]);
});

test('when Hugging Face is down the mirror finishes the file; when everything is down it fails', async () => {
  const size = PIECE * 3;
  const [hf, ms] = sourcesFor(HF, { verified: true });
  const down = server(size, { [hf.url]: { status: 502 } });
  const s = sink(size);
  await fetchPieces({ size, sources: [hf, ms], write: s.write, fetchFn: down.fetchFn, wait: noWait, verify: async (st, b) => goodBytes(st, b) });
  assert.ok(goodBytes(0, s.out));
  const all = server(size, { [hf.url]: { status: 502 }, [ms.url]: { down: true } });
  await assert.rejects(fetchPieces({ size, sources: [hf, ms], write: sink(size).write, fetchFn: all.fetchFn, wait: noWait, verify: async () => true }));
});

test('a wrong piece from Hugging Face itself stops the download, and aborting stops it too', async () => {
  const size = PIECE * 2;
  const bad = server(size, { [HF]: { corrupt: true } });
  await assert.rejects(fetchPieces({ size, sources: sourcesFor(HF), write: sink(size).write, fetchFn: bad.fetchFn, wait: noWait,
    verify: async (st, b) => goodBytes(st, b) }), /does not match/);
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(fetchPieces({ size, sources: sourcesFor(HF), write: sink(size).write, fetchFn: server(size).fetchFn, wait: noWait, signal: ac.signal }),
    (e) => e.name === 'AbortError');
});
