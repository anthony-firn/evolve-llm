import test from 'node:test';
import assert from 'node:assert/strict';
import { CHUNK, Manifests, PAGE_CHUNKS, RangeSet, chunkSpan, decodeFrame, encodeFrame, parseControl, sha256 } from '../web/swarm-protocol.mjs';
import { Signal, SIGNAL_KIND } from '../web/swarm-signal.mjs';
import { Swarm } from '../web/swarm.mjs';

const hex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');

// A file of 5.5 chunks and its manifest, built the way tools/chunk_manifest.py builds them.
const file = new Uint8Array(Math.floor(5.5 * CHUNK)).map((_, i) => (i * 31 + (i >> 9)) & 255);
async function manifestFor(bytes) {
  const page = new Uint8Array(Math.ceil(bytes.length / CHUNK) * 32);
  for (let c = 0; c * CHUNK < bytes.length; c++) page.set(await sha256(bytes.subarray(c * CHUNK, (c + 1) * CHUNK)), c * 32);
  const sha = hex(await sha256(bytes));
  const index = { format: 'evolve-llm-chunks/1', sha256: sha, size: bytes.length, chunk: CHUNK, page_chunks: PAGE_CHUNKS, pages: [hex(await sha256(page))] };
  const files = { [`${sha}/index.json`]: JSON.stringify(index), [`${sha}/0.bin`]: page };
  const fetchFn = async (url) => {
    const body = files[url.replace('https://m.example/v1/', '')];
    if (!body) return { ok: false, status: 404 };
    return { ok: true, status: 200, json: async () => JSON.parse(body), arrayBuffer: async () => (typeof body === 'string' ? new TextEncoder().encode(body) : body).slice().buffer };
  };
  return { sha, files, manifests: new Manifests('https://m.example/v1/', { fetchFn }) };
}

test('chunk ranges merge and answer coverage', () => {
  const r = new RangeSet([[0, 2], [5, 7]]);
  r.add(2, 5);
  assert.deepEqual(r.toJSON(), [[0, 7]]);
  assert.ok(r.has(1, 6) && !r.has(6, 8));
  assert.deepEqual(chunkSpan(CHUNK - 1, 2), [0, 2]);
});

test('frames round-trip and malformed or oversized control messages are refused', () => {
  const f = decodeFrame(encodeFrame(7, 65536, new Uint8Array([1, 2, 3])));
  assert.deepEqual([f.id, f.at, [...f.bytes]], [7, 65536, [1, 2, 3]]);
  const sha = 'a'.repeat(64);
  assert.ok(parseControl(JSON.stringify({ t: 'req', id: 1, sha, s: 0, e: 16 })));
  assert.equal(parseControl(JSON.stringify({ t: 'req', id: 1, sha, s: 0, e: 17 })), null);
  assert.equal(parseControl(JSON.stringify({ t: 'have', sha: 'x', ranges: [] })), null);
  assert.equal(parseControl('not json'), null);
});

test('chunks are checked against their manifest page, and pages against the index', async () => {
  const { sha, files, manifests } = await manifestFor(file);
  assert.ok(await manifests.verify(sha, 0, 3, file.subarray(0, 3 * CHUNK)));
  assert.ok(await manifests.verify(sha, 5, 6, file.subarray(5 * CHUNK)), 'the short last chunk');
  const bad = file.slice(0, 2 * CHUNK);
  bad[CHUNK + 5] ^= 1;
  assert.equal(await manifests.verify(sha, 0, 2, bad), false);
  // A page that does not match the index is refused.
  files[`${sha}/0.bin`] = new Uint8Array(files[`${sha}/0.bin`].length);
  const fresh = new Manifests('https://m.example/v1/', { fetchFn: manifests.fetchFn });
  await assert.rejects(fresh.verify(sha, 0, 1, file.subarray(0, CHUNK)), /does not match/);
  assert.equal(await fresh.index('b'.repeat(64)), null, 'no manifest is not an error');
});

// Two data channels joined in memory.
function channelPair() {
  const mk = () => ({ readyState: 'open', bufferedAmount: 0, binaryType: 'arraybuffer', addEventListener() {}, close() { this.readyState = 'closed'; } });
  const a = mk();
  const b = mk();
  const deliver = (to, data) => queueMicrotask(() => to.onmessage?.({ data: typeof data === 'string' ? data : data.slice().buffer }));
  a.send = (d) => deliver(b, d);
  b.send = (d) => deliver(a, d);
  return [a, b];
}

function linked(swarmA, swarmB) {
  const [da, db] = channelPair();
  const peer = (s, id) => {
    const p = { id, pc: { close() {} }, dc: null, have: new Map(), pending: new Map(), busy: 0 };
    s.peers.set(id, p);
    return p;
  };
  const pa = peer(swarmA, 'b'.repeat(64));
  const pb = peer(swarmB, 'a'.repeat(64));
  swarmA.channel(pa, da);
  swarmB.channel(pb, db);
  da.onopen();
  db.onopen();
  return { pa, pb };
}

const quietSignal = () => ({ pubkey: 'c'.repeat(64), start() {}, announce: async () => {}, send: async () => {}, close() {} });
const settle = () => new Promise((r) => setTimeout(r, 20));

test('a peer fetches chunks another serves from its cache; a peer that lies is dropped', async () => {
  const { sha, manifests } = await manifestFor(file);
  const files = [{ sha, size: file.length }];
  const seed = new Swarm({ files, manifests, signal: quietSignal(), uploadBytesPerSec: 0, read: async (_, s, e) => file.slice(s * CHUNK, Math.min(e * CHUNK, file.length)) });
  const leech = new Swarm({ files, manifests, signal: quietSignal(), read: async () => null });
  seed.addHave(sha, 0, 6);
  linked(seed, leech);
  await settle();
  const got = await leech.get(sha, 2, 6);
  assert.deepEqual([...got], [...file.subarray(2 * CHUNK)]);
  assert.equal(leech.stats.fromPeers, got.byteLength);
  assert.equal(seed.stats.toPeers, got.byteLength);
  assert.equal(await leech.get(sha, 0, 17), null, 'more than one request is refused');

  const liar = new Swarm({ files, manifests, signal: quietSignal(), uploadBytesPerSec: 0, read: async (_, s, e) => new Uint8Array(Math.min(e * CHUNK, file.length) - s * CHUNK) });
  const victim = new Swarm({ files, manifests, signal: quietSignal(), read: async () => null });
  liar.addHave(sha, 0, 6);
  const { pb } = linked(liar, victim);
  await settle();
  assert.equal(await victim.get(sha, 0, 2), null);
  assert.equal(victim.stats.rejected, 1);
  assert.ok(victim.banned.has(pb.id) && !victim.peers.has(pb.id));
});

// A relay bus: every socket gets the events matching its filters.
function fakeRelay() {
  const sockets = new Set();
  const matches = (f, ev) => f.kinds.includes(ev.kind)
    && (!f['#t'] || ev.tags.some((t) => t[0] === 't' && f['#t'].includes(t[1])))
    && (!f['#p'] || ev.tags.some((t) => t[0] === 'p' && f['#p'].includes(t[1])));
  class WS {
    constructor() {
      this.readyState = 1;
      this.filters = [];
      sockets.add(this);
      queueMicrotask(() => this.onopen?.());
    }
    send(text) {
      const m = JSON.parse(text);
      if (m[0] === 'REQ') this.filters = m.slice(2);
      if (m[0] === 'EVENT') for (const s of sockets) if (s.filters.some((f) => matches(f, m[1]))) queueMicrotask(() => s.onmessage?.({ data: JSON.stringify(['EVENT', 'swarm', m[1]]) }));
    }
    close() { sockets.delete(this); }
  }
  return WS;
}

test('signaling: signed presence reaches peers of the same file; offers arrive only decrypted by their recipient', async () => {
  const WS = fakeRelay();
  const got = { a: [], b: [], c: [] };
  const mk = (k) => new Signal({ relays: ['wss://r.example'], WebSocketImpl: WS, onEvent: (kind, from, data) => got[k].push({ kind, from, data }) });
  const a = mk('a');
  const b = mk('b');
  const c = mk('c');
  const sha = 'd'.repeat(64);
  a.start([sha]);
  b.start([sha]);
  c.start(['e'.repeat(64)]);
  await settle();
  await a.announce();
  await settle();
  assert.deepEqual(got.b.map((e) => [e.kind, e.from, e.data]), [['presence', a.pubkey, [sha]]]);
  assert.equal(got.c.length, 0, 'another file is another swarm');
  await b.send(a.pubkey, { type: 'offer', sdp: 'v=0 secret addresses' });
  await settle();
  assert.deepEqual(got.a.at(-1), { kind: 'message', from: b.pubkey, data: { type: 'offer', sdp: 'v=0 secret addresses' } });
  // A forged event (bad signature) is ignored.
  const before = got.b.length;
  const forged = { pubkey: a.pubkey, created_at: Math.floor(Date.now() / 1000), kind: SIGNAL_KIND, tags: [['t', `evolve-swarm:${sha}`]], content: '', id: '0'.repeat(64), sig: '0'.repeat(128) };
  await b.receive(JSON.stringify(['EVENT', 'swarm', forged]));
  assert.equal(got.b.length, before);
  for (const s of [a, b, c]) s.close();
});
