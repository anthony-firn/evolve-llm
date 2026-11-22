import test from 'node:test';
import assert from 'node:assert/strict';
import { Helpers, Helping, Link, Pairing, newCode, normalizeCode } from '../web/devices.mjs';
import { capacityOf, planFill, planSplit } from '../web/split.mjs';
import { Signal } from '../web/swarm-signal.mjs';

const settle = (ms = 30) => new Promise((r) => setTimeout(r, ms));

test('layers are shared in proportion to memory, with the embedding and head on the chat device', () => {
  const lb = { n_layer: 10, embed: 50, head: 50, tied: false, layers: Array(10).fill(100) };
  assert.deepEqual(planSplit(lb, [550, 500]), [{ first: 0, end: 5 }, { first: 5, end: 10 }]);
  // A device with little memory gets no layers; one with none of its own keeps only the ends.
  assert.deepEqual(planSplit(lb, [1100, 10]), [{ first: 0, end: 10 }, { first: 10, end: 10 }]);
  assert.deepEqual(planSplit(lb, [100, 500, 500]), [{ first: 0, end: 0 }, { first: 0, end: 5 }, { first: 5, end: 10 }]);
  assert.equal(capacityOf('auto'), 4 * 2 ** 30);
  assert.equal(capacityOf('6'), 6 * 2 ** 30);
  assert.match(newCode(), /^[A-Z2-9]{4}(-[A-Z2-9]{4}){3}$/);
  assert.equal(normalizeCode('abcd-efgh'), 'ABCDEFGH');
});

// Two data channels joined in memory (opened by `open()`).
function channelPair() {
  const mk = () => ({ readyState: 'connecting', bufferedAmount: 0, addEventListener() {}, close() { this.readyState = 'closed'; this.peer.readyState = 'closed'; this.onclose?.(); this.peer.onclose?.(); } });
  const a = mk();
  const b = mk();
  a.peer = b;
  b.peer = a;
  const deliver = (to, data) => queueMicrotask(() => to.onmessage?.({ data: typeof data === 'string' ? data : data.slice().buffer }));
  a.send = (d) => deliver(b, d);
  b.send = (d) => deliver(a, d);
  const open = () => {
    a.readyState = b.readyState = 'open';
    a.onopen?.();
    b.onopen?.();
  };
  return [a, b, open];
}

test('a link carries messages and payloads of many frames, in order', async () => {
  const [a, b, open] = channelPair();
  open();
  const got = [];
  const la = new Link(a);
  new Link(b, (m, bytes) => got.push([m.t, bytes ? bytes.length : 0, bytes ? bytes[bytes.length - 1] : null]));
  const big = new Uint8Array(300000).map((_, i) => i & 255);
  await Promise.all([la.send({ t: 'one' }, big), la.send({ t: 'two' }), la.send({ t: 'three' }, new Uint8Array([7]))]);
  await settle();
  assert.deepEqual(got, [['one', 300000, 299999 & 255], ['two', 0, null], ['three', 1, 7]]);
});

// Peer connections joined in memory: the offer's text names the offering connection.
function fakeRTC() {
  const offers = new Map();
  let n = 0;
  return class FakePC {
    constructor() {
      this.iceGatheringState = 'complete';
      this.connectionState = 'new';
    }
    createDataChannel() {
      [this.mine, this.theirs, this.open] = channelPair();
      return this.mine;
    }
    async createOffer() { return { type: 'offer', sdp: `offer-${++n}` }; }
    async createAnswer() { return { type: 'answer', sdp: `answer-${++n}` }; }
    async setLocalDescription(d) {
      this.localDescription = d;
      if (d.type === 'offer') offers.set(d.sdp, this);
    }
    async setRemoteDescription(d) {
      if (d.type === 'offer') {
        offers.get(d.sdp).answerer = this;
        return;
      }
      // The answer reached the offering side: the channel opens on both.
      queueMicrotask(() => {
        this.answerer?.ondatachannel?.({ channel: this.theirs });
        this.open();
      });
    }
    close() { this.connectionState = 'closed'; }
  };
}

// A relay bus for signaling.
function fakeRelay() {
  const sockets = new Set();
  const matches = (f, ev) => f.kinds.includes(ev.kind)
    && (!f['#t'] || ev.tags.some((t) => t[0] === 't' && f['#t'].includes(t[1])))
    && (!f['#p'] || ev.tags.some((t) => t[0] === 'p' && f['#p'].includes(t[1])));
  return class WS {
    constructor() {
      this.readyState = 1;
      this.filters = [];
      sockets.add(this);
      queueMicrotask(() => this.onopen?.());
    }
    send(text) {
      const m = JSON.parse(text);
      if (m[0] === 'REQ') this.filters = m.slice(2);
      if (m[0] === 'EVENT') for (const s of sockets) if (s !== this && s.filters.some((f) => matches(f, m[1]))) queueMicrotask(() => s.onmessage?.({ data: JSON.stringify(['EVENT', 'x', m[1]]) }));
    }
    close() { sockets.delete(this); }
  };
}

// A worker whose "layers" add 1 to every value of the hidden state.
function fakeWorker() {
  const listeners = new Set();
  const emit = (data) => queueMicrotask(() => listeners.forEach((l) => l({ data })));
  return {
    loads: [],
    addEventListener: (_, l) => listeners.add(l),
    removeEventListener: (_, l) => listeners.delete(l),
    postMessage(m) {
      if (m.type === 'load') {
        this.loads.push(m);
        emit({ type: 'progress', loaded: 1, total: 2, phase: 'upload' });
        emit({ type: 'stage-loaded', stage: m.stage, placement: { streaming: false } });
      } else if (m.type === 'stage-forward') {
        const h = new Float32Array(m.hidden).map((x) => x + 1);
        emit({ type: 'stage-result', id: m.id, hidden: h.buffer });
      }
    },
  };
}

test('pairing: devices with the code connect (never two helpers), and a pass goes through each helper in layer order', async () => {
  const WS = fakeRelay();
  const RTC = fakeRTC();
  const code = newCode();
  const signal = () => new Signal({ relays: ['wss://r'], WebSocketImpl: WS, prefix: 'evolve-pair:' });
  const helpers = new Helpers();
  const main = new Pairing({ code, role: 'main', RTC, iceServers: [], signal: signal(), onLink: (link, id) => helpers.add(link, id), onClose: (id) => helpers.remove(id) });
  const workers = [fakeWorker(), fakeWorker()];
  const helpings = [];
  const mkHelper = (w, name, c = code) => new Pairing({
    code: c, role: 'helper', RTC, iceServers: [], signal: signal(),
    onLink: (link) => helpings.push(new Helping({ link, worker: w, about: { name, capacity: 1e9, gpu: 'test' }, settings: () => ({ gpu: 'high-performance', gpuMemory: 'auto', cacheGb: 5 }) })),
  });
  const stranger = new Pairing({ code: newCode(), role: 'helper', RTC, iceServers: [], signal: signal(), onLink: () => assert.fail('a device with another code connected') });
  const pages = [main, mkHelper(workers[0], 'Laptop'), mkHelper(workers[1], 'Desktop'), stranger];
  for (const p of pages) await p.start();
  for (const p of pages) p.announce(true);
  await settle(300);
  for (const p of pages) p.announce(true);
  await settle(300);
  assert.deepEqual(helpers.ready().map((h) => h.name).sort(), ['Desktop', 'Laptop']);
  assert.equal(pages[1].peers.size, 1, 'helpers connect only to the chat device');

  const [x, y] = helpers.ready().sort((a, b) => a.name.localeCompare(b.name));
  const source = { kind: 'hf', repo: 'r', file: 'f.gguf', parts: [] };
  await Promise.all([helpers.load(y, source, { first: 4, end: 8, embed: false, head: false }, 2048), helpers.load(x, source, { first: 8, end: 12, embed: false, head: false }, 2048)]);
  assert.deepEqual(workers.map((w) => w.loads.length), [1, 1]);
  assert.equal(workers[0].loads[0].source.stream, true, 'helpers read their layers from the network');
  const out = new Float32Array(await helpers.forward(new Float32Array([1, 2, 3]).buffer, 1, 0));
  assert.deepEqual([...out], [3, 4, 5]);
  for (const p of pages) p.stop();
});

test('a phone drives the computer\'s worker: messages relay both ways, files and page services stay local', async () => {
  const { Hosting, RemoteWorker } = await import('../web/devices.mjs');
  const [a, b, open] = channelPair();
  // The computer's worker: answers hello and generate; tells its page about cached pieces.
  const listeners = new Set();
  const emit = (data) => queueMicrotask(() => listeners.forEach((l) => l({ data })));
  const posted = [];
  const worker = {
    addEventListener: (_, l) => listeners.add(l),
    removeEventListener: (_, l) => listeners.delete(l),
    postMessage(m) {
      posted.push(m);
      if (m.type === 'hello') emit({ type: 'adapter', adapter: { ok: true, description: 'Test GPU' } });
      if (m.type === 'load') { emit({ type: 'swarm-have', file: 0, offset: 0, len: 1 }); emit({ type: 'loaded', model: {}, chat: { name: 'Tiny' }, source: { kind: 'hf', key: 'k' } }); }
      if (m.type === 'generate') { emit({ type: 'token', text: 'Hi' }); emit({ type: 'done' }); }
    },
  };
  const hosting = new Hosting({ worker });
  const states = [];
  const phone = new RemoteWorker({ onConnection: (s, info) => states.push([s, info.loaded?.chat?.name ?? null]) });
  const seen = [];
  phone.onmessage = (e) => seen.push(e.data.type);
  phone.postMessage({ type: 'hello', gpu: 'high-performance' }); // before connecting: queued
  phone.postMessage({ type: 'load', source: { kind: 'file', files: [] } });
  open();
  hosting.attach(new Link(a));
  phone.attach(new Link(b));
  await settle();
  phone.postMessage({ type: 'swarm', on: true });
  phone.postMessage({ type: 'load', source: { kind: 'hf', key: 'k' }, share: true, split: [1e9] });
  phone.postMessage({ type: 'generate', messages: [] });
  await settle();
  assert.deepEqual(posted.map((m) => m.type), ['hello', 'load', 'generate'], 'page services are not the phone\'s to switch on');
  assert.equal(posted[1].share, false);
  assert.equal(posted[1].split, null);
  assert.deepEqual(seen, ['error', 'adapter', 'loaded', 'token', 'done'], 'phone files are refused; the swarm bookkeeping stays on the computer');
  assert.deepEqual(states, [['connected', null]]);
  // A phone that reconnects learns which model the computer has loaded.
  const [c, d, open2] = channelPair();
  open2();
  hosting.attach(new Link(c));
  phone.attach(new Link(d));
  await settle();
  assert.deepEqual(states.at(-1), ['connected', 'Tiny']);
});


test('phone startup waits for the host listener and repeats hello after reconnecting', async () => {
  const { RemoteWorker } = await import('../web/devices.mjs');
  const phone = new RemoteWorker();
  const sent = [];
  const first = { open: true, send: async msg => sent.push(msg.m) };
  phone.postMessage({type:'hello',gpu:'high-performance'});
  phone.attach(first);
  phone.postMessage({type:'cache'});
  assert.deepEqual(sent, [], 'channel opening is not proof the host can receive messages');
  first.onMessage({t:'host',loaded:null});
  assert.deepEqual(sent.map(m=>m.type), ['hello','cache']);
  phone.detach(first);
  const next = { open: true, send: async msg => sent.push(msg.m) };
  phone.attach(next);
  assert.equal(sent.length,2);
  next.onMessage({t:'host',loaded:null});
  assert.deepEqual(sent.map(m=>m.type),['hello','cache','hello'], 'reconnection refreshes GPU and cached models');
});

test('two GPUs: the faster takes as many layers as fit with their attention cache, the other the rest', () => {
  // Qwen3 4B-like: 36 layers of 60 MB in the file, a 300 MB embedding (tied), 1.35x on the GPU,
  // 72 KB of attention cache per token per layer.
  const lb = { n_layer: 36, embed: 300e6, head: 0, tied: true, layers: Array(36).fill(60e6), kv_per_token: 2 * 1024 * 2 * 2 };
  const per = 60e6 * 1.35 + lb.kv_per_token * 8192; // ~115 MB a layer
  const [fast, slow] = planFill(lb, 1.75 * 2 ** 30, 8192, 1.35);
  assert.equal(fast.first, 0);
  assert.equal(fast.end, Math.floor((1.75 * 2 ** 30 * 0.9 - 300e6 * 1.35) / per));
  assert.deepEqual(slow, { first: fast.end, end: 36 });
  // Everything fits: nothing for the other GPU.
  assert.deepEqual(planFill(lb, 8 * 2 ** 30, 8192, 1.35)[1], { first: 36, end: 36 });
  // Not even the embedding fits: every layer on the other GPU.
  assert.deepEqual(planFill(lb, 0.2 * 2 ** 30, 8192, 1.35), [{ first: 0, end: 0 }, { first: 0, end: 36 }]);
});
