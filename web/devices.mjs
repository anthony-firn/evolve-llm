// One model on several of a person's devices. The device the chat runs on keeps the embedding,
// the head and the first layers; each paired device ("helper") runs a later run of layers. For
// every pass the hidden state goes through the helpers in order and comes back.
//
// Pairing: the chat device shows a pairing link holding a code; a device that opens it finds
// the chat device through public Nostr relays, on a topic derived from the code. Each side
// proves it knows the code (an HMAC of its role and per-visit key), so seeing the topic on a
// relay is not enough to join, and only a chat device and a helper connect (not two helpers). Connection offers are encrypted to their recipient, and the data channel between the
// devices is WebRTC (encrypted end to end).

import { FRAME, decodeFrame, encodeFrame } from './swarm-protocol.mjs';
import { Signal } from './swarm-signal.mjs';
import { ICE_SERVERS } from './swarm.mjs';

export const PAIR_PROTOCOL = 'evolve-pair/1';
const ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
const MAX_PAYLOAD = 2 ** 31;

/** A new pairing code: 16 characters (80 bits), shown in groups of four. */
export function newCode() {
  const r = crypto.getRandomValues(new Uint8Array(16));
  const c = Array.from(r, (b) => ALPHABET[b & 31]).join('');
  return c.match(/.{4}/g).join('-');
}

export const normalizeCode = (c) => String(c || '').toUpperCase().replace(/[^A-Z0-9]/g, '');

const hex = (b) => Array.from(new Uint8Array(b), (x) => x.toString(16).padStart(2, '0')).join('');

async function hmac(code, text) {
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(`${PAIR_PROTOCOL}:${code}`), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return hex(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(text)));
}

async function topicFor(code) {
  return hex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`${PAIR_PROTOCOL}:topic:${code}`)));
}

/**
 * Messages over a data channel: a JSON header, then its payload (if any) in binary frames.
 * Sends are queued so two payloads never interleave.
 */
export class Link {
  constructor(dc, onMessage = () => {}) {
    this.dc = dc;
    this.onMessage = onMessage;
    this.incoming = new Map();
    this.seq = 0;
    this.queue = Promise.resolve();
    dc.binaryType = 'arraybuffer';
    dc.bufferedAmountLowThreshold = 1 << 20;
    dc.onmessage = (m) => this.receive(m.data);
  }

  get open() {
    return this.dc.readyState === 'open';
  }

  send(msg, payload = null) {
    const body = payload ? new Uint8Array(payload.buffer ?? payload, payload.byteOffset ?? 0, payload.byteLength) : null;
    const run = async () => {
      if (!this.open) throw new Error('The other device is not connected.');
      const mid = ++this.seq;
      this.dc.send(JSON.stringify({ ...msg, mid, bytes: body ? body.byteLength : 0 }));
      for (let at = 0; body && at < body.byteLength; at += FRAME) {
        if (this.dc.bufferedAmount > 8 << 20) {
          await new Promise((r) => {
            this.dc.addEventListener('bufferedamountlow', r, { once: true });
            this.dc.addEventListener('close', r, { once: true });
          });
        }
        if (!this.open) throw new Error('The other device disconnected.');
        this.dc.send(encodeFrame(mid, at, body.subarray(at, Math.min(at + FRAME, body.byteLength))));
      }
    };
    const p = this.queue.then(run);
    this.queue = p.catch(() => {});
    return p;
  }

  receive(data) {
    if (typeof data === 'string') {
      let m;
      try { m = JSON.parse(data); } catch { return; }
      if (!m || typeof m !== 'object' || typeof m.t !== 'string') return;
      const n = Number(m.bytes) || 0;
      if (n > 0 && n < MAX_PAYLOAD) this.incoming.set(m.mid, { msg: m, buf: new Uint8Array(n), got: 0 });
      else if (n === 0) this.onMessage(m, null);
      return;
    }
    const f = decodeFrame(data);
    const r = this.incoming.get(f.id);
    if (!r || f.at + f.bytes.byteLength > r.buf.byteLength) return;
    r.buf.set(f.bytes, f.at);
    r.got += f.bytes.byteLength;
    if (r.got >= r.buf.byteLength) {
      this.incoming.delete(f.id);
      this.onMessage(r.msg, r.buf);
    }
  }

  close() {
    try { this.dc.close(); } catch { /* closed */ }
  }
}

function gathered(pc, ms) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    pc.addEventListener('icegatheringstatechange', () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); }
    });
  });
}

/** Finds the other devices with the same code and opens a Link to each. */
export class Pairing {
  /**
   * `role`: 'main' (the chat device) and 'helper' (runs layers for it), or 'host' (a computer
   * running models) and 'remote' (a phone using it). `onLink(link, peer)` for each device of the
   * matching role that connects; `onClose(peer)` when one goes.
   */
  constructor({ code, role, onLink, onClose = () => {}, relays, iceServers = ICE_SERVERS, RTC = globalThis.RTCPeerConnection, signal = null }) {
    this.code = normalizeCode(code);
    this.role = role;
    this.other = { main: 'helper', helper: 'main', host: 'remote', remote: 'host' }[role];
    if (!this.other) throw new Error(`unknown pairing role ${role}`);
    this.onLink = onLink;
    this.onClose = onClose;
    this.relays = relays;
    this.iceServers = iceServers;
    this.RTC = RTC;
    this.signal = signal;
    this.peers = new Map();
    this.lastAnnounce = 0;
  }

  async start() {
    this.topic = await topicFor(this.code);
    const onEvent = (kind, from, data, content) => this.onSignal(kind, from, data, content).catch(() => {});
    if (this.signal) this.signal.onEvent = onEvent;
    else this.signal = new Signal({ ...(this.relays && { relays: this.relays }), prefix: 'evolve-pair:', onEvent });
    this.proof = await hmac(this.code, `${this.role}:${this.signal.pubkey}`);
    this.signal.start([this.topic]);
    this.timer = setInterval(() => this.announce(true), 20000);
    setTimeout(() => this.announce(true), 500);
  }

  announce(force = false) {
    const now = Date.now();
    if (!force && now - this.lastAnnounce < 3000) return;
    this.lastAnnounce = now;
    this.signal.announce(`${this.role}.${this.proof}`).catch(() => {});
  }

  /** Whether `peer` (of the other role) showed it knows the code. */
  async knows(peer, proof) {
    return typeof proof === 'string' && proof === await hmac(this.code, `${this.other}:${peer}`);
  }

  async onSignal(kind, from, data, content) {
    if (kind === 'presence') {
      const [role, proof] = String(content).split('.');
      if (role !== this.other || !data.includes(this.topic) || this.peers.has(from) || !(await this.knows(from, proof))) return;
      // The page with the smaller key makes the offer; the other answers its announcement.
      if (this.signal.pubkey < from) await this.connect(from).catch(() => this.drop(from));
      else this.announce();
    } else if (data?.type === 'offer' && typeof data.sdp === 'string' && (await this.knows(from, data.proof))) {
      await this.accept(from, data.sdp).catch(() => this.drop(from));
    } else if (data?.type === 'answer' && typeof data.sdp === 'string' && (await this.knows(from, data.proof))) {
      await this.peers.get(from)?.pc.setRemoteDescription({ type: 'answer', sdp: data.sdp }).catch(() => this.drop(from));
    }
  }

  peer(from) {
    const pc = new this.RTC({ iceServers: this.iceServers });
    const p = { pc, link: null };
    this.peers.set(from, p);
    pc.onconnectionstatechange = () => {
      if (['failed', 'closed'].includes(pc.connectionState)) this.drop(from);
    };
    return p;
  }

  channel(from, p, dc) {
    dc.onopen = () => {
      p.link = new Link(dc);
      this.onLink(p.link, from);
    };
    dc.onclose = () => this.drop(from);
  }

  async connect(from) {
    const p = this.peer(from);
    this.channel(from, p, p.pc.createDataChannel('evolve-pair', { ordered: true }));
    await p.pc.setLocalDescription(await p.pc.createOffer());
    await gathered(p.pc, 3000);
    await this.signal.send(from, { type: 'offer', sdp: p.pc.localDescription.sdp, proof: this.proof });
  }

  async accept(from, sdp) {
    if (this.peers.has(from)) return;
    const p = this.peer(from);
    p.pc.ondatachannel = (e) => this.channel(from, p, e.channel);
    await p.pc.setRemoteDescription({ type: 'offer', sdp });
    await p.pc.setLocalDescription(await p.pc.createAnswer());
    await gathered(p.pc, 3000);
    await this.signal.send(from, { type: 'answer', sdp: p.pc.localDescription.sdp, proof: this.proof });
  }

  drop(from) {
    const p = this.peers.get(from);
    if (!p) return;
    this.peers.delete(from);
    p.link?.close();
    try { p.pc.close(); } catch { /* closed */ }
    if (p.link) this.onClose(from);
  }

  stop() {
    clearInterval(this.timer);
    for (const id of [...this.peers.keys()]) this.drop(id);
    this.signal?.close();
  }
}

/**
 * The chat device's side: the helpers that joined, loading their layers, and passing hidden
 * states through them. `onChange()` after a helper joins, leaves or reports progress.
 */
export class Helpers {
  constructor({ onChange = () => {} } = {}) {
    this.onChange = onChange;
    this.list = []; // {id, link, name, capacity, gpu, stage, state, progress, pending}
    this.seq = 0;
  }

  /** A helper's link opened. */
  add(link, id) {
    const h = { id, link, name: 'Device', capacity: 0, gpu: '', stage: null, state: 'connecting', progress: null, pending: new Map() };
    this.list.push(h);
    link.onMessage = (m, bytes) => this.message(h, m, bytes);
    // The helper answers with its own hello; ask again until it has.
    const hello = () => link.send({ t: 'hello', v: PAIR_PROTOCOL, role: 'main' }).catch(() => {});
    hello();
    let tries = 0;
    const retry = setInterval(() => {
      if (h.state !== 'connecting' || !this.list.includes(h) || ++tries > 10) clearInterval(retry);
      else hello();
    }, 3000);
    this.onChange();
  }

  remove(id) {
    const h = this.list.find((x) => x.id === id);
    if (!h) return;
    this.list = this.list.filter((x) => x !== h);
    for (const p of h.pending.values()) p.reject(new Error(`${h.name} disconnected.`));
    this.onChange();
  }

  /** Helpers ready to take layers. */
  ready() {
    return this.list.filter((h) => h.state === 'ready' || h.state === 'loaded');
  }

  message(h, m, bytes) {
    if (m.t === 'hello' && m.role === 'helper' && m.v === PAIR_PROTOCOL) {
      h.name = String(m.name || 'Device').slice(0, 60);
      h.capacity = Number(m.capacity) || 0;
      h.gpu = String(m.gpu || '').slice(0, 120);
      h.state = 'ready';
    } else if (m.t === 'progress') {
      h.progress = { loaded: Number(m.loaded) || 0, total: Number(m.total) || 0, phase: String(m.phase || '') };
    } else if (m.t === 'loaded' || m.t === 'out' || m.t === 'error') {
      const p = h.pending.get(m.id);
      if (!p) return;
      h.pending.delete(m.id);
      if (m.t === 'error') p.reject(new Error(`${h.name}: ${m.message}`));
      else p.resolve(bytes ?? m);
    }
    this.onChange();
  }

  call(h, msg, payload = null) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      h.pending.set(id, { resolve, reject });
      h.link.send({ ...msg, id }, payload).catch((e) => {
        h.pending.delete(id);
        reject(e);
      });
    });
  }

  /** Load layers `stage` of the model `source` on helper `h`. */
  async load(h, source, stage, ctx) {
    h.stage = stage;
    h.state = 'loading';
    h.progress = null;
    this.onChange();
    try {
      const r = await this.call(h, { t: 'load', source, stage, ctx });
      h.state = 'loaded';
      h.placement = r.placement || null;
    } catch (e) {
      h.state = 'ready';
      h.stage = null;
      throw e;
    } finally {
      this.onChange();
    }
  }

  /** The hidden state through every loaded helper, in layer order (an ArrayBuffer of floats). */
  async forward(hidden, n, pos0) {
    let h = new Uint8Array(hidden);
    for (const x of this.list.filter((y) => y.state === 'loaded').sort((a, b) => a.stage.first - b.stage.first)) {
      h = await this.call(x, { t: 'fwd', n, pos0 }, h);
    }
    return h.buffer.slice(h.byteOffset, h.byteOffset + h.byteLength);
  }

  control(op) {
    for (const h of this.list.filter((x) => x.state === 'loaded')) h.link.send({ t: 'ctl', op }).catch(() => {});
  }

  unloadAll() {
    for (const h of this.list) {
      if (h.state === 'loaded' || h.state === 'loading') h.link.send({ t: 'unload' }).catch(() => {});
      if (h.state !== 'connecting') h.state = 'ready';
      h.stage = null;
    }
    this.onChange();
  }
}

/**
 * Two ends of a link inside one page, for a helper that is this machine's other GPU: what one
 * end sends arrives at the other's onMessage, with the same shape as a data channel's.
 */
export function localLinkPair() {
  const a = { onMessage: null, close() {} };
  const b = { onMessage: null, close() {} };
  a.send = (m, bytes = null) => { queueMicrotask(() => b.onMessage?.(m, bytes)); return Promise.resolve(); };
  b.send = (m, bytes = null) => { queueMicrotask(() => a.onMessage?.(m, bytes)); return Promise.resolve(); };
  return [a, b];
}

/**
 * A helper's side: answers the chat device's messages with its own worker. `about` is what it
 * tells the chat device ({name, capacity, gpu}); `load(msg)` starts loading in the worker.
 * `local`: the helper is this machine's other GPU, which reads the model from where the main
 * worker does (a file, this browser's storage) instead of each part from Hugging Face.
 */
export class Helping {
  constructor({ link, worker, about, settings, onState = () => {}, local = false }) {
    this.link = link;
    this.local = local;
    this.worker = worker;
    this.settings = settings;
    this.onState = onState;
    this.loadId = null;
    this.calls = new Map();
    this.seq = 0;
    this.served = 0;
    this.about = about;
    link.onMessage = (m, bytes) => this.message(m, bytes);
    this.onWorker = (e) => this.fromWorker(e.data || {});
    worker.addEventListener('message', this.onWorker);
    this.hello();
  }

  hello() {
    this.link.send({ ...this.about, t: 'hello', v: PAIR_PROTOCOL, role: 'helper' }).catch(() => {});
  }

  message(m, bytes) {
    if (m.t === 'hello' && m.role === 'main') {
      this.hello();
    } else if (m.t === 'load') {
      this.loadId = m.id;
      this.stage = m.stage;
      this.served = 0;
      this.onState({ state: 'loading', stage: m.stage, name: m.source?.name });
      const s = this.settings();
      this.worker.postMessage({ type: 'load', source: this.local ? m.source : { ...m.source, stream: true }, ctx: m.ctx, stage: m.stage, gpu: s.gpu, gpuMemory: s.gpuMemory, cacheGb: Number(s.cacheGb) });
    } else if (m.t === 'fwd' && bytes) {
      const id = ++this.seq;
      this.calls.set(id, m.id);
      const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      this.worker.postMessage({ type: 'stage-forward', id, hidden: buf, n: m.n, pos0: m.pos0 }, [buf]);
    } else if (m.t === 'ctl') {
      this.worker.postMessage({ type: 'stage-control', op: m.op });
    } else if (m.t === 'unload') {
      this.worker.postMessage({ type: 'unload' });
      this.onState({ state: 'idle' });
    }
  }

  fromWorker(d) {
    if (d.type === 'progress' && this.loadId) {
      this.link.send({ t: 'progress', loaded: d.loaded, total: d.total, phase: d.phase }).catch(() => {});
      this.onState({ state: 'loading', stage: this.stage, progress: d });
    } else if (d.type === 'stage-loaded' && this.loadId) {
      this.link.send({ t: 'loaded', id: this.loadId, placement: d.placement }).catch(() => {});
      this.loadId = null;
      this.onState({ state: 'loaded', stage: d.stage, placement: d.placement });
    } else if (d.type === 'stage-result') {
      const id = this.calls.get(d.id);
      this.calls.delete(d.id);
      this.served++;
      this.link.send({ t: 'out', id }, new Uint8Array(d.hidden)).catch(() => {});
      this.onState({ state: 'loaded', stage: this.stage, served: this.served });
    } else if (d.type === 'error') {
      if (d.during === 'load' && this.loadId) {
        this.link.send({ t: 'error', id: this.loadId, message: d.message }).catch(() => {});
        this.loadId = null;
        this.onState({ state: 'error', message: d.message });
      } else if (d.during === 'stage-forward' && this.calls.has(d.id)) {
        const id = this.calls.get(d.id);
        this.calls.delete(d.id);
        this.link.send({ t: 'error', id, message: d.message }).catch(() => {});
      }
    }
  }

  stop() {
    this.worker.removeEventListener('message', this.onWorker);
  }
}

// ---- A phone using a computer's GPU ----

// What the computer's worker tells its own page, not the phone: requests for the page's swarm
// and paired devices, and the swarm's bookkeeping.
const HOST_ONLY = new Set(['swarm-get', 'swarm-have', 'remote-forward', 'remote-control', 'split-plan']);
// And what the phone's page may not ask of it: the page services above, and helper stages.
const PHONE_BLOCKED = new Set(['swarm', 'swarm-data', 'remote-result', 'stage-forward', 'stage-control']);

/**
 * The computer's side: messages from the phone go to this page's worker, and the worker's
 * replies go back. Local files and this page's swarm and paired devices stay out of it. The
 * model the phone last loaded is kept, and described again when the phone reconnects.
 */
export class Hosting {
  constructor({ worker, onState = () => {} }) {
    this.worker = worker;
    this.onState = onState;
    this.link = null;
    this.loaded = null;
    this.tokens = 0;
    this.onWorker = (e) => this.fromWorker(e.data || {});
    worker.addEventListener('message', this.onWorker);
  }

  /** The phone connected (again). */
  attach(link) {
    this.link?.close();
    this.link = link;
    link.onMessage = (m) => this.message(m);
    link.send({ t: 'host', v: PAIR_PROTOCOL, loaded: this.loaded }).catch(() => {});
    this.onState({ state: 'connected', loaded: this.loaded });
  }

  detach(link) {
    if (this.link !== link) return;
    this.link = null;
    this.onState({ state: 'waiting', loaded: this.loaded });
  }

  message(m) {
    if (m.t !== 'w' || !m.m || typeof m.m.type !== 'string' || PHONE_BLOCKED.has(m.m.type)) return;
    const msg = { ...m.m };
    if (msg.type === 'load') {
      if (msg.source?.kind !== 'hf') {
        this.reply({ type: 'error', during: 'load', message: 'Files on your phone cannot be opened by your computer. Choose a model from Hugging Face or one saved on the computer.' });
        return;
      }
      msg.share = false;
      msg.split = null;
      this.loaded = null;
    }
    this.worker.postMessage(msg);
  }

  reply(msg) {
    this.link?.send({ t: 'w', m: msg }).catch(() => {});
  }

  fromWorker(d) {
    if (HOST_ONLY.has(d.type)) return;
    if (d.type === 'loaded') this.loaded = d;
    if (d.type === 'unloaded' || (d.type === 'error' && d.during === 'load')) this.loaded = null;
    if (d.type === 'token') this.tokens++;
    if (d.type === 'loaded' || d.type === 'unloaded' || d.type === 'done') this.onState({ state: this.link ? 'connected' : 'waiting', loaded: this.loaded, tokens: this.tokens });
    this.reply(d);
  }

  stop() {
    this.worker.removeEventListener('message', this.onWorker);
    this.link?.close();
  }
}

/**
 * The phone's side: an object the page uses as its worker, whose messages go to the computer's
 * worker. Messages wait while the computer is not connected; `onConnection(state, info)` says
 * when it connects ('connected', with the model it has loaded) and goes ('disconnected').
 */
export class RemoteWorker {
  constructor({ onConnection = () => {} } = {}) {
    this.onConnection = onConnection;
    this.listeners = new Set();
    this.onmessage = null;
    this.onerror = null;
    this.queue = [];
    this.link = null;
    this.ready = false;
    this.hello = null;
  }

  attach(link) {
    this.link = link;
    this.ready = false;
    link.onMessage = (m) => {
      if (this.link !== link) return;
      if (m.t === 'w' && m.m) this.emit(m.m);
      else if (m.t === 'host') {
        // The host announces itself only after its message handler is installed.
        // Sending hello on the data channel's open event can race that handler.
        this.ready = true;
        const queued = this.queue.splice(0);
        if (this.hello && !queued.some(msg => msg.type === 'hello')) queued.unshift(this.hello);
        for (const msg of queued) link.send({ t: 'w', m: msg }).catch(() => {});
        this.onConnection('connected', { loaded: m.loaded || null });
      }
    };
  }

  detach(link) {
    if (this.link !== link) return;
    this.link = null;
    this.ready = false;
    this.onConnection('disconnected', {});
  }

  emit(data) {
    const event = { data };
    this.onmessage?.(event);
    for (const l of this.listeners) l(event);
  }

  postMessage(msg) {
    if (msg?.type === 'load' && msg.source?.kind === 'file') {
      queueMicrotask(() => this.emit({ type: 'error', during: 'load', message: 'Files on this phone cannot be opened by your computer. Choose a model from Hugging Face or one saved on the computer.' }));
      return;
    }
    // Only plain data crosses (transfers and files do not).
    const plain = JSON.parse(JSON.stringify(msg));
    if (plain.type === 'hello') this.hello = plain;
    if (this.link?.open && this.ready) this.link.send({ t: 'w', m: plain }).catch(() => this.queue.push(plain));
    else this.queue.push(plain);
  }

  addEventListener(type, l) {
    if (type === 'message') this.listeners.add(l);
  }

  removeEventListener(type, l) {
    if (type === 'message') this.listeners.delete(l);
  }

  terminate() {}
}
