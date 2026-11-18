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
   * `role`: 'main' (the chat device) or 'helper'. `onLink(link, peer)` for each device of the
   * other role that connects; `onClose(peer)` when one goes.
   */
  constructor({ code, role, onLink, onClose = () => {}, relays, iceServers = ICE_SERVERS, RTC = globalThis.RTCPeerConnection, signal = null }) {
    this.code = normalizeCode(code);
    this.role = role;
    this.other = role === 'main' ? 'helper' : 'main';
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
    link.send({ t: 'hello', v: PAIR_PROTOCOL, role: 'main' }).catch(() => {});
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
 * A helper's side: answers the chat device's messages with its own worker. `about` is what it
 * tells the chat device ({name, capacity, gpu}); `load(msg)` starts loading in the worker.
 */
export class Helping {
  constructor({ link, worker, about, settings, onState = () => {} }) {
    this.link = link;
    this.worker = worker;
    this.settings = settings;
    this.onState = onState;
    this.loadId = null;
    this.calls = new Map();
    this.seq = 0;
    this.served = 0;
    link.onMessage = (m, bytes) => this.message(m, bytes);
    this.onWorker = (e) => this.fromWorker(e.data || {});
    worker.addEventListener('message', this.onWorker);
    link.send({ t: 'hello', v: PAIR_PROTOCOL, role: 'helper', ...about }).catch(() => {});
  }

  message(m, bytes) {
    if (m.t === 'load') {
      this.loadId = m.id;
      this.stage = m.stage;
      this.served = 0;
      this.onState({ state: 'loading', stage: m.stage, name: m.source?.name });
      const s = this.settings();
      this.worker.postMessage({ type: 'load', source: { ...m.source, stream: true }, ctx: m.ctx, stage: m.stage, gpu: s.gpu, gpuMemory: s.gpuMemory, cacheGb: Number(s.cacheGb) });
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
