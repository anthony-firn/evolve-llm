// The weight swarm on the page (WebRTC is not available in workers): visitors who turn sharing on
// fetch model chunks from each other and serve what their device cache holds. Every chunk is
// checked against the published manifest before use, and a peer that sends a wrong one is
// dropped for the visit. Whatever no peer delivers comes from Hugging Face as before.

import { CHUNK, FRAME, MANIFESTS, MAX_REQUEST_CHUNKS, Manifests, PROTOCOL, RangeSet, decodeFrame, encodeFrame, parseControl } from './swarm-protocol.mjs';
import { Signal } from './swarm-signal.mjs';

export { MANIFESTS };
export const ICE_SERVERS = [{ urls: ['stun:stun.cloudflare.com:3478', 'stun:stun.l.google.com:19302'] }];
const MAX_PEERS = 8;
const REQUEST_MS = 20000;
// Right after starting, a piece no peer has yet is waited for this long while peers connect (a
// model's first reads, its headers, come before any connection is up).
const GRACE_MS = 6000;

/** Wait until ICE gathering is complete, or `ms`, so one message carries every candidate. */
function gathered(pc, ms) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    pc.addEventListener('icegatheringstatechange', () => {
      if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); }
    });
  });
}

/** Bytes per second for uploads, as a token bucket. */
class Limiter {
  constructor(rate) { this.rate = rate; this.tokens = rate; this.at = Date.now(); }
  async take(n, signal) {
    for (;;) {
      if (!this.rate) return;
      const now = Date.now();
      this.tokens = Math.min(this.rate, this.tokens + ((now - this.at) / 1000) * this.rate);
      this.at = now;
      if (this.tokens >= n || this.tokens >= this.rate) { this.tokens -= n; return; }
      await new Promise((r) => setTimeout(r, Math.min(1000, ((n - this.tokens) / this.rate) * 1000)));
      if (signal?.closed) throw new Error('closed');
    }
  }
}

export class Swarm {
  /**
   * `read(sha, s, e)` returns chunks [s, e) of a file from this device's cache, or null.
   * `files`: [{sha, size}] the swarm is about.
   */
  constructor({ files, read, uploadBytesPerSec = 1 << 20, manifests = new Manifests(MANIFESTS), signal, iceServers = ICE_SERVERS, RTC = globalThis.RTCPeerConnection, onStats = () => {}, graceMs = GRACE_MS }) {
    this.graceMs = graceMs;
    this.graceUntil = 0;
    this.files = new Map(files.map((f) => [f.sha, { ...f, have: new RangeSet() }]));
    this.read = read;
    this.manifests = manifests;
    this.iceServers = iceServers;
    this.RTC = RTC;
    this.onStats = onStats;
    this.limiter = new Limiter(uploadBytesPerSec);
    this.peers = new Map();
    this.banned = new Set();
    this.nextId = 1;
    this.stats = { peers: 0, fromPeers: 0, toPeers: 0, rejected: 0 };
    this.signal = signal || new Signal({ onEvent: (kind, from, data) => this.onSignal(kind, from, data) });
    if (signal) signal.onEvent = (kind, from, data) => this.onSignal(kind, from, data);
    this.lastAnnounce = 0;
  }

  /** Start: only files with a published manifest take part. Returns their SHA-256s. */
  async start() {
    const usable = [];
    for (const sha of this.files.keys()) {
      if (await this.manifests.index(sha).catch(() => null)) usable.push(sha);
      else this.files.delete(sha);
    }
    if (!usable.length) return usable;
    this.signal.start(usable);
    this.graceUntil = Date.now() + this.graceMs;
    this.timer = setInterval(() => this.announce(), 30000);
    setTimeout(() => this.announce(), 1500);
    return usable;
  }

  announce(force = false) {
    const now = Date.now();
    if (!force && now - this.lastAnnounce < 5000) return;
    this.lastAnnounce = now;
    this.signal.announce().catch(() => {});
  }

  report() {
    this.stats.peers = [...this.peers.values()].filter((p) => p.dc?.readyState === 'open').length;
    this.onStats({ ...this.stats });
  }

  /** Chunks this device can now serve (its cache stored them). */
  addHave(sha, s, e) {
    const f = this.files.get(sha);
    if (!f) return;
    f.have.add(s, e);
    for (const p of this.peers.values()) this.sendControl(p, { t: 'have', sha, ranges: [[s, e]] });
  }

  onSignal(kind, from, data) {
    if (this.banned.has(from)) return;
    if (kind === 'presence') {
      if (!data.some((sha) => this.files.has(sha)) || this.peers.has(from)) return;
      if (this.peers.size >= MAX_PEERS) return;
      // The page with the smaller key makes the offer; the other answers its announcement.
      if (this.signal.pubkey < from) this.connect(from).catch(() => this.drop(from));
      else this.announce(true);
    } else if (data?.type === 'offer' && typeof data.sdp === 'string') {
      this.accept(from, data.sdp).catch(() => this.drop(from));
    } else if (data?.type === 'answer' && typeof data.sdp === 'string') {
      const p = this.peers.get(from);
      p?.pc.setRemoteDescription({ type: 'answer', sdp: data.sdp }).catch(() => this.drop(from));
    }
  }

  peer(from) {
    const pc = new this.RTC({ iceServers: this.iceServers });
    const p = { id: from, pc, dc: null, have: new Map(), pending: new Map(), busy: 0 };
    this.peers.set(from, p);
    pc.onconnectionstatechange = () => {
      if (['failed', 'closed', 'disconnected'].includes(pc.connectionState)) this.drop(from);
    };
    return p;
  }

  async connect(from) {
    const p = this.peer(from);
    this.channel(p, p.pc.createDataChannel('evolve-swarm', { ordered: true }));
    await p.pc.setLocalDescription(await p.pc.createOffer());
    await gathered(p.pc, 3000);
    await this.signal.send(from, { type: 'offer', sdp: p.pc.localDescription.sdp });
  }

  async accept(from, sdp) {
    if (this.peers.has(from) || this.peers.size >= MAX_PEERS) return;
    const p = this.peer(from);
    p.pc.ondatachannel = (e) => this.channel(p, e.channel);
    await p.pc.setRemoteDescription({ type: 'offer', sdp });
    await p.pc.setLocalDescription(await p.pc.createAnswer());
    await gathered(p.pc, 3000);
    await this.signal.send(from, { type: 'answer', sdp: p.pc.localDescription.sdp });
  }

  channel(p, dc) {
    p.dc = dc;
    dc.binaryType = 'arraybuffer';
    dc.bufferedAmountLowThreshold = 1 << 20;
    dc.onopen = () => {
      const files = {};
      for (const [sha, f] of this.files) files[sha] = f.have.toJSON();
      this.sendControl(p, { t: 'hello', v: PROTOCOL, files });
      this.report();
    };
    dc.onclose = () => this.drop(p.id);
    dc.onmessage = (m) => {
      if (typeof m.data === 'string') this.control(p, parseControl(m.data));
      else this.data(p, decodeFrame(m.data));
    };
  }

  sendControl(p, msg) {
    if (p.dc?.readyState === 'open') p.dc.send(JSON.stringify(msg));
  }

  control(p, m) {
    if (!m) return;
    if (m.t === 'hello') {
      for (const [sha, ranges] of Object.entries(m.files)) if (this.files.has(sha)) p.have.set(sha, new RangeSet(ranges));
    } else if (m.t === 'have') {
      if (!this.files.has(m.sha)) return;
      if (!p.have.has(m.sha)) p.have.set(m.sha, new RangeSet());
      for (const [s, e] of m.ranges) p.have.get(m.sha).add(s, e);
    } else if (m.t === 'req') {
      this.serve(p, m).catch(() => this.sendControl(p, { t: 'nope', id: m.id }));
    } else if (m.t === 'nope') {
      p.pending.get(m.id)?.done(null);
    }
  }

  async serve(p, m) {
    const f = this.files.get(m.sha);
    if (!f || !f.have.has(m.s, m.e)) return this.sendControl(p, { t: 'nope', id: m.id });
    const bytes = await this.read(m.sha, m.s, m.e);
    if (!bytes) return this.sendControl(p, { t: 'nope', id: m.id });
    for (let at = 0; at < bytes.byteLength; at += FRAME) {
      const part = bytes.subarray(at, Math.min(at + FRAME, bytes.byteLength));
      await this.limiter.take(part.byteLength);
      if (p.dc.readyState !== 'open') return;
      if (p.dc.bufferedAmount > 4 << 20) {
        await new Promise((r) => {
          p.dc.addEventListener('bufferedamountlow', r, { once: true });
          p.dc.addEventListener('close', r, { once: true });
        });
        if (p.dc.readyState !== 'open') return;
      }
      p.dc.send(encodeFrame(m.id, at, part));
      this.stats.toPeers += part.byteLength;
    }
    this.report();
  }

  data(p, { id, at, bytes }) {
    const r = p.pending.get(id);
    if (!r || at + bytes.byteLength > r.buf.byteLength) return;
    r.buf.set(bytes, at);
    r.got += bytes.byteLength;
    if (r.got >= r.buf.byteLength) r.done(r.buf);
  }

  /** Ask one peer for chunks [s, e); resolves to their bytes, checked, or null. */
  request(p, sha, s, e) {
    const size = this.files.get(sha).size;
    const len = Math.min(e * CHUNK, size) - s * CHUNK;
    const id = this.nextId++;
    p.busy++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => done(null), REQUEST_MS);
      const done = (buf) => {
        clearTimeout(timer);
        p.pending.delete(id);
        p.busy--;
        resolve(buf);
      };
      p.pending.set(id, { buf: new Uint8Array(len), got: 0, done });
      this.sendControl(p, { t: 'req', id, sha, s, e });
    });
  }

  /**
   * Chunks [s, e) of a file from peers that have them (at most MAX_REQUEST_CHUNKS), or null when
   * none can provide them. A peer whose bytes fail the manifest check is dropped for the visit.
   */
  async get(sha, s, e) {
    if (!this.files.has(sha) || e - s > MAX_REQUEST_CHUNKS) return null;
    const tried = new Set();
    const pick = () => [...this.peers.values()]
      .filter((p) => p.dc?.readyState === 'open' && !tried.has(p.id) && p.have.get(sha)?.has(s, e))
      .sort((a, b) => a.busy - b.busy)[0];
    for (let attempt = 0; attempt < 3; attempt++) {
      let p = pick();
      while (!p && attempt === 0 && Date.now() < this.graceUntil) {
        await new Promise((r) => setTimeout(r, 200));
        p = pick();
      }
      if (!p) return null;
      tried.add(p.id);
      const buf = await this.request(p, sha, s, e);
      if (!buf) continue;
      if (await this.manifests.verify(sha, s, e, buf).catch(() => false)) {
        this.stats.fromPeers += buf.byteLength;
        this.report();
        return buf;
      }
      this.stats.rejected++;
      this.banned.add(p.id);
      this.drop(p.id);
    }
    return null;
  }

  drop(id) {
    const p = this.peers.get(id);
    if (!p) return;
    this.peers.delete(id);
    for (const r of p.pending.values()) r.done(null);
    try { p.dc?.close(); } catch { /* closed */ }
    try { p.pc.close(); } catch { /* closed */ }
    this.report();
  }

  stop() {
    clearInterval(this.timer);
    for (const id of [...this.peers.keys()]) this.drop(id);
    this.signal.close();
  }
}
