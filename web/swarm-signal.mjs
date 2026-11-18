// Finding swarm peers and opening WebRTC connections through public Nostr relays: no server of
// ours is involved. Each page makes a fresh key per visit. Presence events say which model files
// (by SHA-256) a page shares; connection offers and answers are encrypted to their one recipient
// (ECDH, then AES-GCM), so relays never see addresses. Events are ephemeral (kind 25075): relays
// pass them to current subscribers and do not keep them.

import { getSharedSecret, schnorr } from './vendor/noble-secp256k1.mjs';
import { NOSTR_RELAYS, eventId, verifyEvent } from './relay-directory.mjs';

export const SIGNAL_KIND = 25075;
export const topic = (sha) => `evolve-swarm:${sha}`;
const PREFIX = 'evolve-swarm:';

const hexToBytes = (hex) => Uint8Array.from(hex.match(/../g) ?? [], (b) => parseInt(b, 16));
const bytesToHex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
const b64 = (bytes) => btoa(String.fromCharCode(...bytes));
const unb64 = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));

export class Signal {
  /**
   * `onEvent(kind, from, data, content)`: kind 'presence' (data: the topics' names, such as
   * file SHA-256s; content: what the announcement carried) or 'message' (data: the decrypted
   * object a peer sent to this page). `prefix` names the kind of topic (pairing uses its own).
   */
  constructor({ relays = NOSTR_RELAYS, WebSocketImpl = globalThis.WebSocket, onEvent = () => {}, now = () => Math.floor(Date.now() / 1000), prefix = PREFIX } = {}) {
    this.prefix = prefix;
    this.relays = relays;
    this.WebSocketImpl = WebSocketImpl;
    this.onEvent = onEvent;
    this.now = now;
    this.secret = crypto.getRandomValues(new Uint8Array(32));
    this.pubkey = bytesToHex(schnorr.getPublicKey(this.secret));
    this.sockets = new Map();
    this.seen = new Set();
    this.topics = [];
    this.keys = new Map();
    this.closed = false;
  }

  start(shas) {
    this.topics = shas.map((s) => this.prefix + s);
    for (const url of this.relays) this.connect(url, 0);
  }

  connect(url, attempt) {
    if (this.closed) return;
    let ws;
    try { ws = new this.WebSocketImpl(url); } catch { return; }
    this.sockets.set(url, ws);
    ws.onopen = () => {
      const since = this.now() - 30;
      ws.send(JSON.stringify(['REQ', 'swarm', { kinds: [SIGNAL_KIND], '#t': this.topics, since }, { kinds: [SIGNAL_KIND], '#p': [this.pubkey], since }]));
      // Presence is not kept by relays: say it again on each relay as it connects.
      if (this.presence !== undefined) this.publish(this.topics.map((t) => ['t', t]), this.presence, ws).catch(() => {});
    };
    ws.onmessage = (m) => this.receive(m.data);
    ws.onclose = () => {
      if (this.sockets.get(url) === ws) this.sockets.delete(url);
      if (!this.closed && attempt < 6) setTimeout(() => this.connect(url, attempt + 1), 2000 * 2 ** attempt);
    };
    ws.onerror = () => {};
  }

  async receive(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    if (!Array.isArray(msg) || msg[0] !== 'EVENT' || !msg[2]) return;
    const ev = msg[2];
    if (ev.kind !== SIGNAL_KIND || ev.pubkey === this.pubkey || this.seen.has(ev.id)) return;
    if (Math.abs(this.now() - ev.created_at) > 120) return;
    this.seen.add(ev.id);
    if (!(await verifyEvent(ev).catch(() => false))) return;
    const to = ev.tags.find((t) => t[0] === 'p')?.[1];
    if (to === this.pubkey) {
      const data = await this.decrypt(ev.pubkey, ev.content).catch(() => null);
      if (data) this.onEvent('message', ev.pubkey, data);
    } else if (!to) {
      const shas = ev.tags.filter((t) => t[0] === 't' && t[1]?.startsWith(this.prefix)).map((t) => t[1].slice(this.prefix.length));
      if (shas.length) this.onEvent('presence', ev.pubkey, shas, typeof ev.content === 'string' ? ev.content : '');
    }
  }

  async key(peer) {
    if (!this.keys.has(peer)) {
      const shared = getSharedSecret(this.secret, hexToBytes('02' + peer)).subarray(1, 33);
      const base = await crypto.subtle.importKey('raw', shared, 'HKDF', false, ['deriveKey']);
      this.keys.set(peer, await crypto.subtle.deriveKey(
        { name: 'HKDF', hash: 'SHA-256', salt: new TextEncoder().encode('evolve-swarm/1'), info: new Uint8Array() },
        base, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt'],
      ));
    }
    return this.keys.get(peer);
  }

  async encrypt(peer, data) {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ct = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await this.key(peer), new TextEncoder().encode(JSON.stringify(data))));
    const out = new Uint8Array(12 + ct.length);
    out.set(iv);
    out.set(ct, 12);
    return b64(out);
  }

  async decrypt(peer, content) {
    const raw = unb64(content);
    const pt = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: raw.subarray(0, 12) }, await this.key(peer), raw.subarray(12));
    return JSON.parse(new TextDecoder().decode(pt));
  }

  /** Sign and send an event to every connected relay, or only to `only`. */
  async publish(tags, content, only = null) {
    const ev = { pubkey: this.pubkey, created_at: this.now(), kind: SIGNAL_KIND, tags, content };
    ev.id = await eventId(ev);
    ev.sig = bytesToHex(await schnorr.signAsync(hexToBytes(ev.id), this.secret));
    const text = JSON.stringify(['EVENT', ev]);
    for (const ws of only ? [only] : this.sockets.values()) if (ws.readyState === 1) ws.send(text);
  }

  /** Say which files (topics) this page shares, optionally with a short text. */
  announce(content = '') {
    this.presence = content;
    return this.publish(this.topics.map((t) => ['t', t]), content);
  }

  /** Send an object to one peer, encrypted to it. */
  async send(peer, data) {
    return this.publish([['p', peer]], await this.encrypt(peer, data));
  }

  close() {
    this.closed = true;
    for (const ws of this.sockets.values()) try { ws.close(); } catch { /* closed */ }
    this.sockets.clear();
  }
}
