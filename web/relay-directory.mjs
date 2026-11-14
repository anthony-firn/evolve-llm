/**
 * Community relay directory: finds independently operated evolve-relay instances that announce
 * themselves on Nostr (`evolve-relay --public --announce`), verifies their signed announcements
 * and checks that each relay answers.
 *
 * An announcement is a parameterized replaceable Nostr event of kind {@link ANNOUNCE_KIND}:
 *
 * ```
 * tags: ["d", url] ["u", url] ["proto", "wisp/1"] ["limits", "8/128/2147483648"]
 *       ["ports", "80,443"] ["name", …] ["region", …] ["contact", …]
 *       ["expiration", unix seconds] ["t", "evolve-relay"]
 * ```
 *
 * An announcement is a claim by its operator, not proof of honesty. The engine negotiates TLS
 * itself, so a relay only forwards encrypted bytes for https:// sites, but it does see your IP
 * address and the host names you visit. Offer community relays as an explicit choice.
 *
 * Plain ES module with no dependencies besides the vendored BIP-340 verifier.
 * @module relay-directory
 */

import { schnorr } from './vendor/noble-secp256k1.mjs';

/** Nostr event kind of relay announcements. */
export const ANNOUNCE_KIND = 38787;
/** `t` tag every announcement carries. */
export const ANNOUNCE_TOPIC = 'evolve-relay';
/** Public Nostr relays to query (the same defaults as Fairground). */
export const NOSTR_RELAYS = Object.freeze(['wss://relay.damus.io', 'wss://nos.lol', 'wss://relay.primal.net', 'wss://offchain.pub']);

const HEX64 = /^[0-9a-f]{64}$/;
const HEX128 = /^[0-9a-f]{128}$/;
const MAX_EVENTS_PER_RELAY = 500;
/** Announcements dated further in the future than this are ignored (clock skew allowance). */
const MAX_FUTURE_SECONDS = 600;

const hexToBytes = (hex) => Uint8Array.from(hex.match(/../g) ?? [], (b) => parseInt(b, 16));
const bytesToHex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');

/**
 * NIP-01 serialization used for the event id: `[0, pubkey, created_at, kind, tags, content]`
 * as compact JSON. `JSON.stringify` escapes exactly the characters NIP-01 lists.
 * @param {{pubkey: string, created_at: number, kind: number, tags: string[][], content: string}} event
 * @returns {string}
 */
export function serializeEvent(event) {
  return JSON.stringify([0, event.pubkey, event.created_at, event.kind, event.tags, event.content]);
}

/**
 * The NIP-01 event id (lowercase hex SHA-256 of {@link serializeEvent}).
 * @returns {Promise<string>}
 */
export async function eventId(event) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(serializeEvent(event)));
  return bytesToHex(new Uint8Array(digest));
}

/**
 * True when `event` is well formed, its id matches its contents and its BIP-340 signature by
 * `pubkey` verifies.
 * @returns {Promise<boolean>}
 */
export async function verifyEvent(event) {
  if (!event || typeof event !== 'object') return false;
  const { id, pubkey, sig, created_at: createdAt, kind, tags, content } = event;
  if (!HEX64.test(id ?? '') || !HEX64.test(pubkey ?? '') || !HEX128.test(sig ?? '')) return false;
  if (!Number.isSafeInteger(createdAt) || createdAt < 0 || !Number.isSafeInteger(kind) || typeof content !== 'string') return false;
  if (!Array.isArray(tags) || !tags.every((t) => Array.isArray(t) && t.every((v) => typeof v === 'string'))) return false;
  if ((await eventId(event)) !== id) return false;
  try {
    return await schnorr.verifyAsync(hexToBytes(sig), hexToBytes(id), hexToBytes(pubkey));
  } catch {
    return false;
  }
}

/**
 * @typedef {object} Announcement
 * @property {string} url        wss:// address of the relay
 * @property {string} pubkey     operator's Nostr public key (hex)
 * @property {string} d          the event's `d` tag (the relay's own address)
 * @property {string} [name]
 * @property {string} [region]
 * @property {string} [contact]
 * @property {{connections: number, streams: number, bytesPerHour: number} | null} limits
 *           per client address, 0 = no limit
 * @property {number[]} ports    destination ports the relay allows
 * @property {number} createdAt  unix seconds
 * @property {number} expiresAt  unix seconds
 * @property {string} id         event id
 */

/**
 * Reads a relay announcement from a (verified) event, or returns null when it is not a usable,
 * current announcement: wrong kind or topic, expired, dated in the future, or an address other
 * than `wss://` (`ws://localhost` and `ws://127.0.0.1` only with `allowLocal`).
 * @param {object} event
 * @param {{now?: number, allowLocal?: boolean}} [options]  now in unix seconds
 * @returns {Announcement | null}
 */
export function parseAnnouncement(event, { now = Math.floor(Date.now() / 1000), allowLocal = false } = {}) {
  if (event?.kind !== ANNOUNCE_KIND || !Array.isArray(event.tags)) return null;
  const tag = (name) => event.tags.find((t) => t[0] === name)?.[1];
  if (!event.tags.some((t) => t[0] === 't' && t[1] === ANNOUNCE_TOPIC)) return null;
  const url = tag('u') ?? '';
  const d = tag('d') ?? '';
  if (!url || url !== d || !usableRelayUrl(url, allowLocal)) return null;
  const expiresAt = Number(tag('expiration'));
  if (!Number.isSafeInteger(expiresAt) || expiresAt <= now) return null;
  if (event.created_at > now + MAX_FUTURE_SECONDS) return null;
  const proto = tag('proto');
  if (proto && proto !== 'wisp/1') return null;
  const limitParts = (tag('limits') ?? '').split('/').map(Number);
  const limits = limitParts.length === 3 && limitParts.every((n) => Number.isSafeInteger(n) && n >= 0)
    ? { connections: limitParts[0], streams: limitParts[1], bytesPerHour: limitParts[2] }
    : null;
  const ports = (tag('ports') ?? '').split(',').map(Number).filter((p) => Number.isInteger(p) && p > 0 && p < 65536);
  const text = (name) => {
    const v = tag(name);
    return typeof v === 'string' && v.trim() ? v.trim().slice(0, 120) : undefined;
  };
  return { url, d, pubkey: event.pubkey, name: text('name'), region: text('region'), contact: text('contact'), limits, ports, createdAt: event.created_at, expiresAt, id: event.id };
}

function usableRelayUrl(url, allowLocal) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) return false;
  if (parsed.protocol === 'wss:') return true;
  return allowLocal && parsed.protocol === 'ws:' && ['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname);
}

/**
 * Verifies events and keeps the newest current announcement per operator and relay address
 * (`pubkey` + `d`, as NIP-01 replaceable events work).
 * @param {object[]} events
 * @param {{now?: number, allowLocal?: boolean}} [options]
 * @returns {Promise<Announcement[]>}
 */
export async function selectAnnouncements(events, options = {}) {
  const newest = new Map();
  const checked = new Set(); // id + signature pairs already verified (relays repeat events)
  for (const event of events) {
    if (!event || checked.has(`${event.id}${event.sig}`)) continue;
    const ann = parseAnnouncement(event, options);
    if (!ann) continue;
    const key = `${ann.pubkey} ${ann.d}`;
    const prev = newest.get(key);
    if (prev && (prev.createdAt > ann.createdAt || (prev.createdAt === ann.createdAt && prev.id <= ann.id))) continue;
    // Only verified events count as seen, so a forged copy of an id cannot hide the real one.
    if (!(await verifyEvent(event))) continue;
    checked.add(`${event.id}${event.sig}`);
    newest.set(key, ann);
  }
  return [...newest.values()];
}

/**
 * Asks Nostr relays for announcements. Each relay is queried once, until it sends EOSE or the
 * timeout passes; relays that fail are skipped.
 * @param {{nostrRelays?: readonly string[], timeoutMs?: number, since?: number,
 *          WebSocketImpl?: typeof WebSocket}} [options]
 * @returns {Promise<object[]>} raw (unverified) events
 */
export async function fetchAnnouncements({ nostrRelays = NOSTR_RELAYS, timeoutMs = 4000, since = Math.floor(Date.now() / 1000) - 86400, WebSocketImpl = globalThis.WebSocket } = {}) {
  const filter = { kinds: [ANNOUNCE_KIND], '#t': [ANNOUNCE_TOPIC], since, limit: 200 };
  const lists = await Promise.all(nostrRelays.map((url) => queryRelay(url, filter, timeoutMs, WebSocketImpl)));
  return lists.flat();
}

function queryRelay(url, filter, timeoutMs, WebSocketImpl) {
  return new Promise((resolve) => {
    const events = [];
    const sub = `evolve-${Math.random().toString(36).slice(2, 10)}`;
    let ws;
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      try {
        if (ws?.readyState === 1) ws.send(JSON.stringify(['CLOSE', sub]));
        ws?.close();
      } catch {}
      resolve(events);
    };
    const timer = setTimeout(finish, timeoutMs);
    try {
      ws = new WebSocketImpl(url);
    } catch {
      finish();
      return;
    }
    ws.onopen = () => ws.send(JSON.stringify(['REQ', sub, filter]));
    ws.onmessage = (message) => {
      let frame;
      try {
        frame = JSON.parse(typeof message.data === 'string' ? message.data : '');
      } catch {
        return;
      }
      if (!Array.isArray(frame) || frame[1] !== sub) return;
      if (frame[0] === 'EVENT' && frame[2] && typeof frame[2] === 'object') {
        events.push(frame[2]);
        if (events.length >= MAX_EVENTS_PER_RELAY) finish();
      } else if (frame[0] === 'EOSE' || frame[0] === 'CLOSED') {
        finish();
      }
    };
    ws.onerror = finish;
    ws.onclose = finish;
  });
}

/**
 * Checks that a Wisp v1 relay answers: opens a WebSocket and waits for the CONTINUE packet for
 * stream 0 that Wisp servers send right after the handshake.
 * @param {string} url
 * @param {{timeoutMs?: number, WebSocketImpl?: typeof WebSocket}} [options]
 * @returns {Promise<{ok: boolean, rttMs: number | null, error?: string}>}
 */
export function probeWisp(url, { timeoutMs = 4000, WebSocketImpl = globalThis.WebSocket } = {}) {
  return new Promise((resolve) => {
    const started = performance.now();
    let ws;
    let finished = false;
    const finish = (result) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      try { ws?.close(); } catch {}
      resolve(result);
    };
    const timer = setTimeout(() => finish({ ok: false, rttMs: null, error: 'timeout' }), timeoutMs);
    try {
      ws = new WebSocketImpl(url);
      ws.binaryType = 'arraybuffer';
    } catch (e) {
      finish({ ok: false, rttMs: null, error: String(e?.message || e) });
      return;
    }
    ws.onmessage = (message) => {
      const data = message.data instanceof ArrayBuffer ? new Uint8Array(message.data) : null;
      const isContinue = data && data.length === 9 && data[0] === 0x03 && data[1] === 0 && data[2] === 0 && data[3] === 0 && data[4] === 0;
      finish(isContinue ? { ok: true, rttMs: Math.round(performance.now() - started) } : { ok: false, rttMs: null, error: 'not a Wisp v1 relay' });
    };
    ws.onerror = () => finish({ ok: false, rttMs: null, error: 'unreachable' });
    ws.onclose = () => finish({ ok: false, rttMs: null, error: 'closed' });
  });
}

/**
 * Finds community relays: fetches announcements from Nostr, verifies them, drops expired and
 * non-`wss://` ones, keeps the newest per operator and address, then health-checks each relay.
 * Reachable relays come first, fastest first.
 *
 * @param {object} [options]
 * @param {readonly string[]} [options.nostrRelays]  Nostr relays to ask (default {@link NOSTR_RELAYS})
 * @param {number} [options.timeoutMs]  per Nostr query and per health check (default 4000)
 * @param {(url: string, opts: {timeoutMs: number}) => Promise<{ok: boolean, rttMs: number|null, error?: string}>} [options.probe]
 *        health check (default {@link probeWisp})
 * @param {number} [options.now]  unix seconds (for tests)
 * @param {boolean} [options.allowLocal]  accept ws://localhost and ws://127.0.0.1 (tests only)
 * @param {typeof WebSocket} [options.WebSocketImpl]
 * @returns {Promise<Array<Announcement & {ok: boolean, rttMs: number | null, error?: string}>>}
 */
export async function discoverRelays({ nostrRelays = NOSTR_RELAYS, timeoutMs = 4000, probe, now = Math.floor(Date.now() / 1000), allowLocal = false, WebSocketImpl = globalThis.WebSocket } = {}) {
  const events = await fetchAnnouncements({ nostrRelays, timeoutMs, since: now - 86400, WebSocketImpl });
  const announcements = await selectAnnouncements(events, { now, allowLocal });
  const check = probe ?? ((url, opts) => probeWisp(url, { ...opts, WebSocketImpl }));
  const health = new Map();
  for (const url of new Set(announcements.map((a) => a.url))) health.set(url, check(url, { timeoutMs }).catch((e) => ({ ok: false, rttMs: null, error: String(e?.message || e) })));
  const results = await Promise.all(announcements.map(async (a) => ({ ...a, ...(await health.get(a.url)) })));
  return results.sort((a, b) => (b.ok - a.ok) || ((a.rttMs ?? Infinity) - (b.rttMs ?? Infinity)) || (b.createdAt - a.createdAt));
}
