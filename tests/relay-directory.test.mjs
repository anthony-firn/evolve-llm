import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { schnorr } from '../web/vendor/noble-secp256k1.mjs';
import { ANNOUNCE_KIND, NOSTR_RELAYS, discoverRelays, eventId, parseAnnouncement, probeWisp, selectAnnouncements, serializeEvent, verifyEvent } from '../web/relay-directory.mjs';

const fixture = (name) => new URL(`./fixtures/${name}`, import.meta.url);
const hex = (b) => Buffer.from(b).toString('hex');
const unhex = (s) => Uint8Array.from(Buffer.from(s, 'hex'));
const NOW = 1_700_000_000;

/** Signs an announcement the way `evolve-relay --announce` does. */
async function announcement({ secret = 1, url = 'wss://relay.example.com/', createdAt = NOW - 60, expiresIn = 7200, extra = [] } = {}) {
  const sk = new Uint8Array(32);
  sk[31] = secret;
  const tags = [['d', url], ['u', url], ['proto', 'wisp/1'], ['limits', '8/128/2147483648'], ['ports', '80,443'], ['name', `Relay ${secret}`], ['expiration', String(createdAt + expiresIn)], ['t', 'evolve-relay']].filter((t) => !extra.some((e) => e[0] === t[0])).concat(extra);
  const event = { pubkey: hex(schnorr.getPublicKey(sk)), created_at: createdAt, kind: ANNOUNCE_KIND, tags, content: 'Wisp v1 relay for Evolve Browser' };
  event.id = await eventId(event);
  event.sig = hex(await schnorr.signAsync(unhex(event.id), sk));
  return event;
}

test('BIP-340 official test vectors (bitcoin/bips bip-0340/test-vectors.csv)', async () => {
  const rows = readFileSync(fixture('bip340-vectors.csv'), 'utf8').trim().split('\n').slice(1).map((l) => l.split(','));
  assert.equal(rows.length, 19);
  for (const [index, secret, pub, aux, msg, sig, result, ...comment] of rows) {
    if (secret) {
      assert.equal(hex(schnorr.getPublicKey(unhex(secret))).toUpperCase(), pub, `vector ${index} public key`);
      assert.equal(hex(await schnorr.signAsync(unhex(msg), unhex(secret), unhex(aux))).toUpperCase(), sig, `vector ${index} signature`);
    }
    let ok;
    try { ok = await schnorr.verifyAsync(unhex(sig), unhex(msg), unhex(pub)); } catch { ok = false; }
    assert.equal(ok, result === 'TRUE', `vector ${index}: ${comment.join(',')}`);
  }
});

test('event ids follow the NIP-01 serialization', async () => {
  const event = { pubkey: 'ab', created_at: 7, kind: ANNOUNCE_KIND, tags: [['d', 'wss://r.example/']], content: 'line\nquote" back\\ tab\t é /' };
  const expected = '[0,"ab",7,38787,[["d","wss://r.example/"]],"line\\nquote\\" back\\\\ tab\\t é /"]';
  assert.equal(serializeEvent(event), expected);
  assert.equal(await eventId(event), createHash('sha256').update(expected).digest('hex'));
});

test('an announcement signed by evolve-relay (Rust) verifies here, tampering does not', async () => {
  // The same event is produced by the relay's unit test from secret key 3 and zero aux_rand.
  const event = JSON.parse(readFileSync(fixture('relay-announcement.json'), 'utf8'));
  assert.equal(await verifyEvent(event), true);
  assert.equal(await verifyEvent({ ...event, content: 'changed' }), false);
  assert.equal(await verifyEvent({ ...event, tags: event.tags.map((t) => (t[0] === 'u' ? ['u', 'wss://evil.example/'] : t)) }), false);
  const other = await announcement({ secret: 5 });
  assert.equal(await verifyEvent({ ...event, pubkey: other.pubkey }), false);
  assert.equal(await verifyEvent({ ...event, sig: other.sig }), false);
  assert.equal(await verifyEvent({ ...event, sig: 'zz' + event.sig.slice(2) }), false);
  assert.equal(await verifyEvent(null), false);
  const ann = parseAnnouncement(event, { now: NOW });
  assert.deepEqual(
    { url: ann.url, name: ann.name, region: ann.region, contact: ann.contact, limits: ann.limits, ports: ann.ports, expiresAt: ann.expiresAt },
    { url: 'wss://relay.example.com/', name: 'Example relay', region: 'us-west', contact: 'ops@example.com', limits: { connections: 8, streams: 128, bytesPerHour: 2147483648 }, ports: [80, 443], expiresAt: 1_700_007_200 },
  );
});

test('only current wss:// announcements are kept, newest per operator and address', async () => {
  const fresh = await announcement({ secret: 1, createdAt: NOW - 60 });
  const older = await announcement({ secret: 1, createdAt: NOW - 600 });
  const expired = await announcement({ secret: 2, url: 'wss://two.example/', createdAt: NOW - 8000 });
  const future = await announcement({ secret: 3, url: 'wss://three.example/', createdAt: NOW + 3600 });
  const plain = await announcement({ secret: 4, url: 'ws://four.example/' });
  const local = await announcement({ secret: 4, url: 'ws://127.0.0.1:8787/' });
  const secondOperator = await announcement({ secret: 6 });
  const wrongProto = await announcement({ secret: 7, url: 'wss://seven.example/', extra: [['proto', 'socks5']] });
  // A forged copy of the fresh event's id must not hide the real event.
  const forged = { ...fresh, sig: older.sig };
  const mismatched = { ...(await announcement({ secret: 8 })), tags: fresh.tags.map((t) => (t[0] === 'd' ? ['d', 'wss://other.example/'] : t)) };

  const kept = await selectAnnouncements([older, forged, fresh, fresh, expired, future, plain, local, secondOperator, mismatched], { now: NOW });
  assert.deepEqual(kept.map((a) => [a.url, a.createdAt, a.name]).sort(), [['wss://relay.example.com/', NOW - 60, 'Relay 1'], ['wss://relay.example.com/', NOW - 60, 'Relay 6']]);
  assert.equal(parseAnnouncement(wrongProto, { now: NOW }), null);
  assert.equal(parseAnnouncement({ ...fresh, kind: 1 }, { now: NOW }), null);
  assert.equal(parseAnnouncement({ ...fresh, tags: fresh.tags.filter((t) => t[0] !== 't') }, { now: NOW }), null);
  assert.equal(parseAnnouncement({ ...fresh, tags: fresh.tags.filter((t) => t[0] !== 'expiration') }, { now: NOW }), null);
  // ws:// only for local tests
  assert.equal((await selectAnnouncements([local], { now: NOW, allowLocal: true })).length, 1);
  assert.equal((await selectAnnouncements([plain], { now: NOW, allowLocal: true })).length, 0);
  assert.equal(NOSTR_RELAYS.length, 4);
});

// ---------------------------------------------------------------------------------------------
// A tiny WebSocket server (RFC 6455, unfragmented frames) for a fake Nostr relay and Wisp relay.

function wsServer(onConnection) {
  const server = createServer((_, res) => res.writeHead(426).end());
  server.on('upgrade', (req, socket) => {
    const accept = createHash('sha1').update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
    const send = (data, opcode = typeof data === 'string' ? 1 : 2) => {
      const body = Buffer.from(data);
      const head = body.length < 126 ? Buffer.from([0x80 | opcode, body.length]) : Buffer.from([0x80 | opcode, 126, body.length >> 8, body.length & 255]);
      if (!socket.destroyed) socket.write(Buffer.concat([head, body]));
    };
    const peer = { send, close: () => { send(Buffer.alloc(0), 8); socket.end(); }, onText: () => {} };
    let buf = Buffer.alloc(0);
    socket.on('data', (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      while (buf.length >= 2) {
        let len = buf[1] & 127;
        let off = 2;
        if (len === 126) { len = buf.readUInt16BE(2); off = 4; }
        if (buf.length < off + 4 + len) return;
        const mask = buf.subarray(off, off + 4);
        const payload = Buffer.from(buf.subarray(off + 4, off + 4 + len).map((b, i) => b ^ mask[i % 4]));
        const opcode = buf[0] & 15;
        buf = buf.subarray(off + 4 + len);
        if (opcode === 1) peer.onText(payload.toString());
        if (opcode === 8) socket.end();
      }
    });
    socket.on('error', () => {});
    onConnection(peer, req);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, url: `ws://127.0.0.1:${server.address().port}/` })));
}

/** A Nostr relay holding `events`: answers REQ with matching events and EOSE. */
function fakeNostr(events) {
  const requests = [];
  return wsServer((peer) => {
    peer.onText = (text) => {
      const frame = JSON.parse(text);
      if (frame[0] !== 'REQ') return;
      const [, sub, filter] = frame;
      requests.push(filter);
      for (const ev of events) {
        if (filter.kinds.includes(ev.kind) && ev.created_at >= filter.since) peer.send(JSON.stringify(['EVENT', sub, ev]));
      }
      peer.send(JSON.stringify(['EVENT', 'someone-else', events[0]]));
      peer.send(JSON.stringify(['EOSE', sub]));
    };
  }).then((s) => ({ ...s, requests }));
}

/** A Wisp v1 server: sends CONTINUE for stream 0 (buffer 128) after `delayMs`. */
const fakeWisp = (delayMs = 0) => wsServer((peer) => setTimeout(() => peer.send(Buffer.from([0x03, 0, 0, 0, 0, 128, 0, 0, 0])), delayMs));
/** A WebSocket server that is not a Wisp relay. */
const fakeOther = () => wsServer((peer) => peer.send('hello'));

test('discoverRelays finds, verifies and health-checks announced relays end to end', async (t) => {
  const fast = await fakeWisp(0);
  const slow = await fakeWisp(150);
  const other = await fakeOther();
  const now = Math.floor(Date.now() / 1000);
  const events = [
    await announcement({ secret: 1, url: slow.url, createdAt: now - 30 }),
    await announcement({ secret: 2, url: fast.url, createdAt: now - 30 }),
    await announcement({ secret: 3, url: other.url, createdAt: now - 30 }),
    await announcement({ secret: 4, url: 'ws://127.0.0.1:1/', createdAt: now - 30 }),
    await announcement({ secret: 5, url: 'ws://127.0.0.1:2/', createdAt: now - 90000 }), // older than a day
  ];
  const nostrA = await fakeNostr(events.slice(0, 3));
  const nostrB = await fakeNostr(events.slice(2));
  t.after(() => [fast, slow, other, nostrA, nostrB].forEach((s) => s.server.close()));

  const found = await discoverRelays({ nostrRelays: [nostrA.url, nostrB.url, 'ws://127.0.0.1:3/'], timeoutMs: 2000, allowLocal: true });
  assert.deepEqual(found.slice(0, 2).map((r) => [r.url, r.ok, r.name]), [[fast.url, true, 'Relay 2'], [slow.url, true, 'Relay 1']]);
  assert.deepEqual(found.slice(2).map((r) => r.url).sort(), [other.url, 'ws://127.0.0.1:1/'].sort());
  assert.equal(found.length, 4);
  assert.ok(found[0].rttMs < found[1].rttMs, 'faster relay first');
  assert.deepEqual(found.slice(2).map((r) => r.ok), [false, false]);
  assert.equal(found.find((r) => r.url === other.url).error, 'not a Wisp v1 relay');
  assert.deepEqual(nostrA.requests[0].kinds, [38787]);
  assert.deepEqual(nostrA.requests[0]['#t'], ['evolve-relay']);
  assert.ok(nostrA.requests[0].since <= now - 86400 + 5);

  // Without allowLocal, ws:// announcements are ignored entirely.
  assert.deepEqual(await discoverRelays({ nostrRelays: [nostrA.url], timeoutMs: 2000 }), []);
  // A custom probe replaces the health check.
  const probed = await discoverRelays({ nostrRelays: [nostrA.url], timeoutMs: 2000, allowLocal: true, probe: async (url) => ({ ok: url === slow.url, rttMs: 1 }) });
  assert.equal(probed[0].url, slow.url);
});

test('probeWisp times out on a silent server and fails on a closed port', async (t) => {
  const silent = await wsServer(() => {});
  t.after(() => silent.server.close());
  assert.deepEqual(await probeWisp(silent.url, { timeoutMs: 200 }), { ok: false, rttMs: null, error: 'timeout' });
  const closed = await probeWisp('ws://127.0.0.1:1/', { timeoutMs: 2000 });
  assert.equal(closed.ok, false);
});
