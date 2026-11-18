// A minimal Nostr relay for tests: WebSocket text frames, REQ/EVENT/CLOSE, filters on kinds and
// the #t and #p tags, nothing stored. Prints its port. Usage: node mini-nostr-relay.mjs [port]
import http from 'node:http';
import crypto from 'node:crypto';

const subs = new Map(); // socket -> {id, filters}
const server = http.createServer((_, res) => { res.writeHead(426); res.end(); });

function send(socket, text) {
  const data = Buffer.from(text);
  const head = data.length < 126 ? Buffer.from([0x81, data.length])
    : data.length < 65536 ? Buffer.from([0x81, 126, data.length >> 8, data.length & 255])
    : Buffer.concat([Buffer.from([0x81, 127]), (() => { const b = Buffer.alloc(8); b.writeBigUInt64BE(BigInt(data.length)); return b; })()]);
  socket.write(Buffer.concat([head, data]));
}

function frame(buf) {
  if (buf.length < 2) return null;
  const op = buf[0] & 15;
  let len = buf[1] & 127;
  let at = 2;
  if (len === 126) { if (buf.length < 4) return null; len = buf.readUInt16BE(2); at = 4; }
  else if (len === 127) { if (buf.length < 10) return null; len = Number(buf.readBigUInt64BE(2)); at = 10; }
  const masked = buf[1] & 128;
  const key = masked ? buf.subarray(at, at + 4) : null;
  at += masked ? 4 : 0;
  if (buf.length < at + len) return null;
  const payload = Buffer.from(buf.subarray(at, at + len));
  if (key) for (let i = 0; i < len; i++) payload[i] ^= key[i & 3];
  return { op, payload, used: at + len };
}

const matches = (f, ev) => (!f.kinds || f.kinds.includes(ev.kind))
  && (!f['#t'] || ev.tags.some((t) => t[0] === 't' && f['#t'].includes(t[1])))
  && (!f['#p'] || ev.tags.some((t) => t[0] === 'p' && f['#p'].includes(t[1])));

function onText(socket, text) {
  let m;
  try { m = JSON.parse(text); } catch { return; }
  if (m[0] === 'REQ') { subs.set(socket, { id: m[1], filters: m.slice(2) }); send(socket, JSON.stringify(['EOSE', m[1]])); }
  else if (m[0] === 'CLOSE') subs.set(socket, { id: null, filters: [] });
  else if (m[0] === 'EVENT') {
    const ev = m[1];
    send(socket, JSON.stringify(['OK', ev.id, true, '']));
    for (const [s, sub] of subs) if (s !== socket && sub.filters.some((f) => matches(f, ev))) send(s, JSON.stringify(['EVENT', sub.id, ev]));
  }
}

server.on('upgrade', (req, socket) => {
  const accept = crypto.createHash('sha1').update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
  socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
  subs.set(socket, { id: null, filters: [] });
  let buf = Buffer.alloc(0);
  socket.on('data', (d) => {
    buf = Buffer.concat([buf, d]);
    for (let f; (f = frame(buf));) {
      buf = buf.subarray(f.used);
      if (f.op === 8) { socket.end(); return; }
      if (f.op === 1) onText(socket, f.payload.toString());
    }
  });
  const gone = () => subs.delete(socket);
  socket.on('close', gone);
  socket.on('error', gone);
});

server.listen(Number(process.argv[2] || 0), '127.0.0.1', () => console.log(server.address().port));
