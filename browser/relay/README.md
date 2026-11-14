# evolve-relay

A small [Wisp v1](https://github.com/MercuryWorkshop/wisp-protocol) relay for Evolve Browser.

Evolve Browser is a web browser engine compiled to WebAssembly that runs inside a page. A page
cannot open TCP sockets, so the engine opens one WebSocket to this relay and multiplexes its TCP
connections over it. The relay connects to the destination and forwards bytes in both
directions. That is all it does.

## What the relay can see

For `https://` sites, TLS is negotiated end to end **inside the engine** (rustls compiled to
WebAssembly, with the Mozilla root store). The relay only ever handles the encrypted TLS
records, so it cannot read page contents, form fields, cookies or passwords, and it cannot
change them without breaking the TLS connection.

It does see what any network hop sees: the destination host name and port of each connection,
when connections open and close, and how many bytes flow. Plain `http://` sites are not
encrypted, so their traffic is readable by the relay, as it is by every other hop.

The relay logs nothing about destinations unless you pass `--verbose`.

## Run it

```sh
cd browser/relay
cargo run --release
# evolve-relay listening on ws://127.0.0.1:8787/
```

Or install the binary once and run it from anywhere:

```sh
cargo install --path browser/relay
evolve-relay
```

The browser's default relay is `ws://127.0.0.1:8787/`, so with the defaults nothing else needs
configuring. Pages served from `https://www.evolvetech.dev`, `http://localhost:<port>` and
`http://127.0.0.1:<port>` can use it.

## Options

| Flag | Meaning |
|---|---|
| `--listen ADDR` | Address to listen on. Default `127.0.0.1:8787`. |
| `--allow-origin URL` | Also accept WebSocket connections from this page origin (repeatable). `scheme://host:*` allows any port. |
| `--allow-private` | Allow loopback, private, link-local and other non-public destinations. For local development only. |
| `--allow-port N` | Allow destination port `N` in addition to 80 and 443 (repeatable). |
| `--verbose` | Log connections and destinations to stderr. |
| `--public` | Settings for a relay reachable from the internet behind a CDN: identify clients by the CDN's `CDN-ClientIP` / `X-Real-IP` / `X-Forwarded-For` header, serve up to 1024 connections, allow each client address 8 connections, 128 open streams and 2 GiB per hour (then new streams are refused with 0x49), close connections idle for 15 minutes, and also allow `https://evolvetech.dev`. Only use it behind a proxy that sets those headers. |
| `--help` | Show help. |

## Hosting a public relay

`evolve-relay --public --listen 0.0.0.0:8080` behind an HTTPS CDN that passes WebSocket upgrades
gives visitors a `wss://` relay with nothing to install. A static Linux build:

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

The page origin check stops other websites from using the relay from their visitors' browsers,
but any program can send an allowed `Origin` header, so the per-client limits, the port
allowlist and the private-address block are what protect a public relay from abuse. The workbench
reads its relay from `web/browser-config.json` (`{"relayUrl": "wss://…/"}`); a relay set in the
panel overrides it.

## Security

The relay is an open TCP proxy for the pages allowed to use it, so it is locked down by default:

- **Page origins.** The WebSocket upgrade must carry an allowed `Origin` header. Anything else,
  including a missing or `null` origin, gets HTTP 403. This stops arbitrary websites you visit
  in your normal browser from using a relay running on your machine.
- **Destinations.** Every address a host name resolves to is checked, and the connection is
  refused (Wisp close reason `0x48`) if **any** of them is non-public: loopback, private ranges
  (10/8, 172.16/12, 192.168/16), link-local (169.254/16, fe80::/10), carrier-grade NAT
  (100.64/10), multicast, broadcast, unspecified, reserved and documentation ranges, IPv6
  unique-local (fc00::/7), Teredo, and IPv4 addresses embedded in IPv6 (IPv4-mapped,
  IPv4-compatible, NAT64, 6to4). This keeps pages from reaching your router, local services or
  cloud metadata endpoints. `--allow-private` turns this off.
- **Ports.** Only 80 and 443 unless `--allow-port` adds more, so the relay cannot be used to send
  mail or reach SSH and database ports.
- **Limits.** At most 64 WebSocket connections at once (more get HTTP 503) and 256 streams per
  connection (more get close reason `0x49`). Messages are capped at 1 MiB, the handshake must
  finish within 10 seconds and each TCP connect within 10 seconds.
- **Flow control.** Each stream buffers at most 128 packets. The relay sends `CONTINUE` after
  every 64 packets it writes to the destination; a client that ignores its credit has the
  stream closed.

Listening on a public interface (`--listen 0.0.0.0:8787`) makes the relay reachable by anyone who
can reach that address, subject to the origin check, which a non-browser client can fake.
Put it behind authentication and rate limits before exposing it.

## Protocol notes

- On connect the relay sends `CONTINUE` for stream 0 with a buffer of 128 packets.
- Only TCP streams (`0x01`) are supported; other types are closed with `0x41`.
- Data from the destination is sent in `DATA` packets of at most 32 KiB.
- Close reasons use the Wisp v1 numbering: `0x01` unspecified, `0x02` voluntary (the
  destination closed the connection), `0x03` network error, `0x41` invalid stream information
  or unsupported stream type, `0x42` unreachable host (including DNS failure and invalid host
  names), `0x43` connection timeout, `0x44` connection refused, `0x47` TCP data transfer
  timeout, `0x48` blocked address or port, `0x49` throttled (too many streams).

## Tests

```sh
cargo test                     # codec, address policy, origins, and end-to-end relay tests
cargo test -- --ignored        # also fetch http://example.com through the relay
```
