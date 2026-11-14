# Vendored crates

## webtor and subtle-tls

From [webtor-rs](https://github.com/ethereum/webtor-rs) at commit
`a2432984ee657fec6905269cdfeb013254ef2c26` (webtor 0.5.7), MIT licence (see `LICENSE-webtor`).
webtor is a Tor client for web pages built on Arti's `tor-proto`; `subtle-tls` is the TLS it uses
for the link to the Snowflake bridge. The Arti crates themselves come from the same commit of the
webtor-rs repository through `[patch.crates-io]` in `../Cargo.toml` (they carry webtor's
WebAssembly changes: time without `std::time::Instant`, `?Send` async traits).

Changes for Evolve Browser:

- **Web Worker support.** The engine runs in a dedicated worker, where `window` does not exist.
  `webtor/src/js_global.rs` calls `fetch`, `setTimeout` and `performance.now` on `globalThis`;
  `directory.rs`, `snowflake_broker.rs`, `wasm_runtime.rs` and `time.rs` use it, and subtle-tls's
  `cert.rs` and `handshake.rs` use its existing `globalThis.crypto` fallback.
- **Owned circuits and raw streams.** `TorClient::bootstrap_channel()` connects to the bridge
  without building a circuit, `TorClient::build_circuit()` builds a three-hop circuit the caller
  owns (`CircuitManager::create_unlisted_circuit`), and `has_channel()` reports whether the bridge
  connection is up. The engine keeps its own per-site circuits (`src/net/tor.rs`) and opens
  `DataStream`s with `ClientTunnel::begin_stream`, running its own HTTP and rustls TLS on top, so
  page TLS never uses subtle-tls. (`connect_stream`, which goes through webtor's isolation-key
  manager, is kept but unused.)
- **Snowflake transport fixes.** Without these, any page bigger than a few dozen kilobytes stalled
  the bridge link:
  - `kcp_stream.rs`: each KCP output packet is sent as its own Turbo frame (they were
    concatenated, and the Snowflake server drops frames larger than one packet, including every
    retransmission of them); ACKs and window updates are flushed as soon as data arrives or is
    read, instead of waiting for an update timer that nothing drove; receive window 1024
    segments.
  - `smux.rs`: the `AsyncRead` path sends window updates (UPD) as data is consumed (only an unused
    helper did, so the server stopped after its window); our window is 1 MiB like the Snowflake
    Go client; written frames are capped at 32 KiB (the 16-bit length used to wrap).
  - `subtle-tls/src/stream.rs`: a TLS record carries at most 2^14 bytes of plaintext.
- **Diagnostics.** `stats.rs` counts bytes and frames at each layer (`Browser.torStats()` in the
  engine); the engine can route webtor's `tracing` output to the worker console.
- **Consensus sources.** `directory::set_consensus_sources(urls)` replaces the hard-coded
  snapshot URL; sources are tried in order. The default is the snapshot webtor's CI commits every
  day (raw.githubusercontent.com): the GitHub Pages copy stopped updating when the repository
  moved and holds keys that relays no longer use. A snapshot older than three days is skipped
  while a fresher one exists; if only stale ones load, `cached_consensus_is_stale()` tells the
  engine to fetch a fresh consensus through the bridge (`TorClient::bootstrap`).
- **No `psl` crate.** `IsolationKey::from_url`'s per-domain policy falls back to the host; the
  engine always passes explicit first-party keys computed with its own public suffix list.
- Manifests rewritten without workspace inheritance, benches, fuzzers or tests.

Known limits inherited from webtor: the cached consensus is parsed without checking the directory
authorities' signatures (only its age is checked), and there is no onion-service
support. HTTPS page content is still protected end to end by the engine's own TLS; a forged
consensus could only choose the relays a circuit uses.
