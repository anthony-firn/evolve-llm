# Evolve Browser

A web browser engine compiled to WebAssembly, running in a worker inside the workbench page. It
loads real websites, runs their JavaScript, keeps cookies and lets you log in. You and Pi use
the same live page at the same time.

Open it with the **Browser** button in the workbench toolbar.

## How it works

```
workbench page                    Web Worker                                 how bytes reach websites
┌────────────────────────────┐    ┌──────────────────────────────────────┐ wss ┌─────────────────────┐
│ browser panel: canvas,     │◄──►│ Browser (Rust → WebAssembly)         │◄──►│ Tor (default):       │
│ address bar, menus, input  │    │  HTML parser, DOM, CSS, layout, text │    │ Snowflake bridge →   │
│ Pi browser tools           │    │  JavaScript (Boa)                    │    │ 3 volunteer relays   │
└────────────────────────────┘    │  HTTP/1.1 · TLS (rustls) · cookies   │ ws ├─────────────────────┤
                                  │  Tor client (Arti tor-proto, webtor) │◄──►│ or evolve-relay     │
                                  └──────────────────────────────────────┘    │ (yours or community) │
                                                                               └─────────────────────┘
```

| Part | Implementation |
|---|---|
| HTML, DOM, events, forms | [Blitz](https://github.com/DioxusLabs/blitz) 0.3 (html5ever parser) |
| CSS | Stylo, the style engine of Firefox and Servo |
| Layout | Taffy (block, inline, flexbox, grid, tables, floats) |
| Text | Parley and Swash, with bundled fonts |
| Painting | vello_cpu, rasterized in the worker and drawn on a canvas |
| JavaScript | Boa, with the DOM and web APIs written in JavaScript over a small host API (`browser/src/js/prelude.js`, `host_api.rs`) |
| Network | HTTP/1.1 (gzip, deflate, Brotli, chunked, keep-alive) over rustls TLS, over Tor streams (Arti's `tor-proto` via webtor) or Wisp relay streams |

The engine is `browser/` (crate `evolve-browser`); its interfaces are in
[browser/API.md](../browser/API.md). The panel is `web/browser-client.mjs`, the worker
`web/browser-worker.mjs` and `web/browser-host.mjs`, and Pi's tools are in `web/pi-agent.mjs`.

Fonts: Liberation Sans, Serif and Mono (metric-compatible with Arial, Times New Roman and Courier
New; registered under common family names), DejaVu Sans, and Droid Sans Fallback for Chinese,
Japanese and Korean. Licences are in `browser/fonts/`.

## How pages reach websites

A web page cannot open TCP connections, so something outside the page has to carry the engine's
bytes. Nothing in this workbench is hosted for that: by default the engine joins the **Tor
network**, which thousands of volunteers already run, and it can also use an `evolve-relay` on
your own computer or one that someone else announces. Pick the route with the button at the
right of the address bar.

| Route | What carries the bytes | Who sees what |
|---|---|---|
| **Automatic** (default) | Tor. When the workbench itself is served from localhost and a relay answers on `ws://127.0.0.1:8787/` (1.2 s check), that relay. A public page never probes your local network. | as below |
| **Tor network** | A WebSocket to Tor's Snowflake bridge, then a three-hop circuit through volunteer relays to an exit, which opens the TCP connection. | The bridge sees your address, not your sites; the exit sees the site's name, not you. |
| **Relay on this computer** | `evolve-relay`, Wisp v1 to plain TCP. Fastest; sites see your own address. | Only you. |
| **Community relay** | An `evolve-relay --public` that its operator announces on Nostr (`Find community relays`). | Its operator sees your address and the site names. |
| **Another relay** | Any Wisp v1 server you name (`wss://…`). | Its operator. |

In every route **TLS runs inside the engine** (rustls with the Mozilla root store), so neither a
relay nor a Tor exit can read or change pages, form fields, cookies or passwords on `https://`
sites. They see what any network hop sees: host names, timing and sizes. Plain `http://` pages
are not encrypted.

### The Tor route

The engine contains a Tor client: Arti's `tor-proto` (the Tor Project's Rust implementation)
through [webtor](https://github.com/ethereum/webtor-rs), vendored in `browser/vendor/` with the
changes listed there. On start it:

1. loads the relay list (consensus) from the snapshot webtor's CI commits every day, or from the
   copy bundled with the workbench (`web/browser-tor/`, refreshed at build time), skipping copies
   older than three days; if nothing fresh loads, it fetches the consensus through the bridge;
2. opens a WebSocket to the Snowflake bridge (`wss://snowflake.torproject.net/`, then the second
   bridge) and, inside it, the Tor link (Turbo → KCP → SMUX → TLS), checking the bridge's
   identity fingerprint;
3. builds three-hop circuits and opens one Tor stream per TCP connection. The exit resolves host
   names, so there are no DNS lookups outside Tor.

Circuits are **isolated by first-party site**, like Tor Browser: everything a site's pages load
shares that site's circuits, and two sites never share one. The engine starts connecting when the
panel opens; the toolbar button shows `Tor` with an amber dot until the first circuit is ready.

It uses Snowflake's direct WebSocket bridge rather than its WebRTC volunteer proxies: WebRTC is not
available in a Web Worker, the direct path was more reliable in testing, and it leaves the
volunteer proxies to people in censored countries, who depend on them.

Limits of the Tor route: it is slower than a direct connection (each circuit has three hops), many
sites challenge or block Tor exits (Cloudflare checks, Google, Reddit, some logins), and a site may
ask you to sign in again when a circuit changes. The consensus is read without checking the
directory authorities' signatures (webtor does not implement that yet); a forged list could only
choose which relays carry your already-encrypted traffic.

### Run your own relay

```sh
cargo build --release --manifest-path browser/relay/Cargo.toml
browser/relay/target/release/evolve-relay          # ws://127.0.0.1:8787/
```

Then choose **Relay on this computer** (or use Automatic on a locally served workbench). The relay
refuses private, loopback and link-local addresses (so pages cannot reach your local network),
ports other than 80 and 443, and WebSocket connections from pages other than the workbench
origins. Flags: `--listen`, `--allow-origin`, `--allow-port`, `--allow-private`, `--verbose`,
`--public`. See [browser/relay/README.md](../browser/relay/README.md), which also explains how to
join the community relay network with `--public --announce`.

A deployment can preset the default route in `web/browser-config.json`
(`{"relayUrl": "tor:"}`, `"auto"` or a `wss://` URL); the person's own choice wins.

## Using it with Pi

- The page is drawn on a canvas. Click, scroll and type as usual; text goes through a hidden
  field, so IME input and paste work. `<select>` menus open as panel menus.
- Ask the main chat to browse ("go to news.ycombinator.com and open the top story") or click the
  Browser button and use it yourself. Pi reads the page as a numbered outline
  (`[4] textbox "Email" value=""`) and acts with the same pointer and keyboard events you produce,
  so page scripts see ordinary input. An overlay shows where Pi clicks.
- You and Pi can act at once. If you touch the page while Pi is working, Pi's next action is
  refused as stale ("You took over"), and Pi looks at the page again.
- **You log in; Pi never types passwords.** Password, one-time-code and card fields show as
  `(value hidden)` in Pi's outline, and Pi's typing tool refuses them. Pi asks you before clicks
  that buy, send, post, delete or change an account. Page text is treated as untrusted data.

## Cookies, storage and security

- Cookies follow browser rules: Secure, HttpOnly (hidden from page scripts), path and domain
  matching with the public suffix list, and SameSite (Lax when unspecified), so a site cannot make
  your logged-in requests to another site.
- `fetch()` and `XMLHttpRequest` enforce CORS (preflights included); cross-origin responses are
  readable only when the server allows it, and credentials need its explicit permission.
- Cookies and `localStorage` are saved in this browser's IndexedDB, on this device only, and come
  back after a reload. There is no profile sync.
- The engine has no access to the workbench page, your real browser's cookies or saved passwords.

## What works

Measured in Chrome on Linux (1280×900 panel), over the **Tor route** with nothing installed
(`tests/browser_tor_live_test.py`, two runs on 2026-11-14; Tor ready 1.6–2 s after the panel
opens):

| Site | Loaded over Tor | Requests |
|---|---|---|
| example.com | 3.9–5.6 s | 2 |
| Hacker News front page | 5.5–7.1 s | 6 |
| Wikipedia article (WebAssembly) | 10.9–12.3 s | 41–45 |
| Lobsters | 6–13 s | 33–38 |
| rust-lang.org | 15–41 s | 62 |
| GitHub repository page | 25–33 s | 52–58 |
| MDN (HTML reference) | 35 s | 172, 1 failed |

Through a relay on the same computer the same pages load 2–10 times faster:

| Site | Loaded through a local relay | Requests |
|---|---|---|
| example.com | 2.7 s | 2 |
| Hacker News front page | 2.6 s | 8 |
| Wikipedia article (WebAssembly), including its script-built Appearance panel | 3–4.5 s | 41–53 |
| Lobsters | 3.4–4.6 s | 86 |
| rust-lang.org | 3.3 s | 148 |
| MDN (HTML reference) | 3.1 s | 224 |
| GitHub repository page | first paint < 1 s after the HTML; scripts keep loading for ~15 s | 320 |

A login on a public practice site (the-internet.herokuapp.com) worked end to end, through a
local relay and again over Tor: the person typed the username and password into the canvas, Pi
was refused the password field, Pi clicked **Login**, and the session cookie carried through the
redirect to the secure area.

Diagnostics: setting `localStorage['evolve-browser-tor-log'] = '1'` logs the Tor client
(circuits, extends, streams) to the worker console, and `getBrowser().request('torStats')`
returns byte and frame counters for each Snowflake layer.

Tests: 16 engine tests (`cargo test --manifest-path browser/Cargo.toml --test engine`: scripts,
modules and import maps, DOM APIs, events, forms, timers, fetch, cookies, history, the select
menu, the agent sharing a page), 16 unit tests for HTTP, Wisp, cookie and CORS rules and the
snapshot, 29 relay tests (Wisp, limits, Nostr announcements with the BIP-340 vectors), 6 tests for
finding community relays (`tests/relay-directory.test.mjs`), `tests/browser_view_browser_test.py`,
which drives the real engine in Chrome through the relay with the agent and the person on the same
page, and `tests/browser_tor_live_test.py`, which loads real sites over the live Tor network.

## Limits

- **JavaScript is interpreted.** Boa has no JIT, so script-heavy apps are slow and some large
  bundles hit Boa bugs: GitHub's React-rendered file list and some of MDN's custom elements do not
  appear. Server-rendered pages work best.
- **Heavy script apps can wedge the engine.** After a GitHub repository page, the next navigation
  may not finish, or the engine may stop with a stack overflow; the panel then restarts the engine
  (cookies and storage survive). Reload or navigate again.
- **Missing web APIs:** canvas 2D and WebGL, audio and video playback, Web Workers, page
  WebSockets, IndexedDB, Service Workers and WebAuthn. Shadow DOM is approximated in the light
  DOM. `Intl` is a small built-in subset (no ICU data).
- **HTTP/1.1 only**, six connections per host, with stylesheets and documents first. Pages with
  hundreds of scripts load slowly, more so over Tor, where each connection is a Tor stream with
  its own TLS handshake; HTTP/2 would help.
- **Sign-in limits:** sites with bot checks (Google, Cloudflare challenges, Reddit) may refuse the
  engine's TLS fingerprint or the relay's IP address. Passkeys and your real browser's password
  manager do not reach pages inside the engine.
- **Rendering:** Blitz 0.3 is a beta. Some whitespace between inline elements collapses, icon
  spacing in flex layouts can differ, and there is no printing.
- **Size:** the engine is 18.6 MB of WebAssembly after `wasm-opt`, plus 8.6 MB of fonts, both
  downloaded once.

## Build and test

```sh
node tools/build_workbench.mjs --browser-only     # engine → web/browser-pkg, fonts → web/browser-fonts
EVOLVE_SKIP_BROWSER=1 node tools/build_workbench.mjs   # everything else, without the slow browser build
cargo test --manifest-path browser/Cargo.toml     # engine and unit tests (native)
cargo test --manifest-path browser/relay/Cargo.toml
npm test
python3 tests/browser_view_browser_test.py --runtime web
python3 tests/browser_tor_live_test.py --runtime web   # real pages through the live Tor network
node --test tests/relay-directory.test.mjs            # Nostr announcements, BIP-340 vectors
cargo run --manifest-path browser/Cargo.toml --example render -- https://site/ saved-dir/ out.png
```

Hosting the workbench needs `https://raw.githubusercontent.com wss:` (Tor's consensus snapshot, the
Snowflake bridge, Nostr and community relays) and `ws://127.0.0.1:* ws://localhost:*` (a relay on
the visitor's computer) in the content security policy's `connect-src`.
