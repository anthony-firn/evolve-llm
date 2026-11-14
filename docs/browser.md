# Evolve Browser

A web browser engine compiled to WebAssembly, running in a worker inside the workbench page. It
loads real websites, runs their JavaScript, keeps cookies and lets you log in. You and Pi use
the same live page at the same time.

Open it with the **Browser** button in the workbench toolbar.

## How it works

```
workbench page                    Web Worker                                  evolve-relay
┌────────────────────────────┐    ┌──────────────────────────────────────┐ ws ┌──────────────┐
│ browser panel: canvas,     │◄──►│ Browser (Rust → WebAssembly)         │◄──►│ Wisp v1 →    │
│ address bar, menus, input  │    │  HTML parser, DOM, CSS, layout, text │    │ TCP sockets  │
│ Pi browser tools           │    │  JavaScript (Boa)                    │    └──────┬───────┘
└────────────────────────────┘    │  HTTP/1.1 · TLS (rustls) · cookies   │           │ TLS bytes
                                  └──────────────────────────────────────┘     websites
```

| Part | Implementation |
|---|---|
| HTML, DOM, events, forms | [Blitz](https://github.com/DioxusLabs/blitz) 0.3 (html5ever parser) |
| CSS | Stylo, the style engine of Firefox and Servo |
| Layout | Taffy (block, inline, flexbox, grid, tables, floats) |
| Text | Parley and Swash, with bundled fonts |
| Painting | vello_cpu, rasterized in the worker and drawn on a canvas |
| JavaScript | Boa, with the DOM and web APIs written in JavaScript over a small host API (`browser/src/js/prelude.js`, `host_api.rs`) |
| Network | HTTP/1.1 (gzip, deflate, Brotli, chunked, keep-alive) over rustls TLS over Wisp streams |

The engine is `browser/` (crate `evolve-browser`); its interfaces are in
[browser/API.md](../browser/API.md). The panel is `web/browser-client.mjs`, the worker
`web/browser-worker.mjs` and `web/browser-host.mjs`, and Pi's tools are in `web/pi-agent.mjs`.

Fonts: Liberation Sans, Serif and Mono (metric-compatible with Arial, Times New Roman and Courier
New; registered under common family names), DejaVu Sans, and Droid Sans Fallback for Chinese,
Japanese and Korean. Licences are in `browser/fonts/`.

## The relay

A web page cannot open TCP connections, so the engine opens one WebSocket to a relay and carries
its connections over it ([Wisp v1](https://github.com/MercuryWorkshop/wisp-protocol)).

TLS is negotiated **inside the engine** with rustls and the Mozilla root store. The relay forwards
encrypted bytes only: it cannot read or change pages, form fields, cookies or passwords. It sees
what any network hop sees: host names, ports, timing and sizes. Plain `http://` traffic is not
encrypted.

The workbench uses the relay named in `web/browser-config.json` (`{"relayUrl": "wss://…/"}`), so a
deployment can provide one and visitors install nothing; a hosted relay runs
`evolve-relay --public` with per-visitor limits (see the relay README). Without that file, or to
use your own, run it on your computer:

```sh
cargo build --release --manifest-path browser/relay/Cargo.toml
browser/relay/target/release/evolve-relay          # ws://127.0.0.1:8787/
```

The relay refuses private, loopback and link-local addresses (so pages cannot reach your local
network), ports other than 80 and 443, and WebSocket connections from pages other than the
workbench origins. Flags: `--listen`, `--allow-origin`, `--allow-port`, `--allow-private`,
`--verbose`. See [browser/relay/README.md](../browser/relay/README.md). The panel's **Relay** chip
shows whether it is reachable and lets you point the browser at another relay.

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

Measured in Chrome on Linux through a local relay (1280×900 panel):

| Site | Loaded | Requests |
|---|---|---|
| example.com | 2.7 s | 2 |
| Hacker News front page | 2.6 s | 8 |
| Wikipedia article (WebAssembly), including its script-built Appearance panel | 3–4.5 s | 41–53 |
| Lobsters | 3.4–4.6 s | 86 |
| rust-lang.org | 3.3 s | 148 |
| MDN (HTML reference) | 3.1 s | 224 |
| GitHub repository page | first paint < 1 s after the HTML; scripts keep loading for ~15 s | 320 |

A login on a public practice site (the-internet.herokuapp.com) worked end to end: the person
typed the username and password into the canvas, Pi was refused the password field, Pi clicked
**Login**, and the session cookie carried through the redirect to the secure area.

Tests: 16 engine tests (`cargo test --manifest-path browser/Cargo.toml --test engine`: scripts,
modules and import maps, DOM APIs, events, forms, timers, fetch, cookies, history, the select
menu, the agent sharing a page), 16 unit tests for HTTP, Wisp, cookie and CORS rules and the
snapshot, 16 relay tests, and `tests/browser_view_browser_test.py`, which drives the real engine
in Chrome through the relay with the agent and the person on the same page.

## Limits

- **JavaScript is interpreted.** Boa has no JIT, so script-heavy apps are slow and some large
  bundles hit Boa bugs: GitHub's React-rendered file list and some of MDN's custom elements do not
  appear. Server-rendered pages work best.
- **Missing web APIs:** canvas 2D and WebGL, audio and video playback, Web Workers, page
  WebSockets, IndexedDB, Service Workers and WebAuthn. Shadow DOM is approximated in the light
  DOM. `Intl` is a small built-in subset (no ICU data).
- **HTTP/1.1 only**, six connections per host, with stylesheets and documents first. Pages with
  hundreds of scripts load slowly; HTTP/2 would help.
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
cargo run --manifest-path browser/Cargo.toml --example render -- https://site/ saved-dir/ out.png
```

Hosting the workbench needs `ws://127.0.0.1:* ws://localhost:*` (or your relay's `wss://` URL) in
the content security policy's `connect-src`.
