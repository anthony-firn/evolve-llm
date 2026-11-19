# Evolve Browser: interfaces between parts

This file is the contract between the parts of the browser that are built in parallel. Change it
only together with every side that depends on it.

```
main thread                         Web Worker (browser-worker.mjs)          route to websites
┌───────────────────────────┐ msgs  ┌──────────────────────────────────┐ wss ┌───────────────────┐
│ BrowserClient             │◄────►│ wasm Browser (browser/src/web.rs)│◄──►│ Tor: Snowflake     │
│  canvas, input, overlay   │       │  Blitz DOM/CSS/layout, Boa JS,   │     │ bridge → 3 relays  │
│ Pi tools (pi-agent.mjs)   │       │  rustls, cookies, vello_cpu,     │ ws  ├───────────────────┤
└───────────────────────────┘       │  Tor client (vendor/webtor)      │◄──►│ or evolve-relay    │
                                    └──────────────────────────────────┘     │ (Wisp v1 → TCP)    │
                                                                             └───────────────────┘
```

## 1. WASM exports (`browser/src/web.rs`, wasm-bindgen, `--target web`)

Built to `web/browser-pkg/browser.js` + `browser_bg.wasm`. Everything runs inside one dedicated
worker. All coordinates are CSS pixels relative to the top-left of the page viewport.

```ts
class Browser {
  // fonts: [[fileName, Uint8Array], ...] (see web/browser-fonts/), wake: called (with no args)
  // whenever the engine wants tick() to run soon.
  // relayUrl is the route: "tor:" (the Tor network over Snowflake) or a Wisp relay's ws(s):// URL.
  constructor(fonts: Array<[string, Uint8Array]>, relayUrl: string, cookiesJson: string | null,
              storageJson: string | null, wake: () => void);
  resize(cssWidth: number, cssHeight: number, devicePixelRatio: number): void;
  setRelay(url: string): void;          // switch the route ("tor:" or ws(s)://); drops open connections
  setIdentity(profile: 'legacy' | 'firefox-esr' | 'firefox' | 'chrome' | 'evolve'): void;
  setTlsProfile(profile: 'default' | 'firefox-order'): void;
  // Both profile setters are diagnostic and throw after navigation starts. Restart to change.
  prepareNetwork(): void;              // start connecting now (Tor: bridge + first circuit, ~5–20 s)
  setTorConsensus(baseUrls: string[]): void; // where consensus.txt.br / microdescriptors.txt.br live
  navigate(url: string): void;          // typed into the address bar; "example.com" → https://
  back(): void; forward(): void; reload(): void; stop(): void;

  // Run the event loop: network completions, scripts, timers, rAF, style, layout, paint.
  // Returns ms until tick() should run again for timers (-1 = only when wake() is called).
  tick(nowMs: number): number;
  // Latest rendered frame since the last call, or undefined. RGBA8, premultiplied=false,
  // frameWidth() x frameHeight() device pixels.
  takeFrame(): Uint8Array | undefined;
  frameWidth(): number; frameHeight(): number;

  // Input from the person (and from the agent, see 3.)
  pointer(kind: "down" | "move" | "up" | "leave", x: number, y: number, button: number,
          buttons: number, mods: number): void;   // mods bits: 1 shift, 2 ctrl, 4 alt, 8 meta
  wheel(x: number, y: number, dx: number, dy: number, deltaMode: number, mods: number): void;
  key(kind: "down" | "up", key: string, code: string, mods: number, repeat: boolean): void;
  imeCommit(text: string): void;       // text from the hidden textarea (typing, paste, IME)
  imePreedit(text: string): void;
  chooseOption(id: number, index: number): void; // answer to a "select" event; -1 = cancelled

  // JSON array of events since the last call (see 2.)
  takeEvents(): string;
  cookiesJson(): string;               // persist in IndexedDB, pass back to the constructor
  storageJson(): string;               // localStorage per origin, same
  clearData(): void;                   // forget cookies and storage
  networkLog(): string;                // JSON [{method,url,status,bytes,ms,dest,error}]
  evaluate(code: string): string;      // devtools console: result or error text

  // Agent API (3.)
  snapshot(maxChars: number): string;  // JSON Snapshot
  agentClick(ref: number): string;     // JSON ActionResult
  agentType(ref: number, text: string, submit: boolean): string;
  agentSelect(ref: number, value: string): string;
  agentScroll(dy: number): string;
  agentKey(key: string): string;       // "Enter", "Escape", "Tab", "ArrowDown", ...
}
```

## 2. Engine events (`takeEvents()`), tagged by `type`

```ts
{ type: "title", title }                         { type: "url", url, canBack, canForward }
{ type: "loading", loading: boolean, url }        { type: "cursor", cursor }   // CSS cursor name
{ type: "console", level, text }                  { type: "dialog", kind, text } // alert/confirm/prompt/clipboard
{ type: "clipboard", text }                       { type: "storage" }  { type: "cookies" } // persist now
{ type: "error", text }                           { type: "download", url, name }
{ type: "ime", enabled, x, y, w, h }              // where to put the hidden textarea (CSS px)
{ type: "select", id, x, y, w, h, multiple,       // the person clicked a <select>: show a menu at
  options: [{ label, value, selected, disabled, group }] } // the box; reply with chooseOption(id, i)
{ type: "network", route: "tor", state: "connecting" | "ready" | "failed", detail } // route progress
```

## 3. Agent API (shared page, person and agent at the same time)

`snapshot(maxChars)` returns:

```ts
type Snapshot = {
  url: string; title: string; scrollY: number; pageHeight: number; viewportHeight: number;
  text: string;          // compact outline, see below, at most maxChars characters
  refs: number;          // number of refs in this snapshot
  version: number;       // increments on every person input; actions on an older version fail
};
```

`text` is an indented outline of what a person can see, in document order, e.g.

```
# Hacker News
[1] link "Hacker News"  [2] link "new"  [3] link "login"
## Login
[4] textbox "username" value=""
[5] password "password" (value hidden)
[6] button "login"
Lorem ipsum paragraph text, truncated…
```

Refs number interactive elements: links, buttons, inputs, selects, textareas, checkboxes,
radios, summary, elements with onclick/role=button/tabindex. Password values are never
included. Refs are valid until the next snapshot.

`ActionResult` = `{ ok: boolean, error?: string, x?: number, y?: number, w?: number, h?: number,
navigated?: boolean }`. x/y/w/h is the target's box in viewport CSS px, so the UI can draw the
agent's cursor and highlight. Every agent action is applied through the same input path as the
person's (pointer down/up, key events), so page scripts see real events. Agent actions fail with
`error: "stale"` if the person has interacted since the snapshot; the agent then snapshots again.
`agentType` refuses password fields (`error: "password"`): the person types passwords.

Rust side (`browser/src/agent.rs`), called by the engine:

```rust
pub struct Snapshot { pub text: String, pub refs: Vec<blitz_dom::NodeId> } // ref n = refs[n - 1]
pub fn snapshot(doc: &blitz_dom::BaseDocument, max_chars: usize) -> Snapshot;
/// The element's box in viewport CSS px (x, y, w, h), if it has one.
pub fn element_box(doc: &blitz_dom::BaseDocument, node: blitz_dom::NodeId) -> Option<(f32, f32, f32, f32)>;
pub fn is_password(doc: &blitz_dom::BaseDocument, node: blitz_dom::NodeId) -> bool;
```

The doc passed in is fully resolved (styles and layout up to date).

## 4. Main-thread client (`web/browser-client.mjs`)

```ts
export class BrowserClient extends EventTarget {
  // relayUrl: the default route setting ("auto" when omitted; see docs/browser.md). The person's
  // own choice is saved in localStorage and wins.
  constructor(options?: { relayUrl?: string; identityProfile?: string; tlsProfile?: string });
  identityProfile: string; tlsProfile: string;
  setProfiles(identityProfile: string, tlsProfile?: string): Promise<void>;
  // setProfiles persists cookies/localStorage, restarts the worker and reloads the current URL.
  mount(container: HTMLElement): void;   // address bar, canvas, status, agent overlay
  ready: Promise<void>;
  navigate(url: string): Promise<void>;
  back(): void; forward(): void; reload(): void;
  // choice: "auto" | "tor:" | ws(s):// URL; relayUrl: the route in use ("tor:" or a URL);
  // relayOk: null while connecting; network: the latest "network" event.
  get state(): { url: string; title: string; loading: boolean; relayUrl: string; relayOk: boolean | null;
                 choice: string; network: { route: string; state: string; detail: string } | null };
  setRelay(choice: string): Promise<boolean | null>;
  // Agent API: same names as 3.; each shows the agent cursor/highlight on the canvas and resolves
  // with the parsed JSON. While an agent action runs the panel shows "Pi is using the page".
  snapshot(maxChars?: number): Promise<Snapshot>;
  click(ref: number): Promise<ActionResult>;
  type(ref: number, text: string, opts?: { submit?: boolean }): Promise<ActionResult>;
  select(ref: number, value: string): Promise<ActionResult>;
  scroll(dy: number): Promise<ActionResult>;
  key(key: string): Promise<ActionResult>;
  waitForIdle(timeoutMs?: number): Promise<void>; // until loading is false and the network is quiet
  // events: "title", "url", "loading", "console", "dialog", "agent" ({action, ref, ok})
}
```

## 5. Routes to websites

`tor:` uses `browser/src/net/tor.rs` over the vendored `webtor` crate (`browser/vendor/`): a
WebSocket to Snowflake's bridge (Turbo → KCP → SMUX → TLS → Tor channel), three-hop circuits from
the cached consensus, one Tor stream per TCP connection, circuits isolated by first-party site.
**Automatic** (`auto`, the default) is Tor, except on a page served from localhost, where it first
tries the local relay (1.2 s probe). Public pages never probe the local network.

## 6. Relay (`browser/relay`, binary `evolve-relay`)

Wisp v1 server: `ws://127.0.0.1:8787/` by default. Only TCP streams. Refuses loopback, private,
link-local, CGNAT, multicast and unspecified addresses after DNS resolution (unless
`--allow-private`), ports other than 80/443 (unless `--allow-port N`), and WebSocket upgrades
whose `Origin` is not allowed (default: `https://www.evolvetech.dev`, `http://localhost:*`,
`http://127.0.0.1:*`; `--allow-origin URL` adds more). The local relay URL in the client:
`ws://127.0.0.1:8787/`.
