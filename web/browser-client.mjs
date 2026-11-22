/**
 * Main-thread side of the WASM browser (browser/API.md section 4).
 *
 * The engine runs in a dedicated worker and draws into a transferred OffscreenCanvas. This
 * module forwards the person's pointer, wheel and keyboard input, shows engine events (title,
 * address, loading, console, dialogs), persists cookies and localStorage in IndexedDB, and gives
 * Pi the same page through the agent API, drawing Pi's cursor on top of the canvas.
 */

export const DEFAULT_RELAY = 'ws://127.0.0.1:8787/';
/** Route value for the Tor network (reached over Snowflake; nothing to install or host). */
export const TOR_ROUTE = 'tor:';
/** Route setting: Tor, or the relay on this computer when the page itself is served locally. */
export const AUTO_ROUTE = 'auto';
export const RELAY_DOCS = 'https://github.com/anthony-firn/evolve-llm/blob/main/docs/browser.md#how-pages-reach-websites';
const RELAY_KEY = 'evolve-browser-relay';
export const IDENTITY_PROFILES = Object.freeze({ legacy: 'Evolve + Chrome 131 (default)', 'firefox-esr': 'Firefox ESR 153 (experiment)', firefox: 'Firefox 160 (experiment)', chrome: 'Chrome 131 (experiment)', evolve: 'EvolveBrowser only' });
export const TLS_PROFILES = Object.freeze({ default: 'rustls (default)', 'firefox-order': 'NSS cipher order (partial experiment)' });
const IDENTITY_KEY = 'evolve-browser-identity';
const TLS_KEY = 'evolve-browser-tls';
function validProfile(profiles, value) {
  if (!Object.hasOwn(profiles, value)) throw new TypeError('Unknown browser profile: ' + value);
  return value;
}

/** True when this page is served from this computer (no Local Network Access prompt for a local relay). */
export function isLocalOrigin(loc = globalThis.location) {
  return /^(localhost|127\.\d+\.\d+\.\d+|\[::1\])$/i.test(loc?.hostname || '');
}

/** A short name for a route, for the toolbar chip. */
export function routeLabel(route) {
  if (route === TOR_ROUTE) return 'Tor';
  if (route === DEFAULT_RELAY) return 'Local relay';
  return 'Relay';
}
const MAX_CONSOLE = 300;

/** Address-bar text to a URL: schemes pass through, bare hosts get https://, other text searches. */
export function addressToUrl(text) {
  const t = String(text ?? '').trim();
  if (!t) return '';
  if (/^(localhost|127\.\d+\.\d+\.\d+|\[::1\])(:\d+)?(\/|$)/i.test(t)) return 'http://' + t;
  if (/^[a-z][a-z0-9+.-]*:/i.test(t) && !/^[^/]+\.[a-z]{2,}:\d+/i.test(t)) return t;
  if (/\s/.test(t) || !/^[^/?#]+\.[^/?#.]{2,}([/?#:]|$)/.test(t)) return 'https://html.duckduckgo.com/html/?q=' + encodeURIComponent(t);
  return 'https://' + t;
}

/** Start ordinary Google sign-in in the real browser, without carrying panel session tokens. */
export function externalGoogleSignInUrl(value) {
  let current;
  try { current = new URL(value); } catch { return null; }
  if (current.protocol !== 'https:' || current.hostname !== 'accounts.google.com' || current.port || current.username || current.password) return null;
  if (!['/', '/ServiceLogin', '/signin', '/v3/signin/identifier', '/v3/signin/rejected', '/signin/v2/identifier'].includes(current.pathname)) return null;
  const flow = current.searchParams.get('flowName');
  if (flow && !['GlifWebSignIn', 'WebLiteSignIn'].includes(flow)) return null;
  if (['client_id', 'redirect_uri', 'scope', 'state'].some(key => current.searchParams.has(key))) return null;
  const target = new URL('https://accounts.google.com/ServiceLogin');
  for (const key of ['continue', 'service', 'hl']) {
    if (current.searchParams.has(key)) target.searchParams.set(key, current.searchParams.get(key));
  }
  return target.href;
}

export function modifierBits(e) {
  return (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);
}

function readSetting(key) { try { return globalThis.localStorage?.getItem(key) ?? null; } catch { return null; } }
function writeSetting(key, value) { try { globalThis.localStorage?.setItem(key, value); } catch {} }

/** Cookies and per-origin localStorage, kept in IndexedDB. Every access tolerates failure. */
export function indexedDbPersistence(name = 'evolve-browser-v1') {
  const open = () => new Promise((resolve, reject) => {
    let req;
    try { req = globalThis.indexedDB.open(name, 1); } catch (e) { reject(e); return; }
    req.onupgradeneeded = () => { try { req.result.createObjectStore('state'); } catch {} };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB is unavailable'));
    req.onblocked = () => reject(new Error('IndexedDB is blocked'));
  });
  return {
    async load() {
      try {
        const db = await open();
        return await new Promise((resolve) => {
          try {
            const tx = db.transaction('state', 'readonly');
            const get = tx.objectStore('state').get('current');
            get.onsuccess = () => resolve(get.result || {});
            get.onerror = () => resolve({});
            tx.oncomplete = () => db.close();
          } catch { db.close(); resolve({}); }
        });
      } catch { return {}; }
    },
    async save(value) {
      try {
        const db = await open();
        await new Promise((resolve) => {
          try {
            const tx = db.transaction('state', 'readwrite');
            tx.objectStore('state').put(value, 'current');
            tx.oncomplete = () => { db.close(); resolve(); };
            tx.onerror = tx.onabort = () => { db.close(); resolve(); };
          } catch { db.close(); resolve(); }
        });
      } catch {}
    },
  };
}

/** Resolves true when a WebSocket to the relay opens within the time limit. */
export function probeRelay(url, timeoutMs = 3000) {
  return new Promise((resolve) => {
    let ws, done = false;
    const finish = (ok) => { if (done) return; done = true; clearTimeout(timer); try { ws && ws.close(); } catch {} resolve(ok); };
    const timer = setTimeout(() => finish(false), timeoutMs);
    try { ws = new WebSocket(url); } catch { finish(false); return; }
    ws.onopen = () => finish(true);
    ws.onerror = () => finish(false);
    ws.onclose = () => finish(false);
  });
}

function dialogMessage({ kind, text }) {
  const t = String(text ?? '').trim();
  const shown = t.length > 300 ? t.slice(0, 300) + '…' : t;
  if (kind === 'confirm') return `The page asked “${shown}”. It was answered OK and the page continued.`;
  if (kind === 'prompt') return `The page asked for input: “${shown}”. It was left empty and the page continued.`;
  return `The page showed an alert: “${shown}”. The page continued.`;
}

function safeHttpUrl(url) {
  try { const u = new URL(String(url)); return u.protocol === 'https:' || u.protocol === 'http:' ? u.href : null; } catch { return null; }
}

/**
 * The menu shown when the person clicks a <select>. DOM-free: the view renders `rows`, and
 * calls key()/click()/cancel(). `choose(index)` answers the engine (index into the event's
 * options, which match the select's options in document order).
 */
export class SelectMenu {
  constructor({ id, multiple, options = [], x = 0, y = 0, w = 0, h = 0 }, { choose, close, render = () => {}, now = () => Date.now() }) {
    this.id = id;
    this.multiple = !!multiple;
    this.box = { x, y, w, h };
    this.options = options.map((o, index) => ({ index, label: String(o.label ?? o.value ?? ''), value: String(o.value ?? ''), selected: !!o.selected, disabled: !!o.disabled, group: o.group || '' }));
    this._choose = choose; this._close = close; this._render = render; this._now = now;
    this.open = true;
    this._typed = ''; this._typedAt = 0;
    const selected = this.options.find((o) => o.selected && !o.disabled);
    this.active = selected ? selected.index : this._step(-1, 1);
  }

  /** Options with a header row whenever the optgroup changes. */
  get rows() {
    const rows = [];
    let group = '';
    for (const o of this.options) {
      if (o.group !== group) { group = o.group; if (group) rows.push({ kind: 'group', label: group }); }
      rows.push({ kind: 'option', ...o, active: o.index === this.active });
    }
    return rows;
  }

  _step(from, dir) {
    for (let i = from + dir; i >= 0 && i < this.options.length; i += dir) if (!this.options[i].disabled) return i;
    return from >= 0 && from < this.options.length ? from : -1;
  }

  _edge(dir) { return this._step(dir > 0 ? -1 : this.options.length, dir); }

  _pick(index) {
    const o = this.options[index];
    if (!this.open || !o || o.disabled) return false;
    this.active = index;
    this._choose(index);
    if (this.multiple) { o.selected = !o.selected; this._render(this); }
    else { for (const x of this.options) x.selected = x.index === index; this._finish(); }
    return true;
  }

  _finish() { if (!this.open) return; this.open = false; this._close(); }

  click(index) { return this._pick(index); }

  /** Close without a new choice (Escape, click outside, navigation). */
  cancel() { if (!this.open) return; this._choose(-1); this._finish(); }

  /** Close a multiple-select menu after toggling. */
  done() { this.cancel(); }

  /** Keyboard handling; returns true when the key was used. */
  key(key) {
    if (!this.open) return false;
    const move = (to) => { if (to >= 0) { this.active = to; this._render(this); } return true; };
    switch (key) {
      case 'ArrowDown': return move(this._step(this.active, 1));
      case 'ArrowUp': return move(this._step(this.active, -1));
      case 'Home': case 'PageUp': return move(this._edge(1));
      case 'End': case 'PageDown': return move(this._edge(-1));
      case 'Enter': if (this.multiple) { this.done(); return true; } return this.active >= 0 ? (this._pick(this.active), true) : (this.cancel(), true);
      case ' ': if (this.multiple) return this.active >= 0 ? (this._pick(this.active), true) : true; return this.active >= 0 ? (this._pick(this.active), true) : true;
      case 'Escape': this.cancel(); return true;
      case 'Tab': if (!this.multiple && this.active >= 0) this._pick(this.active); else this.cancel(); return true;
    }
    if (key.length === 1) {
      const t = this._now();
      this._typed = (t - this._typedAt < 700 ? this._typed : '') + key.toLowerCase();
      this._typedAt = t;
      const n = this.options.length;
      const start = this._typed.length > 1 ? this.active : this.active + 1;
      for (let k = 0; k < n; k++) {
        const o = this.options[(((start + k) % n) + n) % n];
        if (!o.disabled && o.label.trim().toLowerCase().startsWith(this._typed)) return move(o.index);
      }
      return true;
    }
    return false;
  }
}

const defaultWorker = () => new Worker(new URL('./browser-worker.mjs', import.meta.url), { type: 'module' });

export class BrowserClient extends EventTarget {
  constructor({ relayUrl, identityProfile, tlsProfile, workerFactory = defaultWorker, persistence = indexedDbPersistence(), probe = probeRelay, timeoutMs = 30000, agentTimeoutMs = 25000 } = {}) {
    super();
    this.identityProfile = validProfile(IDENTITY_PROFILES, identityProfile ?? readSetting(IDENTITY_KEY) ?? 'legacy');
    this.tlsProfile = validProfile(TLS_PROFILES, tlsProfile ?? readSetting(TLS_KEY) ?? 'default');
    this._workerFactory = workerFactory;
    this._persist = persistence;
    this._probe = probe;
    this._timeoutMs = timeoutMs;
    this._agentTimeoutMs = agentTimeoutMs;
    this._seq = 0;
    this._pending = new Map();
    this._worker = null;
    this._starting = null;
    this._ui = null;
    this._size = { width: 1280, height: 800, dpr: 1 };
    this._agentActive = 0;
    this._crashes = [];
    this._console = [];
    this._errors = 0;
    this._lastActivity = 0;
    this._persistTimer = null;
    // The person's own route setting wins, then the site's configured route, then Auto (Tor, or
    // the relay on this computer when the workbench itself is served locally).
    this._defaultRelay = relayUrl || AUTO_ROUTE;
    const choice = readSetting(RELAY_KEY) || this._defaultRelay;
    this._state = { url: '', title: '', loading: false, choice, relayUrl: choice === AUTO_ROUTE ? TOR_ROUTE : choice, relayOk: null, network: null, canBack: false, canForward: false };
    this._ready = new Promise((resolve, reject) => { this._readyResolve = resolve; this._readyReject = reject; });
    this._ready.catch(() => {});
  }

  /** Resolves when the engine has started (starting it if nothing else has). */
  get ready() {
    this._start().catch(() => {});
    return this._ready;
  }

  get state() {
    const { url, title, loading, relayUrl, relayOk, choice, network } = this._state;
    return { url, title, loading, relayUrl, relayOk, choice, network };
  }

  get console() { return this._console.slice(); }

  // ---- worker protocol ----

  _start() {
    if (this._halted) return Promise.reject(new Error('The browser engine stopped repeatedly on this page. Press Reload to start it again.'));
    if (this._starting) return this._starting;
    this._starting = (async () => {
      const [saved, route] = await Promise.all([this._persist.load().catch(() => ({})), this._resolveRoute(this._state.choice)]);
      this._state.relayUrl = route;
      this._spawn(saved || {});
      this.checkRelay();
    })();
    return this._starting;
  }

  _spawn(saved) {
    const worker = this._worker = this._workerFactory();
    worker.onmessage = ({ data }) => { if (this._worker === worker) this._onMessage(data); };
    worker.onerror = (event) => {
      event?.preventDefault?.();
      if (this._worker === worker) this._recover({ error: event?.message || 'The browser worker failed.' });
    };
    const message = { type: 'init', identityProfile: this.identityProfile, tlsProfile: this.tlsProfile, relayUrl: this._state.relayUrl, cookies: saved.cookies ?? null, storage: saved.storage ?? null, torLog: readSetting('evolve-browser-tor-log') === '1', ...this._size };
    const transfer = [];
    const offscreen = this._ui?.takeOffscreen();
    if (offscreen) { message.canvas = offscreen; transfer.push(offscreen); }
    const init = this._call(message, transfer, 120000);
    init.then(() => this._readyResolve(), (e) => {
      this._readyReject(e);
      this._ui?.status(String(e.message || e), 'error');
      this.dispatchEvent(new CustomEvent('error', { detail: { text: String(e.message || e) } }));
    });
    return init;
  }

  _call(message, transfer = [], timeoutMs = this._timeoutMs) {
    return new Promise((resolve, reject) => {
      const worker = this._worker;
      if (!worker) { reject(new Error('The browser is not running.')); return; }
      const id = ++this._seq;
      const timer = setTimeout(() => { if (this._pending.delete(id)) reject(new Error(`The browser did not answer "${message.type}" within ${Math.round(timeoutMs / 1000)} seconds.`)); }, timeoutMs);
      this._pending.set(id, { resolve, reject, timer });
      try { worker.postMessage({ ...message, id }, transfer); } catch (e) { clearTimeout(timer); this._pending.delete(id); reject(e); }
    });
  }

  async request(type, data = {}, { timeoutMs } = {}) {
    await this._start();
    if (!this._worker) await this._start(); // the engine stopped while this request waited
    return this._call({ type, ...data }, [], timeoutMs || this._timeoutMs);
  }

  /** Fire-and-forget input (pointer moves, keys); dropped while the worker is not running. */
  _send(type, data) {
    if (!this._worker) { this._start().catch(() => {}); return; }
    try { this._worker.postMessage({ type, ...data }); } catch {}
  }

  _onMessage(data) {
    if (!data) return;
    if (data.id !== undefined) {
      const p = this._pending.get(data.id);
      if (!p) return;
      this._pending.delete(data.id);
      clearTimeout(p.timer);
      data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
      return;
    }
    if (data.type === 'events') for (const ev of data.events || []) this._apply(ev);
    else if (data.type === 'crashed') this._recover(data);
  }

  _apply(ev) {
    this._lastActivity = Date.now();
    const s = this._state;
    switch (ev.type) {
      case 'title': s.title = ev.title || ''; this._ui?.title(s.title); break;
      case 'url':
        if (ev.url !== s.url) this._closeSelect();
        s.url = ev.url || ''; s.canBack = !!ev.canBack; s.canForward = !!ev.canForward; this._ui?.address(s);
        break;
      case 'loading':
        s.loading = !!ev.loading;
        this._ui?.loading(s.loading, ev.url);
        // A page that finished loading over http(s) proves the route works.
        if (!ev.loading && /^https?:/.test(ev.url || '') && !s.relayOk && !(s.relayUrl === TOR_ROUTE && s.network?.state === 'failed')) { s.relayOk = true; this._ui?.relay(s); }
        break;
      case 'cursor': this._ui?.cursor(ev.cursor); break;
      case 'console':
        this._console.push({ level: ev.level, text: ev.text, time: Date.now() });
        if (this._console.length > MAX_CONSOLE) this._console.splice(0, this._console.length - MAX_CONSOLE);
        if (ev.level === 'error') this._errors++;
        this._ui?.consoleLine(ev, this._errors, this._console.length);
        break;
      case 'dialog':
        if (ev.kind === 'clipboard') this._copy(ev.text);
        else this._ui?.toast(dialogMessage(ev), { sticky: true });
        break;
      case 'clipboard': this._copy(ev.text); break;
      case 'select': this._openSelect(ev); break;
      case 'network':
        s.network = { route: ev.route, state: ev.state, detail: ev.detail || '' };
        if (s.relayUrl === TOR_ROUTE && ev.route === 'tor') {
          s.relayOk = ev.state === 'ready' ? true : ev.state === 'failed' ? false : null;
          this._ui?.relay(s);
        }
        break;
      case 'cookies': case 'storage': this._schedulePersist(); break;
      case 'error':
        s.loading = false;
        this._ui?.loading(false);
        if (/relay/i.test(ev.text || '') && s.relayUrl !== TOR_ROUTE) { s.relayOk = false; this._ui?.relay(s); }
        this._ui?.status(ev.text, 'error');
        break;
      case 'download': {
        const href = safeHttpUrl(ev.url);
        this._ui?.toast(`The browser engine did not download ${ev.name || 'this file'}.${href ? ' Open it in a normal tab to save it.' : ''}`, { sticky: true, link: href ? { href, text: 'Open in a new tab' } : null });
        break;
      }
      case 'ime': this._ui?.ime(ev); break;
    }
    this.dispatchEvent(new CustomEvent(ev.type, { detail: ev }));
  }

  /** Copy text for the page. Writing needs a recent user gesture; when the browser refuses, the
   * text waits for the person's next click or key in the panel, or the toast's Copy button. */
  _copy(text) {
    const value = String(text ?? '');
    const write = () => {
      try { return Promise.resolve(globalThis.navigator?.clipboard?.writeText(value)).then(() => true, () => false); } catch { return Promise.resolve(false); }
    };
    return write().then((ok) => {
      if (ok && globalThis.navigator?.clipboard) { this._pendingCopy = null; this._ui?.note('Copied to the clipboard.'); return true; }
      this._pendingCopy = write;
      this._ui?.toast('The page copied text. Your browser needs a click to put it on the clipboard.', { sticky: true, action: { text: 'Copy', run: () => this._flushCopy() } });
      return false;
    });
  }

  /** Called inside a user gesture: finish a copy the page asked for. */
  _flushCopy() {
    const write = this._pendingCopy;
    if (!write) return;
    this._pendingCopy = null;
    write().then((ok) => { if (ok) { this._ui?.toast(''); this._ui?.note('Copied to the clipboard.'); } });
  }

  // ---- <select> menus ----

  _openSelect(ev) {
    this._closeSelect();
    if (!this._ui) { this.chooseOption(ev.id, -1); return; }
    this._menu = new SelectMenu(ev, {
      choose: (index) => this.chooseOption(ev.id, index),
      close: () => { this._menu = null; this._ui?.hideSelect(); },
      render: (menu) => this._ui?.renderSelect(menu),
    });
    this._ui.showSelect(this._menu);
  }

  _closeSelect() {
    if (this._menu) this._menu.cancel();
  }

  /** Answer a "select" event: the option index, or -1 when the menu was dismissed. */
  chooseOption(id, index) { this._send('chooseOption', { id, index }); }

  _schedulePersist() {
    clearTimeout(this._persistTimer);
    this._persistTimer = setTimeout(() => { this.persist(); }, 800);
  }

  /** Save cookies and localStorage now. */
  async persist() {
    clearTimeout(this._persistTimer);
    if (!this._worker) return;
    try {
      const [cookies, storage] = await Promise.all([this._call({ type: 'cookiesJson' }), this._call({ type: 'storageJson' })]);
      await this._persist.save({ cookies, storage });
    } catch {}
  }

  async _recover({ error, cookies, storage } = {}) {
    const worker = this._worker;
    this._worker = null;
    this._starting = null;
    try { worker?.terminate(); } catch {}
    for (const [id, p] of this._pending) { clearTimeout(p.timer); p.reject(new Error(`The browser engine stopped: ${error || 'unknown error'}`)); this._pending.delete(id); }
    if (cookies != null || storage != null) {
      const previous = await this._persist.load().catch(() => ({}));
      await this._persist.save({ cookies: cookies ?? previous?.cookies ?? null, storage: storage ?? previous?.storage ?? null });
    }
    const now = Date.now();
    this._crashes = this._crashes.filter((t) => now - t < 20000).concat(now);
    const url = this._state.url;
    this._state.loading = false;
    this.dispatchEvent(new CustomEvent('crashed', { detail: { error } }));
    this._ui?.replaceCanvas();
    if (this._crashes.length > 2) {
      this._halted = true;
      this._ui?.status(`The browser engine stopped twice on this page (${error}). Press Reload to try again.`, 'error');
      return;
    }
    this._ui?.status(`The browser engine stopped (${error}). Restarting…`, 'error');
    await this._start();
    if (url) this.request('navigate', { url }).catch(() => {});
  }

  /** Restart the panel before changing identities; cookies and localStorage survive. */
  async setProfiles(identityProfile, tlsProfile = this.tlsProfile) {
    validProfile(IDENTITY_PROFILES, identityProfile);
    validProfile(TLS_PROFILES, tlsProfile);
    if (this._profilesChanging) throw new Error('A browser profile change is already in progress');
    if (identityProfile === this.identityProfile && tlsProfile === this.tlsProfile) return;
    this._profilesChanging = true;
    try {
      await this._start();
      await this.ready;
      await this.persist();
      const url = this._state.url;
      const worker = this._worker;
      this._worker = null;
      this._starting = null;
      worker?.terminate();
      for (const [id, p] of this._pending) { clearTimeout(p.timer); p.reject(new Error('Browser profile changed')); this._pending.delete(id); }
      this._closeSelect();
      this.identityProfile = identityProfile;
      this.tlsProfile = tlsProfile;
      writeSetting(IDENTITY_KEY, identityProfile);
      writeSetting(TLS_KEY, tlsProfile);
      this._halted = false;
      this._crashes = [];
      this._state.loading = false;
      this._state.network = null;
      this._state.relayOk = null;
      this._ready = new Promise((resolve, reject) => { this._readyResolve = resolve; this._readyReject = reject; });
      this._ready.catch(() => {});
      this._ui?.replaceCanvas();
      await this._start();
      await this.ready;
      if (url) await this.navigate(url);
    } finally {
      this._profilesChanging = false;
    }
  }

  // ---- navigation ----

  async navigate(url) {
    const target = addressToUrl(url);
    if (!target) return;
    this._lastActivity = Date.now();
    this._ui?.hideStart();
    await this.request('navigate', { url: target });
  }
  back() { this.request('back').catch(() => {}); }
  forward() { this.request('forward').catch(() => {}); }
  reload() {
    if (this._halted) { this._halted = false; this._crashes = []; this._start().then(() => this._state.url && this.request('navigate', { url: this._state.url })).catch(() => {}); return; }
    this.request('reload').catch(() => {});
  }
  stop() { this.request('stop').catch(() => {}); }
  evaluate(code) { return this.request('evaluate', { code }); }
  networkLog() { return this.request('networkLog'); }
  async clearData() { await this.request('clearData'); await this._persist.save({ cookies: null, storage: null }); }

  /** The route for a setting: Auto is Tor, or the local relay when the page is served locally and the relay answers. */
  async _resolveRoute(choice) {
    if (choice !== AUTO_ROUTE) return choice;
    if (isLocalOrigin()) {
      let ok = false;
      try { ok = await this._probe(DEFAULT_RELAY, 1200); } catch {}
      if (ok) return DEFAULT_RELAY;
    }
    return TOR_ROUTE;
  }

  /** Use a route: "auto", "tor:" or a relay's ws(s):// URL (empty restores the default). */
  async setRelay(url) {
    const value = String(url || '').trim() || this._defaultRelay;
    this._state.choice = value;
    if (value === this._defaultRelay) { try { globalThis.localStorage?.removeItem(RELAY_KEY); } catch {} } else writeSetting(RELAY_KEY, value);
    const route = await this._resolveRoute(value);
    if (route !== this._state.relayUrl) { this._state.network = null; this._state.relayOk = null; }
    this._state.relayUrl = route;
    if (this._worker) await this.request('setRelay', { url: route }).catch(() => {});
    return this.checkRelay();
  }

  async checkRelay() {
    const s = this._state;
    if (s.relayUrl === TOR_ROUTE) {
      // Tor reports its own progress through "network" events.
      if (s.network?.route === 'tor') s.relayOk = s.network.state === 'ready' ? true : s.network.state === 'failed' ? false : null;
      this._ui?.relay(s);
      return s.relayOk;
    }
    this._state.relayOk = null;
    this._ui?.relay(this._state);
    let ok = false;
    try { ok = await this._probe(this._state.relayUrl); } catch {}
    this._state.relayOk = ok;
    this._ui?.relay(this._state);
    this.dispatchEvent(new CustomEvent('relay', { detail: { ok, url: this._state.relayUrl } }));
    return ok;
  }

  resize(width, height, dpr) {
    const w = Math.max(1, Math.round(width)), h = Math.max(1, Math.round(height));
    const d = Math.min(Math.max(dpr || 1, 1), 2);
    if (w === this._size.width && h === this._size.height && d === this._size.dpr) return;
    this._size = { width: w, height: h, dpr: d };
    this._send('resize', this._size);
  }

  // ---- input from the person ----

  pointer(kind, x, y, button, buttons, mods) { this._send('pointer', { kind, x, y, button, buttons, mods }); }
  wheel(x, y, dx, dy, deltaMode, mods) { this._send('wheel', { x, y, dx, dy, deltaMode, mods }); }
  keyEvent(kind, key, code, mods, repeat) { this._send('key', { kind, key, code, mods, repeat }); }
  imeCommit(text) { if (text) this._send('imeCommit', { text }); }
  imePreedit(text) { this._send('imePreedit', { text: text || '' }); }
  get selectMenu() { return this._menu || null; }

  // ---- the agent ----

  async _agent(type, data, action) {
    this._agentActive++;
    this._ui?.agentBusy(true);
    try {
      const result = await this.request(type, data, { timeoutMs: this._agentTimeoutMs });
      if (result && result.ok && Number.isFinite(result.x)) this._ui?.agentPoint(result);
      if (result && result.error === 'stale') this._ui?.note('You took over. Pi will look at the page again.');
      if (action !== 'snapshot') this.dispatchEvent(new CustomEvent('agent', { detail: { action, ref: data.ref, ok: !!(result && result.ok), error: result && result.error } }));
      this._lastActivity = Date.now();
      return result;
    } finally {
      this._agentActive--;
      if (!this._agentActive) this._ui?.agentBusy(false);
    }
  }

  snapshot(maxChars = 6000) { return this._agent('snapshot', { maxChars }, 'snapshot'); }
  click(ref) { return this._agent('agentClick', { ref }, 'click'); }
  type(ref, text, { submit = false } = {}) { return this._agent('agentType', { ref, text: String(text), submit: !!submit }, 'type'); }
  select(ref, value) { return this._agent('agentSelect', { ref, value: String(value) }, 'select'); }
  scroll(dy) { return this._agent('agentScroll', { dy: Number(dy) || 0 }, 'scroll'); }
  key(key) { return this._agent('agentKey', { key: String(key) }, 'key'); }
  /** The number [n] in the latest snapshot of the element at (x, y), viewport CSS px. */
  refAt(x, y) { return this.request('agentRefAt', { x: Number(x), y: Number(y) }, { timeoutMs: this._agentTimeoutMs }); }
  /** The page as drawn, for a vision model: { rgb, width, height }, sides multiples of unit. */
  screenshot({ unit = 32, maxSide = 1024 } = {}) { return this.request('screenshot', { unit, maxSide }, { timeoutMs: this._agentTimeoutMs }); }

  /** Resolves once loading has finished and no engine events arrived for quietMs (or on timeout). */
  waitForIdle(timeoutMs = 15000, quietMs = 500) {
    const start = Date.now();
    return new Promise((resolve) => {
      const check = () => {
        const now = Date.now();
        if ((!this._state.loading && now - this._lastActivity >= quietMs) || now - start >= timeoutMs) { resolve(); return; }
        setTimeout(check, 100);
      };
      setTimeout(check, Math.min(quietMs, 150));
    });
  }

  // ---- UI ----

  /** Build the browser UI in `container`. `onClose` adds a close button. */
  mount(container, { onClose } = {}) {
    if (this._ui) throw new Error('The browser is already mounted.');
    this._ui = new BrowserView(this, container, { onClose });
    if (this._worker) {
      const offscreen = this._ui.takeOffscreen();
      if (offscreen) this._worker.postMessage({ type: 'canvas', canvas: offscreen }, [offscreen]);
    } else {
      this._start();
    }
    this._ui.relay(this._state);
    const save = () => { this.persist(); };
    globalThis.addEventListener?.('pagehide', save);
    globalThis.document?.addEventListener?.('visibilitychange', () => { if (globalThis.document.visibilityState === 'hidden') save(); });
  }

  dispose() {
    clearTimeout(this._persistTimer);
    try { this._worker?.terminate(); } catch {}
    this._worker = null;
    this._ui?.remove();
    this._ui = null;
  }
}

// ---- DOM view (only constructed by mount) ----

function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

const ICONS = {
  back: 'M10 3 5 8l5 5', forward: 'M6 3l5 5-5 5', reload: 'M13 8a5 5 0 1 1-1.6-3.7M13 3v3h-3', stop: 'M4 4l8 8M12 4l-8 8', close: 'M4 4l8 8M12 4l-8 8',
};
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', ICONS[name]);
  svg.append(path);
  return svg;
}

class BrowserView {
  constructor(client, container, { onClose }) {
    this.client = client;
    this.container = container;
    this.transferred = false;
    const button = (name, label, fn) => { const b = el('button', { type: 'button', class: 'browser-icon', 'aria-label': label, title: label, onclick: fn }); b.append(icon(name)); return b; };
    this.backButton = button('back', 'Back', () => client.back());
    this.forwardButton = button('forward', 'Forward', () => client.forward());
    this.reloadButton = button('reload', 'Reload', () => (client.state.loading ? client.stop() : client.reload()));
    this.addressInput = el('input', { type: 'text', class: 'browser-address-input', 'aria-label': 'Address', placeholder: 'Search or enter an address', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false', enterkeyhint: 'go' });
    const form = el('form', { class: 'browser-address', role: 'search', onsubmit: (e) => { e.preventDefault(); const v = this.addressInput.value; if (v.trim()) { client.navigate(v).catch((err) => this.status(String(err.message || err), 'error')); this.focusPage(); } } }, this.addressInput);
    this.relayChip = el('button', { type: 'button', class: 'browser-relay', 'data-state': 'checking', 'aria-expanded': 'false', title: 'How pages reach websites', onclick: () => this.toggleRelaySettings() }, el('i', { 'aria-hidden': 'true' }), el('span', { text: 'Tor' }));
    this.relayChipLabel = this.relayChip.querySelector('span');
    const toolbar = el('header', { class: 'browser-toolbar' }, this.backButton, this.forwardButton, this.reloadButton, form, this.relayChip);
    if (onClose) toolbar.append(button('close', 'Close the browser', onClose));

    this.externalSignIn = el('a', { class: 'browser-button', target: '_blank', rel: 'noopener noreferrer', text: 'Open Google in new tab' });
    this.signInHelp = el('div', { class: 'browser-signin-help', role: 'status', hidden: true },
      el('span', { text: 'Google can block sign-in in this panel. Use your regular browser to sign in. Signing in there does not sign this panel in.' }), this.externalSignIn);

    this.relayInput = el('input', { type: 'url', class: 'browser-relay-input', 'aria-label': 'Relay WebSocket address', placeholder: 'wss://relay.example/', spellcheck: 'false', autocomplete: 'off', oninput: () => { this.routeRadio('custom').checked = true; } });
    const option = (value, title, detail, extra) => el('label', { class: 'browser-route' },
      el('input', { type: 'radio', name: 'browser-route', value }),
      el('span', {}, el('strong', { text: title }), el('small', { text: detail })), ...(extra ? [extra] : []));
    this.communityList = el('div', { class: 'browser-route-community' });
    this.communityButton = el('button', { type: 'button', class: 'browser-button browser-button-quiet', text: 'Find community relays', onclick: () => this.findCommunityRelays() });
    this.identitySelect = el('select', { 'aria-label': 'User-agent profile' }, ...Object.entries(IDENTITY_PROFILES).map(([value, text]) => el('option', { value, text })));
    this.tlsSelect = el('select', { 'aria-label': 'TLS profile' }, ...Object.entries(TLS_PROFILES).map(([value, text]) => el('option', { value, text })));
    this.identitySelect.value = client.identityProfile;
    this.tlsSelect.value = client.tlsProfile;
    this.profileApply = el('button', { type: 'button', class: 'browser-button', text: 'Apply profiles and reload', onclick: async () => {
      this.profileApply.disabled = true;
      try { await client.setProfiles(this.identitySelect.value, this.tlsSelect.value); this.toggleRelaySettings(false); }
      catch (e) { this.status(String(e.message || e), 'error'); }
      finally { this.profileApply.disabled = false; }
    } });
    this.relaySettings = el('form', { class: 'browser-relay-settings', hidden: true, onsubmit: (e) => { e.preventDefault(); this.applyRoute(); } },
      el('fieldset', {}, el('legend', { text: 'How pages reach websites' }),
        option(AUTO_ROUTE, 'Automatic', isLocalOrigin() ? 'The relay on this computer when it is running, otherwise the Tor network.' : 'The Tor network. Nothing to install.'),
        option(TOR_ROUTE, 'Tor network', 'Through three volunteer relays, reached over Snowflake. Slower, and some sites block Tor.'),
        option(DEFAULT_RELAY, 'Relay on this computer', 'Run evolve-relay; fastest, and sites see your own address.'),
        el('div', { class: 'browser-route-group' }, el('div', { class: 'browser-route-head' }, el('strong', { text: 'Community relays' }), this.communityButton), this.communityList),
        option('custom', 'Another relay', '', this.relayInput)),
      el('div', { class: 'browser-route-actions' }, el('button', { type: 'submit', class: 'browser-button', text: 'Use' })),
      el('fieldset', {}, el('legend', { text: 'Compatibility experiments' }),
        el('label', { text: 'User-agent profile' }, this.identitySelect),
        el('label', { text: 'TLS profile' }, this.tlsSelect),
        el('small', { text: 'Changes the identity sent to sites and restarts this panel. Features remain those of Evolve. NSS ordering changes supported ciphers only; it does not reproduce Firefox TLS.' }), this.profileApply),
      el('p', { text: 'TLS runs inside the engine, so no relay or Tor exit can read pages, passwords or cookies on https:// sites. They see which sites you connect to: a relay also sees your address; Tor separates the two.' }));
    this.relayHelp = el('div', { class: 'browser-help', role: 'note', hidden: true },
      el('strong', { text: '' }),
      el('p', { text: '' }),
      el('a', { href: RELAY_DOCS, target: '_blank', rel: 'noopener', text: 'How pages reach websites' }));
    this.relayHelpTitle = this.relayHelp.querySelector('strong');
    this.relayHelpText = this.relayHelp.querySelector('p');

    this.stage = el('div', { class: 'browser-stage' });
    this.canvas = this.makeCanvas();
    this.input = el('textarea', { class: 'browser-keyboard', 'aria-label': 'Keyboard input for the page', autocomplete: 'off', autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false', rows: '1', tabindex: '-1' });
    this.agentBox = el('div', { class: 'browser-agent-box' });
    this.agentCursor = el('div', { class: 'browser-agent-cursor' }, el('span', { text: 'Pi' }));
    const overlay = el('div', { class: 'browser-overlay', 'aria-hidden': 'true' }, this.agentBox, this.agentCursor);
    this.banner = el('div', { class: 'browser-banner', role: 'status', hidden: true, text: 'Pi is using the page. You can still click and type; Pi will notice.' });
    this.toastBox = el('div', { class: 'browser-toast', role: 'status', hidden: true });
    this.menuBox = el('div', { class: 'browser-menu', role: 'listbox', hidden: true });
    this.start = el('div', { class: 'browser-start' },
      el('h2', { text: 'A web browser compiled to WebAssembly' }),
      el('p', { text: 'Pages are parsed, styled, laid out and run (Stylo CSS, Taffy layout, Boa JavaScript) in a worker on this device. You and Pi can use the same page at the same time.' }),
      el('div', { class: 'browser-start-links' }, ...['example.com', 'news.ycombinator.com', 'en.wikipedia.org/wiki/WebAssembly'].map((u) => el('button', { type: 'button', class: 'browser-button', text: u, onclick: () => { this.addressInput.value = u; client.navigate(u).catch((err) => this.status(String(err.message || err), 'error')); } }))));
    this.stage.append(this.canvas, this.input, overlay, this.start, this.banner, this.toastBox, this.menuBox);

    this.statusText = el('span', { class: 'browser-status-text', role: 'status', text: 'Starting the browser engine…' });
    this.consoleSummary = el('summary', { text: 'Console' });
    this.consoleLog = el('pre', { class: 'browser-console-log' });
    this.consoleBox = el('details', { class: 'browser-console' }, this.consoleSummary, this.consoleLog);
    this.noteText = el('span', { class: 'browser-note', role: 'status' });
    const footer = el('footer', { class: 'browser-status' }, this.statusText, this.noteText, this.consoleBox);

    this.root = el('div', { class: 'browser' }, toolbar, this.signInHelp, this.relaySettings, this.relayHelp, this.stage, footer);
    container.append(this.root);
    this.bindInput();
    this.observeSize();
    client._ready.then(() => { if (!client.state.url) this.status('Ready. Enter an address above.'); }, () => {});
  }

  makeCanvas() {
    const c = el('canvas', { class: 'browser-canvas', 'aria-label': 'Web page', role: 'img' });
    this.transferred = false;
    return c;
  }

  takeOffscreen() {
    if (this.transferred || typeof this.canvas.transferControlToOffscreen !== 'function') return null;
    this.transferred = true;
    return this.canvas.transferControlToOffscreen();
  }

  replaceCanvas() {
    const fresh = this.makeCanvas();
    this.canvas.replaceWith(fresh);
    this.canvas = fresh;
  }

  observeSize() {
    let timer = null;
    const measure = () => {
      const r = this.stage.getBoundingClientRect();
      if (r.width < 20 || r.height < 20) return;
      this.client.resize(r.width, r.height, globalThis.devicePixelRatio || 1);
    };
    measure();
    if (typeof ResizeObserver === 'function') {
      this.resizeObserver = new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(measure, 60); });
      this.resizeObserver.observe(this.stage);
    }
    globalThis.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(measure, 60); });
  }

  point(e) {
    const r = this.stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  focusPage() {
    try { this.input.focus({ preventScroll: true }); } catch {}
  }

  bindInput() {
    const c = this.client, stage = this.stage, input = this.input;
    let queued = null, frame = 0;
    const flush = () => { frame = 0; if (queued) { const q = queued; queued = null; c.pointer('move', ...q); } };
    const isUi = (e) => e.target.closest('.browser-start, .browser-banner, .browser-toast, .browser-menu');
    stage.addEventListener('pointerdown', (e) => {
      if (isUi(e)) return;
      if (c.selectMenu?.open) { e.preventDefault(); c._closeSelect(); this.focusPage(); return; }
      e.preventDefault();
      try { stage.setPointerCapture(e.pointerId); } catch {}
      this.focusPage();
      flush();
      c.pointer('down', ...this.point(e), e.button, e.buttons, modifierBits(e));
    });
    stage.addEventListener('pointermove', (e) => {
      if (isUi(e)) return;
      queued = [...this.point(e), e.button < 0 ? 0 : e.button, e.buttons, modifierBits(e)];
      if (!frame) frame = requestAnimationFrame(flush);
    });
    const up = (e) => {
      if (isUi(e)) return;
      flush();
      c.pointer('up', ...this.point(e), e.button < 0 ? 0 : e.button, e.buttons, modifierBits(e));
      try { stage.releasePointerCapture(e.pointerId); } catch {}
    };
    stage.addEventListener('pointerup', up);
    stage.addEventListener('pointercancel', up);
    stage.addEventListener('pointerleave', (e) => { if (!stage.hasPointerCapture?.(e.pointerId)) { queued = null; c.pointer('leave', ...this.point(e), 0, 0, modifierBits(e)); } });
    stage.addEventListener('contextmenu', (e) => { if (!isUi(e)) e.preventDefault(); });
    stage.addEventListener('wheel', (e) => { e.preventDefault(); c.wheel(...this.point(e), e.deltaX, e.deltaY, e.deltaMode, modifierBits(e)); }, { passive: false });

    let composing = false;
    input.addEventListener('keydown', (e) => {
      if (composing || e.isComposing || e.keyCode === 229) return;
      if (c.selectMenu?.open) { if (c.selectMenu.key(e.key)) e.preventDefault(); return; }
      const action = e.ctrlKey || e.metaKey;
      const printable = e.key.length === 1 && !action;
      const paste = action && !e.altKey && e.key.toLowerCase() === 'v';
      c.keyEvent('down', e.key, e.code, modifierBits(e), e.repeat);
      // Text arrives through the input event (so dead keys and IME work); other keys are only
      // sent to the engine, never edited into the hidden textarea.
      if (!printable && !paste) e.preventDefault();
    });
    input.addEventListener('keyup', (e) => { if (!e.isComposing) c.keyEvent('up', e.key, e.code, modifierBits(e), false); });
    input.addEventListener('compositionstart', () => { composing = true; });
    input.addEventListener('compositionupdate', (e) => c.imePreedit(e.data));
    input.addEventListener('compositionend', (e) => { composing = false; c.imePreedit(''); c.imeCommit(e.data); input.value = ''; });
    input.addEventListener('input', (e) => {
      if (composing || e.isComposing) return;
      const text = input.value;
      input.value = '';
      if (text) c.imeCommit(text);
    });
    input.addEventListener('paste', (e) => {
      e.preventDefault();
      const text = e.clipboardData?.getData('text/plain');
      if (text) c.imeCommit(text);
    });
  }

  // ---- updates from engine events ----

  title(t) {
    this.root.dataset.title = t || '';
    this.container.setAttribute?.('aria-label', t ? `Web browser: ${t}` : 'Web browser');
  }

  address(s) {
    if (globalThis.document.activeElement !== this.addressInput) this.addressInput.value = s.url;
    this.backButton.disabled = !s.canBack;
    this.forwardButton.disabled = !s.canForward;
    const externalSignIn = externalGoogleSignInUrl(s.url);
    this.signInHelp.hidden = !externalSignIn;
    if (externalSignIn) this.externalSignIn.href = externalSignIn;
    else this.externalSignIn.removeAttribute('href');
    if (s.url) this.hideStart();
  }

  hideStart() { this.start.hidden = true; }

  loading(on, url) {
    this.root.dataset.loading = on ? 'true' : 'false';
    this.reloadButton.replaceChildren(icon(on ? 'stop' : 'reload'));
    this.reloadButton.setAttribute('aria-label', on ? 'Stop' : 'Reload');
    this.reloadButton.title = on ? 'Stop' : 'Reload';
    if (on) { this.hideStart(); this.status(`Loading ${shortUrl(url || this.client.state.url)}…`); }
    else if (this.statusText.dataset.kind !== 'error') this.status(this.client.state.title || 'Done');
  }

  status(text, kind = '') {
    this.statusText.textContent = text || '';
    this.statusText.dataset.kind = kind;
  }

  note(text) {
    this.noteText.textContent = text;
    clearTimeout(this.noteTimer);
    this.noteTimer = setTimeout(() => { this.noteText.textContent = ''; }, 5000);
  }

  cursor(name) {
    this.stage.style.cursor = /^[a-z-]+$/.test(name || '') ? name : 'default';
  }

  consoleLine(ev, errors, total) {
    this.consoleSummary.textContent = errors ? `Console (${total}, ${errors} error${errors === 1 ? '' : 's'})` : `Console (${total})`;
    const line = el('span', { class: `browser-console-${/^(error|warn)$/.test(ev.level) ? ev.level : 'log'}`, text: `${ev.text}\n` });
    this.consoleLog.append(line);
    while (this.consoleLog.childNodes.length > MAX_CONSOLE) this.consoleLog.firstChild.remove();
  }

  toast(text, { sticky = false, action = null, link = null } = {}) {
    clearTimeout(this.toastTimer);
    if (!text) { this.toastBox.hidden = true; this.toastBox.replaceChildren(); return; }
    const close = el('button', { type: 'button', class: 'browser-toast-close', 'aria-label': 'Dismiss', text: '×', onclick: () => this.toast('') });
    const parts = [el('span', { text })];
    if (action) parts.push(el('button', { type: 'button', class: 'browser-button', text: action.text, onclick: () => { action.run(); } }));
    if (link) parts.push(el('a', { href: link.href, target: '_blank', rel: 'noopener noreferrer', text: link.text }));
    this.toastBox.replaceChildren(...parts, close);
    this.toastBox.hidden = false;
    if (!sticky) this.toastTimer = setTimeout(() => { this.toastBox.hidden = true; }, 5000);
  }

  showSelect(menu) {
    this.renderSelect(menu);
    this.menuBox.hidden = false;
    const stage = this.stage.getBoundingClientRect();
    const { x, y, w, h } = menu.box;
    this.menuBox.style.minWidth = `${Math.max(120, w)}px`;
    this.menuBox.style.left = `${Math.max(0, Math.min(x, stage.width - Math.max(120, w)))}px`;
    const below = stage.height - (y + h), above = y;
    const height = Math.min(this.menuBox.scrollHeight, 320);
    if (below >= height || below >= above) { this.menuBox.style.top = `${y + h + 2}px`; this.menuBox.style.maxHeight = `${Math.max(80, below - 8)}px`; }
    else { this.menuBox.style.top = `${Math.max(4, y - height - 2)}px`; this.menuBox.style.maxHeight = `${Math.max(80, above - 8)}px`; }
    this.menuBox.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' });
    this.focusPage();
  }

  renderSelect(menu) {
    const rows = menu.rows.map((r) => r.kind === 'group'
      ? el('div', { class: 'browser-menu-group', role: 'presentation', text: r.label })
      : el('div', {
        class: 'browser-menu-option', role: 'option', id: `browser-option-${r.index}`, text: r.label || '\u00a0',
        'aria-selected': String(!!r.selected), 'aria-disabled': String(!!r.disabled), 'data-active': String(!!r.active),
        onpointerdown: (e) => { e.preventDefault(); e.stopPropagation(); },
        onclick: (e) => { e.stopPropagation(); menu.click(r.index); },
      }));
    if (menu.multiple) rows.push(el('button', { type: 'button', class: 'browser-button browser-menu-done', text: 'Done', onclick: () => menu.done() }));
    this.menuBox.setAttribute('aria-multiselectable', String(menu.multiple));
    this.menuBox.setAttribute('aria-activedescendant', menu.active >= 0 ? `browser-option-${menu.active}` : '');
    this.menuBox.replaceChildren(...rows);
    this.menuBox.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' });
  }

  hideSelect() {
    this.menuBox.hidden = true;
    this.menuBox.replaceChildren();
  }

  ime({ enabled, x, y, w, h }) {
    if (!enabled) return;
    this.input.style.left = `${Math.max(0, x)}px`;
    this.input.style.top = `${Math.max(0, y)}px`;
    this.input.style.height = `${Math.max(1, h)}px`;
    this.input.style.width = `${Math.max(1, w)}px`;
  }

  relay(s) {
    const state = s.relayOk === null ? 'checking' : s.relayOk ? 'ok' : 'down';
    const tor = s.relayUrl === TOR_ROUTE;
    this.relayChip.dataset.state = state;
    this.relayChipLabel.textContent = routeLabel(s.relayUrl);
    const detail = tor && s.network?.route === 'tor' ? s.network.detail : '';
    this.relayChip.title = tor
      ? (detail || (state === 'ok' ? 'Connected to the Tor network' : 'Connecting to the Tor network…'))
      : state === 'ok' ? `Relay connected: ${s.relayUrl}` : state === 'down' ? `Relay not reachable: ${s.relayUrl}` : `Checking the relay at ${s.relayUrl}`;
    this.relayHelp.hidden = state !== 'down';
    if (tor) {
      this.relayHelpTitle.textContent = 'The Tor network is not reachable. ';
      this.relayHelpText.textContent = `${detail || 'The Snowflake bridge did not answer.'} Reload to try again, or choose another route with the Tor button (for example evolve-relay on this computer).`;
    } else {
      this.relayHelpTitle.textContent = 'Start the relay: ';
      this.relayHelpText.textContent = `evolve-relay — this page cannot reach the relay at ${s.relayUrl}. Run it on this computer (it listens on ws://127.0.0.1:8787/), or choose the Tor network with the Relay button. Chrome may ask to let this site reach devices on your local network; allow it.`;
    }
    if (!this.relaySettings.hidden) return;
    this.syncRouteForm(s);
  }

  routeRadio(value) {
    return [...this.relaySettings.querySelectorAll('input[name=browser-route]')].find((r) => r.value === value) || null;
  }

  syncRouteForm(s = this.client.state) {
    const choice = s.choice || s.relayUrl;
    const radio = this.routeRadio(choice) || (this.relayInput.value = choice, this.routeRadio('custom'));
    if (radio) radio.checked = true;
  }

  applyRoute() {
    const picked = this.relaySettings.querySelector('input[name=browser-route]:checked')?.value || AUTO_ROUTE;
    const value = picked === 'custom' ? this.relayInput.value.trim() : picked;
    if (!value) { this.relayInput.focus(); return; }
    this.client.setRelay(value);
    this.toggleRelaySettings(false);
  }

  async findCommunityRelays() {
    this.communityButton.disabled = true;
    this.communityList.replaceChildren(el('small', { text: 'Asking Nostr relays for announced evolve-relay servers…' }));
    try {
      const { discoverRelays } = await import('./relay-directory.mjs');
      const found = (await discoverRelays({})).filter((r) => r.ok);
      if (!found.length) { this.communityList.replaceChildren(el('small', { text: 'No community relays answered right now. Anyone can run one: evolve-relay --public --announce.' })); return; }
      this.communityList.replaceChildren(...found.slice(0, 8).map((r) => {
        const where = [r.region, `${Math.round(r.rttMs)} ms`].filter(Boolean).join(' · ');
        const label = el('label', { class: 'browser-route' },
          el('input', { type: 'radio', name: 'browser-route', value: r.url }),
          el('span', {}, el('strong', { text: r.name || new URL(r.url).host }), el('small', { text: `${new URL(r.url).host}${where ? ' · ' + where : ''} · operator ${String(r.pubkey || '').slice(0, 12)}… sees your address and the sites you open.` })));
        return label;
      }));
      this.syncRouteForm();
    } catch (e) {
      this.communityList.replaceChildren(el('small', { text: `Could not look up community relays: ${e.message || e}` }));
    } finally {
      this.communityButton.disabled = false;
    }
  }

  toggleRelaySettings(force) {
    const open = typeof force === 'boolean' ? force : this.relaySettings.hidden;
    this.relaySettings.hidden = !open;
    this.relayChip.setAttribute('aria-expanded', String(open));
    if (open) { this.syncRouteForm(); this.relaySettings.querySelector('input[name=browser-route]:checked')?.focus(); }
  }

  agentBusy(on) {
    clearTimeout(this.bannerTimer);
    if (on) { this.banner.hidden = false; this.root.dataset.agent = 'true'; }
    else this.bannerTimer = setTimeout(() => { this.banner.hidden = true; this.root.dataset.agent = 'false'; }, 900);
  }

  agentPoint({ x, y, w = 0, h = 0 }) {
    const cx = x + w / 2, cy = y + h / 2;
    this.agentCursor.style.transform = `translate(${cx}px, ${cy}px)`;
    this.agentCursor.classList.add('visible');
    if (w > 0 && h > 0) {
      this.agentBox.style.left = `${x}px`;
      this.agentBox.style.top = `${y}px`;
      this.agentBox.style.width = `${w}px`;
      this.agentBox.style.height = `${h}px`;
      this.agentBox.classList.remove('visible');
      void this.agentBox.offsetWidth;
      this.agentBox.classList.add('visible');
    }
    clearTimeout(this.pointTimer);
    this.pointTimer = setTimeout(() => { this.agentBox.classList.remove('visible'); this.agentCursor.classList.remove('visible'); }, 1200);
  }

  remove() {
    this.resizeObserver?.disconnect();
    this.root.remove();
  }
}

function shortUrl(url) {
  try { const u = new URL(url); return u.host + (u.pathname === '/' ? '' : u.pathname); } catch { return url || ''; }
}
