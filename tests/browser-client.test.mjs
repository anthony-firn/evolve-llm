// The WASM browser's main-thread client and worker host, with a fake worker and a fake engine.
import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserClient, addressToUrl, externalGoogleSignInUrl, modifierBits, isLocalOrigin, routeLabel, TOR_ROUTE, AUTO_ROUTE, DEFAULT_RELAY } from '../web/browser-client.mjs';
import { createBrowserHost, torConsensusSources } from '../web/browser-host.mjs';

const tick = () => new Promise((r) => setTimeout(r, 0));

function fakeWorkers(respond = () => undefined) {
  const workers = [];
  const factory = () => {
    const worker = {
      messages: [],
      terminated: false,
      postMessage(message) {
        this.messages.push(message);
        if (message.id === undefined) return;
        const answer = message.type === 'init' ? { fonts: 11 } : respond(message, worker);
        if (answer !== undefined) queueMicrotask(() => worker.onmessage({ data: answer instanceof Error ? { id: message.id, error: answer.message } : { id: message.id, result: answer } }));
      },
      terminate() { this.terminated = true; },
      emit(events) { this.onmessage({ data: { type: 'events', events } }); },
    };
    workers.push(worker);
    return worker;
  };
  return { workers, factory };
}

function memoryPersistence(initial = {}) {
  const store = { value: initial, saves: [] };
  return { store, load: async () => store.value, save: async (v) => { store.value = v; store.saves.push(v); } };
}

const client = (opts = {}) => new BrowserClient({ probe: async () => true, persistence: memoryPersistence(opts.saved), ...opts });

test('external Google sign-in keeps the destination but starts a separate browser session', () => {
  for (const path of ['/v3/signin/identifier', '/v3/signin/rejected']) {
    const original = new URL(`https://accounts.google.com${path}?hl=en&flowName=GlifWebSignIn&service=mail&continue=https%3A%2F%2Fmail.google.com%2Fmail%2F&dsh=session&TL=token&idnf=private-identifier&epd=session-data`);
    const external = new URL(externalGoogleSignInUrl(original.href));
    assert.equal(external.origin, original.origin);
    assert.equal(external.pathname, '/ServiceLogin');
    for (const key of ['hl', 'service', 'continue']) assert.equal(external.searchParams.get(key), original.searchParams.get(key));
    assert.deepEqual([...external.searchParams.keys()].sort(), ['continue', 'hl', 'service']);
  }
  assert.equal(externalGoogleSignInUrl('https://accounts.google.com/'), 'https://accounts.google.com/ServiceLogin');
});

test('external sign-in does not rewrite lookalikes, OAuth grants or account challenges', () => {
  for (const url of [
    'not a URL', 'http://accounts.google.com/', 'https://accounts.google.com.evil.test/',
    'https://accounts.google.com:8443/', 'https://person:password@accounts.google.com/',
    'https://accounts.google.com/v3/signin/challenge/pwd', 'https://accounts.google.com/o/oauth2/v2/auth',
    'https://accounts.google.com/v3/signin/identifier?flowName=GeneralOAuthFlow',
    ...['client_id', 'redirect_uri', 'scope', 'state'].map(key => `https://accounts.google.com/ServiceLogin?${key}=value`),
  ]) assert.equal(externalGoogleSignInUrl(url), null, url);
});

test('address bar text becomes a URL or a search', () => {
  assert.equal(addressToUrl('example.com'), 'https://example.com');
  assert.equal(addressToUrl('  news.ycombinator.com/login?goto=news '), 'https://news.ycombinator.com/login?goto=news');
  assert.equal(addressToUrl('https://a.b/c'), 'https://a.b/c');
  assert.equal(addressToUrl('about:blank'), 'about:blank');
  assert.equal(addressToUrl('localhost:3000/x'), 'http://localhost:3000/x');
  assert.equal(addressToUrl('example.com:8443/x'), 'https://example.com:8443/x');
  assert.equal(addressToUrl('rust wasm'), 'https://html.duckduckgo.com/html/?q=rust%20wasm');
  assert.equal(addressToUrl('webassembly'), 'https://html.duckduckgo.com/html/?q=webassembly');
  assert.equal(addressToUrl(''), '');
  assert.equal(modifierBits({ shiftKey: true, ctrlKey: false, altKey: true, metaKey: true }), 13);
});

test('requests carry ids, start the worker with saved state and reject engine errors', async () => {
  const { workers, factory } = fakeWorkers((m) => (m.type === 'evaluate' ? (m.code === 'bad' ? new Error('SyntaxError: x') : '2') : null));
  const c = client({ workerFactory: factory, saved: { cookies: '[{"a":1}]', storage: '{}' } });
  assert.equal(await c.evaluate('1+1'), '2');
  const init = workers[0].messages[0];
  assert.equal(init.type, 'init');
  assert.equal(init.cookies, '[{"a":1}]');
  // Automatic route on a public origin is Tor: no relay to install or host.
  assert.equal(init.relayUrl, 'tor:');
  assert.equal(init.width, 1280);
  await assert.rejects(c.evaluate('bad'), /SyntaxError/);
  await c.ready;
  c.dispose();
});

test('input is fire-and-forget and resize is deduplicated', async () => {
  const { workers, factory } = fakeWorkers(() => null);
  const c = client({ workerFactory: factory });
  await c.navigate('example.com');
  const w = workers[0];
  assert.deepEqual(w.messages.find((m) => m.type === 'navigate').url, 'https://example.com');
  c.pointer('down', 10, 20, 0, 1, 2);
  c.keyEvent('down', 'Enter', 'Enter', 0, false);
  c.imeCommit('héllo');
  c.imeCommit('');
  c.resize(800.4, 600, 3);
  c.resize(800, 600, 2);
  const sent = w.messages.filter((m) => m.id === undefined);
  assert.deepEqual(sent.map((m) => m.type), ['pointer', 'key', 'imeCommit', 'resize']);
  assert.deepEqual(sent[0], { type: 'pointer', kind: 'down', x: 10, y: 20, button: 0, buttons: 1, mods: 2 });
  assert.deepEqual(sent[3], { type: 'resize', width: 800, height: 600, dpr: 2 });
  c.dispose();
});

test('engine events update state, dispatch DOM events and trigger persistence', async () => {
  const { workers, factory } = fakeWorkers((m) => (m.type === 'cookiesJson' ? 'COOKIES' : m.type === 'storageJson' ? 'STORAGE' : null));
  const persistence = memoryPersistence();
  const c = new BrowserClient({ relayUrl: 'ws://127.0.0.1:8787/', workerFactory: factory, probe: async () => false, persistence });
  await c.ready;
  const seen = [];
  for (const t of ['title', 'url', 'loading', 'console']) c.addEventListener(t, (e) => seen.push([t, e.detail]));
  const w = workers[0];
  w.emit([
    { type: 'loading', loading: true, url: 'https://example.com/' },
    { type: 'url', url: 'https://example.com/', canBack: true, canForward: false },
    { type: 'title', title: 'Example Domain' },
    { type: 'console', level: 'error', text: 'boom' },
    { type: 'loading', loading: false, url: 'https://example.com/' },
    { type: 'cookies' },
  ]);
  assert.deepEqual(c.state, { url: 'https://example.com/', title: 'Example Domain', loading: false, relayUrl: 'ws://127.0.0.1:8787/', relayOk: true, choice: 'ws://127.0.0.1:8787/', network: null });
  assert.equal(seen.length, 5);
  assert.equal(c.console[0].text, 'boom');
  await c.persist();
  assert.deepEqual(persistence.store.value, { cookies: 'COOKIES', storage: 'STORAGE' });
  w.emit([{ type: 'error', text: 'Could not reach the relay at ws://127.0.0.1:8787/' }]);
  assert.equal(c.state.relayOk, false);
  c.dispose();
});

test('automatic route: Tor on public origins, the local relay on a local origin when it answers', async () => {
  assert.equal(isLocalOrigin({ hostname: 'localhost' }), true);
  assert.equal(isLocalOrigin({ hostname: '127.0.0.1' }), true);
  assert.equal(isLocalOrigin({ hostname: 'www.evolvetech.dev' }), false);
  assert.equal(routeLabel(TOR_ROUTE), 'Tor');
  assert.equal(routeLabel(DEFAULT_RELAY), 'Local relay');
  assert.equal(routeLabel('wss://relay.example/'), 'Relay');
  const probes = [];
  const probe = async (url) => { probes.push(url); return true; };
  const c = new BrowserClient({ probe, persistence: memoryPersistence() });
  assert.equal(await c._resolveRoute(AUTO_ROUTE), TOR_ROUTE);
  assert.deepEqual(probes, [], 'a public origin never probes the local network');
  const saved = globalThis.location;
  try {
    globalThis.location = { hostname: '127.0.0.1' };
    assert.equal(await c._resolveRoute(AUTO_ROUTE), DEFAULT_RELAY);
    const down = new BrowserClient({ probe: async () => false, persistence: memoryPersistence() });
    assert.equal(await down._resolveRoute(AUTO_ROUTE), TOR_ROUTE);
  } finally {
    if (saved === undefined) delete globalThis.location; else globalThis.location = saved;
  }
  assert.equal(await c._resolveRoute('wss://relay.example/'), 'wss://relay.example/');
});

test('Tor progress events drive the route state, and switching routes reaches the worker', async () => {
  const { workers, factory } = fakeWorkers(() => null);
  const c = client({ workerFactory: factory });
  await c.ready;
  const w = workers[0];
  assert.equal(c.state.relayUrl, TOR_ROUTE);
  assert.equal(c.state.relayOk, null);
  const seen = [];
  c.addEventListener('network', (e) => seen.push(e.detail.state));
  w.emit([{ type: 'network', route: 'tor', state: 'connecting', detail: 'Connecting to the Tor network…' }]);
  assert.equal(c.state.relayOk, null);
  w.emit([{ type: 'network', route: 'tor', state: 'ready', detail: 'Connected to Tor in 7.1 s' }]);
  assert.equal(c.state.relayOk, true);
  assert.equal(c.state.network.detail, 'Connected to Tor in 7.1 s');
  // A Tor failure is not mistaken for a relay failure, and vice versa.
  w.emit([{ type: 'error', text: 'Could not reach the relay at x' }]);
  assert.equal(c.state.relayOk, true);
  w.emit([{ type: 'network', route: 'tor', state: 'failed', detail: 'Could not connect to the Tor network: timeout' }]);
  assert.equal(c.state.relayOk, false);
  assert.deepEqual(seen, ['connecting', 'ready', 'failed']);
  await c.setRelay('wss://relay.example/');
  assert.equal(w.messages.at(-1).type, 'setRelay');
  assert.equal(w.messages.at(-1).url, 'wss://relay.example/');
  assert.equal(c.state.choice, 'wss://relay.example/');
  assert.equal(c.state.relayOk, true, 'probed');
  await c.setRelay(AUTO_ROUTE);
  assert.equal(w.messages.at(-1).url, TOR_ROUTE);
  assert.equal(c.state.network, null);
  c.dispose();
});

test('Tor consensus sources: webtor snapshot first, then the bundled copy', () => {
  const list = torConsensusSources(['https://mirror.example/tor']);
  assert.equal(list[0], 'https://mirror.example/tor');
  assert.equal(list[1], 'https://raw.githubusercontent.com/ethereum/webtor-rs/main/webtor/src/cached');
  assert.match(list[2], /\/browser-tor\/$/);
  assert.equal(new Set(list).size, list.length);
});

test('agent actions resolve with the engine result and report what happened', async () => {
  const results = { agentClick: { ok: true, x: 5, y: 6, w: 10, h: 12, navigated: false }, agentType: { ok: false, error: 'stale' }, snapshot: { url: 'u', title: 't', text: '[1] link "a"', refs: 1, version: 3 } };
  const { factory, workers } = fakeWorkers((m) => results[m.type] ?? null);
  const c = client({ workerFactory: factory });
  const agentEvents = [];
  c.addEventListener('agent', (e) => agentEvents.push(e.detail));
  const snap = await c.snapshot(1000);
  assert.equal(snap.refs, 1);
  assert.equal(workers[0].messages.find((m) => m.type === 'snapshot').maxChars, 1000);
  assert.deepEqual(await c.click(1), results.agentClick);
  assert.deepEqual(await c.type(1, 'hi', { submit: true }), results.agentType);
  assert.deepEqual(workers[0].messages.find((m) => m.type === 'agentType'), { type: 'agentType', ref: 1, text: 'hi', submit: true, id: workers[0].messages.find((m) => m.type === 'agentType').id });
  assert.deepEqual(agentEvents, [{ action: 'click', ref: 1, ok: true, error: undefined }, { action: 'type', ref: 1, ok: false, error: 'stale' }]);
  c.dispose();
});

test('a crashed engine saves cookies, restarts and returns to the page, but stops after repeated crashes', async () => {
  const { workers, factory } = fakeWorkers((m) => (m.type === 'evaluate' ? undefined : null));
  const persistence = memoryPersistence({ cookies: 'old', storage: 'S' });
  const c = new BrowserClient({ workerFactory: factory, probe: async () => true, persistence });
  await c.ready;
  workers[0].emit([{ type: 'url', url: 'https://example.com/', canBack: false, canForward: false }]);
  const pending = c.evaluate('slow');
  await tick();
  workers[0].onmessage({ data: { type: 'crashed', error: 'unreachable', cookies: 'saved-cookies', storage: null } });
  await assert.rejects(pending, /stopped/);
  for (let i = 0; i < 5; i++) await tick();
  assert(workers[0].terminated);
  assert.equal(workers.length, 2);
  assert.equal(workers[1].messages[0].cookies, 'saved-cookies');
  assert.equal(workers[1].messages[0].storage, 'S');
  assert.equal(workers[1].messages.find((m) => m.type === 'navigate').url, 'https://example.com/');
  workers[1].onmessage({ data: { type: 'crashed', error: 'again' } });
  for (let i = 0; i < 5; i++) await tick();
  assert.equal(workers.length, 3);
  workers[2].onmessage({ data: { type: 'crashed', error: 'third' } });
  for (let i = 0; i < 5; i++) await tick();
  assert.equal(workers.length, 3, 'no restart loop');
  c.pointer('move', 1, 1, 0, 0, 0);
  await assert.rejects(c.evaluate('x'), /Press Reload/);
  assert.equal(workers.length, 3, 'input does not restart a halted engine');
  c.reload();
  for (let i = 0; i < 5; i++) await tick();
  assert.equal(workers.length, 4, 'Reload starts it again');
  assert.equal(workers[3].messages.find((m) => m.type === 'navigate').url, 'https://example.com/');
  c.dispose();
});

test('waitForIdle waits for loading to finish and the page to go quiet', async () => {
  const { workers, factory } = fakeWorkers(() => null);
  const c = client({ workerFactory: factory });
  await c.ready;
  workers[0].emit([{ type: 'loading', loading: true, url: 'https://a.test/' }]);
  let done = false;
  const wait = c.waitForIdle(3000, 50).then(() => { done = true; });
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(done, false);
  workers[0].emit([{ type: 'loading', loading: false, url: 'https://a.test/' }]);
  await wait;
  assert.equal(done, true);
  c.dispose();
});

// ---- worker host ----

function fakeEngine({ panicOn } = {}) {
  const calls = [];
  class Browser {
    constructor(fonts, relay, cookies, storage, wake) { calls.push(['new', fonts.length, relay, cookies, storage]); this.wake = wake; this.frames = 0; }
    resize(...a) { calls.push(['resize', ...a]); }
    tick() { calls.push(['tick']); if (panicOn === 'tick') throw new Error('RuntimeError: unreachable'); return this.frames++ === 0 ? 16 : -1; }
    takeFrame() { return this.frames === 1 ? new Uint8Array(2 * 1 * 4).fill(255) : undefined; }
    frameWidth() { return 2; } frameHeight() { return 1; }
    takeEvents() { return this.frames === 1 ? '[{"type":"title","title":"T"}]' : '[]'; }
    navigate(u) { calls.push(['navigate', u]); if (panicOn === 'navigate') throw new Error('RuntimeError: unreachable'); }
    agentClick(ref) { return JSON.stringify({ ok: true, x: 1, y: 2, ref }); }
    cookiesJson() { return 'C'; } storageJson() { return 'S'; }
  }
  return { Browser, calls };
}

function hostHarness(engine) {
  const posted = [];
  const timers = [];
  globalThis.ImageData ??= class { constructor(data, w, h) { this.data = data; this.width = w; this.height = h; } };
  const host = createBrowserHost({
    load: async () => ({ Browser: engine.Browser, fonts: [['a.ttf', new Uint8Array(1)]] }),
    post: (m) => posted.push(m),
    now: () => 100,
    setTimer: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
    clearTimer: () => {},
  });
  const drawn = [];
  const canvas = { width: 0, height: 0, getContext: () => ({ putImageData: (img) => drawn.push(img) }) };
  const runTimers = () => { while (timers.length) timers.shift().fn(); };
  return { host, posted, timers, drawn, canvas, runTimers };
}

test('worker host starts the engine, ticks on wake and timers, draws frames and forwards events', async () => {
  const engine = fakeEngine();
  const h = hostHarness(engine);
  await h.host.handle({ id: 1, type: 'init', canvas: h.canvas, relayUrl: 'ws://r/', cookies: 'C0', storage: null, width: 640, height: 480, dpr: 2 });
  assert.deepEqual(engine.calls.slice(0, 2), [['new', 1, 'ws://r/', 'C0', null], ['resize', 640, 480, 2]]);
  assert.deepEqual(h.posted.find((m) => m.id === 1), { id: 1, result: { fonts: 1 } });
  assert(h.posted.some((m) => m.type === 'ready'));
  const first = h.timers.shift();
  assert.equal(first.ms, 0);
  first.fn();
  assert.equal(h.drawn.length, 1);
  assert.equal(h.canvas.width, 2);
  assert.deepEqual(h.posted.find((m) => m.type === 'events').events, [{ type: 'title', title: 'T' }]);
  assert.equal(h.timers.at(-1).ms, 16, 'honors the next-timer delay');
  await h.host.handle({ id: 2, type: 'agentClick', ref: 4 });
  assert.deepEqual(h.posted.find((m) => m.id === 2).result, { ok: true, x: 1, y: 2, ref: 4 });
  await h.host.handle({ id: 3, type: 'nope' });
  assert.match(h.posted.find((m) => m.id === 3).error, /Unknown/);
});

test('worker host reports an engine panic once, with the cookies it could save', async () => {
  const engine = fakeEngine({ panicOn: 'navigate' });
  const h = hostHarness(engine);
  await h.host.handle({ id: 1, type: 'init', relayUrl: 'ws://r/' });
  await h.host.handle({ id: 2, type: 'navigate', url: 'https://x.test/' });
  const crash = h.posted.filter((m) => m.type === 'crashed');
  assert.equal(crash.length, 1);
  assert.deepEqual([crash[0].cookies, crash[0].storage], ['C', 'S']);
  assert.match(h.posted.find((m) => m.id === 2).error, /unreachable/);
  await h.host.handle({ id: 3, type: 'navigate', url: 'https://x.test/' });
  assert.match(h.posted.find((m) => m.id === 3).error, /stopped/);
  assert.equal(h.posted.filter((m) => m.type === 'crashed').length, 1);
  assert(h.host.crashed);
});
