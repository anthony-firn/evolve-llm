/**
 * The worker side of the WASM browser, independent of how the engine is loaded so it can be
 * tested with a fake engine. One Browser instance per worker; see browser/API.md.
 *
 * Messages from the main thread: { id?, type, ...args }. With an id, the worker replies
 * { id, result } or { id, error }. Without an id (pointer moves, wheel, keys) nothing is sent
 * back. The worker also sends { type: 'events', events }, { type: 'ready' }, { type: 'crashed' }.
 */

const AGENT = { snapshot: 'snapshot', agentClick: 'agentClick', agentType: 'agentType', agentSelect: 'agentSelect', agentScroll: 'agentScroll', agentKey: 'agentKey' };

/**
 * Where the Tor route gets its cached consensus: the snapshot webtor's CI commits daily first,
 * then the copy bundled next to this module at build time (web/browser-tor/).
 */
export function torConsensusSources(extra) {
  const list = Array.isArray(extra) ? extra.map(String) : [];
  list.push('https://raw.githubusercontent.com/ethereum/webtor-rs/main/webtor/src/cached');
  try { list.push(new URL('./browser-tor/', import.meta.url).href); } catch {}
  return [...new Set(list)];
}

export function createBrowserHost({ load, post, now = () => performance.now(), setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (id) => clearTimeout(id) }) {
  let browser = null;
  let canvas = null;
  let ctx = null;
  let crashed = false;
  let wakeQueued = false;
  let timer = null;
  let starting = null;
  let size = { width: 1280, height: 800, dpr: 1 };

  const isPanic = (e) => (typeof WebAssembly !== 'undefined' && e instanceof WebAssembly.RuntimeError) || /unreachable|RuntimeError|already borrowed|panicked/i.test(String(e && (e.message || e)));

  function crash(error) {
    if (crashed) return;
    crashed = true;
    clearTimer(timer);
    let cookies = null, storage = null;
    // After a panic the engine's state may be poisoned; saving is best effort.
    try { cookies = browser.cookiesJson(); } catch {}
    try { storage = browser.storageJson(); } catch {}
    post({ type: 'crashed', error: String((error && (error.message || error)) || 'The browser engine stopped.'), cookies, storage });
  }

  // Calls into the engine; a panic in the engine is reported once and later calls fail fast.
  function engine(fn) {
    if (crashed) throw new Error('The browser engine stopped. Reload the browser to continue.');
    if (!browser) throw new Error('The browser engine is still starting.');
    try { return fn(browser); } catch (e) { if (isPanic(e)) crash(e); throw e; }
  }

  function wake() {
    if (wakeQueued || crashed) return;
    wakeQueued = true;
    setTimer(run, 0);
  }

  function draw() {
    const pixels = browser.takeFrame();
    if (!pixels) return false;
    const width = browser.frameWidth(), height = browser.frameHeight();
    if (!ctx || !width || !height || pixels.length !== width * height * 4) return false;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const data = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.byteLength);
    ctx.putImageData(new ImageData(data, width, height), 0, 0);
    return true;
  }

  function flushEvents() {
    const text = browser.takeEvents();
    if (!text || text === '[]') return;
    let events;
    try { events = JSON.parse(text); } catch { return; }
    if (events.length) post({ type: 'events', events });
  }

  function run() {
    wakeQueued = false;
    if (!browser || crashed) return;
    let next = -1;
    try {
      next = browser.tick(now());
      draw();
      flushEvents();
    } catch (e) {
      crash(e);
      return;
    }
    clearTimer(timer);
    timer = null;
    if (typeof next === 'number' && next >= 0) timer = setTimer(run, Math.min(Math.max(0, next), 2147483647));
  }

  async function start(message) {
    if (message.canvas) attachCanvas(message.canvas);
    if (message.width) size = { width: message.width, height: message.height, dpr: message.dpr || 1 };
    const { Browser, fonts } = await load();
    browser = new Browser(fonts, message.relayUrl, message.cookies ?? null, message.storage ?? null, wake);
    browser.setTorConsensus?.(torConsensusSources(message.torConsensus));
    if (message.torLog) browser.enableTorLog?.();
    browser.resize(size.width, size.height, size.dpr);
    browser.prepareNetwork?.();
    post({ type: 'ready' });
    wake();
    return { fonts: fonts.length };
  }

  function attachCanvas(offscreen) {
    canvas = offscreen;
    ctx = canvas.getContext('2d', { alpha: false });
  }

  const handlers = {
    navigate: (b, m) => b.navigate(String(m.url)),
    back: (b) => b.back(),
    forward: (b) => b.forward(),
    reload: (b) => b.reload(),
    stop: (b) => b.stop(),
    resize: (b, m) => { size = { width: m.width, height: m.height, dpr: m.dpr || 1 }; b.resize(size.width, size.height, size.dpr); },
    setRelay: (b, m) => { b.setRelay(String(m.url)); b.prepareNetwork?.(); },
    pointer: (b, m) => b.pointer(m.kind, m.x, m.y, m.button | 0, m.buttons | 0, m.mods | 0),
    wheel: (b, m) => b.wheel(m.x, m.y, m.dx, m.dy, m.deltaMode | 0, m.mods | 0),
    key: (b, m) => b.key(m.kind, String(m.key), String(m.code || ''), m.mods | 0, !!m.repeat),
    imeCommit: (b, m) => b.imeCommit(String(m.text)),
    imePreedit: (b, m) => b.imePreedit(String(m.text)),
    chooseOption: (b, m) => b.chooseOption(Number(m.id), m.index | 0),
    evaluate: (b, m) => b.evaluate(String(m.code)),
    networkLog: (b) => JSON.parse(b.networkLog() || '[]'),
    torStats: (b) => JSON.parse(b.torStats?.() || '{}'),
    cookiesJson: (b) => b.cookiesJson(),
    storageJson: (b) => b.storageJson(),
    clearData: (b) => b.clearData(),
    snapshot: (b, m) => JSON.parse(b.snapshot(m.maxChars || 6000)),
    agentClick: (b, m) => JSON.parse(b.agentClick(m.ref)),
    agentType: (b, m) => JSON.parse(b.agentType(m.ref, String(m.text), !!m.submit)),
    agentSelect: (b, m) => JSON.parse(b.agentSelect(m.ref, String(m.value))),
    agentScroll: (b, m) => JSON.parse(b.agentScroll(m.dy)),
    agentKey: (b, m) => JSON.parse(b.agentKey(String(m.key))),
  };

  async function handle(message) {
    const { id, type } = message || {};
    const reply = (body) => { if (id !== undefined) post({ id, ...body }); };
    try {
      let result;
      if (type === 'init') {
        starting ||= start(message);
        result = await starting;
      } else if (type === 'canvas') {
        attachCanvas(message.canvas);
        if (browser) { try { browser.resize(size.width, size.height, size.dpr); } catch (e) { if (isPanic(e)) crash(e); } }
        result = true;
      } else if (handlers[type]) {
        if (starting) await starting;
        result = engine((b) => handlers[type](b, message));
        if (AGENT[type]) run(); // show the effect of an agent action right away
      } else {
        throw new Error(`Unknown browser request ${type}`);
      }
      reply({ result: result === undefined ? null : result });
    } catch (e) {
      reply({ error: String((e && (e.message || e)) || e) });
    } finally {
      if (browser && !crashed) wake();
    }
  }

  return { handle, wake, get crashed() { return crashed; } };
}
