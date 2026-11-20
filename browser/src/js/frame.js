// Minimal iframe realm for classic inline scripts and origin-checked messaging.
// The host interface has no embedding-document capabilities. Cross-origin windows are
// realm-local facades; only serialized message values cross the boundary.
(function () {
  'use strict';
  const G = globalThis;
  const H = G.__h;
  delete G.__h;
  class DOMException extends Error { constructor(message, name) { super(message); this.name = name; } }
  G.DOMException = DOMException;
  /*__WINDOW_BRIDGE__*/
  const parentWindow = windowProxy(0);
  const topWindow = parentWindow;
  const origin = String(G.__frameOrigin || '');
  const href = String(G.__frameUrl || '');
  for (const k of ['__parentWindow', '__topWindow', '__frameOrigin', '__frameUrl', '__frameId_value']) { try { delete G[k]; } catch (e) {} }

  const listeners = new Map();
  const add = (type, cb) => { if (!cb || (typeof cb !== 'function' && typeof cb.handleEvent !== 'function')) return; const a = listeners.get(type) || []; if (!a.includes(cb)) a.push(cb); listeners.set(type, a); };
  const rm = (type, cb) => { const a = listeners.get(type); if (a) { const i = a.indexOf(cb); if (i >= 0) a.splice(i, 1); } };
  const dispatch = (ev) => {
    const a = (listeners.get(ev.type) || []).slice();
    for (const cb of a) { try { (typeof cb === 'function' ? cb : cb.handleEvent).call(G, ev); } catch (e) { H.log('error', 'Uncaught ' + (e && e.stack || e)); } }
    const on = G['on' + ev.type];
    if (typeof on === 'function') { try { on.call(G, ev); } catch (e) { H.log('error', 'Uncaught ' + (e && e.stack || e)); } }
    return true;
  };

  class MessageEvent {
    constructor(type, init = {}) { this.type = type; this.data = init.data; this.origin = init.origin || ''; this.source = init.source || null; this.lastEventId = init.lastEventId || ''; this.ports = init.ports || []; this.bubbles = false; this.cancelable = false; this.defaultPrevented = false; }
    stopPropagation() {} stopImmediatePropagation() {} preventDefault() { this.defaultPrevented = true; }
  }
  G.MessageEvent = MessageEvent;

  const u = (() => { try { return new URL(href); } catch (e) { return null; } })();
  const location = {
    href, origin, protocol: u ? u.protocol : '', host: u ? u.host : '', hostname: u ? u.hostname : '',
    port: u ? u.port : '', pathname: u ? u.pathname : '', search: u ? u.search : '', hash: u ? u.hash : '',
    toString() { return href; }, assign() {}, replace() {}, reload() {},
  };

  G.window = G; G.self = G; G.frames = G; G.parent = parentWindow; G.top = topWindow;
  G.opener = null; G.frameElement = null; G.closed = false; G.length = 0; G.name = ''; G.origin = origin;
  G.location = location;
  G.addEventListener = add; G.removeEventListener = rm; G.dispatchEvent = dispatch;
  G.postMessage = function (data, targetOrigin) { postToWindow(sourceFrame, data, targetOrigin); };
  // Timers run as microtasks: the delay is ignored, which suffices for the handshakes frames use.
  G.setTimeout = function (fn) { if (typeof fn === 'function') Promise.resolve().then(() => { try { fn(); } catch (e) { H.log('error', 'Uncaught ' + (e && e.stack || e)); } }); return 0; };
  G.setInterval = function () { return 0; };
  G.clearTimeout = function () {}; G.clearInterval = function () {};
  G.queueMicrotask = function (fn) { if (typeof fn === 'function') Promise.resolve().then(fn); };
  const out = (level) => (...a) => H.log(level, a.map((x) => { try { return typeof x === 'string' ? x : JSON.stringify(x); } catch (e) { return String(x); } }).join(' '));
  G.console = { log: out('log'), info: out('info'), warn: out('warn'), error: out('error'), debug: out('debug'), trace: out('log'), dir: out('log'), group() {}, groupEnd() {}, groupCollapsed() {}, table: out('log'), assert() {}, count() {}, time() {}, timeEnd() {}, timeLog() {} };

  // A near-empty document: just enough that scripts reading document.* do not throw. These
  // frames (bscframe, CheckConnection) do their work through postMessage, not the DOM.
  const stubEl = () => ({ style: {}, setAttribute() {}, getAttribute() { return null; }, appendChild(x) { return x; }, removeChild(x) { return x; }, addEventListener() {}, removeEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; }, getElementsByTagName() { return []; }, classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} } });
  G.document = {
    cookie: '', referrer: topWindow === parentWindow ? '' : '', readyState: 'complete', location, URL: href, documentURI: href, characterSet: 'UTF-8',
    title: '', body: stubEl(), head: stubEl(), documentElement: stubEl(),
    getElementById() { return null; }, getElementsByTagName() { return []; }, getElementsByClassName() { return []; }, querySelector() { return null; }, querySelectorAll() { return []; },
    createElement() { return stubEl(); }, createTextNode() { return {}; }, createElementNS() { return stubEl(); },
    addEventListener() {}, removeEventListener() {}, write() {}, writeln() {}, open() {}, close() {},
  };
  const ua = (() => { try { return JSON.parse(H.identity()).userAgent || ''; } catch (e) { return ''; } })();
  G.navigator = { userAgent: ua, languages: ['en-US', 'en'], language: 'en-US', onLine: true, cookieEnabled: true, platform: '' };
  G.screen = { width: 0, height: 0, availWidth: 0, availHeight: 0, colorDepth: 24, pixelDepth: 24 };
  const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  G.btoa = (s) => { s = String(s); let o = ''; for (let i = 0; i < s.length;) { const a = s.charCodeAt(i++), b = i < s.length ? s.charCodeAt(i++) : NaN, c = i < s.length ? s.charCodeAt(i++) : NaN; const n = (a << 16) | ((b || 0) << 8) | (c || 0); o += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (isNaN(b) ? '=' : B64[(n >> 6) & 63]) + (isNaN(c) ? '=' : B64[n & 63]); } return o; };
  G.atob = (s) => { s = String(s).replace(/[^A-Za-z0-9+/]/g, ''); let o = ''; for (let i = 0; i < s.length;) { const n = (B64.indexOf(s[i++]) << 18) | (B64.indexOf(s[i++]) << 12) | (B64.indexOf(s[i++]) << 6) | B64.indexOf(s[i++]); o += String.fromCharCode((n >> 16) & 255); if (s[i - 2] !== undefined) o += String.fromCharCode((n >> 8) & 255); if (s[i - 1] !== undefined) o += String.fromCharCode(n & 255); } return o; };

  // Called by the host's posted-message task in this realm, after sender/target epoch checks.
  Object.defineProperty(G, '__deliverMessage', { value: (wire, origin, source) => { dispatch(new MessageEvent('message', { data: decodeWindowMessage(wire), origin, source: windowProxy(source) })); }, enumerable: false });
})();
