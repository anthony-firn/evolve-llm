// Evolve Browser: the DOM and web platform APIs, implemented in JavaScript over the engine's
// host functions (`__h`). Pages see standard globals; the host object is captured here and
// removed from the global scope.
(function () {
  'use strict';
  const H = globalThis.__h;
  delete globalThis.__h;
  const G = globalThis;
  const ID = Symbol('node');
  const identity = JSON.parse(H.identity());
  const UA = identity.userAgent;
  const define = (obj, props) => { for (const k of Object.keys(props)) Object.defineProperty(obj, k, { value: props[k], writable: true, configurable: true, enumerable: false }); };
  const getters = (obj, props) => { for (const k of Object.keys(props)) { const d = props[k]; Object.defineProperty(obj, k, typeof d === 'function' ? { get: d, configurable: true, enumerable: true } : Object.assign({ configurable: true, enumerable: true }, d)); } };
  const fmt = (v, depth = 0) => {
    if (typeof v === 'string') return depth ? JSON.stringify(v) : v;
    if (v instanceof Error) return (v.name || 'Error') + ': ' + v.message + (v.stack ? '\n' + v.stack : '');
    if (v && v[ID] !== undefined) return '<' + (v.nodeName || 'node').toLowerCase() + '>';
    if (typeof v === 'function') return 'function ' + (v.name || '(anonymous)');
    if (typeof v === 'symbol') return v.toString();
    if (v && typeof v === 'object') {
      if (depth > 2) return Array.isArray(v) ? '[…]' : '{…}';
      try {
        if (Array.isArray(v)) return '[' + v.slice(0, 50).map(x => fmt(x, depth + 1)).join(', ') + ']';
        return '{' + Object.keys(v).slice(0, 30).map(k => k + ': ' + fmt(v[k], depth + 1)).join(', ') + '}';
      } catch (e) { return String(v); }
    }
    return String(v);
  };
  const reportError = (e) => {
    try {
      H.log('error', 'Uncaught ' + fmt(e));
      const ev = new ErrorEvent('error', { message: e && e.message || String(e), error: e, cancelable: true });
      dispatchTo(G, ev);
    } catch (_) {}
  };
  const call = (fn, self, args) => { try { return fn.apply(self, args); } catch (e) { reportError(e); } };
  const kebab = (s) => s === 'cssFloat' ? 'float' : s.replace(/^(webkit|moz|ms)(?=[A-Z])/, '-$1').replace(/[A-Z]/g, m => '-' + m.toLowerCase());
  const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

  // ---------------------------------------------------------------- DOMException
  class DOMException extends Error {
    constructor(message = '', name = 'Error') { super(message); this.name = name; }
    get code() { return { IndexSizeError: 1, HierarchyRequestError: 3, NotFoundError: 8, NotSupportedError: 9, InvalidStateError: 11, SyntaxError: 12, SecurityError: 18, NetworkError: 19, AbortError: 20, QuotaExceededError: 22, TimeoutError: 23 }[this.name] || 0; }
  }

  // ---------------------------------------------------------------- Events
  class Event {
    constructor(type, init = {}) {
      if (arguments.length === 0) throw new TypeError("Failed to construct 'Event': 1 argument required");
      this._type = String(type); this._bubbles = !!init.bubbles; this._cancelable = !!init.cancelable; this._composed = !!init.composed;
      this._target = null; this._current = null; this._phase = 0; this._stop = false; this._stopNow = false; this._canceled = false;
      this._trusted = false; this._time = performance.now(); this._passive = false;
    }
    get type() { return this._type; } get bubbles() { return this._bubbles; } get cancelable() { return this._cancelable; } get composed() { return this._composed; }
    get target() { return this._target; } get srcElement() { return this._target; } get currentTarget() { return this._current; } get eventPhase() { return this._phase; }
    get defaultPrevented() { return this._canceled; } get isTrusted() { return this._trusted; } get timeStamp() { return this._time; }
    get returnValue() { return !this._canceled; } set returnValue(v) { if (!v) this.preventDefault(); }
    get cancelBubble() { return this._stop; } set cancelBubble(v) { if (v) this._stop = true; }
    composedPath() { return this._path ? this._path.slice() : []; }
    stopPropagation() { this._stop = true; }
    stopImmediatePropagation() { this._stop = true; this._stopNow = true; }
    preventDefault() { if (this._cancelable && !this._passive) this._canceled = true; }
    initEvent(type, bubbles, cancelable) { this._type = String(type); this._bubbles = !!bubbles; this._cancelable = !!cancelable; }
  }
  define(Event, { NONE: 0, CAPTURING_PHASE: 1, AT_TARGET: 2, BUBBLING_PHASE: 3 });
  class CustomEvent extends Event { constructor(t, i = {}) { super(t, i); this.detail = i.detail === undefined ? null : i.detail; } initCustomEvent(t, b, c, d) { this.initEvent(t, b, c); this.detail = d; } }
  class UIEvent extends Event { constructor(t, i = {}) { super(t, i); this.view = i.view || G; this.detail = i.detail || 0; this.which = i.which || 0; } }
  class FocusEvent extends UIEvent { constructor(t, i = {}) { super(t, i); this.relatedTarget = i.relatedTarget || null; } }
  class MouseEvent extends UIEvent {
    constructor(t, i = {}) {
      super(t, i);
      for (const k of ['screenX', 'screenY', 'clientX', 'clientY', 'offsetX', 'offsetY', 'movementX', 'movementY', 'button', 'buttons']) this[k] = i[k] || 0;
      for (const k of ['ctrlKey', 'shiftKey', 'altKey', 'metaKey']) this[k] = !!i[k];
      this.relatedTarget = i.relatedTarget || null;
      this.pageX = i.pageX !== undefined ? i.pageX : this.clientX + (G.scrollX || 0); this.pageY = i.pageY !== undefined ? i.pageY : this.clientY + (G.scrollY || 0);
      this.x = this.clientX; this.y = this.clientY; this.which = this.button + 1;
    }
    getModifierState(k) { return { Control: this.ctrlKey, Shift: this.shiftKey, Alt: this.altKey, Meta: this.metaKey }[k] || false; }
  }
  class PointerEvent extends MouseEvent {
    constructor(t, i = {}) { super(t, i); this.pointerId = i.pointerId || 1; this.pointerType = i.pointerType || 'mouse'; this.isPrimary = i.isPrimary !== false; this.width = i.width || 1; this.height = i.height || 1; this.pressure = i.pressure || 0; this.tiltX = 0; this.tiltY = 0; this.twist = 0; }
    getCoalescedEvents() { return [this]; } getPredictedEvents() { return []; }
  }
  class WheelEvent extends MouseEvent { constructor(t, i = {}) { super(t, i); this.deltaX = i.deltaX || 0; this.deltaY = i.deltaY || 0; this.deltaZ = 0; this.deltaMode = i.deltaMode || 0; } }
  define(WheelEvent, { DOM_DELTA_PIXEL: 0, DOM_DELTA_LINE: 1, DOM_DELTA_PAGE: 2 });
  class KeyboardEvent extends UIEvent {
    constructor(t, i = {}) {
      super(t, i); this.key = i.key || ''; this.code = i.code || ''; this.location = i.location || 0; this.repeat = !!i.repeat; this.isComposing = !!i.isComposing;
      for (const k of ['ctrlKey', 'shiftKey', 'altKey', 'metaKey']) this[k] = !!i[k];
      const kc = i.keyCode || keyCodeOf(this.key, this.code); this.keyCode = kc; this.which = kc; this.charCode = t === 'keypress' && this.key.length === 1 ? this.key.charCodeAt(0) : 0;
    }
    getModifierState(k) { return { Control: this.ctrlKey, Shift: this.shiftKey, Alt: this.altKey, Meta: this.metaKey }[k] || false; }
  }
  const KEYCODES = { Backspace: 8, Tab: 9, Enter: 13, Shift: 16, Control: 17, Alt: 18, Pause: 19, CapsLock: 20, Escape: 27, ' ': 32, PageUp: 33, PageDown: 34, End: 35, Home: 36, ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Insert: 45, Delete: 46, Meta: 91 };
  function keyCodeOf(key, code) {
    if (KEYCODES[key] !== undefined) return KEYCODES[key];
    if (key.length === 1) { const c = key.toUpperCase().charCodeAt(0); if ((c >= 65 && c <= 90) || (c >= 48 && c <= 57)) return c; }
    const m = /^F(\d+)$/.exec(key); if (m) return 111 + +m[1];
    return 0;
  }
  class InputEvent extends UIEvent { constructor(t, i = {}) { super(t, i); this.data = i.data === undefined ? null : i.data; this.inputType = i.inputType || ''; this.isComposing = !!i.isComposing; } getTargetRanges() { return []; } }
  class CompositionEvent extends UIEvent { constructor(t, i = {}) { super(t, i); this.data = i.data || ''; } }
  class SubmitEvent extends Event { constructor(t, i = {}) { super(t, i); this.submitter = i.submitter || null; } }
  class ErrorEvent extends Event { constructor(t, i = {}) { super(t, i); this.message = i.message || ''; this.filename = i.filename || ''; this.lineno = i.lineno || 0; this.colno = i.colno || 0; this.error = i.error; } }
  class ProgressEvent extends Event { constructor(t, i = {}) { super(t, i); this.lengthComputable = !!i.lengthComputable; this.loaded = i.loaded || 0; this.total = i.total || 0; } }
  class MessageEvent extends Event { constructor(t, i = {}) { super(t, i); this.data = i.data; this.origin = i.origin || ''; this.lastEventId = ''; this.source = i.source || null; this.ports = []; } }
  class PopStateEvent extends Event { constructor(t, i = {}) { super(t, i); this.state = i.state === undefined ? null : i.state; } }
  class HashChangeEvent extends Event { constructor(t, i = {}) { super(t, i); this.oldURL = i.oldURL || ''; this.newURL = i.newURL || ''; } }
  class PageTransitionEvent extends Event { constructor(t, i = {}) { super(t, i); this.persisted = !!i.persisted; } }
  class StorageEvent extends Event { constructor(t, i = {}) { super(t, i); this.key = i.key || null; this.oldValue = i.oldValue || null; this.newValue = i.newValue || null; this.url = i.url || ''; this.storageArea = i.storageArea || null; } }
  class PromiseRejectionEvent extends Event { constructor(t, i = {}) { super(t, i); this.promise = i.promise; this.reason = i.reason; } }
  class AnimationEvent extends Event { constructor(t, i = {}) { super(t, i); this.animationName = i.animationName || ''; this.elapsedTime = 0; this.pseudoElement = ''; } }
  class TransitionEvent extends Event { constructor(t, i = {}) { super(t, i); this.propertyName = i.propertyName || ''; this.elapsedTime = 0; this.pseudoElement = ''; } }
  class ClipboardEvent extends Event { constructor(t, i = {}) { super(t, i); this.clipboardData = i.clipboardData || null; } }
  class DragEvent extends MouseEvent { constructor(t, i = {}) { super(t, i); this.dataTransfer = i.dataTransfer || null; } }
  class TouchEvent extends UIEvent { constructor(t, i = {}) { super(t, i); this.touches = i.touches || []; this.targetTouches = i.targetTouches || []; this.changedTouches = i.changedTouches || []; for (const k of ['ctrlKey', 'shiftKey', 'altKey', 'metaKey']) this[k] = !!i[k]; } }

  const listeners = new WeakMap();
  const handlers = new WeakMap();
  class EventTarget {
    addEventListener(type, cb, opts) {
      if (cb == null) return;
      const capture = typeof opts === 'boolean' ? opts : !!(opts && opts.capture);
      const once = !!(opts && typeof opts === 'object' && opts.once);
      const passive = !!(opts && typeof opts === 'object' && opts.passive);
      const signal = opts && typeof opts === 'object' ? opts.signal : null;
      if (signal && signal.aborted) return;
      let map = listeners.get(this); if (!map) { map = new Map(); listeners.set(this, map); }
      let list = map.get(type); if (!list) { list = []; map.set(type, list); }
      if (list.some(l => l.cb === cb && l.capture === capture)) return;
      const entry = { cb, capture, once, passive, removed: false };
      list.push(entry);
      if (signal) signal.addEventListener('abort', () => this.removeEventListener(type, cb, capture));
      H.listen(String(type));
    }
    removeEventListener(type, cb, opts) {
      const capture = typeof opts === 'boolean' ? opts : !!(opts && opts.capture);
      const list = listeners.get(this) && listeners.get(this).get(type); if (!list) return;
      const i = list.findIndex(l => l.cb === cb && l.capture === capture);
      if (i >= 0) { list[i].removed = true; list.splice(i, 1); }
    }
    dispatchEvent(ev) {
      if (!(ev instanceof Event)) throw new TypeError("Failed to execute 'dispatchEvent': parameter 1 is not of type 'Event'.");
      return dispatchTo(this, ev);
    }
  }
  const HANDLER_EVENTS = ['abort', 'animationend', 'animationstart', 'beforeinput', 'beforeunload', 'blur', 'change', 'click', 'close', 'contextmenu', 'dblclick', 'drag', 'dragend', 'dragenter', 'dragleave', 'dragover', 'dragstart', 'drop', 'error', 'focus', 'focusin', 'focusout', 'hashchange', 'input', 'invalid', 'keydown', 'keypress', 'keyup', 'load', 'loadend', 'loadstart', 'message', 'mousedown', 'mouseenter', 'mouseleave', 'mousemove', 'mouseout', 'mouseover', 'mouseup', 'pagehide', 'pageshow', 'paste', 'copy', 'cut', 'pointercancel', 'pointerdown', 'pointerenter', 'pointerleave', 'pointermove', 'pointerout', 'pointerover', 'pointerup', 'popstate', 'progress', 'readystatechange', 'reset', 'resize', 'scroll', 'select', 'storage', 'submit', 'toggle', 'touchcancel', 'touchend', 'touchmove', 'touchstart', 'transitionend', 'unhandledrejection', 'unload', 'visibilitychange', 'wheel', 'timeout'];
  for (const name of HANDLER_EVENTS) {
    Object.defineProperty(EventTarget.prototype, 'on' + name, {
      configurable: true, enumerable: true,
      get() { const m = handlers.get(this); return (m && m[name]) || null; },
      set(fn) { let m = handlers.get(this); if (!m) { m = {}; handlers.set(this, m); } m[name] = typeof fn === 'function' ? fn : null; H.listen(name); },
    });
  }
  function inlineHandler(target, type) {
    const m = handlers.get(target);
    if (m && m[type]) return m[type];
    if (target[ID] === undefined || target.nodeType !== 1) return null;
    const code = H.getAttr(target[ID], 'on' + type);
    if (code == null) return null;
    const cache = target._inline || (target._inline = {});
    if (cache[type] && cache[type].code === code) return cache[type].fn;
    let fn = null;
    try { fn = new Function('event', 'with (this.ownerDocument || document) { with (this.form || {}) { with (this) {\n' + code + '\n} } }'); } catch (e) { reportError(e); }
    cache[type] = { code, fn };
    return fn;
  }
  function invoke(node, ev, phase) {
    ev._current = node;
    ev._phase = node === ev._target ? 2 : phase;
    const inline = phase !== 1 ? inlineHandler(node, ev._type) : null;
    if (inline) {
      const r = call(inline, node, [ev]);
      if (r === false && ev._type !== 'mouseover') ev.preventDefault();
      if (ev._stopNow) return;
    }
    const map = listeners.get(node); const list = map && map.get(ev._type); if (!list) return;
    for (const l of list.slice()) {
      if (l.removed) continue;
      if (ev._phase === 1 && !l.capture) continue;
      if (ev._phase === 3 && l.capture) continue;
      if (l.once) node.removeEventListener(ev._type, l.cb, l.capture);
      ev._passive = l.passive;
      if (typeof l.cb === 'function') call(l.cb, node, [ev]); else if (l.cb && typeof l.cb.handleEvent === 'function') call(l.cb.handleEvent, l.cb, [ev]);
      ev._passive = false;
      if (ev._stopNow) break;
    }
  }
  function eventPath(target) {
    const path = [target];
    if (target === G) return path;
    if (target[ID] === undefined) return path;
    let id = target[ID];
    for (;;) { const p = H.parent(id); if (p < 0) break; id = p; path.push(wrap(id)); }
    if (path[path.length - 1] === document) path.push(G);
    return path;
  }
  function dispatchTo(target, ev) {
    ev._target = target; ev._stop = false; ev._stopNow = false;
    const path = eventPath(target); ev._path = path;
    for (let i = path.length - 1; i > 0 && !ev._stop; i--) invoke(path[i], ev, 1);
    if (!ev._stop) invoke(target, ev, 2);
    if (ev._bubbles) for (let i = 1; i < path.length && !ev._stop; i++) invoke(path[i], ev, 3);
    ev._current = null; ev._phase = 0;
    return !ev._canceled;
  }

  // ---------------------------------------------------------------- Nodes
  const nodes = new Map();
  let upgrading = -1;
  const idOf = (n, what = 'Node') => { if (n == null || n[ID] === undefined) throw new TypeError("parameter is not of type '" + what + "'."); return n[ID]; };
  const TAG_CLASS = {};
  function wrap(id) {
    if (id == null || id < 0) return null;
    let n = nodes.get(id); if (n) return n;
    const info = H.info(id);
    let proto;
    switch (info[0]) {
      case 1: {
        const tag = info[1];
        if (info[2] === 'svg') proto = (tag === 'svg' ? SVGSVGElement : SVGElement).prototype;
        else {
          const def = customDefs.get(tag);
          if (def && def.ready) { upgrading = id; try { n = Reflect.construct(def.cls, []); } catch (e) { reportError(e); } finally { upgrading = -1; } if (n) { nodes.set(id, n); return n; } }
          proto = (TAG_CLASS[tag] || (tag.includes('-') ? HTMLElement : HTMLUnknownElement)).prototype;
        }
        break;
      }
      case 3: proto = Text.prototype; break;
      case 8: proto = Comment.prototype; break;
      case 9: return document;
      default: return null;
    }
    n = Object.create(proto);
    n[ID] = id;
    nodes.set(id, n);
    return n;
  }
  const fragments = new WeakSet();
  const toNode = (v) => (v != null && v[ID] !== undefined) ? v : document.createTextNode(String(v));

  // MutationObserver bookkeeping
  const observers = [];
  function queueRecord(rec) {
    if (!observers.length) return;
    for (const o of observers) o._consider(rec);
  }

  class Node extends EventTarget {
    get nodeType() { return H.info(this[ID])[0]; }
    get nodeName() { return this.nodeType === 1 ? this.tagName : H.info(this[ID])[1]; }
    get nodeValue() { const t = this.nodeType; return t === 3 || t === 8 ? H.text(this[ID]) : null; }
    set nodeValue(v) { const t = this.nodeType; if (t === 3 || t === 8) this.data = v; }
    get textContent() { return this.nodeType === 9 ? null : H.text(this[ID]); }
    set textContent(v) {
      const old = this.nodeType === 1 ? this.childNodes : [];
      H.setText(this[ID], v == null ? '' : String(v));
      if (this.nodeType === 1) queueRecord({ type: 'childList', target: this, removedNodes: old, addedNodes: this.childNodes });
      else queueRecord({ type: 'characterData', target: this });
    }
    get parentNode() { const p = H.parent(this[ID]); return p < 0 ? null : wrap(p); }
    get parentElement() { const p = this.parentNode; return p && p.nodeType === 1 ? p : null; }
    get childNodes() { return makeNodeList(H.children(this[ID]).map(wrap)); }
    get firstChild() { const c = H.children(this[ID]); return c.length ? wrap(c[0]) : null; }
    get lastChild() { const c = H.children(this[ID]); return c.length ? wrap(c[c.length - 1]) : null; }
    get nextSibling() { return sibling(this, 1); }
    get previousSibling() { return sibling(this, -1); }
    get ownerDocument() { return this === document ? null : document; }
    get isConnected() { let id = this[ID]; const root = H.documentId(); for (;;) { if (id === root) return true; const p = H.parent(id); if (p < 0) return false; id = p; } }
    get baseURI() { return document.baseURI; }
    hasChildNodes() { return H.children(this[ID]).length > 0; }
    getRootNode() { let n = this; for (;;) { const p = n.parentNode; if (!p) return n; n = p; } }
    contains(other) { if (other == null) return false; let id = other[ID]; for (;;) { if (id === this[ID]) return true; const p = H.parent(id); if (p < 0) return false; id = p; } }
    isSameNode(o) { return o === this; }
    isEqualNode(o) { return !!o && o.nodeType === this.nodeType && (this.nodeType === 1 ? o.outerHTML === this.outerHTML : o.textContent === this.textContent); }
    compareDocumentPosition(other) {
      if (other === this) return 0;
      if (this.contains(other)) return 20; if (other.contains(this)) return 10;
      if (!this.isConnected || !other.isConnected) return 1 | 32;
      const all = [...document.querySelectorAll('*')]; const a = this.nodeType === 1 ? this : this.parentElement; const b = other.nodeType === 1 ? other : other.parentElement;
      return all.indexOf(a) < all.indexOf(b) ? 4 : 2;
    }
    appendChild(child) { insert(this, child, null); return child; }
    insertBefore(child, ref) { insert(this, child, ref == null ? null : ref); return child; }
    removeChild(child) {
      if (!child || H.parent(idOf(child)) !== this[ID]) throw new DOMException("The node to be removed is not a child of this node.", 'NotFoundError');
      const connected = child.nodeType === 1 && customDefs.size && child.isConnected;
      const prev = child.previousSibling, next = child.nextSibling;
      H.remove(child[ID]);
      if (connected) walkCustom(child, 'disconnectedCallback');
      queueRecord({ type: 'childList', target: this, removedNodes: [child], addedNodes: [], previousSibling: prev, nextSibling: next });
      return child;
    }
    replaceChild(newChild, oldChild) { this.insertBefore(newChild, oldChild); this.removeChild(oldChild); return oldChild; }
    cloneNode(deep = false) {
      if (this.nodeType === 3) return document.createTextNode(this.data);
      if (this.nodeType === 8) return document.createComment(this.data);
      if (fragments.has(this)) { const f = document.createDocumentFragment(); if (deep) for (const c of this.childNodes) f.appendChild(c.cloneNode(true)); return f; }
      return wrap(H.clone(this[ID], !!deep));
    }
    normalize() {
      let prev = null;
      for (const c of this.childNodes) {
        if (c.nodeType === 3) { if (c.data === '') { c.remove(); continue; } if (prev) { prev.data += c.data; c.remove(); continue; } prev = c; }
        else { prev = null; if (c.nodeType === 1) c.normalize(); }
      }
    }
    lookupNamespaceURI() { return null; } lookupPrefix() { return null; } isDefaultNamespace() { return true; }
  }
  define(Node, { ELEMENT_NODE: 1, ATTRIBUTE_NODE: 2, TEXT_NODE: 3, CDATA_SECTION_NODE: 4, PROCESSING_INSTRUCTION_NODE: 7, COMMENT_NODE: 8, DOCUMENT_NODE: 9, DOCUMENT_TYPE_NODE: 10, DOCUMENT_FRAGMENT_NODE: 11, DOCUMENT_POSITION_DISCONNECTED: 1, DOCUMENT_POSITION_PRECEDING: 2, DOCUMENT_POSITION_FOLLOWING: 4, DOCUMENT_POSITION_CONTAINS: 8, DOCUMENT_POSITION_CONTAINED_BY: 16, DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC: 32 });
  define(Node.prototype, { ELEMENT_NODE: 1, TEXT_NODE: 3, COMMENT_NODE: 8, DOCUMENT_NODE: 9, DOCUMENT_FRAGMENT_NODE: 11 });

  function sibling(node, dir) {
    const p = H.parent(node[ID]); if (p < 0) return null;
    const c = H.children(p); const i = c.indexOf(node[ID]);
    return i < 0 ? null : wrap(c[i + dir] === undefined ? -1 : c[i + dir]);
  }
  function elementSibling(node, dir) { let n = sibling(node, dir); while (n && n.nodeType !== 1) n = sibling(n, dir); return n; }
  function makeNodeList(arr) {
    const list = Object.create(NodeList.prototype);
    for (let i = 0; i < arr.length; i++) list[i] = arr[i];
    Object.defineProperty(list, 'length', { value: arr.length });
    return list;
  }
  class NodeList {
    item(i) { return this[i] || null; }
    forEach(fn, self) { for (let i = 0; i < this.length; i++) fn.call(self, this[i], i, this); }
    *[Symbol.iterator]() { for (let i = 0; i < this.length; i++) yield this[i]; }
    *entries() { for (let i = 0; i < this.length; i++) yield [i, this[i]]; }
    *keys() { for (let i = 0; i < this.length; i++) yield i; }
    *values() { for (let i = 0; i < this.length; i++) yield this[i]; }
  }
  class HTMLCollection {
    item(i) { return this[i] || null; }
    namedItem(name) { for (let i = 0; i < this.length; i++) if (this[i].id === name || this[i].getAttribute('name') === name) return this[i]; return null; }
    *[Symbol.iterator]() { for (let i = 0; i < this.length; i++) yield this[i]; }
  }
  function makeCollection(arr) {
    const list = Object.create(HTMLCollection.prototype);
    for (let i = 0; i < arr.length; i++) list[i] = arr[i];
    Object.defineProperty(list, 'length', { value: arr.length });
    return list;
  }

  function insert(parent, child, ref) {
    if (child == null || child[ID] === undefined) throw new TypeError("Failed to execute 'appendChild' on 'Node': parameter 1 is not of type 'Node'.");
    if (ref != null && ref[ID] === undefined) throw new TypeError("parameter 2 is not of type 'Node'.");
    const pid = parent === document ? H.documentId() : parent[ID];
    if (fragments.has(child)) {
      const kids = child.childNodes;
      for (const k of kids) insert(parent, k, ref);
      return;
    }
    const wasConnected = customDefs.size && child.nodeType === 1 && child.isConnected;
    const oldParent = child.parentNode;
    if (ref) H.insertBefore(pid, child[ID], ref[ID]); else H.append(pid, child[ID]);
    if (oldParent && observers.length) queueRecord({ type: 'childList', target: oldParent, removedNodes: [child], addedNodes: [] });
    queueRecord({ type: 'childList', target: parent, addedNodes: [child], removedNodes: [], nextSibling: ref || null, previousSibling: child.previousSibling });
    if (child.nodeType === 1 && parent.isConnected) {
      if (child.localName === 'script') H.runScript(child[ID]);
      else if (H.children(child[ID]).length) for (const s of H.queryAll(child[ID], 'script')) H.runScript(s);
      if (customDefs.size && !wasConnected) { upgradeTree(child); walkCustom(child, 'connectedCallback'); }
      if (child.localName === 'img' || child.localName === 'iframe') scheduleLoad(child);
    }
  }
  function scheduleLoad(el) { if (!el.getAttribute('src')) return; setTimeout(() => el.dispatchEvent(new Event('load')), 0); }

  class CharacterData extends Node {
    get data() { return H.text(this[ID]); } set data(v) { H.setText(this[ID], String(v)); queueRecord({ type: 'characterData', target: this }); }
    get length() { return this.data.length; }
    get nextElementSibling() { return elementSibling(this, 1); } get previousElementSibling() { return elementSibling(this, -1); }
    appendData(s) { this.data += s; } insertData(o, s) { const d = this.data; this.data = d.slice(0, o) + s + d.slice(o); }
    deleteData(o, n) { const d = this.data; this.data = d.slice(0, o) + d.slice(o + n); } replaceData(o, n, s) { const d = this.data; this.data = d.slice(0, o) + s + d.slice(o + n); }
    substringData(o, n) { return this.data.substr(o, n); }
    remove() { const p = this.parentNode; if (p) p.removeChild(this); }
    before(...n) { const p = this.parentNode; if (p) for (const x of n) p.insertBefore(toNode(x), this); }
    after(...n) { const p = this.parentNode; if (!p) return; let ref = this.nextSibling; for (const x of n) p.insertBefore(toNode(x), ref); }
    replaceWith(...n) { const p = this.parentNode; if (!p) return; this.before(...n); this.remove(); }
  }
  class Text extends CharacterData {
    constructor(data = '') { super(); const id = H.createText(String(data)); this[ID] = id; nodes.set(id, this); }
    get wholeText() { return this.data; }
    splitText(offset) { const d = this.data; this.data = d.slice(0, offset); const t = document.createTextNode(d.slice(offset)); this.after(t); return t; }
    get assignedSlot() { return null; }
  }
  class Comment extends CharacterData { constructor(data = '') { super(); const id = H.createComment(String(data)); this[ID] = id; nodes.set(id, this); } }
  class CDATASection extends Text {}
  class ProcessingInstruction extends CharacterData {}

  // ---------------------------------------------------------------- Attributes, tokens, styles
  class Attr {
    constructor(owner, name) { this.ownerElement = owner; this.name = name; this.localName = name; this.namespaceURI = null; this.prefix = null; this.specified = true; }
    get value() { return this.ownerElement ? this.ownerElement.getAttribute(this.name) : this._v; } set value(v) { if (this.ownerElement) this.ownerElement.setAttribute(this.name, v); else this._v = v; }
    get nodeValue() { return this.value; } get nodeName() { return this.name; } get nodeType() { return 2; } get textContent() { return this.value; }
  }
  class NamedNodeMap {
    item(i) { return this[i] || null; }
    getNamedItem(n) { for (let i = 0; i < this.length; i++) if (this[i].name === n.toLowerCase()) return this[i]; return null; }
    getNamedItemNS(_, n) { return this.getNamedItem(n); }
    setNamedItem(a) { this._el.setAttribute(a.name, a.value); } removeNamedItem(n) { this._el.removeAttribute(n); }
    *[Symbol.iterator]() { for (let i = 0; i < this.length; i++) yield this[i]; }
  }
  class DOMTokenList {
    constructor(el, attr) { this._el = el; this._attr = attr; }
    _get() { const v = this._el.getAttribute(this._attr); return v ? v.split(/\s+/).filter(Boolean) : []; }
    _set(list) { this._el.setAttribute(this._attr, [...new Set(list)].join(' ')); }
    get length() { return this._get().length; } get value() { return this._el.getAttribute(this._attr) || ''; } set value(v) { this._el.setAttribute(this._attr, v); }
    item(i) { return this._get()[i] || null; } contains(t) { return this._get().includes(t); }
    add(...t) { const l = this._get(); for (const x of t) { if (!x || /\s/.test(x)) throw new DOMException('Invalid token', 'InvalidCharacterError'); if (!l.includes(x)) l.push(x); } this._set(l); }
    remove(...t) { this._set(this._get().filter(x => !t.includes(x))); }
    toggle(t, force) { const has = this.contains(t); const want = force === undefined ? !has : !!force; if (want && !has) this.add(t); else if (!want && has) this.remove(t); return want; }
    replace(a, b) { const l = this._get(); const i = l.indexOf(a); if (i < 0) return false; l[i] = b; this._set(l); return true; }
    supports() { return true; }
    forEach(fn, self) { this._get().forEach((v, i) => fn.call(self, v, i, this)); }
    entries() { return this._get().entries(); } keys() { return this._get().keys(); } values() { return this._get().values(); }
    [Symbol.iterator]() { return this._get()[Symbol.iterator](); }
    toString() { return this.value; }
  }
  function styleProxy(el) {
    const id = el[ID];
    const target = {
      getPropertyValue: (p) => H.styleGet(id, p.startsWith('--') ? p : kebab(p)),
      setProperty: (p, v, pri) => { H.styleSet(id, p.startsWith('--') ? p : kebab(p), v == null ? '' : String(v) + (pri === 'important' ? ' !important' : '')); },
      removeProperty: (p) => { const old = H.styleGet(id, p); H.styleRemove(id, p); return old; },
      getPropertyPriority: () => '',
      item: (i) => (H.styleGet(id, '').split(';').map(s => s.split(':')[0].trim()).filter(Boolean))[i] || '',
    };
    return new Proxy(target, {
      get(t, k) {
        if (typeof k === 'symbol') return undefined;
        if (k in t) return t[k];
        if (k === 'cssText') return H.styleGet(id, '');
        if (k === 'length') return H.styleGet(id, '').split(';').filter(s => s.trim()).length;
        if (k === 'parentRule') return null;
        if (/^\d+$/.test(k)) return t.item(+k);
        return H.styleGet(id, kebab(k));
      },
      set(t, k, v) {
        if (k === 'cssText') { el.setAttribute('style', String(v)); return true; }
        H.styleSet(id, String(k).startsWith('--') ? k : kebab(k), v == null ? '' : String(v));
        return true;
      },
      has(t, k) { return typeof k === 'string'; },
    });
  }
  class CSSStyleDeclaration {}
  // Constructed stylesheets are backed by real style elements so replacement, insertion,
  // adoption and removal go through Stylo and affect layout, rather than just feature tests.
  // Shadow DOM currently uses the engine's light-DOM approximation (see docs/browser.md).
  const adoptedSheets = new WeakMap();
  const adoptedNodes = new WeakSet();
  function cssRuleTexts(text) {
    const rules = [];
    let start = 0, depth = 0, parens = 0, quote = '', comment = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i], next = text[i + 1];
      if (comment) { if (c === '*' && next === '/') { comment = false; i++; } continue; }
      if (quote) { if (c === '\\') i++; else if (c === quote) quote = ''; continue; }
      if (c === '/' && next === '*') { comment = true; i++; continue; }
      if (c === '\\') { i++; continue; }
      if (c === '"' || c === "'") { quote = c; continue; }
      if (c === '(' || c === '[') parens++;
      if (c === ')' || c === ']') parens--;
      if (parens) continue;
      if (c === '{') depth++;
      if (c === '}') depth--;
      if (depth < 0) { start = i + 1; depth = 0; continue; }
      if ((!depth && c === '}') || (!depth && c === ';')) {
        const rule = text.slice(start, i + 1).replace(/^(?:\s|\/\*[\s\S]*?\*\/)+/, '').trim();
        if (rule && (rule.includes('{') || rule.startsWith('@'))) rules.push(rule);
        start = i + 1;
      }
    }
    return rules;
  }
  class CSSRule {
    constructor(text, sheet) { this.cssText = text; this.parentStyleSheet = sheet; this.parentRule = null; }
    get type() { return this.cssText.startsWith('@') ? 0 : 1; }
  }
  define(CSSRule, { STYLE_RULE: 1, IMPORT_RULE: 3, MEDIA_RULE: 4, FONT_FACE_RULE: 5, KEYFRAMES_RULE: 7, SUPPORTS_RULE: 12 });
  class StyleSheet {
    get type() { return 'text/css'; }
  }
  class CSSStyleSheet extends StyleSheet {
    constructor(options = {}) {
      super();
      this.ownerNode = null; this.href = null; this.ownerRule = null; this.parentStyleSheet = null; this.title = '';
      this._rules = []; this._copies = new Map(); this._disabled = !!options.disabled;
      const sheet = this;
      this.media = {
        _text: String(options.media || ''),
        get mediaText() { return this._text; },
        set mediaText(v) { this._text = String(v); sheet._sync(); },
        toString() { return this.mediaText; },
      };
      this._ruleList = new Proxy([], {
        get(_, k) {
          if (k === 'item') return i => sheet._rules[i] || null;
          const value = Reflect.get(sheet._rules, k);
          return typeof value === 'function' ? value.bind(sheet._rules) : value;
        },
        set() { return false; },
      });
    }
    get cssRules() { return this._ruleList; }
    get rules() { return this.cssRules; }
    get disabled() { return this._disabled; }
    set disabled(v) { this._disabled = !!v; this._sync(); }
    replaceSync(text) {
      this._rules = cssRuleTexts(String(text)).filter(r => !/^@import\b/i.test(r)).map(r => new CSSRule(r, this));
      this._sync();
    }
    replace(text) { try { this.replaceSync(text); return Promise.resolve(this); } catch (e) { return Promise.reject(e); } }
    insertRule(text, index = 0) {
      index = Number(index) >>> 0;
      if (index > this._rules.length) throw new DOMException('Rule index is out of range', 'IndexSizeError');
      const rules = cssRuleTexts(String(text));
      if (rules.length !== 1) throw new DOMException('Expected one CSS rule', 'SyntaxError');
      if (/^@import\b/i.test(rules[0])) throw new DOMException('Constructed sheets cannot import stylesheets', 'SyntaxError');
      this._rules.splice(index, 0, new CSSRule(rules[0], this));
      this._sync(); return index;
    }
    deleteRule(index) {
      index = Number(index) >>> 0;
      if (index >= this._rules.length) throw new DOMException('Rule index is out of range', 'IndexSizeError');
      this._rules.splice(index, 1); this._sync();
    }
    addRule(selector, style, index = this._rules.length) { this.insertRule(String(selector) + '{' + String(style) + '}', index); return -1; }
    removeRule(index = 0) { this.deleteRule(index); }
    _sync() {
      let text = this._rules.map(r => r.cssText).join('\n');
      // Blitz does not yet restyle a <style> when only its media attribute changes.
      // Put the condition in the stylesheet so Stylo evaluates it on every update.
      if (this.media.mediaText) text = '@media ' + this.media.mediaText + '{' + text + '}';
      if (this.disabled) text = '';
      for (const node of this._copies.values()) {
        node.setAttribute('media', this.disabled ? 'not all' : this.media.mediaText);
        node.textContent = text;
      }
    }
  }
  function adoptedList(root) {
    let state = adoptedSheets.get(root);
    if (state) return state.proxy;
    state = { list: [], active: new Set() };
    const validate = list => {
      for (const sheet of list) if (!(sheet instanceof CSSStyleSheet)) throw new DOMException('Only constructed stylesheets can be adopted', 'NotAllowedError');
    };
    const sync = () => {
      const wanted = new Set(state.list);
      for (const sheet of state.active) if (!wanted.has(sheet)) { sheet._copies.get(root)?.remove(); sheet._copies.delete(root); }
      const parent = root.nodeType === 9 ? root.head || root.documentElement : root;
      for (const sheet of state.list) {
        let node = sheet._copies.get(root);
        if (!node) { node = document.createElement('style'); adoptedNodes.add(node); sheet._copies.set(root, node); }
        parent.appendChild(node); sheet._sync();
      }
      state.active = wanted;
    };
    const mutators = new Set(['push', 'pop', 'shift', 'unshift', 'splice', 'sort', 'reverse', 'copyWithin', 'fill']);
    state.proxy = new Proxy(state.list, {
      get(list, k, receiver) {
        if (mutators.has(k)) return (...args) => {
          const next = list.slice(), result = Array.prototype[k].apply(next, args);
          validate(next); list.splice(0, list.length, ...next); sync();
          return result === next ? state.proxy : result;
        };
        return Reflect.get(list, k, receiver);
      },
      set(list, k, value) {
        const next = list.slice(); Reflect.set(next, k, value); validate(next);
        Reflect.set(list, k, value); sync(); return true;
      },
      deleteProperty() { throw new TypeError('An adopted stylesheet list cannot contain holes'); },
    });
    state.replace = list => { const next = Array.from(list); validate(next); state.list.splice(0, state.list.length, ...next); sync(); };
    adoptedSheets.set(root, state);
    return state.proxy;
  }
  function setAdoptedSheets(root, value) { adoptedList(root); adoptedSheets.get(root).replace(value); }
  function computedProxy(el) {
    const id = el[ID];
    const read = (p) => H.computed(id, p.startsWith('--') ? p : kebab(p));
    return new Proxy({ getPropertyValue: read, getPropertyPriority: () => '' }, {
      get(t, k) { if (typeof k === 'symbol') return undefined; if (k in t) return t[k]; if (k === 'length') return 0; if (k === 'cssText') return ''; return read(k); },
      set() { return true; },
    });
  }

  // ---------------------------------------------------------------- Element
  const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);
  class Element extends Node {
    get tagName() { const [, n, ns] = H.info(this[ID]); return ns === 'html' ? n.toUpperCase() : n; }
    get localName() { return H.info(this[ID])[1]; }
    get namespaceURI() { const ns = H.info(this[ID])[2]; return ns === 'svg' ? 'http://www.w3.org/2000/svg' : ns === 'math' ? 'http://www.w3.org/1998/Math/MathML' : 'http://www.w3.org/1999/xhtml'; }
    get prefix() { return null; }
    get id() { return this.getAttribute('id') || ''; } set id(v) { this.setAttribute('id', v); }
    get className() { return this.getAttribute('class') || ''; } set className(v) { this.setAttribute('class', v); }
    get classList() { return this._cl || (this._cl = new DOMTokenList(this, 'class')); }
    get slot() { return this.getAttribute('slot') || ''; } set slot(v) { this.setAttribute('slot', v); }
    get part() { return new DOMTokenList(this, 'part'); }
    get attributes() {
      const a = H.attrs(this[ID]); const m = Object.create(NamedNodeMap.prototype); m._el = this; let n = 0;
      for (let i = 0; i < a.length; i += 2) m[n++] = new Attr(this, a[i]);
      Object.defineProperty(m, 'length', { value: n });
      return m;
    }
    _name(n) { n = String(n); return H.info(this[ID])[2] === 'html' ? n.toLowerCase() : n; }
    getAttribute(n) { return H.getAttr(this[ID], this._name(n)); }
    getAttributeNS(_, n) { return this.getAttribute(n); }
    getAttributeNode(n) { return this.hasAttribute(n) ? new Attr(this, this._name(n)) : null; }
    getAttributeNames() { const a = H.attrs(this[ID]); const out = []; for (let i = 0; i < a.length; i += 2) out.push(a[i]); return out; }
    hasAttribute(n) { return H.getAttr(this[ID], this._name(n)) !== null; }
    hasAttributeNS(_, n) { return this.hasAttribute(n); }
    hasAttributes() { return H.attrs(this[ID]).length > 0; }
    setAttribute(n, v) {
      n = this._name(n); v = String(v);
      const old = observers.length || this._observed ? H.getAttr(this[ID], n) : null;
      H.setAttr(this[ID], n, v);
      if (this._inline) delete this._inline[n.slice(2)];
      queueRecord({ type: 'attributes', target: this, attributeName: n, oldValue: old });
      attributeChanged(this, n, old, v);
    }
    setAttributeNS(_, n, v) { this.setAttribute(n.includes(':') ? n.split(':')[1] : n, v); }
    setAttributeNode(a) { this.setAttribute(a.name, a.value); }
    removeAttribute(n) {
      n = this._name(n);
      const old = H.getAttr(this[ID], n); if (old === null) return;
      H.removeAttr(this[ID], n);
      queueRecord({ type: 'attributes', target: this, attributeName: n, oldValue: old });
      attributeChanged(this, n, old, null);
    }
    removeAttributeNS(_, n) { this.removeAttribute(n); }
    toggleAttribute(n, force) { const has = this.hasAttribute(n); const want = force === undefined ? !has : !!force; if (want && !has) this.setAttribute(n, ''); if (!want && has) this.removeAttribute(n); return want; }
    get children() { return makeCollection(H.children(this[ID]).map(wrap).filter(n => n && n.nodeType === 1)); }
    get childElementCount() { return this.children.length; }
    get firstElementChild() { return this.children[0] || null; }
    get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
    get nextElementSibling() { return elementSibling(this, 1); }
    get previousElementSibling() { return elementSibling(this, -1); }
    get innerHTML() { return H.innerHTML(this[ID]); }
    set innerHTML(html) {
      const old = observers.length ? this.childNodes : [];
      H.setInnerHTML(this[ID], html == null ? '' : String(html));
      if (customDefs.size) upgradeTree(this);
      queueRecord({ type: 'childList', target: this, removedNodes: old, addedNodes: this.childNodes });
    }
    get outerHTML() { return H.outerHTML(this[ID]); }
    set outerHTML(html) { const p = this.parentNode; if (!p) return; const t = document.createElement('template'); t.innerHTML = html; const kids = [...t.childNodes]; for (const k of kids) p.insertBefore(k, this); this.remove(); }
    get innerText() { return this.textContent; } set innerText(v) { this.textContent = v; }
    get outerText() { return this.textContent; }
    insertAdjacentHTML(pos, html) {
      const t = document.createElement('template'); t.innerHTML = html;
      const kids = [...t.childNodes];
      const p = pos.toLowerCase();
      if (p === 'beforebegin') { for (const k of kids) this.parentNode && this.parentNode.insertBefore(k, this); }
      else if (p === 'afterbegin') { const f = this.firstChild; for (const k of kids) this.insertBefore(k, f); }
      else if (p === 'beforeend') { for (const k of kids) this.appendChild(k); }
      else if (p === 'afterend') { const n = this.nextSibling; for (const k of kids) this.parentNode && this.parentNode.insertBefore(k, n); }
      else throw new DOMException('Invalid position', 'SyntaxError');
    }
    insertAdjacentElement(pos, el) {
      const p = pos.toLowerCase();
      if (p === 'beforebegin') { if (!this.parentNode) return null; this.parentNode.insertBefore(el, this); }
      else if (p === 'afterbegin') this.insertBefore(el, this.firstChild);
      else if (p === 'beforeend') this.appendChild(el);
      else if (p === 'afterend') { if (!this.parentNode) return null; this.parentNode.insertBefore(el, this.nextSibling); }
      return el;
    }
    insertAdjacentText(pos, text) { this.insertAdjacentElement(pos, document.createTextNode(text)); }
    append(...n) { for (const x of n) this.appendChild(toNode(x)); }
    prepend(...n) { const f = this.firstChild; for (const x of n) this.insertBefore(toNode(x), f); }
    before(...n) { const p = this.parentNode; if (p) for (const x of n) p.insertBefore(toNode(x), this); }
    after(...n) { const p = this.parentNode; if (!p) return; const ref = this.nextSibling; for (const x of n) p.insertBefore(toNode(x), ref); }
    replaceWith(...n) { const p = this.parentNode; if (!p) return; const ref = this.nextSibling; this.remove(); for (const x of n) p.insertBefore(toNode(x), ref); }
    replaceChildren(...n) { for (const c of this.childNodes) this.removeChild(c); this.append(...n); }
    remove() { const p = this.parentNode; if (p) p.removeChild(this); }
    querySelector(s) { return wrap(H.query(this[ID], String(s))); }
    querySelectorAll(s) { return makeNodeList(H.queryAll(this[ID], String(s)).map(wrap)); }
    getElementsByTagName(t) { return makeCollection(t === '*' ? H.queryAll(this[ID], '*').map(wrap) : H.queryAll(this[ID], cssName(t)).map(wrap)); }
    getElementsByTagNameNS(_, t) { return this.getElementsByTagName(t); }
    getElementsByClassName(c) { const sel = String(c).trim().split(/\s+/).filter(Boolean).map(x => '.' + cssEscape(x)).join(''); return makeCollection(sel ? H.queryAll(this[ID], sel).map(wrap) : []); }
    matches(s) { return H.matches(this[ID], String(s)); }
    webkitMatchesSelector(s) { return this.matches(s); } msMatchesSelector(s) { return this.matches(s); }
    closest(s) { let n = this; while (n && n.nodeType === 1) { if (n.matches(s)) return n; n = n.parentElement; } return null; }
    getBoundingClientRect() { const r = H.rect(this[ID]); return new DOMRect(r[0], r[1], r[2], r[3]); }
    getClientRects() { const r = this.getBoundingClientRect(); const l = r.width || r.height ? [r] : []; l.item = i => l[i] || null; return l; }
    _box() { return H.box(this[ID]); }
    get clientWidth() { return this === document.documentElement ? G.innerWidth : this._box()[5]; }
    get clientHeight() { return this === document.documentElement ? G.innerHeight : this._box()[6]; }
    get clientTop() { return 0; } get clientLeft() { return 0; }
    get scrollWidth() { return this._box()[7]; } get scrollHeight() { return this._box()[8]; }
    get scrollTop() { return this === document.documentElement || this === document.body ? G.scrollY : this._box()[9]; }
    set scrollTop(v) { if (this === document.documentElement || this === document.body) G.scrollTo(G.scrollX, v); else H.setScroll(this[ID], +v, NaN); }
    get scrollLeft() { return this === document.documentElement || this === document.body ? G.scrollX : this._box()[10]; }
    set scrollLeft(v) { if (this === document.documentElement || this === document.body) G.scrollTo(v, G.scrollY); else H.setScroll(this[ID], NaN, +v); }
    scroll(x, y) { if (typeof x === 'object' && x) { if (x.top !== undefined) this.scrollTop = x.top; if (x.left !== undefined) this.scrollLeft = x.left; } else { this.scrollLeft = x; this.scrollTop = y; } }
    scrollTo(x, y) { this.scroll(x, y); }
    scrollBy(x, y) { if (typeof x === 'object' && x) { this.scrollTop += x.top || 0; this.scrollLeft += x.left || 0; } else { this.scrollLeft += x || 0; this.scrollTop += y || 0; } }
    scrollIntoView(opts) { H.scrollIntoView(this[ID], typeof opts === 'object' && opts && opts.block || (opts === false ? 'end' : 'start')); }
    scrollIntoViewIfNeeded() { H.scrollIntoView(this[ID], 'nearest'); }
    attachShadow(init) {
      if (this._shadow) throw new DOMException('Shadow root already attached', 'NotSupportedError');
      const host = document.createElement('evolve-shadow-root'); host.setAttribute('style', 'display: contents');
      this.insertBefore(host, this.firstChild);
      const root = Object.create(ShadowRoot.prototype); root[ID] = host[ID]; root._host = this; root.mode = init && init.mode || 'open';
      fragments.add(host);
      this._shadow = root; return root;
    }
    get shadowRoot() { return this._shadow && this._shadow.mode === 'open' ? this._shadow : null; }
    get assignedSlot() { return null; }
    animate(keyframes, opts) { return new Animation(); }
    getAnimations() { return []; }
    requestFullscreen() { return Promise.reject(new DOMException('Fullscreen is not supported', 'NotSupportedError')); }
    requestPointerLock() {}
    setPointerCapture() {} releasePointerCapture() {} hasPointerCapture() { return false; }
    checkVisibility() { const r = this.getBoundingClientRect(); return r.width > 0 || r.height > 0; }
    computedStyleMap() { return new Map(); }
  }
  const cssEscape = (s) => String(s).replace(/([^a-zA-Z0-9_\u00A0-\uFFFF-])/g, '\\$1').replace(/^(\d)/, '\\3$1 ');
  const cssName = (t) => /^[a-zA-Z][a-zA-Z0-9-]*$/.test(t) ? t : cssEscape(t);
  class ShadowRoot extends Node {
    get host() { return this._host; } get innerHTML() { return H.innerHTML(this[ID]); } set innerHTML(v) { H.setInnerHTML(this[ID], String(v)); if (customDefs.size) upgradeTree(this); }
    get nodeType() { return 11; } get nodeName() { return '#document-fragment'; }
    querySelector(s) { return wrap(H.query(this[ID], s)); } querySelectorAll(s) { return makeNodeList(H.queryAll(this[ID], s).map(wrap)); }
    getElementById(i) { return this.querySelector('#' + cssEscape(i)); }
    get children() { return makeCollection(H.children(this[ID]).map(wrap).filter(n => n.nodeType === 1)); }
    append(...n) { for (const x of n) this.appendChild(toNode(x)); }
    get activeElement() { return document.activeElement; } get adoptedStyleSheets() { return adoptedList(this); } set adoptedStyleSheets(v) { setAdoptedSheets(this, v); }
  }

  // reflected attributes
  function reflect(cls, props) {
    for (const p of props) {
      const [prop, attr, kind] = Array.isArray(p) ? p : [p, p.toLowerCase(), 'string'];
      Object.defineProperty(cls.prototype, prop, {
        configurable: true, enumerable: true,
        get() {
          const v = this.getAttribute(attr);
          if (kind === 'bool') return v !== null;
          if (kind === 'url') return v === null ? '' : resolveUrl(v);
          if (kind === 'int') { const n = parseInt(v, 10); return isNaN(n) ? (p[3] !== undefined ? p[3] : 0) : n; }
          return v === null ? (p[3] !== undefined ? p[3] : '') : v;
        },
        set(v) { if (kind === 'bool') { if (v) this.setAttribute(attr, ''); else this.removeAttribute(attr); } else this.setAttribute(attr, String(v)); },
      });
    }
  }
  const resolveUrl = (v) => { try { return new URL(v, document.baseURI).href; } catch (e) { return v; } };

  class HTMLElement extends Element {
    constructor() {
      super();
      if (upgrading >= 0) { this[ID] = upgrading; nodes.set(upgrading, this); upgrading = -1; return; }
      const def = defsByClass.get(new.target);
      if (!def) throw new TypeError('Illegal constructor');
      const id = H.create(def.name, '');
      this[ID] = id; nodes.set(id, this);
    }
    get style() { return this._style || (this._style = styleProxy(this)); }
    set style(v) { this.setAttribute('style', String(v)); }
    get dataset() {
      const el = this;
      return this._ds || (this._ds = new Proxy({}, {
        get(_, k) { if (typeof k !== 'string') return undefined; const v = el.getAttribute('data-' + kebab(k)); return v === null ? undefined : v; },
        set(_, k, v) { el.setAttribute('data-' + kebab(k), v); return true; },
        deleteProperty(_, k) { el.removeAttribute('data-' + kebab(k)); return true; },
        has(_, k) { return el.hasAttribute('data-' + kebab(k)); },
        ownKeys() { return el.getAttributeNames().filter(n => n.startsWith('data-')).map(n => camel(n.slice(5))); },
        getOwnPropertyDescriptor(_, k) { const v = el.getAttribute('data-' + kebab(k)); return v === null ? undefined : { value: v, enumerable: true, configurable: true, writable: true }; },
      }));
    }
    get offsetParent() { const p = this._box()[4]; return p < 0 ? null : wrap(p); }
    get offsetTop() { return this._box()[0]; } get offsetLeft() { return this._box()[1]; }
    get offsetWidth() { return this._box()[2]; } get offsetHeight() { return this._box()[3]; }
    get isContentEditable() { return false; }
    get contentEditable() { return this.getAttribute('contenteditable') || 'inherit'; } set contentEditable(v) { this.setAttribute('contenteditable', v); }
    get tabIndex() { const v = parseInt(this.getAttribute('tabindex'), 10); return isNaN(v) ? (/^(a|button|input|select|textarea)$/.test(this.localName) ? 0 : -1) : v; }
    set tabIndex(v) { this.setAttribute('tabindex', v); }
    get inert() { return this.hasAttribute('inert'); } set inert(v) { this.toggleAttribute('inert', !!v); }
    focus() { if (H.focus(this[ID])) { this.dispatchEvent(new FocusEvent('focus')); this.dispatchEvent(new FocusEvent('focusin', { bubbles: true })); } }
    blur() { if (document.activeElement === this) { H.blur(this[ID]); this.dispatchEvent(new FocusEvent('blur')); this.dispatchEvent(new FocusEvent('focusout', { bubbles: true })); } }
    click() {
      if (this.hasAttribute('disabled') && /^(button|input|select|textarea)$/.test(this.localName)) return;
      const ok = this.dispatchEvent(new PointerEvent('click', { bubbles: true, cancelable: true, composed: true, view: G, detail: 1 }));
      if (ok) defaultActivation(this);
    }
    get accessKey() { return this.getAttribute('accesskey') || ''; }
    get autofocus() { return this.hasAttribute('autofocus'); }
    showPopover() {} hidePopover() {} togglePopover() {}
    attachInternals() { return { setFormValue() {}, setValidity() {}, checkValidity: () => true, reportValidity: () => true, states: new Set(), form: null, labels: [] }; }
  }
  reflect(HTMLElement, ['title', 'lang', 'dir', ['hidden', 'hidden', 'bool'], ['draggable', 'draggable', 'string'], ['spellcheck', 'spellcheck', 'string'], ['translate', 'translate', 'string'], ['nonce', 'nonce', 'string'], ['popover', 'popover', 'string'], ['enterKeyHint', 'enterkeyhint', 'string'], ['inputMode', 'inputmode', 'string']]);
  function defaultActivation(el) {
    // Submit buttons and links: the form submit event is dispatched here so scripts can cancel it.
    const sub = el.closest && el.closest('button, input[type=submit], input[type=image]');
    if (sub && sub.form && (sub.localName === 'input' || (sub.getAttribute('type') || 'submit').toLowerCase() === 'submit')) {
      if (!sub.form.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true, submitter: sub }))) return;
    }
    H.defaultClick(el[ID]);
  }

  class SVGElement extends Element {
    get style() { return this._style || (this._style = styleProxy(this)); }
    get dataset() { return HTMLElement.prototype.__lookupGetter__ ? Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'dataset').get.call(this) : {}; }
    get ownerSVGElement() { let p = this.parentElement; while (p && p.localName !== 'svg') p = p.parentElement; return p; }
    get viewportElement() { return this.ownerSVGElement; }
    getBBox() { const r = this.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; }
    get className() { const v = this.getAttribute('class') || ''; return { baseVal: v, animVal: v }; }
    set className(v) { this.setAttribute('class', v); }
    focus() { H.focus(this[ID]); } blur() { H.blur(this[ID]); }
  }
  class SVGSVGElement extends SVGElement { createSVGPoint() { return { x: 0, y: 0, matrixTransform() { return this; } }; } getScreenCTM() { return null; } }

  // HTML element classes
  const ELEMENT_CLASSES = {
    HTMLAnchorElement: ['a'], HTMLAreaElement: ['area'], HTMLAudioElement: ['audio'], HTMLBRElement: ['br'], HTMLBaseElement: ['base'], HTMLBodyElement: ['body'],
    HTMLButtonElement: ['button'], HTMLCanvasElement: ['canvas'], HTMLDListElement: ['dl'], HTMLDataElement: ['data'], HTMLDataListElement: ['datalist'],
    HTMLDetailsElement: ['details'], HTMLDialogElement: ['dialog'], HTMLDivElement: ['div'], HTMLEmbedElement: ['embed'], HTMLFieldSetElement: ['fieldset'],
    HTMLFormElement: ['form'], HTMLHRElement: ['hr'], HTMLHeadElement: ['head'], HTMLHeadingElement: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], HTMLHtmlElement: ['html'],
    HTMLIFrameElement: ['iframe'], HTMLImageElement: ['img'], HTMLInputElement: ['input'], HTMLLIElement: ['li'], HTMLLabelElement: ['label'], HTMLLegendElement: ['legend'],
    HTMLLinkElement: ['link'], HTMLMapElement: ['map'], HTMLMenuElement: ['menu'], HTMLMetaElement: ['meta'], HTMLMeterElement: ['meter'], HTMLModElement: ['ins', 'del'],
    HTMLOListElement: ['ol'], HTMLObjectElement: ['object'], HTMLOptGroupElement: ['optgroup'], HTMLOptionElement: ['option'], HTMLOutputElement: ['output'],
    HTMLParagraphElement: ['p'], HTMLPictureElement: ['picture'], HTMLPreElement: ['pre', 'listing', 'xmp'], HTMLProgressElement: ['progress'], HTMLQuoteElement: ['blockquote', 'q'],
    HTMLScriptElement: ['script'], HTMLSelectElement: ['select'], HTMLSlotElement: ['slot'], HTMLSourceElement: ['source'], HTMLSpanElement: ['span'], HTMLStyleElement: ['style'],
    HTMLTableCaptionElement: ['caption'], HTMLTableCellElement: ['td', 'th'], HTMLTableColElement: ['col', 'colgroup'], HTMLTableElement: ['table'], HTMLTableRowElement: ['tr'],
    HTMLTableSectionElement: ['thead', 'tbody', 'tfoot'], HTMLTemplateElement: ['template'], HTMLTextAreaElement: ['textarea'], HTMLTimeElement: ['time'], HTMLTitleElement: ['title'],
    HTMLTrackElement: ['track'], HTMLUListElement: ['ul'], HTMLUnknownElement: [], HTMLVideoElement: ['video'], HTMLMediaElement: [], HTMLPictureSourceElement: [],
  };
  const C = {};
  for (const name of Object.keys(ELEMENT_CLASSES)) {
    const base = name === 'HTMLAudioElement' || name === 'HTMLVideoElement' ? null : HTMLElement;
    C[name] = base ? ({ [name]: class extends HTMLElement {} })[name] : null;
  }
  C.HTMLMediaElement = class HTMLMediaElement extends HTMLElement {};
  C.HTMLAudioElement = class HTMLAudioElement extends C.HTMLMediaElement {};
  C.HTMLVideoElement = class HTMLVideoElement extends C.HTMLMediaElement {};
  for (const name of Object.keys(ELEMENT_CLASSES)) for (const tag of ELEMENT_CLASSES[name]) TAG_CLASS[tag] = C[name];
  const HTMLUnknownElement = C.HTMLUnknownElement;

  const formOf = (el) => { const f = el.getAttribute('form'); if (f) return document.getElementById(f); return el.closest('form'); };
  const validity = () => ({ valid: true, valueMissing: false, typeMismatch: false, patternMismatch: false, tooLong: false, tooShort: false, rangeUnderflow: false, rangeOverflow: false, stepMismatch: false, badInput: false, customError: false });
  const controlBits = {
    get form() { return formOf(this); },
    get validity() { return validity(); }, get validationMessage() { return ''; }, get willValidate() { return true; },
    checkValidity() { return true; }, reportValidity() { return true; }, setCustomValidity() {},
    get labels() { const id = this.id; return makeNodeList([...document.querySelectorAll('label')].filter(l => (id && l.htmlFor === id) || l.contains(this))); },
  };
  const mix = (cls, obj) => { for (const k of Object.getOwnPropertyNames(obj)) Object.defineProperty(cls.prototype, k, Object.getOwnPropertyDescriptor(obj, k)); };

  // a / area
  for (const cls of [C.HTMLAnchorElement, C.HTMLAreaElement]) {
    reflect(cls, [['href', 'href', 'url'], 'target', 'rel', 'download', 'hreflang', 'type', 'ping', ['referrerPolicy', 'referrerpolicy', 'string']]);
    for (const part of ['protocol', 'host', 'hostname', 'port', 'pathname', 'search', 'hash', 'origin', 'username', 'password']) {
      Object.defineProperty(cls.prototype, part, { configurable: true, get() { try { return new URL(this.href)[part]; } catch (e) { return ''; } }, set(v) { try { const u = new URL(this.href); u[part] = v; this.href = u.href; } catch (e) {} } });
    }
    Object.defineProperty(cls.prototype, 'relList', { get() { return new DOMTokenList(this, 'rel'); } });
    Object.defineProperty(cls.prototype, 'text', { get() { return this.textContent; }, set(v) { this.textContent = v; } });
    cls.prototype.toString = function () { return this.href; };
  }
  // img
  reflect(C.HTMLImageElement, [['src', 'src', 'url'], 'alt', 'srcset', 'sizes', ['crossOrigin', 'crossorigin', 'string'], ['useMap', 'usemap', 'string'], ['isMap', 'ismap', 'bool'], 'loading', 'decoding', ['referrerPolicy', 'referrerpolicy', 'string'], ['fetchPriority', 'fetchpriority', 'string']]);
  getters(C.HTMLImageElement.prototype, {
    width: { get() { return parseInt(this.getAttribute('width'), 10) || Math.round(this.getBoundingClientRect().width); }, set(v) { this.setAttribute('width', v); } },
    height: { get() { return parseInt(this.getAttribute('height'), 10) || Math.round(this.getBoundingClientRect().height); }, set(v) { this.setAttribute('height', v); } },
    naturalWidth() { return Math.round(this.getBoundingClientRect().width); }, naturalHeight() { return Math.round(this.getBoundingClientRect().height); },
    complete() { return true; }, currentSrc() { return this.src; },
  });
  C.HTMLImageElement.prototype.decode = function () { return Promise.resolve(); };
  // script
  reflect(C.HTMLScriptElement, [['src', 'src', 'url'], 'type', 'charset', ['async', 'async', 'bool'], ['defer', 'defer', 'bool'], ['noModule', 'nomodule', 'bool'], ['crossOrigin', 'crossorigin', 'string'], 'integrity', ['referrerPolicy', 'referrerpolicy', 'string'], 'nonce', 'event', ['htmlFor', 'for', 'string']]);
  getters(C.HTMLScriptElement.prototype, { text: { get() { return this.textContent; }, set(v) { this.textContent = v; } } });
  getters(C.HTMLScriptElement.prototype, {
    async: {
      get() { return this.hasAttribute('async') || H.scriptAsync(this[ID]); },
      set(v) { H.scriptAsync(this[ID], false); this.toggleAttribute('async', !!v); },
    },
  });
  // link, style, meta, base, iframe, canvas, media, source, track
  reflect(C.HTMLLinkElement, [['href', 'href', 'url'], 'rel', 'as', 'media', 'type', 'sizes', 'hreflang', ['crossOrigin', 'crossorigin', 'string'], 'integrity', ['disabled', 'disabled', 'bool'], ['imageSrcset', 'imagesrcset', 'string']]);
  getters(C.HTMLLinkElement.prototype, { relList() { return new DOMTokenList(this, 'rel'); }, sheet() { return null; } });
  reflect(C.HTMLStyleElement, ['media', 'type', ['disabled', 'disabled', 'bool']]);
  getters(C.HTMLStyleElement.prototype, { sheet() { const el = this; return { cssRules: [], rules: [], insertRule(rule, i) { el.textContent += '\n' + rule; return i || 0; }, deleteRule() {}, addRule(s, b) { el.textContent += '\n' + s + '{' + b + '}'; } }; } });
  reflect(C.HTMLMetaElement, ['name', 'content', ['httpEquiv', 'http-equiv', 'string'], 'charset', 'media']);
  reflect(C.HTMLBaseElement, [['href', 'href', 'url'], 'target']);
  reflect(C.HTMLIFrameElement, [['src', 'src', 'url'], 'srcdoc', 'name', 'allow', ['allowFullscreen', 'allowfullscreen', 'bool'], 'width', 'height', 'loading', ['referrerPolicy', 'referrerpolicy', 'string']]);
  // contentWindow is the frame's realm window (shared heap, so identity holds for event.source);
  // contentDocument stays null (cross-origin frames expose no document here).
  getters(C.HTMLIFrameElement.prototype, { contentWindow() { return H.frameWindow(this[ID]) || null; }, contentDocument() { return null; }, sandbox() { return new DOMTokenList(this, 'sandbox'); } });
  reflect(C.HTMLCanvasElement, [['width', 'width', 'int', 300], ['height', 'height', 'int', 150]]);
  Object.assign(C.HTMLCanvasElement.prototype, { getContext() { return null; }, toDataURL() { return 'data:,'; }, toBlob(cb) { setTimeout(() => cb(null), 0); }, captureStream() { return null; }, transferControlToOffscreen() { throw new DOMException('Not supported', 'NotSupportedError'); } });
  reflect(C.HTMLMediaElement, [['src', 'src', 'url'], ['autoplay', 'autoplay', 'bool'], ['loop', 'loop', 'bool'], ['controls', 'controls', 'bool'], ['muted', 'muted', 'bool'], 'preload', ['crossOrigin', 'crossorigin', 'string'], ['playsInline', 'playsinline', 'bool'], 'poster']);
  Object.assign(C.HTMLMediaElement.prototype, { play() { return Promise.reject(new DOMException('Media playback is not supported', 'NotSupportedError')); }, pause() {}, load() {}, canPlayType() { return ''; }, addTextTrack() { return {}; } });
  getters(C.HTMLMediaElement.prototype, { paused() { return true; }, ended() { return false; }, duration() { return NaN; }, readyState() { return 0; }, networkState() { return 3; }, buffered() { return { length: 0, start() { return 0; }, end() { return 0; } }; }, currentSrc() { return this.src; }, textTracks() { return []; }, currentTime: { get() { return 0; }, set(v) {} }, volume: { get() { return 1; }, set(v) {} }, playbackRate: { get() { return 1; }, set(v) {} } });
  reflect(C.HTMLSourceElement, [['src', 'src', 'url'], 'type', 'srcset', 'sizes', 'media']);
  reflect(C.HTMLTrackElement, [['src', 'src', 'url'], 'kind', 'srclang', 'label', ['default', 'default', 'bool']]);
  reflect(C.HTMLLabelElement, [['htmlFor', 'for', 'string']]);
  getters(C.HTMLLabelElement.prototype, { control() { const f = this.htmlFor; return f ? document.getElementById(f) : this.querySelector('input, select, textarea, button'); }, form() { const c = this.control; return c ? c.form : null; } });
  reflect(C.HTMLOListElement, [['start', 'start', 'int', 1], ['reversed', 'reversed', 'bool'], 'type']);
  reflect(C.HTMLLIElement, [['value', 'value', 'int']]);
  reflect(C.HTMLTableCellElement, [['colSpan', 'colspan', 'int', 1], ['rowSpan', 'rowspan', 'int', 1], 'headers', 'scope', 'abbr']);
  reflect(C.HTMLTimeElement, [['dateTime', 'datetime', 'string']]);
  reflect(C.HTMLDataElement, ['value']);
  reflect(C.HTMLQuoteElement, [['cite', 'cite', 'url']]);
  reflect(C.HTMLProgressElement, [['max', 'max', 'int', 1]]);
  reflect(C.HTMLDetailsElement, [['open', 'open', 'bool'], 'name']);
  reflect(C.HTMLDialogElement, [['open', 'open', 'bool']]);
  Object.assign(C.HTMLDialogElement.prototype, {
    show() { this.setAttribute('open', ''); }, showModal() { this.setAttribute('open', ''); },
    close(rv) { if (!this.hasAttribute('open')) return; this.removeAttribute('open'); if (rv !== undefined) this.returnValue = rv; this.dispatchEvent(new Event('close')); },
    requestClose(rv) { this.close(rv); },
  });
  getters(C.HTMLTableElement.prototype, {
    rows() { return makeCollection([...this.querySelectorAll('tr')]); }, tBodies() { return makeCollection([...this.children].filter(c => c.localName === 'tbody')); },
    tHead() { return [...this.children].find(c => c.localName === 'thead') || null; }, tFoot() { return [...this.children].find(c => c.localName === 'tfoot') || null; },
    caption() { return [...this.children].find(c => c.localName === 'caption') || null; },
  });
  getters(C.HTMLTableRowElement.prototype, { cells() { return makeCollection([...this.children].filter(c => c.localName === 'td' || c.localName === 'th')); }, rowIndex() { const t = this.closest('table'); return t ? [...t.querySelectorAll('tr')].indexOf(this) : -1; } });
  getters(C.HTMLTableSectionElement.prototype, { rows() { return makeCollection([...this.children].filter(c => c.localName === 'tr')); } });
  C.HTMLTableElement.prototype.insertRow = function (i = -1) { const tr = document.createElement('tr'); const body = this.tBodies[0] || this; const rows = body.querySelectorAll('tr'); if (i < 0 || i >= rows.length) body.appendChild(tr); else body.insertBefore(tr, rows[i]); return tr; };
  C.HTMLTableRowElement.prototype.insertCell = function (i = -1) { const td = document.createElement('td'); const cells = this.cells; if (i < 0 || i >= cells.length) this.appendChild(td); else this.insertBefore(td, cells[i]); return td; };
  getters(C.HTMLTemplateElement.prototype, {
    content() {
      if (this._content) return this._content;
      const f = document.createDocumentFragment();
      for (const c of [...this.childNodes]) f.appendChild(c);
      this._content = f; return f;
    },
  });
  Object.defineProperty(C.HTMLTemplateElement.prototype, 'innerHTML', {
    get() { return this._content ? H.innerHTML(this._content[ID]) : H.innerHTML(this[ID]); },
    set(v) { if (this._content) H.setInnerHTML(this._content[ID], String(v)); else H.setInnerHTML(this[ID], String(v)); },
  });

  // form controls
  const INPUT_TYPES = new Set(['hidden', 'text', 'search', 'tel', 'url', 'email', 'password', 'date', 'month', 'week', 'time', 'datetime-local', 'number', 'range', 'color', 'checkbox', 'radio', 'file', 'submit', 'image', 'reset', 'button']);
  mix(C.HTMLInputElement, controlBits);
  reflect(C.HTMLInputElement, ['name', 'placeholder', 'accept', 'alt', 'autocomplete', 'min', 'max', 'step', 'pattern', ['src', 'src', 'url'], ['multiple', 'multiple', 'bool'], ['required', 'required', 'bool'], ['readOnly', 'readonly', 'bool'], ['disabled', 'disabled', 'bool'], ['autofocus', 'autofocus', 'bool'], ['maxLength', 'maxlength', 'int', -1], ['minLength', 'minlength', 'int', -1], ['size', 'size', 'int', 20], ['defaultValue', 'value', 'string'], ['defaultChecked', 'checked', 'bool'], ['formAction', 'formaction', 'url'], ['formMethod', 'formmethod', 'string'], ['formNoValidate', 'formnovalidate', 'bool'], ['dirName', 'dirname', 'string'], ['inputMode', 'inputmode', 'string']]);
  getters(C.HTMLInputElement.prototype, {
    type: { get() { const t = (this.getAttribute('type') || 'text').toLowerCase(); return INPUT_TYPES.has(t) ? t : 'text'; }, set(v) { this.setAttribute('type', v); } },
    value: {
      get() { const t = this.type; if (t === 'checkbox' || t === 'radio') { const v = this.getAttribute('value'); return v === null ? 'on' : v; } const v = H.value(this[ID]); return v == null ? '' : v; },
      set(v) { const t = this.type; if (t === 'checkbox' || t === 'radio' || t === 'hidden' || t === 'submit' || t === 'button' || t === 'reset' || t === 'image') this.setAttribute('value', v); else H.setValue(this[ID], v == null ? '' : String(v)); },
    },
    checked: { get() { return H.checked(this[ID]); }, set(v) { H.setChecked(this[ID], !!v); } },
    indeterminate: { get() { return !!this._ind; }, set(v) { this._ind = !!v; } },
    valueAsNumber: { get() { const n = parseFloat(this.value); return isNaN(n) ? NaN : n; }, set(v) { this.value = String(v); } },
    valueAsDate: { get() { const d = new Date(this.value); return isNaN(d) ? null : d; }, set(v) { this.value = v ? v.toISOString().slice(0, 10) : ''; } },
    selectionStart: { get() { return this.value.length; }, set(v) {} }, selectionEnd: { get() { return this.value.length; }, set(v) {} }, selectionDirection: { get() { return 'none'; }, set(v) {} },
    files() { return this.type === 'file' ? Object.assign([], { item() { return null; } }) : null; },
    list() { const l = this.getAttribute('list'); return l ? document.getElementById(l) : null; },
  });
  Object.assign(C.HTMLInputElement.prototype, { select() {}, setSelectionRange() {}, setRangeText() {}, showPicker() {}, stepUp(n = 1) { this.value = String((parseFloat(this.value) || 0) + n * (parseFloat(this.step) || 1)); }, stepDown(n = 1) { this.stepUp(-n); } });
  mix(C.HTMLTextAreaElement, controlBits);
  reflect(C.HTMLTextAreaElement, ['name', 'placeholder', 'autocomplete', 'wrap', ['rows', 'rows', 'int', 2], ['cols', 'cols', 'int', 20], ['required', 'required', 'bool'], ['readOnly', 'readonly', 'bool'], ['disabled', 'disabled', 'bool'], ['maxLength', 'maxlength', 'int', -1], ['minLength', 'minlength', 'int', -1], ['autofocus', 'autofocus', 'bool']]);
  getters(C.HTMLTextAreaElement.prototype, {
    type() { return 'textarea'; },
    value: { get() { const v = H.value(this[ID]); return v == null ? this.textContent : v; }, set(v) { H.setValue(this[ID], v == null ? '' : String(v)); } },
    defaultValue: { get() { return this.textContent; }, set(v) { this.textContent = v; } },
    textLength() { return this.value.length; }, selectionStart: { get() { return this.value.length; }, set(v) {} }, selectionEnd: { get() { return this.value.length; }, set(v) {} },
  });
  Object.assign(C.HTMLTextAreaElement.prototype, { select() {}, setSelectionRange() {}, setRangeText() {} });
  mix(C.HTMLButtonElement, controlBits);
  reflect(C.HTMLButtonElement, ['name', 'value', ['disabled', 'disabled', 'bool'], ['formAction', 'formaction', 'url'], ['formMethod', 'formmethod', 'string'], ['formNoValidate', 'formnovalidate', 'bool'], ['formTarget', 'formtarget', 'string'], ['autofocus', 'autofocus', 'bool'], ['popoverTargetElement', 'popovertarget', 'string']]);
  getters(C.HTMLButtonElement.prototype, { type: { get() { const t = (this.getAttribute('type') || 'submit').toLowerCase(); return ['submit', 'reset', 'button'].includes(t) ? t : 'submit'; }, set(v) { this.setAttribute('type', v); } } });
  mix(C.HTMLSelectElement, controlBits);
  reflect(C.HTMLSelectElement, ['name', ['multiple', 'multiple', 'bool'], ['required', 'required', 'bool'], ['disabled', 'disabled', 'bool'], ['size', 'size', 'int', 0], ['autofocus', 'autofocus', 'bool']]);
  getters(C.HTMLSelectElement.prototype, {
    type() { return this.multiple ? 'select-multiple' : 'select-one'; },
    options() { const o = [...this.querySelectorAll('option')]; const c = makeCollection(o); c.add = (opt, before) => this.add(opt, before); c.remove = (i) => this.remove(i); Object.defineProperty(c, 'selectedIndex', { get: () => this.selectedIndex, set: (v) => { this.selectedIndex = v; } }); return c; },
    length() { return this.querySelectorAll('option').length; },
    selectedOptions() { return makeCollection([...this.querySelectorAll('option')].filter(o => o.selected)); },
    selectedIndex: {
      get() { const o = [...this.querySelectorAll('option')]; const i = o.findIndex(x => x.hasAttribute('selected')); return i >= 0 ? i : (o.length && !this.multiple ? 0 : -1); },
      set(i) { const o = [...this.querySelectorAll('option')]; o.forEach((x, j) => { if (j === +i) x.setAttribute('selected', ''); else x.removeAttribute('selected'); }); },
    },
    value: {
      get() { const o = this.options[this.selectedIndex]; return o ? o.value : ''; },
      set(v) { const o = [...this.querySelectorAll('option')]; const i = o.findIndex(x => x.value === String(v)); this.selectedIndex = i; },
    },
  });
  Object.assign(C.HTMLSelectElement.prototype, {
    item(i) { return this.options[i] || null; }, namedItem(n) { return this.querySelector('option[name="' + n + '"], option#' + cssEscape(n)); },
    add(opt, before) { if (before == null) this.appendChild(opt); else this.insertBefore(opt, typeof before === 'number' ? this.options[before] : before); },
    remove(i) { if (i === undefined) { Element.prototype.remove.call(this); return; } const o = this.options[i]; if (o) o.remove(); },
    showPicker() {},
  });
  getters(C.HTMLOptionElement.prototype, {
    value: { get() { const v = this.getAttribute('value'); return v === null ? this.text : v; }, set(v) { this.setAttribute('value', v); } },
    text: { get() { return this.textContent.replace(/\s+/g, ' ').trim(); }, set(v) { this.textContent = v; } },
    label: { get() { return this.getAttribute('label') || this.text; }, set(v) { this.setAttribute('label', v); } },
    selected: { get() { if (this.hasAttribute('selected')) return true; const s = this.closest('select'); return !!s && !s.multiple && s.selectedIndex === this.index; }, set(v) { const s = this.closest('select'); if (v && s && !s.multiple) for (const o of s.querySelectorAll('option')) o.removeAttribute('selected'); this.toggleAttribute('selected', !!v); } },
    defaultSelected: { get() { return this.hasAttribute('selected'); }, set(v) { this.toggleAttribute('selected', !!v); } },
    disabled: { get() { return this.hasAttribute('disabled'); }, set(v) { this.toggleAttribute('disabled', !!v); } },
    index() { const s = this.closest('select'); return s ? [...s.querySelectorAll('option')].indexOf(this) : 0; },
    form() { const s = this.closest('select'); return s ? s.form : null; },
  });
  mix(C.HTMLFieldSetElement, controlBits);
  reflect(C.HTMLFieldSetElement, ['name', ['disabled', 'disabled', 'bool']]);
  getters(C.HTMLFieldSetElement.prototype, { elements() { return makeCollection([...this.querySelectorAll('input, select, textarea, button')]); }, type() { return 'fieldset'; } });
  mix(C.HTMLOutputElement, controlBits);
  getters(C.HTMLOutputElement.prototype, { value: { get() { return this.textContent; }, set(v) { this.textContent = v; } } });

  reflect(C.HTMLFormElement, [['action', 'action', 'url'], 'method', 'enctype', 'target', 'name', ['acceptCharset', 'accept-charset', 'string'], 'autocomplete', ['noValidate', 'novalidate', 'bool'], 'rel']);
  getters(C.HTMLFormElement.prototype, {
    action: { get() { const a = this.getAttribute('action'); return a ? resolveUrl(a) : document.URL; }, set(v) { this.setAttribute('action', v); } },
    method: { get() { const m = (this.getAttribute('method') || 'get').toLowerCase(); return m === 'post' || m === 'dialog' ? m : 'get'; }, set(v) { this.setAttribute('method', v); } },
    encoding: { get() { return this.enctype; }, set(v) { this.enctype = v; } },
    elements() {
      const els = [...this.querySelectorAll('input, select, textarea, button, fieldset, output, object')];
      if (this.id) for (const e of document.querySelectorAll('[form="' + this.id + '"]')) if (!els.includes(e)) els.push(e);
      const c = makeCollection(els);
      for (const e of els) { const n = e.getAttribute('name') || e.id; if (n && !(n in c)) Object.defineProperty(c, n, { value: e }); }
      return c;
    },
    length() { return this.elements.length; },
  });
  Object.assign(C.HTMLFormElement.prototype, {
    submit() { H.submit(this[ID], -1); },
    requestSubmit(submitter) { if (this.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true, submitter: submitter || null }))) H.submit(this[ID], submitter ? submitter[ID] : -1); },
    reset() { if (!this.dispatchEvent(new Event('reset', { bubbles: true, cancelable: true }))) return; for (const e of this.elements) { if (e.localName === 'input') { if (e.type === 'checkbox' || e.type === 'radio') e.checked = e.defaultChecked; else if (e.type !== 'submit' && e.type !== 'button') e.value = e.defaultValue; } else if (e.localName === 'textarea') e.value = e.defaultValue; } },
    checkValidity() { return true; }, reportValidity() { return true; },
  });
  // Named form controls are reachable as properties of the form (form.username).
  const formProxyHandler = {};

  // ---------------------------------------------------------------- Document
  class DocumentFragment extends Node {
    constructor() { super(); const id = H.create('evolve-fragment', ''); this[ID] = id; nodes.set(id, this); fragments.add(this); }
    get nodeType() { return 11; } get nodeName() { return '#document-fragment'; }
    get children() { return makeCollection(H.children(this[ID]).map(wrap).filter(n => n.nodeType === 1)); }
    get childElementCount() { return this.children.length; }
    get firstElementChild() { return this.children[0] || null; } get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
    get textContent() { return H.text(this[ID]); } set textContent(v) { H.setText(this[ID], String(v)); }
    get innerHTML() { return H.innerHTML(this[ID]); }
    querySelector(s) { return wrap(H.query(this[ID], s)); } querySelectorAll(s) { return makeNodeList(H.queryAll(this[ID], s).map(wrap)); }
    getElementById(i) { return this.querySelector('#' + cssEscape(i)); }
    append(...n) { for (const x of n) this.appendChild(toNode(x)); } prepend(...n) { const f = this.firstChild; for (const x of n) this.insertBefore(toNode(x), f); }
    replaceChildren(...n) { for (const c of this.childNodes) this.removeChild(c); this.append(...n); }
    get isConnected() { return false; } get parentNode() { return null; }
  }
  let currentScript = null;
  let docLocation;
  class Document extends Node {
    constructor() { super(); throw new TypeError('Illegal constructor'); }
    get nodeType() { return 9; } get nodeName() { return '#document'; }
    get documentElement() { const kids = H.children(H.documentId()).map(wrap).filter(n => n && n.nodeType === 1); return kids[0] || null; }
    get head() { return this.querySelector('head'); } get body() { return this.querySelector('body') || this.querySelector('frameset'); }
    set body(el) { const b = this.body; if (b) b.replaceWith(el); else this.documentElement.appendChild(el); }
    get title() { return H.title(); }
    set title(v) { let t = this.querySelector('title'); if (!t) { t = this.createElement('title'); (this.head || this.documentElement).appendChild(t); } t.textContent = v; }
    get cookie() { return H.cookie(); } set cookie(v) { H.setCookie(String(v)); }
    get URL() { return H.url(); } get documentURI() { return H.url(); } get location() { return docLocation; } set location(v) { docLocation.href = v; }
    get baseURI() { const b = H.query(H.documentId(), 'base[href]'); if (b >= 0) { try { return new URL(H.getAttr(b, 'href'), H.url()).href; } catch (e) {} } return H.url(); }
    get referrer() { return H.referrer(); } get domain() { return docLocation.hostname; } set domain(v) {}
    get readyState() { return H.readyState(); } get visibilityState() { return 'visible'; } get hidden() { return false; }
    get characterSet() { return 'UTF-8'; } get charset() { return 'UTF-8'; } get inputEncoding() { return 'UTF-8'; } get contentType() { return 'text/html'; }
    get compatMode() { return 'CSS1Compat'; } get defaultView() { return G; } get doctype() { return { name: 'html', publicId: '', systemId: '', nodeType: 10 }; }
    get activeElement() { const a = H.activeElement(); return a >= 0 ? wrap(a) : this.body; }
    get currentScript() { return currentScript; } get scrollingElement() { return this.documentElement; }
    get forms() { return makeCollection([...this.querySelectorAll('form')]); } get images() { return makeCollection([...this.querySelectorAll('img')]); }
    get links() { return makeCollection([...this.querySelectorAll('a[href], area[href]')]); } get scripts() { return makeCollection([...this.querySelectorAll('script')]); }
    get anchors() { return makeCollection([...this.querySelectorAll('a[name]')]); } get embeds() { return makeCollection([...this.querySelectorAll('embed')]); }
    get styleSheets() { return Object.assign([...this.querySelectorAll('style, link[rel~=stylesheet]')].filter(el => !adoptedNodes.has(el)).map(el => ({ ownerNode: el, href: el.href || null, cssRules: [], rules: [], disabled: false, media: { mediaText: '' }, insertRule() { return 0; }, deleteRule() {} })), { item(i) { return this[i] || null; } }); }
    get fonts() { return fontFaceSet; } get timeline() { return { currentTime: performance.now() }; }
    get fullscreenElement() { return null; } get fullscreenEnabled() { return false; } get pictureInPictureEnabled() { return false; } get pointerLockElement() { return null; }
    get designMode() { return 'off'; } set designMode(v) {}
    get children() { return makeCollection([this.documentElement].filter(Boolean)); } get childElementCount() { return this.documentElement ? 1 : 0; }
    get firstElementChild() { return this.documentElement; } get lastElementChild() { return this.documentElement; }
    get implementation() { return { hasFeature: () => true, createHTMLDocument: (t) => createDetachedDocument('<title>' + (t || '') + '</title>'), createDocument: () => createDetachedDocument(''), createDocumentType: (n) => ({ name: n }) }; }
    get adoptedStyleSheets() { return adoptedList(this); } set adoptedStyleSheets(v) { setAdoptedSheets(this, v); }
    createElement(tag, opts) {
      tag = String(tag).toLowerCase();
      if (!/^[a-z][a-z0-9._\-]*$/.test(tag) && !/^[a-z][^\s>/]*$/.test(tag)) throw new DOMException("The tag name provided ('" + tag + "') is not a valid name.", 'InvalidCharacterError');
      const is = opts && opts.is;
      const def = customDefs.get(is || tag);
      if (def && def.ready && !is) { try { return new def.cls(); } catch (e) { reportError(e); } }
      return wrap(H.create(tag, ''));
    }
    createElementNS(ns, qname) { const local = String(qname).split(':').pop(); if (ns === 'http://www.w3.org/1999/xhtml' || ns == null) return this.createElement(local); return wrap(H.create(local, ns)); }
    createTextNode(t) { return new Text(t); } createComment(t) { return new Comment(t); } createCDATASection(t) { return new Text(t); }
    createDocumentFragment() { return new DocumentFragment(); }
    createProcessingInstruction() { return new Comment(''); }
    createAttribute(n) { const a = new Attr(null, String(n).toLowerCase()); a._v = ''; return a; }
    createEvent(kind) { const k = String(kind).toLowerCase(); const cls = { mouseevent: MouseEvent, mouseevents: MouseEvent, keyboardevent: KeyboardEvent, uievent: UIEvent, uievents: UIEvent, customevent: CustomEvent, focusevent: FocusEvent, touchevent: TouchEvent }[k] || Event; const e = new cls(''); return e; }
    createRange() { return new Range(); }
    createTreeWalker(root, what = 0xFFFFFFFF, filter = null) { return new TreeWalker(root, what, filter); }
    createNodeIterator(root, what = 0xFFFFFFFF, filter = null) { const w = new TreeWalker(root, what, filter); return { root, referenceNode: root, nextNode: () => w.nextNode(), previousNode: () => w.previousNode(), detach() {} }; }
    importNode(n, deep) { return n.cloneNode(deep); } adoptNode(n) { if (n.parentNode) n.parentNode.removeChild(n); return n; }
    getElementById(i) { return wrap(H.byId(String(i))); }
    getElementsByName(n) { return makeNodeList(H.queryAll(H.documentId(), '[name="' + String(n).replace(/"/g, '\\"') + '"]').map(wrap)); }
    getElementsByTagName(t) { return Element.prototype.getElementsByTagName.call({ [ID]: H.documentId() }, t); }
    getElementsByTagNameNS(_, t) { return this.getElementsByTagName(t); }
    getElementsByClassName(c) { return Element.prototype.getElementsByClassName.call({ [ID]: H.documentId() }, c); }
    querySelector(s) { return wrap(H.query(H.documentId(), String(s))); }
    querySelectorAll(s) { return makeNodeList(H.queryAll(H.documentId(), String(s)).map(wrap)); }
    elementFromPoint(x, y) { let best = null; for (const el of this.querySelectorAll('body *')) { const r = el.getBoundingClientRect(); if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) best = el; } return best || this.body; }
    elementsFromPoint(x, y) { const e = this.elementFromPoint(x, y); return e ? [e] : []; }
    caretRangeFromPoint() { return null; } caretPositionFromPoint() { return null; }
    hasFocus() { return true; } execCommand() { return false; } queryCommandSupported() { return false; } queryCommandState() { return false; }
    getSelection() { return selection; }
    write(...s) { if (H.readyState() === 'loading') H.write(s.join('')); else if (this.body) this.body.insertAdjacentHTML('beforeend', s.join('')); }
    writeln(...s) { this.write(s.join('') + '\n'); }
    open() { return this; } close() {}
    exitFullscreen() { return Promise.resolve(); } exitPointerLock() {}
    startViewTransition(cb) { const p = Promise.resolve().then(() => cb && cb()); return { finished: p, ready: p, updateCallbackDone: p, skipTransition() {} }; }
    append(...n) { for (const x of n) this.appendChild(toNode(x)); }
  }
  const document = Object.create(Document.prototype);
  Object.defineProperty(document, ID, { get: () => H.documentId() });
  nodes.set(H.documentId(), document);
  class HTMLDocument extends Document {}
  Object.setPrototypeOf(document, HTMLDocument.prototype);
  class XMLDocument extends Document {}
  function createDetachedDocument(html) {
    const root = document.createElement('html');
    root.innerHTML = '<head></head><body></body>';
    const d = Object.create(Document.prototype);
    const rid = root[ID];
    Object.defineProperty(d, ID, { value: rid });
    Object.defineProperty(d, 'documentElement', { value: root });
    Object.defineProperty(d, 'head', { get: () => root.querySelector('head') });
    Object.defineProperty(d, 'body', { get: () => root.querySelector('body') });
    Object.defineProperty(d, 'title', { get: () => { const t = root.querySelector('title'); return t ? t.textContent : ''; } });
    d.querySelector = (s) => root.querySelector(s); d.querySelectorAll = (s) => root.querySelectorAll(s);
    d.getElementById = (i) => root.querySelector('#' + cssEscape(i));
    d.getElementsByTagName = (t) => root.getElementsByTagName(t); d.getElementsByClassName = (c) => root.getElementsByClassName(c);
    d.createElement = (t) => document.createElement(t); d.createTextNode = (t) => document.createTextNode(t);
    if (html) {
      const m = /<head[^>]*>([\s\S]*?)<\/head>/i.exec(html); const b = /<body[^>]*>([\s\S]*)<\/body>/i.exec(html);
      if (m || b) { if (m) d.head.innerHTML = m[1]; d.body.innerHTML = b ? b[1] : html.replace(/<head[^>]*>[\s\S]*?<\/head>/i, ''); }
      else d.body.innerHTML = html;
    }
    return d;
  }
  class DOMParser { parseFromString(str, type) { return createDetachedDocument(String(str)); } }
  class XMLSerializer { serializeToString(n) { return n.outerHTML !== undefined ? n.outerHTML : n.textContent; } }

  class Range {
    constructor() { this.startContainer = document; this.startOffset = 0; this.endContainer = document; this.endOffset = 0; this.collapsed = true; this.commonAncestorContainer = document; }
    setStart(n, o) { this.startContainer = n; this.startOffset = o; } setEnd(n, o) { this.endContainer = n; this.endOffset = o; this.collapsed = false; }
    setStartBefore(n) { this.startContainer = n; } setStartAfter(n) { this.startContainer = n; } setEndBefore(n) { this.endContainer = n; } setEndAfter(n) { this.endContainer = n; }
    selectNode(n) { this.startContainer = this.endContainer = n; } selectNodeContents(n) { this.startContainer = this.endContainer = n; }
    collapse() { this.collapsed = true; } cloneRange() { return Object.assign(new Range(), this); } detach() {}
    deleteContents() {} insertNode(n) { if (this.startContainer && this.startContainer.appendChild) this.startContainer.appendChild(n); }
    surroundContents(n) {} extractContents() { return document.createDocumentFragment(); } cloneContents() { return document.createDocumentFragment(); }
    createContextualFragment(html) { const t = document.createElement('template'); t.innerHTML = html; return t.content; }
    getBoundingClientRect() { const n = this.startContainer; return n && n.getBoundingClientRect ? n.getBoundingClientRect() : new DOMRect(); }
    getClientRects() { return [this.getBoundingClientRect()]; } toString() { return ''; } compareBoundaryPoints() { return 0; } intersectsNode() { return false; } isPointInRange() { return false; }
  }
  define(Range, { START_TO_START: 0, START_TO_END: 1, END_TO_END: 2, END_TO_START: 3 });
  const selection = { rangeCount: 0, isCollapsed: true, type: 'None', anchorNode: null, focusNode: null, anchorOffset: 0, focusOffset: 0, getRangeAt() { return new Range(); }, addRange() {}, removeAllRanges() {}, removeRange() {}, empty() {}, collapse() {}, collapseToEnd() {}, collapseToStart() {}, selectAllChildren() {}, extend() {}, setBaseAndExtent() {}, containsNode() { return false; }, toString() { return ''; }, modify() {} };
  class TreeWalker {
    constructor(root, what, filter) { this.root = root; this.whatToShow = what; this.filter = filter; this.currentNode = root; }
    _ok(n) { const bit = 1 << (n.nodeType - 1); if (!(this.whatToShow & bit)) return 3; if (!this.filter) return 1; const f = typeof this.filter === 'function' ? this.filter : this.filter.acceptNode.bind(this.filter); return f(n); }
    _next(n) { if (n.firstChild) return n.firstChild; while (n && n !== this.root) { if (n.nextSibling) return n.nextSibling; n = n.parentNode; } return null; }
    nextNode() { let n = this.currentNode; for (;;) { n = this._next(n); if (!n) return null; const r = this._ok(n); if (r === 1) { this.currentNode = n; return n; } } }
    previousNode() { return null; } parentNode() { const p = this.currentNode.parentNode; if (p && p !== this.root) { this.currentNode = p; return p; } return null; }
    firstChild() { const c = this.currentNode.firstChild; if (c) this.currentNode = c; return c; } lastChild() { const c = this.currentNode.lastChild; if (c) this.currentNode = c; return c; }
    nextSibling() { const c = this.currentNode.nextSibling; if (c) this.currentNode = c; return c; } previousSibling() { const c = this.currentNode.previousSibling; if (c) this.currentNode = c; return c; }
  }
  const NodeFilter = { FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3, SHOW_ALL: 0xFFFFFFFF, SHOW_ELEMENT: 1, SHOW_ATTRIBUTE: 2, SHOW_TEXT: 4, SHOW_CDATA_SECTION: 8, SHOW_COMMENT: 128, SHOW_DOCUMENT: 256, SHOW_DOCUMENT_FRAGMENT: 1024 };

  // ---------------------------------------------------------------- Custom elements
  const customDefs = new Map();
  const defsByClass = new Map();
  function upgradeTree(root) {
    if (!customDefs.size) return;
    const sel = [...customDefs.keys()].map(cssName).join(',');
    const id = root[ID];
    const found = (root.nodeType === 1 && root.matches(sel) ? [id] : []).concat(H.queryAll(id, sel));
    for (const fid of found) {
      const existing = nodes.get(fid);
      const def = customDefs.get(H.info(fid)[1]);
      if (!def || !def.ready || (existing && existing instanceof def.cls)) continue;
      nodes.delete(fid);
      upgrading = fid;
      let el = null;
      try { el = Reflect.construct(def.cls, []); } catch (e) { reportError(e); } finally { upgrading = -1; }
      if (!el) { if (existing) nodes.set(fid, existing); continue; }
      if (existing) { for (const k of Object.keys(existing)) { try { el[k] = existing[k]; } catch (e) {} } }
      for (const a of def.observed) { const v = H.getAttr(fid, a); if (v !== null && el.attributeChangedCallback) call(el.attributeChangedCallback, el, [a, null, v]); }
    }
  }
  function walkCustom(root, cb) {
    if (!customDefs.size) return;
    const sel = [...customDefs.keys()].map(cssName).join(',');
    const ids = (root.nodeType === 1 && root.matches(sel) ? [root[ID]] : []).concat(H.queryAll(root[ID], sel));
    for (const id of ids) { const el = wrap(id); if (el && typeof el[cb] === 'function') call(el[cb], el, []); }
  }
  function attributeChanged(el, name, old, value) {
    if (!customDefs.size) return;
    const def = customDefs.get(el.localName);
    if (def && def.observed.includes(name) && typeof el.attributeChangedCallback === 'function' && el instanceof def.cls) call(el.attributeChangedCallback, el, [name, old, value]);
  }
  const whenDefined = new Map();
  const customElements = {
    define(name, cls, opts) {
      name = String(name);
      if (!/^[a-z][a-z0-9._]*-[a-z0-9._-]*$/.test(name)) throw new DOMException("'" + name + "' is not a valid custom element name", 'SyntaxError');
      if (customDefs.has(name)) throw new DOMException("the name '" + name + "' has already been used with this registry", 'NotSupportedError');
      const observed = [].concat(cls.observedAttributes || []).map(String);
      const def = { name, cls, observed, ready: true };
      customDefs.set(name, def); defsByClass.set(cls, def);
      const root = document.documentElement;
      if (root) { upgradeTree(root); for (const id of H.queryAll(H.documentId(), cssName(name))) { const el = wrap(id); if (el && el.connectedCallback) call(el.connectedCallback, el, []); } }
      const w = whenDefined.get(name); if (w) { w.resolve(cls); whenDefined.delete(name); }
    },
    get(name) { const d = customDefs.get(name); return d ? d.cls : undefined; },
    getName(cls) { const d = defsByClass.get(cls); return d ? d.name : null; },
    whenDefined(name) { const d = customDefs.get(name); if (d) return Promise.resolve(d.cls); let w = whenDefined.get(name); if (!w) { let resolve; const p = new Promise(r => { resolve = r; }); w = { p, resolve }; whenDefined.set(name, w); } return w.p; },
    upgrade(root) { upgradeTree(root); },
  };

  // ---------------------------------------------------------------- Geometry, animation, observers
  class DOMRectReadOnly {
    constructor(x = 0, y = 0, w = 0, h = 0) { this.x = x; this.y = y; this.width = w; this.height = h; }
    get top() { return Math.min(this.y, this.y + this.height); } get left() { return Math.min(this.x, this.x + this.width); }
    get bottom() { return Math.max(this.y, this.y + this.height); } get right() { return Math.max(this.x, this.x + this.width); }
    toJSON() { return { x: this.x, y: this.y, width: this.width, height: this.height, top: this.top, left: this.left, bottom: this.bottom, right: this.right }; }
    static fromRect(r = {}) { return new this(r.x, r.y, r.width, r.height); }
  }
  class DOMRect extends DOMRectReadOnly {}
  class DOMPoint { constructor(x = 0, y = 0, z = 0, w = 1) { Object.assign(this, { x, y, z, w }); } matrixTransform() { return this; } static fromPoint(p = {}) { return new DOMPoint(p.x, p.y, p.z, p.w); } }
  class DOMMatrix { constructor() { Object.assign(this, { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0, m11: 1, m12: 0, m21: 0, m22: 1, m41: 0, m42: 0, is2D: true, isIdentity: true }); } translate() { return this; } scale() { return this; } rotate() { return this; } multiply() { return this; } inverse() { return this; } transformPoint(p) { return new DOMPoint(p && p.x, p && p.y); } toString() { return 'matrix(1, 0, 0, 1, 0, 0)'; } }
  class Animation extends EventTarget {
    constructor() { super(); this.playState = 'finished'; this.currentTime = 0; this.playbackRate = 1; this.finished = Promise.resolve(this); this.ready = Promise.resolve(this); this.effect = null; this.timeline = null; this.id = ''; setTimeout(() => { if (this.onfinish) call(this.onfinish, this, [new Event('finish')]); this.dispatchEvent(new Event('finish')); }, 0); }
    play() {} pause() {} cancel() {} finish() {} reverse() {} updatePlaybackRate() {} commitStyles() {} persist() {}
  }
  class KeyframeEffect { constructor(target, frames, opts) { this.target = target; } getKeyframes() { return []; } getTiming() { return {}; } updateTiming() {} }

  class MutationObserver {
    constructor(cb) { if (typeof cb !== 'function') throw new TypeError("Failed to construct 'MutationObserver': parameter 1 is not of type 'Function'."); this._cb = cb; this._targets = []; this._records = []; this._scheduled = false; }
    observe(target, opts = {}) {
      if (!target || target[ID] === undefined) throw new TypeError("Failed to execute 'observe' on 'MutationObserver': parameter 1 is not of type 'Node'.");
      const o = Object.assign({}, opts);
      if (o.attributeOldValue || o.attributeFilter) o.attributes = true;
      if (o.characterDataOldValue) o.characterData = true;
      this._targets = this._targets.filter(t => t.node !== target);
      this._targets.push({ node: target, opts: o });
      if (o.attributes) target._observed = true;
      if (!observers.includes(this)) observers.push(this);
    }
    disconnect() { this._targets = []; const i = observers.indexOf(this); if (i >= 0) observers.splice(i, 1); this._records = []; }
    takeRecords() { const r = this._records; this._records = []; return r; }
    _consider(rec) {
      for (const { node, opts } of this._targets) {
        const direct = rec.target === node;
        if (!direct && !(opts.subtree && node.contains(rec.target))) continue;
        if (rec.type === 'childList' && !opts.childList) continue;
        if (rec.type === 'attributes' && (!opts.attributes || (opts.attributeFilter && !opts.attributeFilter.includes(rec.attributeName)))) continue;
        if (rec.type === 'characterData' && !opts.characterData) continue;
        this._records.push({ type: rec.type, target: rec.target, addedNodes: makeNodeList(rec.addedNodes ? [...rec.addedNodes] : []), removedNodes: makeNodeList(rec.removedNodes ? [...rec.removedNodes] : []), previousSibling: rec.previousSibling || null, nextSibling: rec.nextSibling || null, attributeName: rec.attributeName || null, attributeNamespace: null, oldValue: (opts.attributeOldValue || opts.characterDataOldValue) ? rec.oldValue : null });
        if (!this._scheduled) { this._scheduled = true; Promise.resolve().then(() => { this._scheduled = false; const r = this.takeRecords(); if (r.length) call(this._cb, this, [r, this]); }); }
        return;
      }
    }
  }
  class IntersectionObserver {
    constructor(cb, opts = {}) { this._cb = cb; this.root = opts.root || null; this.rootMargin = opts.rootMargin || '0px'; this.thresholds = [].concat(opts.threshold || 0); this._els = new Set(); }
    observe(el) {
      if (this._els.has(el)) return; this._els.add(el);
      setTimeout(() => {
        if (!this._els.has(el)) return;
        const r = el.getBoundingClientRect(); const vh = G.innerHeight, vw = G.innerWidth;
        const visible = r.bottom > -200 && r.top < vh + 200 && r.right > -200 && r.left < vw + 200;
        call(this._cb, this, [[{ target: el, isIntersecting: visible || true, intersectionRatio: 1, boundingClientRect: r, intersectionRect: r, rootBounds: new DOMRect(0, 0, vw, vh), time: performance.now() }], this]);
      }, 0);
    }
    unobserve(el) { this._els.delete(el); } disconnect() { this._els.clear(); } takeRecords() { return []; }
  }
  class ResizeObserver {
    constructor(cb) { this._cb = cb; this._els = new Set(); }
    observe(el) {
      this._els.add(el);
      setTimeout(() => {
        if (!this._els.has(el)) return;
        const r = el.getBoundingClientRect(); const size = [{ inlineSize: r.width, blockSize: r.height }];
        call(this._cb, this, [[{ target: el, contentRect: new DOMRectReadOnly(0, 0, r.width, r.height), borderBoxSize: size, contentBoxSize: size, devicePixelContentBoxSize: size }], this]);
      }, 0);
    }
    unobserve(el) { this._els.delete(el); } disconnect() { this._els.clear(); }
  }
  class PerformanceObserver { constructor(cb) { this._cb = cb; } observe() {} disconnect() {} takeRecords() { return []; } }
  PerformanceObserver.supportedEntryTypes = [];

  // ---------------------------------------------------------------- Timers
  let timerSeq = 1;
  const timers = new Map();
  function setTimeout(fn, ms = 0, ...args) { const id = timerSeq++; timers.set(id, { fn, args, interval: false, ms: +ms || 0 }); H.setTimer(id, +ms || 0); return id; }
  function setInterval(fn, ms = 0, ...args) { const id = timerSeq++; const d = Math.max(+ms || 0, 4); timers.set(id, { fn, args, interval: true, ms: d }); H.setTimer(id, d); return id; }
  function clearTimeout(id) { if (timers.delete(id)) H.clearTimer(id); }
  let rafSeq = 1; let rafs = [];
  function requestAnimationFrame(cb) { const id = rafSeq++; rafs.push({ id, cb }); H.requestFrame(); return id; }
  function cancelAnimationFrame(id) { rafs = rafs.filter(r => r.id !== id); }
  function requestIdleCallback(cb, opts) { return setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 12 }), 1); }
  function queueMicrotask(cb) { Promise.resolve().then(() => call(cb, undefined, [])); }

  // ---------------------------------------------------------------- Encoding, URLs
  class TextEncoder { get encoding() { return 'utf-8'; } encode(s = '') { return H.encode(String(s)); } encodeInto(s, dest) { const b = H.encode(String(s)); const n = Math.min(b.length, dest.length); dest.set(b.subarray(0, n)); return { read: s.length, written: n }; } }
  class TextDecoder {
    constructor(label = 'utf-8', opts = {}) { this.encoding = String(label).toLowerCase(); this.fatal = !!opts.fatal; this.ignoreBOM = !!opts.ignoreBOM; }
    decode(buf, opts) { if (buf == null) return ''; let u8 = buf instanceof Uint8Array ? buf : ArrayBuffer.isView(buf) ? new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength) : new Uint8Array(buf); return H.decode(u8, this.encoding); }
  }
  const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  function btoa(s) {
    s = String(s); let out = '';
    for (let i = 0; i < s.length; i += 3) {
      const a = s.charCodeAt(i), b = s.charCodeAt(i + 1), c = s.charCodeAt(i + 2);
      if (a > 255 || b > 255 || c > 255) throw new DOMException('The string to be encoded contains characters outside of the Latin1 range.', 'InvalidCharacterError');
      const n = (a << 16) | ((b || 0) << 8) | (c || 0);
      out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (i + 1 < s.length ? B64[(n >> 6) & 63] : '=') + (i + 2 < s.length ? B64[n & 63] : '=');
    }
    return out;
  }
  function atob(s) {
    s = String(s).replace(/[\t\n\f\r ]/g, '');
    if (s.length % 4 === 0) s = s.replace(/==?$/, '');
    if (s.length % 4 === 1 || /[^A-Za-z0-9+/]/.test(s)) throw new DOMException('The string to be decoded is not correctly encoded.', 'InvalidCharacterError');
    let out = '', buf = 0, bits = 0;
    for (const ch of s) { buf = (buf << 6) | B64.indexOf(ch); bits += 6; if (bits >= 8) { bits -= 8; out += String.fromCharCode((buf >> bits) & 255); } }
    return out;
  }
  class URLSearchParams {
    constructor(init = '') {
      this._list = [];
      if (init instanceof URLSearchParams) this._list = init._list.map(p => p.slice());
      else if (Array.isArray(init) || (init && typeof init === 'object' && typeof init[Symbol.iterator] === 'function')) for (const [k, v] of init) this._list.push([String(k), String(v)]);
      else if (init && typeof init === 'object') for (const k of Object.keys(init)) this._list.push([k, String(init[k])]);
      else { const s = String(init).replace(/^\?/, ''); if (s) for (const part of s.split('&')) { if (!part) continue; const i = part.indexOf('='); const k = i < 0 ? part : part.slice(0, i); const v = i < 0 ? '' : part.slice(i + 1); this._list.push([dec(k), dec(v)]); } }
    }
    _update() { if (this._url) this._url._setSearch(this.toString()); }
    append(k, v) { this._list.push([String(k), String(v)]); this._update(); }
    delete(k, v) { this._list = this._list.filter(p => p[0] !== String(k) || (v !== undefined && p[1] !== String(v))); this._update(); }
    get(k) { const p = this._list.find(p => p[0] === String(k)); return p ? p[1] : null; }
    getAll(k) { return this._list.filter(p => p[0] === String(k)).map(p => p[1]); }
    has(k, v) { return this._list.some(p => p[0] === String(k) && (v === undefined || p[1] === String(v))); }
    set(k, v) { k = String(k); const i = this._list.findIndex(p => p[0] === k); if (i < 0) this._list.push([k, String(v)]); else { this._list[i][1] = String(v); this._list = this._list.filter((p, j) => j <= i || p[0] !== k); } this._update(); }
    sort() { this._list.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0); this._update(); }
    forEach(fn, self) { for (const [k, v] of this._list) fn.call(self, v, k, this); }
    keys() { return this._list.map(p => p[0])[Symbol.iterator](); } values() { return this._list.map(p => p[1])[Symbol.iterator](); }
    entries() { return this._list.map(p => p.slice())[Symbol.iterator](); } [Symbol.iterator]() { return this.entries(); }
    get size() { return this._list.length; }
    toString() { return this._list.map(([k, v]) => enc(k) + '=' + enc(v)).join('&'); }
  }
  const enc = (s) => encodeURIComponent(s).replace(/%20/g, '+').replace(/[!'()~]/g, c => '%' + c.charCodeAt(0).toString(16).toUpperCase());
  const dec = (s) => { try { return decodeURIComponent(s.replace(/\+/g, ' ')); } catch (e) { return s; } };
  const URL_PARTS = ['href', 'protocol', 'username', 'password', 'host', 'hostname', 'port', 'pathname', 'search', 'hash', 'origin'];
  class URL {
    constructor(url, base) {
      const p = H.parseUrl(String(url), base === undefined ? null : String(base instanceof URL ? base.href : base));
      if (!p) throw new TypeError("Failed to construct 'URL': Invalid URL");
      this._p = p;
    }
    _set(part, v) { const p = H.setUrlPart(this._p[0], part, String(v)); if (p) { this._p = p; if (this._sp) this._sp._list = new URLSearchParams(p[8])._list; } }
    _setSearch(s) { const p = H.setUrlPart(this._p[0], 'search', s); if (p) this._p = p; }
    get searchParams() { if (!this._sp) { this._sp = new URLSearchParams(this._p[8]); this._sp._url = this; } return this._sp; }
    toString() { return this._p[0]; } toJSON() { return this._p[0]; }
    static canParse(u, b) { try { new URL(u, b); return true; } catch (e) { return false; } }
    static parse(u, b) { try { return new URL(u, b); } catch (e) { return null; } }
    static createObjectURL(blob) { const id = 'blob:' + location.origin + '/' + crypto.randomUUID(); blobUrls.set(id, blob); return id; }
    static revokeObjectURL(u) { blobUrls.delete(u); }
  }
  URL_PARTS.forEach((part, i) => Object.defineProperty(URL.prototype, part, { enumerable: true, configurable: true, get() { return this._p[i]; }, set(v) { if (part !== 'origin') this._set(part, v); } }));
  const blobUrls = new Map();

  // ---------------------------------------------------------------- Blob, FormData, Headers, fetch
  class Blob {
    constructor(parts = [], opts = {}) {
      const chunks = [];
      for (const p of parts) {
        if (p instanceof Blob) chunks.push(p._bytes);
        else if (p instanceof ArrayBuffer) chunks.push(new Uint8Array(p));
        else if (ArrayBuffer.isView(p)) chunks.push(new Uint8Array(p.buffer, p.byteOffset, p.byteLength));
        else chunks.push(H.encode(String(p)));
      }
      const len = chunks.reduce((n, c) => n + c.length, 0); const out = new Uint8Array(len); let o = 0;
      for (const c of chunks) { out.set(c, o); o += c.length; }
      this._bytes = out; this.type = String(opts.type || '').toLowerCase();
    }
    get size() { return this._bytes.length; }
    text() { return Promise.resolve(H.decode(this._bytes, 'utf-8')); }
    arrayBuffer() { return Promise.resolve(this._bytes.slice().buffer); }
    bytes() { return Promise.resolve(this._bytes.slice()); }
    slice(a = 0, b = this.size, type = '') { const r = new Blob([], { type }); r._bytes = this._bytes.slice(a, b); return r; }
    stream() { const bytes = this._bytes; return new ReadableStream({ start(c) { c.enqueue(bytes); c.close(); } }); }
  }
  class File extends Blob { constructor(parts, name, opts = {}) { super(parts, opts); this.name = String(name); this.lastModified = opts.lastModified || Date.now(); this.webkitRelativePath = ''; } }
  class FileReader extends EventTarget {
    constructor() { super(); this.readyState = 0; this.result = null; this.error = null; }
    _read(blob, kind) {
      this.readyState = 1;
      setTimeout(() => {
        const b = blob._bytes;
        this.result = kind === 'text' ? H.decode(b, 'utf-8') : kind === 'buffer' ? b.slice().buffer : kind === 'url' ? 'data:' + (blob.type || 'application/octet-stream') + ';base64,' + btoa(String.fromCharCode.apply(null, Array.from(b))) : String.fromCharCode.apply(null, Array.from(b));
        this.readyState = 2;
        for (const t of ['load', 'loadend']) { const e = new ProgressEvent(t, { loaded: b.length, total: b.length }); if (this['on' + t]) call(this['on' + t], this, [e]); this.dispatchEvent(e); }
      }, 0);
    }
    readAsText(b) { this._read(b, 'text'); } readAsArrayBuffer(b) { this._read(b, 'buffer'); } readAsDataURL(b) { this._read(b, 'url'); } readAsBinaryString(b) { this._read(b, 'binary'); } abort() {}
  }
  class FormData {
    constructor(form, submitter) {
      this._list = [];
      if (form) {
        for (const el of form.elements) {
          const name = el.getAttribute('name'); if (!name || el.disabled || el.closest('datalist')) continue;
          const tag = el.localName; const type = (el.getAttribute('type') || '').toLowerCase();
          if (tag === 'button' || (tag === 'input' && ['submit', 'image', 'button', 'reset'].includes(type))) { if (el === submitter) this._list.push([name, el.value]); continue; }
          if (tag === 'input' && (type === 'checkbox' || type === 'radio') && !el.checked) continue;
          if (tag === 'input' && type === 'file') { continue; }
          if (tag === 'select') { for (const o of el.options) if (o.selected) this._list.push([name, o.value]); continue; }
          if (tag === 'fieldset' || tag === 'output' || tag === 'object') continue;
          this._list.push([name, el.value]);
        }
      }
    }
    append(k, v, fname) { this._list.push([String(k), v instanceof Blob ? (fname ? new File([v], fname, { type: v.type }) : v) : String(v)]); }
    set(k, v, fname) { this.delete(k); this.append(k, v, fname); }
    delete(k) { this._list = this._list.filter(p => p[0] !== String(k)); }
    get(k) { const p = this._list.find(p => p[0] === String(k)); return p ? p[1] : null; }
    getAll(k) { return this._list.filter(p => p[0] === String(k)).map(p => p[1]); }
    has(k) { return this._list.some(p => p[0] === String(k)); }
    forEach(fn, self) { for (const [k, v] of this._list) fn.call(self, v, k, this); }
    keys() { return this._list.map(p => p[0])[Symbol.iterator](); } values() { return this._list.map(p => p[1])[Symbol.iterator](); }
    entries() { return this._list.map(p => p.slice())[Symbol.iterator](); } [Symbol.iterator]() { return this.entries(); }
  }
  class Headers {
    constructor(init) {
      this._m = new Map();
      if (init instanceof Headers) init.forEach((v, k) => this.append(k, v));
      else if (Array.isArray(init)) for (const [k, v] of init) this.append(k, v);
      else if (init && typeof init === 'object') for (const k of Object.keys(init)) this.append(k, init[k]);
    }
    append(k, v) { k = String(k).toLowerCase(); v = String(v).trim(); const cur = this._m.get(k); this._m.set(k, cur === undefined ? v : (k === 'set-cookie' ? cur + '\n' : cur + ', ') + v); }
    set(k, v) { this._m.set(String(k).toLowerCase(), String(v).trim()); }
    get(k) { const v = this._m.get(String(k).toLowerCase()); return v === undefined ? null : v; }
    has(k) { return this._m.has(String(k).toLowerCase()); }
    delete(k) { this._m.delete(String(k).toLowerCase()); }
    forEach(fn, self) { for (const [k, v] of [...this._m].sort()) fn.call(self, v, k, this); }
    keys() { return [...this._m.keys()].sort()[Symbol.iterator](); } values() { return [...this._m].sort().map(p => p[1])[Symbol.iterator](); }
    entries() { return [...this._m].sort()[Symbol.iterator](); } [Symbol.iterator]() { return this.entries(); }
    getSetCookie() { return []; }
  }
  function bodyToHost(body, headers) {
    if (body == null) return null;
    if (typeof body === 'string') { if (!headers.has('content-type')) headers.set('content-type', 'text/plain;charset=UTF-8'); return body; }
    if (body instanceof URLSearchParams) { if (!headers.has('content-type')) headers.set('content-type', 'application/x-www-form-urlencoded;charset=UTF-8'); return body.toString(); }
    if (body instanceof FormData) {
      const boundary = '----EvolveFormBoundary' + Math.random().toString(36).slice(2);
      const parts = [];
      for (const [k, v] of body._list) {
        if (v instanceof Blob) { parts.push('--' + boundary + '\r\nContent-Disposition: form-data; name="' + k + '"; filename="' + (v.name || 'blob') + '"\r\nContent-Type: ' + (v.type || 'application/octet-stream') + '\r\n\r\n', v, '\r\n'); }
        else parts.push('--' + boundary + '\r\nContent-Disposition: form-data; name="' + k + '"\r\n\r\n' + v + '\r\n');
      }
      parts.push('--' + boundary + '--\r\n');
      headers.set('content-type', 'multipart/form-data; boundary=' + boundary);
      return new Blob(parts)._bytes;
    }
    if (body instanceof Blob) { if (body.type && !headers.has('content-type')) headers.set('content-type', body.type); return body._bytes; }
    if (body instanceof ArrayBuffer) return new Uint8Array(body);
    if (ArrayBuffer.isView(body)) return new Uint8Array(body.buffer, body.byteOffset, body.byteLength);
    if (body instanceof ReadableStream) return body._bytes ? body._bytes() : null;
    return String(body);
  }
  class ReadableStream {
    constructor(source = {}) {
      this._queue = []; this._closed = false; this._waiting = null;
      const ctl = { enqueue: (c) => { this._queue.push(c); this._wake(); }, close: () => { this._closed = true; this._wake(); }, error: (e) => { this._error = e; this._wake(); }, desiredSize: 1 };
      this._ctl = ctl;
      if (source.start) try { source.start(ctl); } catch (e) { this._error = e; }
      this._pull = source.pull ? () => source.pull(ctl) : null;
    }
    _wake() { if (this._waiting) { const w = this._waiting; this._waiting = null; w(); } }
    get locked() { return !!this._reader; }
    getReader() {
      const s = this; s._reader = true;
      return {
        read() {
          return new Promise((resolve, reject) => {
            const tryRead = () => {
              if (s._error) return reject(s._error);
              if (s._queue.length) return resolve({ value: s._queue.shift(), done: false });
              if (s._closed) return resolve({ value: undefined, done: true });
              s._waiting = tryRead; if (s._pull) s._pull();
            };
            tryRead();
          });
        },
        releaseLock() { s._reader = false; }, cancel() { s._closed = true; return Promise.resolve(); }, get closed() { return Promise.resolve(); },
      };
    }
    cancel() { this._closed = true; return Promise.resolve(); }
    pipeThrough(t) { return t.readable; } pipeTo() { return Promise.resolve(); } tee() { return [this, this]; }
    async *[Symbol.asyncIterator]() { const r = this.getReader(); for (;;) { const { value, done } = await r.read(); if (done) return; yield value; } }
  }
  class WritableStream { constructor() {} getWriter() { return { write() { return Promise.resolve(); }, close() { return Promise.resolve(); }, releaseLock() {}, ready: Promise.resolve() }; } }
  class TransformStream { constructor() { this.readable = new ReadableStream(); this.writable = new WritableStream(); } }
  class Body {
    _consume() { if (this._used) return Promise.reject(new TypeError('Body has already been consumed.')); this._used = true; return Promise.resolve(this._b || new Uint8Array(0)); }
    get bodyUsed() { return !!this._used; }
    get body() { if (this._b == null) return null; const b = this._b; const s = new ReadableStream({ start(c) { c.enqueue(b); c.close(); } }); return s; }
    text() { return this._consume().then(b => H.decode(b, 'utf-8')); }
    json() { return this.text().then(t => JSON.parse(t)); }
    arrayBuffer() { return this._consume().then(b => b.slice().buffer); }
    bytes() { return this._consume().then(b => b.slice()); }
    blob() { return this._consume().then(b => { const r = new Blob([], { type: this.headers.get('content-type') || '' }); r._bytes = b; return r; }); }
    formData() { return this.text().then(t => { const f = new FormData(); for (const [k, v] of new URLSearchParams(t)) f.append(k, v); return f; }); }
  }
  class Request extends Body {
    constructor(input, init = {}) {
      super();
      const base = input instanceof Request ? input : null;
      this.url = base ? base.url : new URL(String(input), document.baseURI).href;
      this.method = String(init.method || (base && base.method) || 'GET').toUpperCase();
      this.headers = new Headers(init.headers || (base && base.headers));
      this.credentials = init.credentials || (base && base.credentials) || 'same-origin';
      this.mode = init.mode || (base && base.mode) || 'cors';
      this.redirect = init.redirect || (base && base.redirect) || 'follow';
      this.signal = init.signal || (base && base.signal) || null;
      this.cache = init.cache || 'default'; this.referrer = 'about:client'; this.referrerPolicy = ''; this.integrity = ''; this.keepalive = !!init.keepalive; this.destination = '';
      const body = init.body !== undefined ? init.body : base ? base._rawBody : null;
      if (body != null && (this.method === 'GET' || this.method === 'HEAD')) throw new TypeError('Request with GET/HEAD method cannot have body.');
      this._rawBody = body;
      this._b = body == null ? null : (() => { const h = this.headers; const v = bodyToHost(body, h); return typeof v === 'string' ? H.encode(v) : v; })();
    }
    clone() { return new Request(this, { body: this._rawBody }); }
  }
  class Response extends Body {
    constructor(body = null, init = {}) {
      super();
      this.status = init.status === undefined ? 200 : init.status; this.statusText = init.statusText || '';
      this.headers = init.headers instanceof Headers ? init.headers : new Headers(init.headers);
      this.url = init.url || ''; this.redirected = !!init.redirected; this.type = init.type || 'basic';
      if (body instanceof Uint8Array && init._raw) this._b = body;
      else if (body != null) { const v = bodyToHost(body, this.headers); this._b = typeof v === 'string' ? H.encode(v) : v; }
      else this._b = null;
    }
    get ok() { return this.status >= 200 && this.status < 300; }
    clone() { const r = new Response(null, { status: this.status, statusText: this.statusText, headers: new Headers(this.headers), url: this.url, redirected: this.redirected }); r._b = this._b; return r; }
    static error() { return new Response(null, { status: 0, type: 'error' }); }
    static redirect(url, status = 302) { return new Response(null, { status, headers: { location: url } }); }
    static json(data, init = {}) { const h = new Headers(init.headers); h.set('content-type', 'application/json'); return new Response(JSON.stringify(data), Object.assign({}, init, { headers: h })); }
  }
  let fetchSeq = 1;
  const fetches = new Map();
  function fetch(input, init = {}) {
    let req;
    try { req = input instanceof Request && !Object.keys(init).length ? input : new Request(input, init); } catch (e) { return Promise.reject(e); }
    return new Promise((resolve, reject) => {
      const signal = req.signal;
      if (signal && signal.aborted) return reject(signal.reason || new DOMException('The operation was aborted.', 'AbortError'));
      if (req.url.startsWith('blob:')) { const b = blobUrls.get(req.url); return b ? resolve(new Response(b._bytes, { status: 200, _raw: true, headers: { 'content-type': b.type } })) : reject(new TypeError('Failed to fetch')); }
      const id = fetchSeq++;
      fetches.set(id, { resolve, reject, kind: 'fetch' });
      if (signal) signal.addEventListener('abort', () => { if (fetches.delete(id)) reject(signal.reason || new DOMException('The operation was aborted.', 'AbortError')); });
      const headers = []; req.headers.forEach((v, k) => headers.push([k, v]));
      H.fetch(id, req.method, req.url, headers, req._b, req.credentials, req.redirect);
    });
  }
  function hostFetchDone(id, error, status, statusText, url, headers, body, redirected) {
    const f = fetches.get(id); if (!f) return; fetches.delete(id);
    if (f.kind === 'xhr') return f.done(error, status, statusText, url, headers, body);
    if (error) return f.reject(new TypeError('Failed to fetch' + (error ? ' (' + error + ')' : '')));
    if (status === 0 && redirected) return f.resolve(new Response(null, { status: 0, type: 'opaqueredirect', url }));
    f.resolve(new Response(body, { status, statusText, headers, url, redirected, _raw: true }));
  }
  class AbortSignal extends EventTarget {
    constructor() { super(); this.aborted = false; this.reason = undefined; }
    throwIfAborted() { if (this.aborted) throw this.reason; }
    static abort(reason) { const c = new AbortController(); c.abort(reason); return c.signal; }
    static timeout(ms) { const c = new AbortController(); setTimeout(() => c.abort(new DOMException('The operation timed out.', 'TimeoutError')), ms); return c.signal; }
    static any(signals) { const c = new AbortController(); for (const s of signals) { if (s.aborted) { c.abort(s.reason); break; } s.addEventListener('abort', () => c.abort(s.reason)); } return c.signal; }
  }
  class AbortController {
    constructor() { this.signal = new AbortSignal(); }
    abort(reason) { const s = this.signal; if (s.aborted) return; s.aborted = true; s.reason = reason === undefined ? new DOMException('signal is aborted without reason', 'AbortError') : reason; const e = new Event('abort'); if (s.onabort) call(s.onabort, s, [e]); s.dispatchEvent(e); }
  }
  class XMLHttpRequest extends EventTarget {
    constructor() { super(); this.readyState = 0; this.status = 0; this.statusText = ''; this.responseText = ''; this.response = ''; this.responseType = ''; this.responseURL = ''; this.responseXML = null; this.timeout = 0; this.withCredentials = false; this._headers = []; this.upload = new EventTarget(); }
    open(method, url, async = true) { this._method = String(method).toUpperCase(); this._url = new URL(String(url), document.baseURI).href; this._async = async !== false; this._headers = []; this._resp = null; this._set(1); }
    setRequestHeader(k, v) { this._headers.push([String(k), String(v)]); }
    overrideMimeType() {}
    getResponseHeader(k) { if (!this._resp) return null; return this._resp.get(k); }
    getAllResponseHeaders() { if (!this._resp) return ''; let s = ''; this._resp.forEach((v, k) => { s += k + ': ' + v + '\r\n'; }); return s; }
    _set(s) { this.readyState = s; this._fire('readystatechange'); }
    _fire(t, init) { const e = t === 'readystatechange' ? new Event(t) : new ProgressEvent(t, init || {}); const h = this['on' + t]; if (h) call(h, this, [e]); this.dispatchEvent(e); }
    send(body = null) {
      if (this.readyState !== 1) throw new DOMException("Failed to execute 'send' on 'XMLHttpRequest': The object's state must be OPENED.", 'InvalidStateError');
      if (!this._async) H.log('warn', 'Synchronous XMLHttpRequest is not supported; it will complete asynchronously.');
      const id = fetchSeq++; this._id = id;
      const headers = new Headers(this._headers);
      const b = this._method === 'GET' || this._method === 'HEAD' ? null : bodyToHost(body, headers);
      if (!headers.has('accept')) headers.set('accept', '*/*');
      fetches.set(id, { kind: 'xhr', done: (error, status, statusText, url, hdrs, bytes) => this._done(error, status, statusText, url, hdrs, bytes) });
      if (this.timeout > 0) this._timer = setTimeout(() => { if (fetches.delete(id)) { this._set(4); this._fire('timeout'); this._fire('loadend'); } }, this.timeout);
      this._fire('loadstart');
      const list = []; headers.forEach((v, k) => list.push([k, v]));
      H.fetch(id, this._method, this._url, list, b, this.withCredentials ? 'include' : 'same-origin', 'follow');
    }
    _done(error, status, statusText, url, headers, bytes) {
      clearTimeout(this._timer);
      if (error) { this.status = 0; this._set(4); this._fire('error'); this._fire('loadend'); return; }
      this.status = status; this.statusText = statusText; this.responseURL = url; this._resp = new Headers(headers);
      this._set(2); this._set(3);
      const t = this.responseType;
      const text = t === '' || t === 'text' || t === 'json' || t === 'document' ? H.decode(bytes, 'utf-8') : '';
      if (t === '' || t === 'text') { this.responseText = text; this.response = text; }
      else if (t === 'json') { try { this.response = JSON.parse(text); } catch (e) { this.response = null; } }
      else if (t === 'arraybuffer') this.response = bytes.slice().buffer;
      else if (t === 'blob') { const bl = new Blob([], { type: this._resp.get('content-type') || '' }); bl._bytes = bytes; this.response = bl; }
      else if (t === 'document') { this.response = this.responseXML = createDetachedDocument(text); }
      this._set(4);
      const n = bytes ? bytes.length : 0;
      this._fire('progress', { loaded: n, total: n, lengthComputable: true }); this._fire('load', { loaded: n, total: n }); this._fire('loadend', { loaded: n, total: n });
    }
    abort() { if (this._id && fetches.delete(this._id)) { this._set(4); this.status = 0; this._fire('abort'); this._fire('loadend'); } this.readyState = 0; }
  }
  define(XMLHttpRequest, { UNSENT: 0, OPENED: 1, HEADERS_RECEIVED: 2, LOADING: 3, DONE: 4 });
  class WebSocket extends EventTarget {
    constructor(url) { super(); this.url = String(url); this.readyState = 3; this.bufferedAmount = 0; this.protocol = ''; this.extensions = ''; this.binaryType = 'blob'; setTimeout(() => { const e = new Event('error'); if (this.onerror) call(this.onerror, this, [e]); this.dispatchEvent(e); const c = new CloseEvent('close', { code: 1006 }); if (this.onclose) call(this.onclose, this, [c]); this.dispatchEvent(c); }, 0); }
    send() { throw new DOMException('WebSocket is not supported in this browser yet', 'InvalidStateError'); } close() {}
  }
  define(WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  class CloseEvent extends Event { constructor(t, i = {}) { super(t, i); this.code = i.code || 1000; this.reason = i.reason || ''; this.wasClean = !!i.wasClean; } }
  class EventSource extends EventTarget { constructor(url) { super(); this.url = String(url); this.readyState = 2; this.withCredentials = false; } close() {} }
  class BroadcastChannel extends EventTarget { constructor(name) { super(); this.name = name; } postMessage() {} close() {} }
  class MessageChannel { constructor() { this.port1 = new MessagePort(); this.port2 = new MessagePort(); this.port1._other = this.port2; this.port2._other = this.port1; } }
  class MessagePort extends EventTarget { postMessage(data) { const o = this._other; setTimeout(() => { const e = new MessageEvent('message', { data }); if (o.onmessage) call(o.onmessage, o, [e]); o.dispatchEvent(e); }, 0); } start() {} close() {} }
  class Worker extends EventTarget { constructor() { super(); throw new DOMException('Web Workers are not supported in this browser yet', 'NotSupportedError'); } }

  // ---------------------------------------------------------------- Storage
  class Storage {
    constructor(kind) { Object.defineProperty(this, '_k', { value: kind }); }
    getItem(k) { return H.storage(this._k, 'get', String(k), ''); }
    setItem(k, v) { H.storage(this._k, 'set', String(k), String(v)); }
    removeItem(k) { H.storage(this._k, 'remove', String(k), ''); }
    clear() { H.storage(this._k, 'clear', '', ''); }
    key(i) { return H.storage(this._k, 'keys', '', '')[i] || null; }
    get length() { return H.storage(this._k, 'length', '', ''); }
  }
  const storageProxy = (kind) => new Proxy(new Storage(kind), {
    get(t, k) { if (typeof k === 'symbol' || k in t) { const v = t[k]; return typeof v === 'function' ? v.bind(t) : v; } return t.getItem(k) === null ? undefined : t.getItem(k); },
    set(t, k, v) { t.setItem(k, v); return true; },
    deleteProperty(t, k) { t.removeItem(k); return true; },
    has(t, k) { return k in t || t.getItem(k) !== null; },
    ownKeys(t) { return H.storage(t._k, 'keys', '', ''); },
    getOwnPropertyDescriptor(t, k) { const v = t.getItem(k); return v === null ? undefined : { value: v, enumerable: true, configurable: true, writable: true }; },
  });
  const localStorage = storageProxy(0);
  const sessionStorage = storageProxy(1);

  // ---------------------------------------------------------------- Location, history, navigator, screen
  class Location {
    get href() { return H.url(); } set href(v) { H.navigate(new URL(String(v), document.baseURI).href, false); }
    assign(v) { this.href = v; } replace(v) { H.navigate(new URL(String(v), document.baseURI).href, true); } reload() { H.navigate(H.url(), true); }
    toString() { return H.url(); } get ancestorOrigins() { return []; }
  }
  for (const part of ['protocol', 'host', 'hostname', 'port', 'pathname', 'search', 'hash', 'origin', 'username', 'password']) {
    Object.defineProperty(Location.prototype, part, { enumerable: true, get() { return new URL(H.url())[part]; }, set(v) { const u = new URL(H.url()); u[part] = v; H.navigate(u.href, false); } });
  }
  docLocation = new Location();
  let historyState = null;
  let historyLength = 1;
  const history = {
    get length() { return historyLength; }, get state() { return historyState; }, scrollRestoration: 'auto',
    pushState(state, _t, url) { historyState = state === undefined ? null : JSON.parse(JSON.stringify(state)); historyLength++; H.history(0, url == null ? H.url() : String(url), JSON.stringify(historyState)); },
    replaceState(state, _t, url) { historyState = state === undefined ? null : JSON.parse(JSON.stringify(state)); H.history(1, url == null ? H.url() : String(url), JSON.stringify(historyState)); },
    back() { H.history(2, '-1', ''); }, forward() { H.history(2, '1', ''); }, go(n = 0) { H.history(2, String(n | 0), ''); },
  };
  const navigator = {
    userAgent: UA, appVersion: identity.appVersion, appName: 'Netscape', appCodeName: 'Mozilla', product: 'Gecko', productSub: '20030107', vendor: identity.vendor, vendorSub: '',
    platform: 'Linux x86_64', language: 'en-US', languages: ['en-US', 'en'], onLine: true, cookieEnabled: true, doNotTrack: null, hardwareConcurrency: 4, deviceMemory: 8, maxTouchPoints: 0,
    webdriver: false, pdfViewerEnabled: false, plugins: Object.assign([], { item() { return null; }, namedItem() { return null; }, refresh() {} }), mimeTypes: Object.assign([], { item() { return null; }, namedItem() { return null; } }),
    javaEnabled: () => false, sendBeacon(url, data) { const h = new Headers(); const b = bodyToHost(data == null ? null : data, h); const l = []; h.forEach((v, k) => l.push([k, v])); H.fetch(fetchSeq++, 'POST', new URL(String(url), document.baseURI).href, l, b, 'include', 'follow'); return true; },
    vibrate: () => false, share: () => Promise.reject(new DOMException('Share is not supported', 'NotAllowedError')), canShare: () => false,
    clipboard: { writeText: (t) => { H.dialog('clipboard', String(t)); return Promise.resolve(); }, readText: () => Promise.reject(new DOMException('Clipboard read is not allowed', 'NotAllowedError')), write: () => Promise.resolve(), read: () => Promise.resolve([]) },
    permissions: { query: (p) => Promise.resolve({ name: p && p.name, state: 'prompt', onchange: null, addEventListener() {}, removeEventListener() {} }) },
    userAgentData: { brands: [{ brand: 'Chromium', version: '131' }, { brand: 'Not_A Brand', version: '24' }], mobile: false, platform: 'Linux', getHighEntropyValues: () => Promise.resolve({ platform: 'Linux', architecture: 'x86', bitness: '64', model: '', platformVersion: '', uaFullVersion: '131.0.0.0', fullVersionList: [] }), toJSON() { return { brands: this.brands, mobile: false, platform: 'Linux' }; } },
    connection: { effectiveType: '4g', downlink: 10, rtt: 50, saveData: false, type: 'wifi', addEventListener() {}, removeEventListener() {} },
    geolocation: { getCurrentPosition: (ok, err) => err && setTimeout(() => err({ code: 1, message: 'User denied Geolocation' }), 0), watchPosition: () => 0, clearWatch() {} },
    mediaDevices: undefined, serviceWorker: undefined, storage: { estimate: () => Promise.resolve({ quota: 5242880, usage: 0 }), persist: () => Promise.resolve(false), persisted: () => Promise.resolve(false) },
    locks: { request: (n, o, cb) => Promise.resolve((typeof o === 'function' ? o : cb)({ name: n })), query: () => Promise.resolve({ held: [], pending: [] }) },
    getGamepads: () => [], requestMediaKeySystemAccess: () => Promise.reject(new DOMException('Not supported', 'NotSupportedError')),
    registerProtocolHandler() {}, getBattery: () => Promise.resolve({ charging: true, level: 1, chargingTime: 0, dischargingTime: Infinity, addEventListener() {} }),
  };
  if (!identity.chromium) {
    delete navigator.userAgentData;
    delete navigator.deviceMemory;
    delete navigator.connection;
  }
  const screen = {
    get width() { return Math.max(H.viewport()[0], 1280); }, get height() { return Math.max(H.viewport()[1], 800); }, get availWidth() { return this.width; }, get availHeight() { return this.height; },
    availLeft: 0, availTop: 0, colorDepth: 24, pixelDepth: 24, isExtended: false,
    orientation: { type: 'landscape-primary', angle: 0, addEventListener() {}, removeEventListener() {}, lock: () => Promise.reject(new DOMException('Not supported', 'NotSupportedError')), unlock() {} },
  };
  const timeOrigin = Date.now() - H.now();
  const marks = [];
  const performance = {
    now: () => H.now(), timeOrigin,
    mark(name, opts) { const m = { name, entryType: 'mark', startTime: H.now(), duration: 0, detail: opts && opts.detail }; marks.push(m); return m; },
    measure(name, a, b) { const s = typeof a === 'string' ? (marks.find(m => m.name === a) || { startTime: 0 }).startTime : 0; const e = typeof b === 'string' ? (marks.find(m => m.name === b) || { startTime: H.now() }).startTime : H.now(); const m = { name, entryType: 'measure', startTime: s, duration: e - s }; marks.push(m); return m; },
    getEntries: () => marks.slice(), getEntriesByType: (t) => t === 'navigation' ? [{ name: H.url(), entryType: 'navigation', startTime: 0, duration: 0, type: 'navigate', domContentLoadedEventEnd: 0, loadEventEnd: 0, transferSize: 0 }] : marks.filter(m => m.entryType === t),
    getEntriesByName: (n) => marks.filter(m => m.name === n), clearMarks() { marks.length = 0; }, clearMeasures() {}, clearResourceTimings() {}, setResourceTimingBufferSize() {},
    timing: { navigationStart: timeOrigin, fetchStart: timeOrigin, domLoading: timeOrigin, domInteractive: 0, domContentLoadedEventStart: 0, domContentLoadedEventEnd: 0, domComplete: 0, loadEventStart: 0, loadEventEnd: 0, responseStart: timeOrigin, responseEnd: timeOrigin, requestStart: timeOrigin, connectStart: timeOrigin, connectEnd: timeOrigin, domainLookupStart: timeOrigin, domainLookupEnd: timeOrigin, unloadEventStart: 0, unloadEventEnd: 0, redirectStart: 0, redirectEnd: 0, secureConnectionStart: 0 },
    navigation: { type: 0, redirectCount: 0 }, memory: { jsHeapSizeLimit: 2172649472, totalJSHeapSize: 10000000, usedJSHeapSize: 5000000 },
    addEventListener() {}, removeEventListener() {}, toJSON() { return {}; },
  };
  const crypto = {
    getRandomValues(arr) {
      if (!ArrayBuffer.isView(arr)) throw new TypeError("Failed to execute 'getRandomValues' on 'Crypto': parameter 1 is not of type 'ArrayBufferView'.");
      if (arr.byteLength > 65536) throw new DOMException('The ArrayBufferView byte length exceeds 65536', 'QuotaExceededError');
      const bytes = H.random(arr.byteLength); new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength).set(bytes); return arr;
    },
    randomUUID() { const b = H.random(16); b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80; const h = Array.from(b, x => x.toString(16).padStart(2, '0')).join(''); return h.slice(0, 8) + '-' + h.slice(8, 12) + '-' + h.slice(12, 16) + '-' + h.slice(16, 20) + '-' + h.slice(20); },
    subtle: { digest: () => Promise.reject(new DOMException('SubtleCrypto is not supported yet', 'NotSupportedError')), importKey: () => Promise.reject(new DOMException('SubtleCrypto is not supported yet', 'NotSupportedError')), generateKey: () => Promise.reject(new DOMException('SubtleCrypto is not supported yet', 'NotSupportedError')) },
  };
  function matchMedia(q) {
    const query = String(q);
    const evaluate = () => {
      const [w, h, , , dpr] = H.viewport();
      return query.split(',').some(part => {
        let p = part.trim().toLowerCase(); let neg = false;
        if (p.startsWith('not ')) { neg = true; p = p.slice(4); }
        p = p.replace(/^only\s+/, '');
        const conds = p.split(/\s+and\s+/);
        const ok = conds.every(c => {
          c = c.trim();
          if (c === 'all' || c === 'screen' || c === '') return true;
          if (c === 'print' || c === 'speech') return false;
          const m = /^\(\s*([a-z-]+)\s*(?::\s*([^)]+))?\)$/.exec(c);
          if (!m) return false;
          const f = m[1], v = (m[2] || '').trim();
          const px = (s) => { const n = parseFloat(s); return /em$/.test(s) ? n * 16 : n; };
          switch (f) {
            case 'min-width': return w >= px(v); case 'max-width': return w <= px(v);
            case 'min-height': return h >= px(v); case 'max-height': return h <= px(v);
            case 'width': return Math.abs(w - px(v)) < 1; case 'height': return Math.abs(h - px(v)) < 1;
            case 'orientation': return v === (w >= h ? 'landscape' : 'portrait');
            case 'prefers-color-scheme': return v === 'light';
            case 'prefers-reduced-motion': return v === 'no-preference' || v === '';
            case 'prefers-contrast': return v === 'no-preference';
            case 'prefers-reduced-transparency': case 'prefers-reduced-data': return v === 'no-preference';
            case 'forced-colors': return v === 'none';
            case 'hover': case 'any-hover': return v === '' || v === 'hover';
            case 'pointer': case 'any-pointer': return v === '' || v === 'fine';
            case 'color': return true; case 'monochrome': return false; case 'display-mode': return v === 'browser'; case 'scripting': return v === 'enabled';
            case 'min-resolution': return dpr * 96 >= parseFloat(v) * (/dppx|x$/.test(v) ? 96 : 1);
            case 'max-resolution': return dpr * 96 <= parseFloat(v) * (/dppx|x$/.test(v) ? 96 : 1);
            case '-webkit-min-device-pixel-ratio': case 'min-device-pixel-ratio': return dpr >= parseFloat(v);
            case '-webkit-max-device-pixel-ratio': case 'max-device-pixel-ratio': return dpr <= parseFloat(v);
            case 'min-aspect-ratio': { const [a, b] = v.split('/').map(Number); return w / h >= a / (b || 1); }
            case 'max-aspect-ratio': { const [a, b] = v.split('/').map(Number); return w / h <= a / (b || 1); }
            default: return false;
          }
        });
        return neg ? !ok : ok;
      });
    };
    const mql = Object.create(MediaQueryList.prototype);
    Object.defineProperty(mql, 'matches', { get: evaluate });
    mql.media = query; mql.onchange = null;
    mqls.push(mql);
    return mql;
  }
  class MediaQueryList extends EventTarget { addListener(cb) { this.addEventListener('change', cb); } removeListener(cb) { this.removeEventListener('change', cb); } }
  const mqls = [];
  function getComputedStyle(el, pseudo) { if (!el || el[ID] === undefined) throw new TypeError("Failed to execute 'getComputedStyle' on 'Window': parameter 1 is not of type 'Element'."); return computedProxy(el); }
  const fontFaceSet = Object.assign(new EventTarget(), { ready: Promise.resolve(), status: 'loaded', size: 0, load: () => Promise.resolve([]), check: () => true, add() {}, delete() {}, clear() {}, forEach() {}, has: () => false, [Symbol.iterator]: function* () {} });
  class FontFace { constructor(family, source, desc) { this.family = family; this.status = 'loaded'; this.loaded = Promise.resolve(this); } load() { return Promise.resolve(this); } }
  const CSS = { supports: (p, v) => v === undefined ? H.supports(String(p), '') : H.supports(String(p), String(v)), escape: cssEscape, registerProperty() {}, px: (n) => n + 'px', number: (n) => n };
  const consoleObj = {};
  const counters = {}; const timersC = {}; let groupDepth = 0;
  const out = (level) => (...a) => H.log(level, '  '.repeat(groupDepth) + a.map(x => fmt(x)).join(' '));
  Object.assign(consoleObj, {
    log: out('log'), info: out('info'), warn: out('warn'), error: out('error'), debug: out('debug'), trace: out('debug'), dir: out('log'), dirxml: out('log'), table: out('log'),
    assert: (c, ...a) => { if (!c) H.log('error', 'Assertion failed: ' + a.map(x => fmt(x)).join(' ')); },
    count: (l = 'default') => { counters[l] = (counters[l] || 0) + 1; H.log('log', l + ': ' + counters[l]); }, countReset: (l = 'default') => { counters[l] = 0; },
    time: (l = 'default') => { timersC[l] = H.now(); }, timeLog: (l = 'default') => H.log('log', l + ': ' + (H.now() - (timersC[l] || 0)).toFixed(1) + ' ms'),
    timeEnd: (l = 'default') => { H.log('log', l + ': ' + (H.now() - (timersC[l] || 0)).toFixed(1) + ' ms'); delete timersC[l]; },
    group: (...a) => { if (a.length) out('log')(...a); groupDepth++; }, groupCollapsed: (...a) => { if (a.length) out('log')(...a); groupDepth++; }, groupEnd: () => { groupDepth = Math.max(0, groupDepth - 1); },
    clear() {}, profile() {}, profileEnd() {}, timeStamp() {},
  });
  function structuredClone(v) {
    if (v === undefined) return undefined;
    const seen = new Map();
    const clone = (x) => {
      if (x === null || typeof x !== 'object') { if (typeof x === 'function' || typeof x === 'symbol') throw new DOMException('The object could not be cloned.', 'DataCloneError'); return x; }
      if (seen.has(x)) return seen.get(x);
      if (x instanceof Date) return new Date(x.getTime());
      if (x instanceof RegExp) return new RegExp(x.source, x.flags);
      if (ArrayBuffer.isView(x)) return x.slice();
      if (x instanceof ArrayBuffer) return x.slice(0);
      if (x instanceof Map) { const m = new Map(); seen.set(x, m); x.forEach((v, k) => m.set(clone(k), clone(v))); return m; }
      if (x instanceof Set) { const s = new Set(); seen.set(x, s); x.forEach(v => s.add(clone(v))); return s; }
      if (x instanceof Blob) return x;
      if (x[ID] !== undefined) throw new DOMException('The object could not be cloned.', 'DataCloneError');
      const o = Array.isArray(x) ? [] : {}; seen.set(x, o);
      for (const k of Object.keys(x)) o[k] = clone(x[k]);
      return o;
    };
    return clone(v);
  }

  // ---------------------------------------------------------------- Intl (minimal; Boa is built without ICU data)
  if (typeof G.Intl === 'undefined') {
    const pad = (n, l = 2) => String(n).padStart(l, '0');
    const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    class NumberFormat {
      constructor(locale, o = {}) { this.o = o; }
      format(n) {
        const o = this.o; n = Number(n);
        if (!isFinite(n)) return String(n);
        const min = o.minimumFractionDigits !== undefined ? o.minimumFractionDigits : (o.style === 'currency' ? 2 : 0);
        const max = o.maximumFractionDigits !== undefined ? Math.max(o.maximumFractionDigits, min) : (o.style === 'currency' ? 2 : o.style === 'percent' ? 0 : Math.max(min, 3));
        let v = o.style === 'percent' ? n * 100 : n;
        if (o.notation === 'compact') { const units = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]; for (const [d, s] of units) if (Math.abs(v) >= d) { const r = v / d; return (Math.round(r * 10) / 10).toString() + s; } }
        let s = v.toFixed(max);
        if (s.includes('.')) { s = s.replace(/0+$/, ''); const dec = s.split('.')[1] || ''; if (dec.length < min) s = (s.endsWith('.') ? s : s) + '0'.repeat(min - dec.length); if (s.endsWith('.')) s = s.slice(0, -1); }
        if (min > 0 && !s.includes('.')) s += '.' + '0'.repeat(min);
        const [i, d] = s.split('.');
        const grouped = o.useGrouping === false ? i : i.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        let r = d ? grouped + '.' + d : grouped;
        if (o.style === 'currency') { const sym = { USD: '$', EUR: '€', GBP: '£', JPY: '¥', INR: '₹', CAD: 'CA$', AUD: 'A$' }[o.currency] || (o.currency + ' '); r = r.startsWith('-') ? '-' + sym + r.slice(1) : sym + r; }
        if (o.style === 'percent') r += '%';
        if (o.style === 'unit' && o.unit) r += ' ' + o.unit;
        return r;
      }
      formatToParts(n) { return [{ type: 'literal', value: this.format(n) }]; }
      resolvedOptions() { return Object.assign({ locale: 'en-US', numberingSystem: 'latn' }, this.o); }
    }
    class DateTimeFormat {
      constructor(locale, o = {}) { this.o = o; }
      format(d) {
        d = d === undefined ? new Date() : new Date(d);
        const o = this.o; const parts = [];
        const has = (k) => o[k] !== undefined;
        if (o.dateStyle || o.timeStyle) {
          if (o.dateStyle === 'full') parts.push(DAYS[d.getDay()] + ', ' + MONTHS[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear());
          else if (o.dateStyle === 'long') parts.push(MONTHS[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear());
          else if (o.dateStyle === 'medium') parts.push(MONTHS[d.getMonth()].slice(0, 3) + ' ' + d.getDate() + ', ' + d.getFullYear());
          else if (o.dateStyle) parts.push((d.getMonth() + 1) + '/' + d.getDate() + '/' + String(d.getFullYear()).slice(2));
          if (o.timeStyle) { const h = d.getHours() % 12 || 12; parts.push(h + ':' + pad(d.getMinutes()) + (o.timeStyle !== 'short' ? ':' + pad(d.getSeconds()) : '') + ' ' + (d.getHours() < 12 ? 'AM' : 'PM')); }
          return parts.join(', ');
        }
        if (!has('year') && !has('month') && !has('day') && !has('hour') && !has('minute') && !has('weekday')) return (d.getMonth() + 1) + '/' + d.getDate() + '/' + d.getFullYear();
        const date = [];
        if (has('weekday')) date.push(o.weekday === 'long' ? DAYS[d.getDay()] : DAYS[d.getDay()].slice(0, 3));
        if (has('month') && (o.month === 'long' || o.month === 'short')) { const m = o.month === 'long' ? MONTHS[d.getMonth()] : MONTHS[d.getMonth()].slice(0, 3); date.push(m + (has('day') ? ' ' + d.getDate() : '') + (has('year') ? ', ' + d.getFullYear() : '')); }
        else if (has('month') || has('day') || has('year')) date.push([has('month') ? (o.month === '2-digit' ? pad(d.getMonth() + 1) : d.getMonth() + 1) : null, has('day') ? (o.day === '2-digit' ? pad(d.getDate()) : d.getDate()) : null, has('year') ? (o.year === '2-digit' ? String(d.getFullYear()).slice(2) : d.getFullYear()) : null].filter(x => x !== null).join('/'));
        if (has('hour') || has('minute')) { const h12 = o.hour12 !== false && o.hourCycle !== 'h23'; const h = h12 ? (d.getHours() % 12 || 12) : pad(d.getHours()); date.push((has('hour') ? h : '') + (has('minute') ? ':' + pad(d.getMinutes()) : '') + (has('second') ? ':' + pad(d.getSeconds()) : '') + (h12 && has('hour') ? ' ' + (d.getHours() < 12 ? 'AM' : 'PM') : '')); }
        return date.join(', ');
      }
      formatToParts(d) { return [{ type: 'literal', value: this.format(d) }]; }
      formatRange(a, b) { return this.format(a) + ' – ' + this.format(b); }
      resolvedOptions() { return Object.assign({ locale: 'en-US', calendar: 'gregory', numberingSystem: 'latn', timeZone: 'UTC' }, this.o); }
    }
    class RelativeTimeFormat { constructor(l, o = {}) { this.o = o; } format(v, unit) { unit = String(unit).replace(/s$/, ''); const n = Math.abs(v); if (this.o.numeric === 'auto' && v === -1 && unit === 'day') return 'yesterday'; if (this.o.numeric === 'auto' && v === 1 && unit === 'day') return 'tomorrow'; const s = n + ' ' + unit + (n === 1 ? '' : 's'); return v < 0 ? s + ' ago' : 'in ' + s; } formatToParts(v, u) { return [{ type: 'literal', value: this.format(v, u) }]; } resolvedOptions() { return { locale: 'en-US' }; } }
    class PluralRules { constructor(l, o = {}) { this.o = o; } select(n) { if (this.o.type === 'ordinal') { const m10 = n % 10, m100 = n % 100; if (m10 === 1 && m100 !== 11) return 'one'; if (m10 === 2 && m100 !== 12) return 'two'; if (m10 === 3 && m100 !== 13) return 'few'; return 'other'; } return n === 1 ? 'one' : 'other'; } resolvedOptions() { return { locale: 'en-US', pluralCategories: ['one', 'other'] }; } }
    class Collator { constructor(l, o = {}) { this.o = o; } compare(a, b) { a = String(a); b = String(b); if (this.o.sensitivity === 'base' || this.o.sensitivity === 'accent') { a = a.toLowerCase(); b = b.toLowerCase(); } if (this.o.numeric) { const na = parseFloat(a), nb = parseFloat(b); if (!isNaN(na) && !isNaN(nb) && na !== nb) return na - nb; } return a < b ? -1 : a > b ? 1 : 0; } resolvedOptions() { return { locale: 'en-US' }; } }
    class ListFormat { constructor(l, o = {}) { this.o = o; } format(list) { const a = [...list]; if (a.length < 2) return a.join(''); const word = this.o.type === 'disjunction' ? 'or' : 'and'; return a.length === 2 ? a[0] + ' ' + word + ' ' + a[1] : a.slice(0, -1).join(', ') + ', ' + word + ' ' + a[a.length - 1]; } }
    class Segmenter { constructor(l, o = {}) { this.o = o; } segment(s) { const g = this.o.granularity || 'grapheme'; const parts = g === 'word' ? String(s).split(/(\s+)/).filter(Boolean) : g === 'sentence' ? String(s).split(/(?<=[.!?])\s+/) : [...String(s)]; let i = 0; return parts.map(p => { const r = { segment: p, index: i, input: s, isWordLike: g === 'word' ? /\w/.test(p) : undefined }; i += p.length; return r; }); } }
    class DisplayNames { constructor(l, o = {}) { this.o = o; } of(code) { return code; } }
    class Locale { constructor(tag) { this.baseName = String(tag); this.language = String(tag).split('-')[0]; this.region = String(tag).split('-')[1]; } toString() { return this.baseName; } maximize() { return this; } minimize() { return this; } }
    G.Intl = { NumberFormat, DateTimeFormat, RelativeTimeFormat, PluralRules, Collator, ListFormat, Segmenter, DisplayNames, Locale, getCanonicalLocales: (l) => [].concat(l || []), supportedValuesOf: () => [] };
    for (const k of ['NumberFormat', 'DateTimeFormat', 'Collator', 'PluralRules', 'RelativeTimeFormat', 'ListFormat']) G.Intl[k].supportedLocalesOf = (l) => [].concat(l || []);
    Number.prototype.toLocaleString = function (l, o) { return new NumberFormat(l, o).format(this); };
    Date.prototype.toLocaleDateString = function (l, o) { return new DateTimeFormat(l, Object.assign({ year: 'numeric', month: 'numeric', day: 'numeric' }, o)).format(this); };
    Date.prototype.toLocaleTimeString = function (l, o) { return new DateTimeFormat(l, Object.assign({ hour: 'numeric', minute: '2-digit', second: '2-digit' }, o)).format(this); };
    Date.prototype.toLocaleString = function (l, o) { return o ? new DateTimeFormat(l, o).format(this) : this.toLocaleDateString(l) + ', ' + this.toLocaleTimeString(l); };
    String.prototype.localeCompare = function (b, l, o) { return new Collator(l, o).compare(this, b); };
  }

  // ---------------------------------------------------------------- Window
  const win = {
    window: G, self: G, globalThis: G, frames: G, parent: G, top: G, opener: null, frameElement: null, closed: false, length: 0, name: '', origin: '',
    document, location: docLocation, history, navigator, clientInformation: navigator, screen, performance, crypto, localStorage, sessionStorage, console: consoleObj, customElements, CSS,
    setTimeout, setInterval, clearTimeout, clearInterval: clearTimeout, requestAnimationFrame, cancelAnimationFrame, requestIdleCallback, cancelIdleCallback: clearTimeout, queueMicrotask,
    atob, btoa, fetch, matchMedia, getComputedStyle, structuredClone, reportError,
    Event, CustomEvent, UIEvent, FocusEvent, MouseEvent, PointerEvent, WheelEvent, KeyboardEvent, InputEvent, CompositionEvent, SubmitEvent, ErrorEvent, ProgressEvent, MessageEvent,
    PopStateEvent, HashChangeEvent, PageTransitionEvent, StorageEvent, PromiseRejectionEvent, AnimationEvent, TransitionEvent, ClipboardEvent, DragEvent, TouchEvent, CloseEvent,
    EventTarget, Node, Element, HTMLElement, SVGElement, SVGSVGElement, CharacterData, Text, Comment, CDATASection, ProcessingInstruction, Document, HTMLDocument, XMLDocument, DocumentFragment, ShadowRoot,
    Attr, NamedNodeMap, DOMTokenList, NodeList, HTMLCollection, CSSStyleDeclaration, CSSStyleSheet, StyleSheet, CSSRule, DOMRect, DOMRectReadOnly, DOMPoint, DOMPointReadOnly: DOMPoint, DOMMatrix, DOMMatrixReadOnly: DOMMatrix, DOMException,
    DOMParser, XMLSerializer, Range, TreeWalker, NodeFilter, Animation, KeyframeEffect, MutationObserver, IntersectionObserver, ResizeObserver, PerformanceObserver,
    TextEncoder, TextDecoder, URL, URLSearchParams, Blob, File, FileReader, FormData, Headers, Request, Response, AbortController, AbortSignal, XMLHttpRequest,
    ReadableStream, WritableStream, TransformStream, WebSocket, EventSource, BroadcastChannel, MessageChannel, MessagePort, Worker, Storage, Location, MediaQueryList, FontFace,
    Image: function Image(w, h) { const img = document.createElement('img'); if (w !== undefined) img.width = w; if (h !== undefined) img.height = h; return img; },
    Audio: function Audio(src) { const a = document.createElement('audio'); if (src) a.src = src; return a; },
    Option: function Option(text = '', value, defSel = false, sel = false) { const o = document.createElement('option'); o.textContent = text; if (value !== undefined) o.value = value; if (defSel) o.setAttribute('selected', ''); return o; },
    alert: (m) => H.dialog('alert', m === undefined ? '' : String(m)),
    confirm: (m) => { H.dialog('confirm', m === undefined ? '' : String(m)); return true; },
    prompt: (m, d) => { H.dialog('prompt', m === undefined ? '' : String(m)); return d === undefined ? null : String(d); },
    print() {}, stop() {}, focus() {}, blur() {}, close() {}, moveTo() {}, moveBy() {}, resizeTo() {}, resizeBy() {}, captureEvents() {}, releaseEvents() {},
    open(url, target) { if (url) H.navigate(new URL(String(url), document.baseURI).href, false); return G; },
    postMessage(data, targetOrigin) { H.postMessage(this == null ? G : this, data, String(targetOrigin == null ? '' : targetOrigin)); },
    getSelection: () => selection,
    scroll(x, y) { if (typeof x === 'object' && x) H.scrollTo(x.left === undefined ? NaN : x.left, x.top === undefined ? NaN : x.top); else H.scrollTo(x, y); },
    scrollTo(x, y) { G.scroll(x, y); },
    scrollBy(x, y) { const v = H.viewport(); if (typeof x === 'object' && x) H.scrollTo(v[2] + (x.left || 0), v[3] + (x.top || 0)); else H.scrollTo(v[2] + (x || 0), v[3] + (y || 0)); },
    addEventListener(...a) { return EventTarget.prototype.addEventListener.apply(this == null ? G : this, a); },
    removeEventListener(...a) { return EventTarget.prototype.removeEventListener.apply(this == null ? G : this, a); },
    dispatchEvent(e) { return EventTarget.prototype.dispatchEvent.call(this == null ? G : this, e); },
    isSecureContext: true, crossOriginIsolated: false, devicePixelRatio: 1, visualViewport: null, speechSynthesis: undefined, indexedDB: undefined, caches: undefined, trustedTypes: undefined,
  };
  Object.assign(win, C);
  for (const k of Object.keys(win)) Object.defineProperty(G, k, { value: win[k], writable: true, configurable: true, enumerable: false });
  getters(G, {
    innerWidth() { return H.viewport()[0]; }, innerHeight() { return H.viewport()[1]; }, outerWidth() { return H.viewport()[0]; }, outerHeight() { return H.viewport()[1] + 80; },
    scrollX() { return H.viewport()[2]; }, scrollY() { return H.viewport()[3]; }, pageXOffset() { return H.viewport()[2]; }, pageYOffset() { return H.viewport()[3]; },
    screenX() { return 0; }, screenY() { return 0; }, screenLeft() { return 0; }, screenTop() { return 0; },
    devicePixelRatio() { return H.viewport()[4]; }, origin() { return docLocation.origin; },
    event() { return currentEvent; },
  });
  Object.defineProperty(G, 'location', { get: () => docLocation, set: (v) => { docLocation.href = v; }, configurable: true });
  Object.defineProperty(G, 'name', { value: '', writable: true, configurable: true });
  Object.setPrototypeOf(G, Object.create(EventTarget.prototype));
  // Some code checks `window instanceof Window`.
  function Window() { throw new TypeError('Illegal constructor'); }
  Window.prototype = Object.getPrototypeOf(G);
  Object.defineProperty(G, 'Window', { value: Window, writable: true, configurable: true });
  // Cross-frame messaging: the top window is frame 0; the host delivers posted messages here.
  // Source is carried in the event, so delivering on a later turn keeps event.source correct.
  Object.defineProperty(G, '__frameId', { value: 0, enumerable: false, configurable: true });
  Object.defineProperty(G, '__deliverMessage', { value: (data, origin, source) => { setTimeout(() => dispatchTo(G, new MessageEvent('message', { data, origin, source })), 0); }, enumerable: false, configurable: true });
  let currentEvent;

  // ---------------------------------------------------------------- Engine entry points
  const EVENT_CLASS = { click: PointerEvent, dblclick: MouseEvent, contextmenu: PointerEvent, auxclick: PointerEvent, mousedown: MouseEvent, mouseup: MouseEvent, mousemove: MouseEvent, mouseover: MouseEvent, mouseout: MouseEvent, mouseenter: MouseEvent, mouseleave: MouseEvent, pointerdown: PointerEvent, pointerup: PointerEvent, pointermove: PointerEvent, pointerover: PointerEvent, pointerout: PointerEvent, pointerenter: PointerEvent, pointerleave: PointerEvent, pointercancel: PointerEvent, wheel: WheelEvent, keydown: KeyboardEvent, keyup: KeyboardEvent, keypress: KeyboardEvent, input: InputEvent, beforeinput: InputEvent, focus: FocusEvent, blur: FocusEvent, focusin: FocusEvent, focusout: FocusEvent, touchstart: TouchEvent, touchend: TouchEvent, touchmove: TouchEvent, touchcancel: TouchEvent, submit: SubmitEvent };
  const NO_BUBBLE = new Set(['focus', 'blur', 'mouseenter', 'mouseleave', 'pointerenter', 'pointerleave', 'load', 'scroll']);
  const evolve = {
    // Dispatch an engine-generated (trusted) event. Returns 1 when the default action was prevented.
    dispatch(type, target, init) {
      const node = target < 0 ? G : wrap(target);
      if (!node) return 0;
      const opts = init ? JSON.parse(init) : {};
      if (opts.submitter !== undefined) opts.submitter = wrap(opts.submitter);
      opts.bubbles = opts.bubbles !== undefined ? opts.bubbles : !NO_BUBBLE.has(type);
      opts.cancelable = opts.cancelable !== undefined ? opts.cancelable : !['input', 'focus', 'blur', 'focusin', 'focusout', 'mouseenter', 'mouseleave', 'scroll', 'load'].includes(type);
      opts.composed = true;
      if (opts.relatedTarget !== undefined) opts.relatedTarget = wrap(opts.relatedTarget);
      const cls = EVENT_CLASS[type] || Event;
      const ev = new cls(type, opts);
      ev._trusted = true;
      if (type === 'input' && opts.inputType === undefined) ev.inputType = 'insertText';
      const prev = currentEvent; currentEvent = ev;
      try { dispatchTo(node, ev); } finally { currentEvent = prev; }
      if (type === 'input' && node.localName && /^(input|textarea|select)$/.test(node.localName)) node._dirty = true;
      return ev.defaultPrevented ? 1 : 0;
    },
    // A form is about to submit through the engine's default action.
    submit(formId, submitterId) {
      const form = wrap(formId); if (!form) return 0;
      const ev = new SubmitEvent('submit', { bubbles: true, cancelable: true, submitter: submitterId >= 0 ? wrap(submitterId) : null });
      ev._trusted = true;
      return dispatchTo(form, ev) ? 0 : 1;
    },
    change(id, force) { const n = wrap(id); if (n && (n._dirty || force)) { n._dirty = false; const e = new Event('change', { bubbles: true }); e._trusted = true; dispatchTo(n, e); } },
    node(id) { return wrap(id); },
    blobText(url) { const b = blobUrls.get(url); return b ? H.decode(b._bytes, 'utf-8') : null; },
    fireTimer(id) {
      const t = timers.get(id); if (!t) return;
      if (t.interval) H.setTimer(id, t.ms); else timers.delete(id);
      if (typeof t.fn === 'function') call(t.fn, G, t.args);
      else if (typeof t.fn === 'string') { try { (0, eval)(t.fn); } catch (e) { reportError(e); } }
    },
    frames(ts) { const list = rafs; rafs = []; for (const r of list) call(r.cb, G, [ts]); },
    fetchDone: hostFetchDone,
    currentScript(id) { currentScript = id < 0 ? null : wrap(id); },
    scriptEvent(id, ok) { const el = wrap(id); if (!el) return; const e = new Event(ok ? 'load' : 'error'); e._trusted = true; dispatchTo(el, e); },
    docEvent(type) { const e = new Event(type, { bubbles: type !== 'readystatechange' }); e._trusted = true; dispatchTo(document, e); },
    winEvent(type, init) {
      const opts = init ? JSON.parse(init) : {};
      const cls = { popstate: PopStateEvent, hashchange: HashChangeEvent, pageshow: PageTransitionEvent, pagehide: PageTransitionEvent, storage: StorageEvent, message: MessageEvent }[type] || Event;
      if (type === 'popstate') historyState = opts.state === undefined ? null : opts.state;
      const e = new cls(type, opts); e._trusted = true; dispatchTo(G, e);
    },
    resized() { for (const m of mqls) { const e = new Event('change'); e.matches = m.matches; e.media = m.media; if (m.onchange) call(m.onchange, m, [e]); m.dispatchEvent(e); } evolve.winEvent('resize'); },
    historyLength(n) { historyLength = n; },
    rejection(reason) { const e = new PromiseRejectionEvent('unhandledrejection', { reason, cancelable: true }); dispatchTo(G, e); },
    upgrade() { if (customDefs.size && document.documentElement) upgradeTree(document.documentElement); },
    has(type) { const m = listeners.get(G); return !!(m && m.get(type) && m.get(type).length) || !!(handlers.get(G) && handlers.get(G)[type]); },
  };
  Object.defineProperty(G, '__evolve', { value: evolve, writable: false, configurable: false, enumerable: false });
})();
