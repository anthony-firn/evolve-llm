  // A realm-local WindowProxy facade. Raw globals never cross an origin boundary.
  const sourceFrame = Number(G.__frameId_value || 0);
  const sourceEpoch = Number(G.__frameEpoch_value || 0);
  delete G.__frameId_value; delete G.__frameEpoch_value;
  const windowProxies = new Map();
  const jsonEncode = JSON.stringify, jsonDecode = JSON.parse;
  const cloneError = () => new DOMException('This message value cannot be cloned by this browser.', 'DataCloneError');
  // Until the full structured-clone wire format is implemented, accept JSON-compatible
  // records/arrays and primitives, and reject other objects instead of sharing their heap.
  function encodeWindowMessage(data) {
    const seen = new Set();
    const copy = value => {
      if (value === undefined) return ['undefined'];
      if (value === null || typeof value === 'string' || typeof value === 'boolean') return ['value', value];
      if (typeof value === 'number') return ['number', String(value)];
      if (typeof value === 'bigint') return ['bigint', String(value)];
      if (typeof value !== 'object' || seen.has(value)) throw cloneError();
      const proto = Object.getPrototypeOf(value);
      if (!Array.isArray(value) && proto !== Object.prototype && proto !== null) throw cloneError();
      seen.add(value);
      const result = Array.isArray(value) ? ['array', value.map(copy)] : ['object', Object.keys(value).map(key => [key, copy(value[key])])];
      seen.delete(value);
      return result;
    };
    return jsonEncode(copy(data));
  }
  function decodeWindowMessage(wire) {
    const copy = value => {
      switch (value[0]) {
        case 'undefined': return undefined;
        case 'value': return value[1];
        case 'number': return Number(value[1]);
        case 'bigint': return BigInt(value[1]);
        case 'array': return value[1].map(copy);
        case 'object': {
          const result = {};
          for (const [key, item] of value[1]) Object.defineProperty(result, key, {value: copy(item), enumerable:true, writable:true, configurable:true});
          return result;
        }
        default: throw cloneError();
      }
    };
    return copy(jsonDecode(wire));
  }
  function postToWindow(id, data, options) {
    const targetOrigin = typeof options === 'object' && options !== null ? (options.targetOrigin === undefined ? '/' : String(options.targetOrigin)) : options === undefined ? '/' : String(options);
    // Transferables are not implemented; fail explicitly rather than leak an object reference.
    if (options && typeof options === 'object' && options.transfer && options.transfer.length) throw cloneError();
    H.postMessage(id, sourceFrame, sourceEpoch, encodeWindowMessage(data), targetOrigin);
  }
  function windowProxy(id) {
    if (id === null || id === undefined || id < 0) return null;
    if (id === sourceFrame) return G;
    if (windowProxies.has(id)) return windowProxies.get(id);
    const deny = () => { throw new DOMException('Blocked access to a cross-origin window.', 'SecurityError'); };
    const post = (data, targetOrigin) => postToWindow(id, data, targetOrigin);
    const proxy = new Proxy(Object.create(null), {
      get(_, key) {
        if (key === 'postMessage') return post;
        if (key === 'window' || key === 'self' || key === 'frames') return proxy;
        if (key === 'top' || key === 'parent') return windowProxy(0);
        if (key === 'closed') return false;
        if (key === 'length') return 0;
        if (key === 'opener' || key === 'frameElement') return null;
        if (key === 'then' || key === Symbol.toStringTag) return key === 'then' ? undefined : 'Window';
        if (!H.windowAccess(id, sourceFrame, sourceEpoch)) return deny();
        return H.windowGet(id, sourceFrame, sourceEpoch, key);
      },
      set(_, key, value) {
        if (!H.windowAccess(id, sourceFrame, sourceEpoch)) return deny();
        H.windowSet(id, sourceFrame, sourceEpoch, key, value); return true;
      },
      getPrototypeOf() { return null; },
      setPrototypeOf: deny, defineProperty: deny, deleteProperty: deny, preventExtensions: deny,
      ownKeys: deny, getOwnPropertyDescriptor: deny,
    });
    windowProxies.set(id, proxy);
    return proxy;
  }
