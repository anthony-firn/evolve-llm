import { MODELS, LARGE, cacheKey, hfUrl, splitParts } from './models.js';

export class EngineError extends Error {
  constructor(code, message) { super(message); this.name = 'EngineError'; this.code = code; }
}
const fail = (code, message) => new EngineError(code, message);

/** One worker and one in-flight operation per engine. Internet tools require explicit per-call opt-in. */
export function createEngine({ workerURL = new URL('./worker.js', import.meta.url), workerFactory = url => new Worker(url, { type: 'module' }), timeoutMs = 120000 } = {}) {
  if (!Number.isFinite(timeoutMs) || timeoutMs < 1 || timeoutMs > 3600000) throw new TypeError('timeoutMs must be 1–3600000');
  let worker = null, operation = null, loaded = false, disposed = false;
  function disconnect() { worker?.terminate(); worker = null; loaded = false; }
  function attach() {
    if (disposed) throw fail('disposed', 'This engine has been disposed.');
    if (worker) return;
    try { worker = workerFactory(workerURL); } catch { throw fail('worker', 'The model worker could not be started. Serve its files from the same origin.'); }
    const owned = worker;
    worker.onmessage = ({ data }) => {
      if (worker !== owned) return;
      const op = operation;
      if (!op || !data || typeof data.type !== 'string') return;
      if (data.type === 'error') { op.end(fail('engine', 'The engine operation failed. Check model compatibility and available GPU memory.'), true); return; }
      if (op.events.has(data.type)) op.push(data);
      if (data.type === op.finish) op.end(null, false, data);
    };
    worker.onerror = event => { if (worker !== owned) return; event.preventDefault?.(); operation?.end(fail('worker', 'The model worker failed.'), true); disconnect(); };
    worker.onmessageerror = () => { if (worker !== owned) return; operation?.end(fail('protocol', 'The model worker returned an unreadable message.'), true); disconnect(); };
  }
  function start(message, finish, { signal, events = [], onProgress, stream = false } = {}) {
    if (operation) throw fail('busy', 'Wait for the current operation to finish.');
    if (signal?.aborted) throw fail('aborted', 'The operation was cancelled.');
    attach();
    let resolve, reject, wake = null, completed = false, error = null, result;
    const queue = [];
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    // Streams consume errors through next(); requests consume promise.
    promise.catch(() => {});
    const abort = () => op.end(fail('aborted', 'The operation was cancelled; reload the model to continue.'), true);
    const timer = setTimeout(() => op.end(fail('timeout', 'The operation timed out; reload the model to continue.'), true), timeoutMs);
    const op = {
      finish, events: new Set(events),
      push(data) {
        if (onProgress) { try { onProgress(data); } catch { /* consumer callbacks do not break worker cleanup */ } }
        if (stream) {
          if (queue.length >= 1024) { op.end(fail('backpressure', 'The event consumer fell behind; reload the model to continue.'), true); return; }
          queue.push(data); wake?.(); wake = null;
        }
      },
      end(reason, terminate, value) {
        if (completed) return;
        completed = true; error = reason; result = value;
        clearTimeout(timer); signal?.removeEventListener('abort', abort);
        if (operation === op) operation = null;
        if (terminate) disconnect();
        reason ? reject(reason) : resolve(value);
        wake?.(); wake = null;
      },
      async next() {
        while (!queue.length && !completed) await new Promise(yes => wake = yes);
        if (error) throw error;
        if (queue.length) return { value: queue.shift(), done: false };
        return { value: result, done: true };
      },
      async return() { if (!completed) abort(); return { done: true }; },
      [Symbol.asyncIterator]() { return this; },
      promise,
    };
    operation = op; signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    else { try { worker.postMessage(message); } catch { op.end(fail('protocol', 'The worker request could not be sent.'), true); } }
    return op;
  }
  return Object.freeze({
    async initialize({ gpu = 'high-performance', signal } = {}) {
      const result = await start({ type: 'hello', gpu }, 'adapter', { signal }).promise;
      if (!result.adapter?.ok) throw fail('unsupported_gpu', 'WebGPU is unavailable. Use a supported browser and GPU.');
      return result.adapter;
    },
    async load(model, { contextSize = 2048, gpu = 'high-performance', gpuMemoryMB = 'auto', signal, onProgress } = {}) {
      if (!Number.isInteger(contextSize) || contextSize < 256 || contextSize > 32768) throw new TypeError('contextSize must be 256–32768');
      let source;
      if (typeof model === 'string') {
        const m = [...MODELS, ...LARGE].find(m => m.id === model);
        if (!m) throw new TypeError('Unknown catalog model ID.');
        source = { kind: 'hf', id: m.id, key: cacheKey(m.repo, m.file), repo: m.repo, file: m.file,
          revision: 'main', url: hfUrl(m.repo, m.file), bytes: m.bytes, name: m.name, resolve: Boolean(splitParts(m.file)) };
      } else if (Array.isArray(model?.files) && model.files.length && model.files.every(f => typeof File !== 'undefined' && f instanceof File)) {
        source = { kind: 'file', files: model.files };
      } else throw new TypeError('Use a catalog model ID or { files: File[] }.');
      if (gpuMemoryMB !== 'auto' && (!Number.isInteger(gpuMemoryMB) || gpuMemoryMB < 1 || gpuMemoryMB > 1048576)) throw new TypeError('gpuMemoryMB must be auto or a positive MB budget');
      const pending = start({ type: 'load', source, ctx: contextSize, gpu, gpuMemory: gpuMemoryMB === 'auto' ? 'auto' : gpuMemoryMB / 1024 }, 'loaded', { signal, onProgress, events: ['download', 'progress', 'inspected'] });
      loaded = false;
      const result = await pending.promise;
      loaded = true; return result;
    },
    generate(messages, { signal, maxTokens = 256, temperature = 0.7, seed = 1, thinking = false, internet = { enabled: false } } = {}) {
      if (disposed) throw fail('disposed', 'This engine has been disposed.');
      if (!loaded) throw fail('not_loaded', 'Load a model before generating.');
      if (!Array.isArray(messages) || !messages.length || !messages.every(m => m && ['system', 'user', 'assistant'].includes(m.role) && typeof m.content === 'string'))
        throw new TypeError('messages must contain system, user or assistant text messages.');
      if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 32768 || !Number.isFinite(temperature) || temperature < 0 || temperature > 2 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
        throw new TypeError('Invalid token, temperature or seed setting.');
      if (typeof internet?.enabled !== 'boolean' || (internet.jinaKey !== undefined && (typeof internet.jinaKey !== 'string' || internet.jinaKey.length > 512 || /[\r\n]/.test(internet.jinaKey)))) throw new TypeError('Invalid internet options.');
      return start({ type: 'generate', messages, options: { enable_thinking: thinking }, maxTokens,
        sampler: { temperature, seed, top_p: 0.95, top_k: 40, min_p: 0.05, repeat_penalty: 1.1, repeat_last_n: 64 }, internet: { enabled: internet.enabled, ...(internet.jinaKey && {jinaKey:internet.jinaKey}) } }, 'done',
        { signal, stream: true, events: ['token', 'prefill', 'done', 'tool', 'tool-result', 'round'] });
    },
    reset() {
      if (disposed) throw fail('disposed', 'This engine has been disposed.');
      if (operation) throw fail('busy', 'Wait for the current operation to finish.');
      worker?.postMessage({ type: 'reset' });
    },
    dispose() {
      disposed = true;
      operation?.end(fail('disposed', 'This engine has been disposed.'), true);
      disconnect();
    },
  });
}
