// Runs the WebAssembly engine off the main thread: downloads, GPU upload and generation.
import init, { create_loader, inspect } from './pkg/evolve_llm.js';
import { formatBytes } from './models.js';

const CHUNK = 16 << 20; // bytes read from the file per upload call
const post = (type, data = {}) => self.postMessage({ type, ...data });

let wasm = null;
let chat = null;
let stopRequested = false;
let cancelLoad = false;
let downloadAbort = null;

async function ensureWasm() {
  wasm ??= init();
  await wasm;
}

async function describeAdapter() {
  if (!('gpu' in navigator) || !navigator.gpu) return { ok: false, reason: 'no-webgpu' };
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return { ok: false, reason: 'no-adapter', detail: String(e?.message || e) };
  }
  if (!adapter) return { ok: false, reason: 'no-adapter' };
  const info = adapter.info || {};
  return {
    ok: true,
    vendor: info.vendor || '',
    architecture: info.architecture || '',
    device: info.device || '',
    description: info.description || '',
    fallback: Boolean(adapter.isFallbackAdapter ?? info.isFallbackAdapter),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBinding: adapter.limits.maxStorageBufferBindingSize,
  };
}

// ---- Model storage in the origin-private file system ----

async function modelsDir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('models', { create: true });
}

async function readText(dir, name) {
  try {
    const file = await (await dir.getFileHandle(name)).getFile();
    return await file.text();
  } catch {
    return null;
  }
}

async function writeText(dir, name, text) {
  const handle = await dir.getFileHandle(name, { create: true });
  const access = await handle.createSyncAccessHandle();
  try {
    const bytes = new TextEncoder().encode(text);
    access.truncate(0);
    access.write(bytes, { at: 0 });
    access.flush();
  } finally {
    access.close();
  }
}

async function readMeta(dir, key) {
  const text = await readText(dir, `${key}.json`);
  try { return text ? JSON.parse(text) : null; } catch { return null; }
}

async function listCached() {
  if (!navigator.storage?.getDirectory) return { models: [], supported: false };
  const dir = await modelsDir();
  const models = [];
  for await (const [name, handle] of dir.entries()) {
    if (handle.kind !== 'file' || !name.endsWith('.json')) continue;
    const key = name.slice(0, -5);
    const meta = await readMeta(dir, key);
    if (!meta) continue;
    let size = 0;
    try { size = (await (await dir.getFileHandle(`${key}.gguf`)).getFile()).size; } catch { /* missing data file */ }
    models.push({ ...meta, key, stored: size });
  }
  let estimate = null;
  try { estimate = await navigator.storage.estimate(); } catch { /* not available */ }
  let persisted = false;
  try { persisted = await navigator.storage.persisted?.(); } catch { /* not available */ }
  return { models, supported: true, usage: estimate?.usage ?? null, quota: estimate?.quota ?? null, persisted };
}

async function removeCached(key) {
  const dir = await modelsDir();
  for (const name of [`${key}.gguf`, `${key}.json`]) {
    try { await dir.removeEntry(name); } catch { /* already gone */ }
  }
}

function httpError(status) {
  if (status === 404) return 'Hugging Face does not have that file. Check the repository and file name.';
  if (status === 401 || status === 403) return 'That model needs a Hugging Face sign-in or license approval. Download it there, then open the file here.';
  if (status === 429) return 'Hugging Face is limiting downloads right now. Wait a minute and try again.';
  return `The download failed with HTTP ${status}.`;
}

async function download(entry) {
  if (!navigator.storage?.getDirectory) throw new Error('This browser cannot store downloaded models. Open a GGUF file from your device instead.');
  const dir = await modelsDir();
  const { key, url } = entry;
  const meta = await readMeta(dir, key);
  const handle = await dir.getFileHandle(`${key}.gguf`, { create: true });
  let have = (await handle.getFile()).size;
  if (meta?.complete && meta.url === url && have === meta.bytes) return handle.getFile();
  if (!meta || meta.url !== url) have = 0;

  const estimate = await navigator.storage.estimate().catch(() => null);
  const need = Math.max(0, (entry.bytes || 0) - have);
  if (estimate?.quota && need > estimate.quota - (estimate.usage || 0)) {
    throw new Error(`This browser can store ${formatBytes(estimate.quota - (estimate.usage || 0))} more, and the model needs ${formatBytes(need)}. Remove a downloaded model or open the file from your device.`);
  }
  try { await navigator.storage.persist?.(); } catch { /* best effort */ }

  downloadAbort = new AbortController();
  let res;
  try {
    res = await fetch(url, { headers: have > 0 ? { Range: `bytes=${have}-` } : {}, signal: downloadAbort.signal });
  } catch (e) {
    if (e?.name === 'AbortError') throw new Error('Download paused. Choose the model again to resume.');
    throw new Error('Could not reach Hugging Face. Check your connection and try again.');
  }
  if (!res.ok) throw new Error(httpError(res.status));
  let total;
  if (res.status === 206) {
    total = Number((res.headers.get('content-range') || '').split('/')[1]) || entry.bytes;
  } else {
    have = 0;
    total = Number(res.headers.get('content-length')) || entry.bytes || 0;
  }
  const record = { url, repo: entry.repo, file: entry.file, revision: entry.revision, name: entry.name, bytes: total, complete: false, started: meta?.started || Date.now() };
  await writeText(dir, `${key}.json`, JSON.stringify(record));
  const access = await handle.createSyncAccessHandle();
  let pos = have;
  try {
    if (have === 0) access.truncate(0);
    const reader = res.body.getReader();
    const t0 = performance.now();
    let last = 0;
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (e) {
        if (e?.name === 'AbortError') throw new Error('Download paused. Choose the model again to resume.');
        throw new Error('The connection dropped. Choose the model again to resume the download.');
      }
      if (chunk.done) break;
      access.write(chunk.value, { at: pos });
      pos += chunk.value.byteLength;
      const now = performance.now();
      if (now - last > 150) {
        last = now;
        post('progress', { phase: 'download', loaded: pos, total, rate: (pos - have) / Math.max(0.001, (now - t0) / 1000) });
      }
    }
    access.flush();
  } finally {
    access.close();
    downloadAbort = null;
  }
  if (total && pos !== total) throw new Error('The download ended early. Choose the model again to resume it.');
  await writeText(dir, `${key}.json`, JSON.stringify({ ...record, bytes: pos, complete: true, saved: Date.now() }));
  return (await dir.getFileHandle(`${key}.gguf`)).getFile();
}

// ---- Loading onto the GPU ----

async function readHeader(blob) {
  let size = Math.min(blob.size, 8 << 20);
  for (;;) {
    const bytes = new Uint8Array(await blob.slice(0, size).arrayBuffer());
    const result = JSON.parse(inspect(bytes));
    if (result.need) {
      if (size >= blob.size) throw new Error('The file ends inside its header. It may be incomplete or damaged.');
      size = Math.min(blob.size, Math.max(result.need, size * 2));
      continue;
    }
    if (result.error) throw new Error(result.error);
    return { bytes, model: result.model };
  }
}

async function loadBlob(blob, wantCtx) {
  const { bytes, model } = await readHeader(blob);
  post('inspected', { model });
  const ctx = Math.max(256, Math.min(wantCtx, model.config.ctx_train));
  let loader;
  try {
    loader = await create_loader(bytes);
  } catch (e) {
    throw new Error(`Could not start the model on the GPU: ${e?.message || e}`);
  }
  try {
    const plan = JSON.parse(loader.plan());
    const total = plan.reduce((sum, p) => sum + p.bytes, 0);
    let done = 0;
    let last = 0;
    const t0 = performance.now();
    for (const p of plan) {
      const rows = Math.max(1, Math.floor(CHUNK / p.row_bytes));
      for (let r = 0; r < p.rows; r += rows) {
        if (cancelLoad) throw new Error('Loading was cancelled.');
        const n = Math.min(rows, p.rows - r);
        const start = p.offset + r * p.row_bytes;
        const buf = await blob.slice(start, start + n * p.row_bytes).arrayBuffer();
        if (buf.byteLength !== n * p.row_bytes) throw new Error('The file is shorter than its header says. It may be incomplete.');
        loader.upload(p.name, r, new Uint8Array(buf));
        done += buf.byteLength;
        const now = performance.now();
        if (now - last > 100) {
          last = now;
          post('progress', { phase: 'upload', loaded: done, total, rate: done / Math.max(0.001, (now - t0) / 1000) });
        }
      }
    }
    post('progress', { phase: 'prepare', loaded: total, total });
    chat = loader.finish(ctx);
  } finally {
    loader.free();
  }
  return { model, chat: JSON.parse(chat.info()) };
}

function unload() {
  if (chat) {
    chat.free();
    chat = null;
  }
}

// ---- Generation ----

async function generate({ messages, options, sampler, maxTokens }) {
  if (!chat) throw new Error('Load a model first.');
  stopRequested = false;
  chat.set_sampler(JSON.stringify(sampler));
  const ctx = chat.context_length();
  const opts = JSON.stringify(options || {});
  let msgs = messages;
  let tokens;
  let dropped = 0;
  for (;;) {
    tokens = chat.prompt(JSON.stringify(msgs), opts);
    if (tokens.length + Math.min(maxTokens, 256) < ctx) break;
    const first = msgs.findIndex((m) => m.role !== 'system');
    if (first === -1 || msgs.length - first <= 1) {
      throw new Error(`This message is ${tokens.length} tokens, more than the ${ctx}-token context holds with room for a reply. Shorten it, or reload the model with a longer context in Settings.`);
    }
    msgs = [...msgs.slice(0, first), ...msgs.slice(first + 2)];
    dropped += 2;
  }
  const t0 = performance.now();
  const pre = JSON.parse(await chat.prefill(tokens));
  const t1 = performance.now();
  post('prefill', { ...pre, ms: t1 - t0, dropped, promptTokens: tokens.length });
  const limit = Math.min(maxTokens, ctx - tokens.length - 1);
  let n = 0;
  let reason = 'length';
  while (n < limit) {
    if (stopRequested) { reason = 'stopped'; break; }
    const id = await chat.next();
    n++;
    if (chat.is_end(id)) { reason = 'end'; break; }
    const text = chat.piece(id);
    if (text) post('token', { text });
  }
  const tail = chat.flush();
  if (tail) post('token', { text: tail });
  const t2 = performance.now();
  post('done', { reason, tokens: n, ms: t2 - t1, used: chat.context_used(), context: ctx });
}

// ---- Messages from the page ----

const handlers = {
  async hello() {
    const adapter = await describeAdapter();
    await ensureWasm();
    post('adapter', { adapter });
    post('cache', await listCached().catch(() => ({ models: [], supported: false })));
  },
  async load({ source, ctx }) {
    cancelLoad = false;
    unload();
    await ensureWasm();
    let blob;
    if (source.kind === 'file') {
      blob = source.file;
    } else {
      blob = await download(source);
      post('cache', await listCached().catch(() => ({ models: [], supported: false })));
    }
    const result = await loadBlob(blob, ctx);
    post('loaded', { ...result, source: source.kind === 'file' ? { kind: 'file', name: source.file.name } : { kind: 'hf', key: source.key, id: source.id } });
  },
  async generate(msg) {
    await generate(msg);
  },
  stop() {
    stopRequested = true;
  },
  cancel() {
    cancelLoad = true;
    downloadAbort?.abort();
  },
  reset() {
    chat?.reset();
  },
  unload() {
    unload();
    post('unloaded');
  },
  async listCache() {
    post('cache', await listCached());
  },
  async removeCache({ key }) {
    await removeCached(key);
    post('cache', await listCached());
  },
};

self.onmessage = async (event) => {
  const msg = event.data || {};
  const handler = handlers[msg.type];
  if (!handler) return;
  try {
    await handler(msg);
  } catch (e) {
    const message = String(e?.message || e);
    post('error', { during: msg.type, message });
  }
};
