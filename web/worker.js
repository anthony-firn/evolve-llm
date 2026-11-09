// Runs the WebAssembly engine off the main thread: downloads, GPU upload and generation.
import init, { create_loader, inspect, inspect_parts } from './pkg/evolve_llm.js';
import { formatBytes } from './models.js';

const CHUNK = 16 << 20; // bytes read from the file per upload call
const post = (type, data = {}) => self.postMessage({ type, ...data });

let wasm = null;
let chat = null;
// Synchronous readers of the loaded model's files, kept open while it runs: weights that do not
// fit on the GPU are read again from them.
let readers = [];
let stopRequested = false;
let cancelLoad = false;
let downloadAbort = null;

async function ensureWasm() {
  wasm ??= init();
  await wasm;
}

async function describeAdapter(preference) {
  if (!('gpu' in navigator) || !navigator.gpu) return { ok: false, reason: 'no-webgpu' };
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: preference === 'low-power' ? 'low-power' : 'high-performance' });
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

// File name of part `i` of a model in storage: single files keep `${key}.gguf`.
function partName(key, i, count) {
  return count > 1 ? `${key}.p${i + 1}.gguf` : `${key}.gguf`;
}

async function storedSize(dir, name) {
  try { return (await (await dir.getFileHandle(name)).getFile()).size; } catch { return 0; }
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
    const count = meta.parts?.length || 1;
    let size = 0;
    for (let i = 0; i < count; i++) size += await storedSize(dir, partName(key, i, count));
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
  const meta = await readMeta(dir, key);
  const count = meta?.parts?.length || 1;
  const names = [`${key}.json`, `${key}.gguf`];
  for (let i = 0; i < count; i++) names.push(partName(key, i, count));
  for (const name of names) {
    try { await dir.removeEntry(name); } catch { /* already gone */ }
  }
}

function httpError(status) {
  if (status === 404) return 'Hugging Face does not have that file. Check the repository and file name.';
  if (status === 401 || status === 403) return 'That model needs a Hugging Face sign-in or license approval. Download it there, then open the file here.';
  if (status === 429) return 'Hugging Face is limiting downloads right now. Wait a minute and try again.';
  return `The download failed with HTTP ${status}.`;
}

// Download every part of a model into storage, resuming partial parts. Returns the part names.
async function download(entry) {
  if (!navigator.storage?.getDirectory) throw new Error('This browser cannot store downloaded models. Open a GGUF file from your device instead.');
  const dir = await modelsDir();
  const { key } = entry;
  const parts = entry.parts?.length ? entry.parts : [{ file: entry.file, url: entry.url, bytes: entry.bytes }];
  const count = parts.length;
  const names = parts.map((_, i) => partName(key, i, count));
  const meta = await readMeta(dir, key);
  const same = meta && meta.url === entry.url && (meta.parts?.length || 1) === count;
  if (same && meta.complete) {
    let ok = true;
    for (let i = 0; i < count; i++) ok &&= (await storedSize(dir, names[i])) === (meta.parts?.[i]?.bytes ?? meta.bytes);
    if (ok) return names;
  }
  const total = parts.reduce((t, p) => t + (p.bytes || 0), 0);
  let have = 0;
  for (let i = 0; i < count; i++) have += same ? await storedSize(dir, names[i]) : 0;
  const estimate = await navigator.storage.estimate().catch(() => null);
  if (estimate?.quota && total - have > estimate.quota - (estimate.usage || 0)) {
    throw new Error(`This browser can store ${formatBytes(estimate.quota - (estimate.usage || 0))} more, and the model needs ${formatBytes(total - have)}. Remove a downloaded model or open the files from your device.`);
  }
  try { await navigator.storage.persist?.(); } catch { /* best effort */ }
  const record = { url: entry.url, repo: entry.repo, file: entry.file, revision: entry.revision, name: entry.name, bytes: total,
    parts: parts.map((p) => ({ file: p.file, bytes: p.bytes })), complete: false, started: meta?.started || Date.now() };
  await writeText(dir, `${key}.json`, JSON.stringify(record));
  const t0 = performance.now();
  let done = have;
  let last = 0;
  for (let i = 0; i < count; i++) {
    const part = parts[i];
    const handle = await dir.getFileHandle(names[i], { create: true });
    let pos = same ? (await handle.getFile()).size : 0;
    if (part.bytes && pos === part.bytes) continue;
    if (part.bytes && pos > part.bytes) pos = 0;
    downloadAbort = new AbortController();
    let res;
    try {
      res = await fetch(part.url, { headers: pos > 0 ? { Range: `bytes=${pos}-` } : {}, signal: downloadAbort.signal });
    } catch (e) {
      if (e?.name === 'AbortError') throw new Error('Download paused. Choose the model again to resume.');
      throw new Error('Could not reach Hugging Face. Check your connection and try again.');
    }
    if (!res.ok) throw new Error(httpError(res.status));
    if (res.status !== 206) { done -= pos; pos = 0; }
    const access = await handle.createSyncAccessHandle();
    try {
      if (pos === 0) access.truncate(0);
      const reader = res.body.getReader();
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
        done += chunk.value.byteLength;
        const now = performance.now();
        if (now - last > 150) {
          last = now;
          post('progress', { phase: 'download', loaded: done, total, part: i + 1, parts: count,
            rate: (done - have) / Math.max(0.001, (now - t0) / 1000) });
        }
      }
      access.flush();
    } finally {
      access.close();
      downloadAbort = null;
    }
    if (part.bytes && pos !== part.bytes) throw new Error('The download ended early. Choose the model again to resume it.');
  }
  await writeText(dir, `${key}.json`, JSON.stringify({ ...record, complete: true, saved: Date.now() }));
  return names;
}

// ---- Reading model files ----

// A synchronous random-access reader: an OPFS access handle for stored models, or
// FileReaderSync for files opened from the device. Both fill a view of WebAssembly memory.
async function openReaders(source) {
  const out = [];
  if (source.names) {
    const dir = await modelsDir();
    for (const name of source.names) {
      const handle = await dir.getFileHandle(name);
      let access;
      try { access = await handle.createSyncAccessHandle({ mode: 'read-only' }); } catch { access = await handle.createSyncAccessHandle(); }
      out.push({
        size: access.getSize(),
        read(offset, target) {
          const n = access.read(target, { at: offset });
          if (n !== target.length) throw new Error('The stored model is shorter than its header says.');
        },
        close() { access.close(); },
      });
    }
  } else {
    const sync = new FileReaderSync();
    for (const file of source.files) {
      out.push({
        size: file.size,
        read(offset, target) {
          const buf = sync.readAsArrayBuffer(file.slice(offset, offset + target.length));
          if (buf.byteLength !== target.length) throw new Error('The file is shorter than its header says. It may be incomplete.');
          target.set(new Uint8Array(buf));
        },
        close() {},
      });
    }
  }
  return out;
}

function closeReaders() {
  for (const r of readers) {
    try { r.close(); } catch { /* already closed */ }
  }
  readers = [];
}

// ---- Loading onto the GPU ----

function readBytes(reader, offset, length) {
  const out = new Uint8Array(length);
  reader.read(offset, out);
  return out;
}

// The header of one file: read a growing prefix until it parses.
function readHeader(reader) {
  let size = Math.min(reader.size, 8 << 20);
  for (;;) {
    const bytes = readBytes(reader, 0, size);
    const result = JSON.parse(inspect(bytes));
    if (result.need) {
      if (size >= reader.size) throw new Error('The file ends inside its header. It may be incomplete or damaged.');
      size = Math.min(reader.size, Math.max(result.need, size * 2));
      continue;
    }
    return { bytes, result };
  }
}

async function loadModel(source, wantCtx, lowPower, budgetPref) {
  readers = await openReaders(source);
  const first = readHeader(readers[0]);
  if (first.result.error) throw new Error(first.result.error);
  const headers = [first.bytes];
  let model = first.result.model;
  if (first.result.split) {
    if (readers.length !== first.result.split) {
      throw new Error(`This model is split into ${first.result.split} files. Open all ${first.result.split} parts together.`);
    }
    for (let i = 1; i < readers.length; i++) {
      const h = readHeader(readers[i]);
      if (h.result.error && !h.result.split) throw new Error(h.result.error);
      headers.push(h.bytes);
    }
    const merged = JSON.parse(inspect_parts(headers));
    if (merged.error) throw new Error(merged.error);
    model = merged.model;
  }
  post('inspected', { model });
  const ctx = Math.max(256, Math.min(wantCtx, model.config.ctx_train));
  // Auto keeps models up to 4 GB fully on the GPU and streams the rest of larger ones.
  const AUTO = 4 * 2 ** 30;
  const budget = budgetPref === 'all' ? null
    : budgetPref && budgetPref !== 'auto' ? Number(budgetPref) * 2 ** 30
    : (model.fast_bytes > AUTO ? AUTO : null);
  let loader;
  try {
    loader = await create_loader(headers, JSON.stringify({ low_power: lowPower, budget }));
  } catch (e) {
    throw new Error(`Could not start the model on the GPU: ${e?.message || e}`);
  }
  try {
    const plan = JSON.parse(loader.plan()).filter((p) => p.resident);
    const total = plan.reduce((sum, p) => sum + p.bytes, 0);
    let done = 0;
    let last = 0;
    const t0 = performance.now();
    for (const p of plan) {
      const rows = Math.max(1, Math.floor(CHUNK / p.row_bytes));
      for (let r = 0; r < p.rows; r += rows) {
        if (cancelLoad) throw new Error('Loading was cancelled.');
        const n = Math.min(rows, p.rows - r);
        const bytes = readBytes(readers[p.file], p.offset + r * p.row_bytes, n * p.row_bytes);
        loader.upload(p.name, r, bytes);
        done += bytes.length;
        const now = performance.now();
        if (now - last > 100) {
          last = now;
          post('progress', { phase: 'upload', loaded: done, total, rate: done / Math.max(0.001, (now - t0) / 1000) });
          // Let progress and cancel messages through.
          await new Promise((r) => setTimeout(r, 0));
        }
      }
    }
    const files = readers;
    loader.set_source((file, offset, target) => files[file].read(offset, target));
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
  closeReaders();
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
  post('done', { reason, tokens: n, ms: t2 - t1, used: chat.context_used(), context: ctx, stream: JSON.parse(chat.stats()) });
}

// ---- Messages from the page ----

const handlers = {
  async hello({ gpu }) {
    const adapter = await describeAdapter(gpu);
    await ensureWasm();
    post('adapter', { adapter });
    post('cache', await listCached().catch(() => ({ models: [], supported: false })));
  },
  async adapter({ gpu }) {
    post('adapter', { adapter: await describeAdapter(gpu) });
  },
  async load({ source, ctx, gpu, gpuMemory }) {
    cancelLoad = false;
    unload();
    await ensureWasm();
    let files;
    if (source.kind === 'file') {
      files = { files: source.files || [source.file] };
    } else {
      files = { names: await download(source) };
      post('cache', await listCached().catch(() => ({ models: [], supported: false })));
    }
    const result = await loadModel(files, ctx, gpu === 'low-power', gpuMemory);
    const name = source.kind === 'file' ? (source.files || [source.file])[0].name : null;
    post('loaded', { ...result, source: source.kind === 'file' ? { kind: 'file', name } : { kind: 'hf', key: source.key, id: source.id } });
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
