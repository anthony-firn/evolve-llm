import { LARGE, MODELS, cacheKey, formatBytes, hfUrl, listRepo, parseHfReference, resolveParts, splitParts } from './models.js';
import { renderMarkdown, splitThinking } from './markdown.js';
import { stripToolCalls } from './tools.js';
import { wantsSnakeStarter, wantsBrowser, createWorkView } from './chat-development.mjs';
import { trace } from './trace.mjs';
import { newChatId, saveChat, getChat, listChats, deleteChat } from './chats.mjs';

const $ = (id) => document.getElementById(id);
const app = $('app');
// A page opened from a "use this computer" link (#remote=CODE) runs nothing itself: its worker
// is the computer's, reached over the paired connection (see devices.mjs).
const REMOTE_CODE = (() => {
  const c = new URLSearchParams(location.hash.slice(1)).get('remote');
  return c ? c.toUpperCase().replace(/[^A-Z0-9-]/g, '').slice(0, 24) : null;
})();
const remoteWorker = REMOTE_CODE ? new (await import('./devices.mjs')).RemoteWorker({ onConnection: (s, info) => onRemoteConnection(s, info) }) : null;
const newWorker = () => remoteWorker || new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let worker = newWorker();
if (REMOTE_CODE) app.dataset.remote = '';
const IS_MOBILE = navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);

const DEFAULTS = { system: '', temperature: 0.7, top_p: 0.95, top_k: 40, min_p: 0.05, repeat_penalty: 1.1, maxTokens: 1024, ctx: 4096, thinking: true, gpu: 'high-performance', gpuMemory: 'auto', cacheGb: 10, share: false, shareUpload: 2, internet: false, jinaKey: '', githubToken: '', autonomy: 1 };
const SUGGESTIONS = [
  'Make me a Snake game I can play.',
  'Write a haiku about running AI on my own laptop.',
  'Give me three tips for writing clear Rust code.',
  'Plan a relaxed weekend in Lisbon in five bullet points.',
];

const state = {
  phase: 'starting', // starting | idle | loading | ready | generating | preparing | agent
  adapter: null,
  cache: { models: [], supported: true },
  loading: null, // { label, id, key }
  loaded: null, // { id, key, label, info, chat }
  messages: [], // { role, content, stats?, error? }
  chatId: newChatId(), // where this conversation is saved (web/chats.mjs)
  live: null, // the assistant message being written
  settings: loadSettings(),
};

// The Linux terminal's VM takes the GitHub token and autonomy from Settings (when it starts,
// and on every change); the VM is created only when the terminal opens.
function configureVm() {
  import('./vm/client.mjs').then(({ provideConfig }) => provideConfig(() => ({ github: { token: state.settings.githubToken || '', autonomy: Number(state.settings.autonomy ?? 1) } }))).catch(() => {});
}

function loadSettings() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('evolve-llm-settings') || '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}

function saveSettings() {
  try { localStorage.setItem('evolve-llm-settings', JSON.stringify(state.settings)); } catch { /* storage unavailable */ }
}

function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

function recall(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

function setPhase(phase) {
  state.phase = phase;
  app.dataset.phase = phase;
  // A phone using its computer keeps its screen on while the computer works for it: a locked
  // phone drops the connection.
  if (REMOTE_CODE) keepAwake(['loading', 'generating', 'agent', 'preparing'].includes(phase));
  const ready = phase === 'ready';
  const agentBusy = phase === 'agent' || phase === 'preparing';
  const generating = phase === 'generating' || (agentBusy && Boolean(state.live?.assistant));
  $('file-input').disabled = agentBusy;
  $('hf-input').disabled = agentBusy;
  $('prompt').disabled = !(ready || phase === 'generating');
  $('prompt').placeholder = state.loaded ? 'Ask anything, or describe what to build…' : 'Choose a model to start';
  $('send').disabled = !ready;
  $('send').hidden = generating;
  $('stop').hidden = !generating;
  $('new-chat').disabled = !(ready && state.messages.length);
  $('welcome').hidden = Boolean(state.loaded) || phase === 'loading';
  $('loading').hidden = phase !== 'loading';
  $('ready').hidden = !(state.loaded && !state.messages.length && phase !== 'loading');
  renderModels();
}

// ---- GPU status ----

function describeGpu(a) {
  const parts = [a.description, [a.vendor, a.architecture].filter(Boolean).join(' ')].filter(Boolean);
  return parts[0] ? `${parts[0]} · WebGPU` : 'WebGPU ready';
}

function showGpu(adapter) {
  state.adapter = adapter;
  const chip = $('gpu-chip');
  const notice = $('gpu-notice');
  notice.replaceChildren();
  if (!adapter.ok) {
    chip.dataset.state = 'error';
    $('gpu-name').textContent = 'WebGPU unavailable';
    notice.className = 'notice error';
    notice.append(
      el('h2', { text: adapter.reason === 'no-webgpu' ? 'This browser does not offer WebGPU' : 'No GPU adapter is available' }),
      el('p', { text: 'Evolve LLM needs WebGPU to run models on your graphics processor. Try one of these:' }),
      el('ul', {},
        el('li', { text: 'Chrome or Edge 113 or later on Windows, macOS, ChromeOS or Android.' }),
        el('li', { text: 'Safari 26 or later on macOS, iOS or iPadOS.' }),
        el('li', { text: 'Firefox 141 or later on Windows.' }),
        el('li', {}, 'On Linux, Chrome may need ', el('code', { text: 'chrome://flags/#enable-unsafe-webgpu' }), ' and the Vulkan flag. Updated GPU drivers help.'),
      ),
    );
    notice.hidden = false;
    $('start-recommended').hidden = true;
    return;
  }
  chip.dataset.state = adapter.fallback ? 'warn' : 'ok';
  $('gpu-name').textContent = REMOTE_CODE ? `Your computer · ${describeGpu(adapter)}` : describeGpu(adapter);
  chip.title = `Largest GPU buffer ${formatBytes(adapter.maxBufferSize)}`;
  if (adapter.fallback) {
    notice.className = 'notice';
    notice.append(el('p', { text: 'This browser is using a software GPU. Models will run, but slowly. A hardware GPU or updated drivers make a large difference.' }));
    notice.hidden = false;
  } else {
    notice.hidden = true;
  }
  $('start-recommended').hidden = Boolean(state.loaded);
  setPhase(state.phase === 'starting' ? 'idle' : state.phase);
}

// ---- Model list ----

function cachedEntry(key) {
  return state.cache.models.find((m) => m.key === key);
}

function catalogEntry(m) {
  const key = cacheKey(m.repo, m.file);
  const split = Boolean(splitParts(m.file));
  // Models marked `stream` are read from Hugging Face as they run instead of downloaded.
  const stream = Boolean(m.stream) && !cachedEntry(key)?.complete;
  // Streamed models always list their parts: each part's SHA-256 names its swarm.
  return { kind: 'hf', id: m.id, key, repo: m.repo, file: m.file, revision: 'main', url: hfUrl(m.repo, m.file), bytes: m.bytes, name: m.name, resolve: split || stream, stream };
}

const ALL = [...MODELS, ...LARGE];

function renderModels() {
  const busy = state.phase === 'loading' || state.phase === 'generating' || ['agent','preparing'].includes(state.phase) || !state.adapter?.ok;
  const item = (m) => {
    const key = cacheKey(m.repo, m.file);
    const cached = cachedEntry(key);
    const saved = cached?.complete && cached.stored === cached.bytes;
    const partial = cached && !saved && cached.stored > 0;
    const running = state.loaded?.key === key;
    const tags = [
      el('span', { class: 'tag', text: m.quant }),
      running ? el('span', { class: 'tag run', text: 'Running' }) : null,
      saved && !running ? el('span', { class: 'tag ok', text: 'Saved' }) : null,
      m.stream && !saved ? el('span', { class: 'tag', text: 'Runs from Hugging Face' }) : null,
      partial ? el('span', { class: 'tag', text: `Paused at ${Math.round((cached.stored / cached.bytes) * 100)}%` }) : null,
      m.recommended && !saved && !running ? el('span', { class: 'tag rec', text: 'Recommended' }) : null,
      m.params ? el('span', { class: 'tag', text: m.params }) : null,
      el('span', { class: 'tag', text: `${m.maker} · ${m.license}` }),
    ];
    const button = el('button', {
      class: 'model', type: 'button', disabled: busy || running, 'aria-current': running ? 'true' : null,
      onclick: () => startLoad(catalogEntry(m)),
    },
    el('span', { class: 'row' }, el('span', { class: 'name', text: m.name }), el('span', { class: 'size', text: formatBytes(m.bytes) })),
    el('span', { class: 'note', text: m.note }),
    el('span', { class: 'meta' }, ...tags));
    const li = el('li', {}, button);
    if (cached && !running) {
      li.append(el('div', { class: 'model-actions' }, el('button', {
        class: 'remove', type: 'button', disabled: busy, text: saved ? 'Remove download' : 'Discard partial download',
        'aria-label': `Remove ${m.name} from this browser`,
        onclick: () => worker.postMessage({ type: 'removeCache', key }),
      })));
    }
    return li;
  };
  const items = MODELS.map(item);
  $('large-list').replaceChildren(...LARGE.map(item));
  const list = $('model-list');
  // Downloads that are not in the catalog.
  for (const c of state.cache.models) {
    if (ALL.some((m) => cacheKey(m.repo, m.file) === c.key)) continue;
    const running = state.loaded?.key === c.key;
    const entry = { kind: 'hf', id: c.key, key: c.key, repo: c.repo, file: c.file, revision: c.revision || 'main', url: c.url, bytes: c.bytes, name: c.file, parts: c.parts?.length > 1 ? c.parts.map((p) => ({ ...p, url: hfUrl(c.repo, p.file, c.revision || 'main') })) : null };
    items.push(el('li', {},
      el('button', { class: 'model', type: 'button', disabled: busy || running, 'aria-current': running ? 'true' : null, onclick: () => startLoad(entry) },
        el('span', { class: 'row' }, el('span', { class: 'name', text: c.file.split('/').pop().replace(/(-\d{5}-of-\d{5})?\.gguf$/i, '') }), el('span', { class: 'size', text: formatBytes(c.bytes) })),
        el('span', { class: 'note', text: c.repo }),
        el('span', { class: 'meta' }, running ? el('span', { class: 'tag run', text: 'Running' }) : el('span', { class: 'tag ok', text: c.complete ? 'Saved' : 'Partial' }))),
      running ? null : el('div', { class: 'model-actions' }, el('button', { class: 'remove', type: 'button', disabled: busy, text: 'Remove download', onclick: () => worker.postMessage({ type: 'removeCache', key: c.key }) }))));
  }
  list.replaceChildren(...items);
  $('hf-form').querySelector('button').disabled = busy;
  $('file-input').disabled = busy;
  renderStorage();
}

function renderStorage() {
  const box = $('storage');
  const { usage, quota, supported, models } = state.cache;
  if (!supported) {
    box.hidden = false;
    box.textContent = 'This browser cannot save models. Open GGUF files from your device instead.';
    return;
  }
  if (!quota) { box.hidden = true; return; }
  const stored = models.reduce((sum, m) => sum + (m.stored || 0), 0);
  box.hidden = false;
  const meter = el('div', { class: 'meter' }, el('span'));
  meter.firstChild.style.setProperty('width', `${Math.min(100, (usage / quota) * 100).toFixed(1)}%`);
  box.replaceChildren(
    el('div', { text: `Saved models: ${formatBytes(stored)}` }),
    meter,
    el('div', { text: `${formatBytes(quota - usage)} of browser storage free` }),
  );
}

// ---- Loading ----

let loadStarted = 0;

async function startLoad(source) {
  if (['agent','preparing'].includes(state.phase)) { $('status-line').textContent = 'Stop the current reply before loading another model.'; return; }
  if (!state.adapter?.ok) return;
  closeDrawer();
  if (state.phase === 'generating') return;
  // A file from disk that is one of the catalog models gets its friendly name.
  const first = source.kind === 'file' ? source.files[0].name : '';
  const known = source.kind === 'file' ? ALL.find((m) => m.file.split('/').pop().toLowerCase() === first.toLowerCase()) : null;
  const label = known ? known.name : source.kind === 'file' ? first.replace(/(-\d{5}-of-\d{5})?\.gguf$/i, '') : source.name;
  state.loading = { label, key: known ? cacheKey(known.repo, known.file) : source.key, id: known ? known.id : source.id, kind: source.kind, stream: Boolean(source.stream) };
  state.loaded = null;
  for(const m of state.messages)m.workView?.dispose();
  state.messages = [];state.chatId=newChatId();assistant?.reset();
  renderMessages();
  $('loading-title').textContent = label;
  for (const li of $('loading-steps').children) li.dataset.state = '';
  const saved = source.kind === 'hf' && cachedEntry(source.key)?.complete;
  const local = source.kind === 'file' || saved || source.stream;
  setStep(local ? 'upload' : 'download', local);
  setProgress(0, 1, '');
  $('loading-step').textContent = saved ? 'Saved on this device' : source.kind === 'file' ? 'From this device' : source.stream ? 'Connecting to Hugging Face' : 'Downloading';
  loadStarted = performance.now();
  setPhase('loading');
  if (source.kind === 'hf') remember('evolve-llm-last', source.id);
  // With paired devices, each reads only its share of a model from Hugging Face.
  state.helpers?.unloadAll();
  const helpers = source.kind === 'hf' ? state.helpers?.ready() ?? [] : [];
  if (helpers.length && !source.stream) source = { ...source, stream: true, resolve: true, parts: null };
  if (source.kind === 'hf' && (source.resolve || !source.bytes) && !source.parts) {
    // Split models and pasted files: ask Hugging Face for every part and its size first.
    try {
      const r = await resolveParts(source.repo, source.file, source.revision);
      source = { ...source, bytes: r.bytes, parts: r.parts.length > 1 || source.stream ? r.parts : null, url: r.parts[0].url };
    } catch (e) {
      onError({ during: 'load', message: e.message });
      return;
    }
  }
  stopSharing();
  const share = Boolean(!REMOTE_CODE && source.stream && state.settings.share && source.parts?.length);
  if (share) await startSharing(source);
  state.split = helpers.length ? { source, helpers, pending: null, plan: null } : null;
  worker.postMessage({ type: 'load', source, ctx: Number(state.settings.ctx), gpu: state.settings.gpu, gpuMemory: state.settings.gpuMemory, cacheGb: Number(state.settings.cacheGb), share, split: helpers.length ? helpers.map((h) => h.capacity) : null });
}

// ---- Your other devices: pairing, and running part of a model on them ----

let pairing = null;
// Advanced: other Nostr relays or ICE servers (self-hosting, tests), as JSON in
// localStorage['evolve-llm-swarm'] = {"relays": [...], "iceServers": [...]}.
function networkOverrides() {
  try { return JSON.parse(localStorage.getItem('evolve-llm-swarm') || '{}') || {}; } catch { return {}; }
}

async function startPairing() {
  if (pairing) return;
  const { Helpers, Pairing, newCode } = await import('./devices.mjs');
  const code = recall('evolve-llm-pair-code') || newCode();
  remember('evolve-llm-pair-code', code);
  state.helpers ??= new Helpers({ onChange: renderDevices });
  const o = networkOverrides();
  pairing = new Pairing({
    code,
    role: 'main',
    ...(Array.isArray(o.relays) && { relays: o.relays }),
    ...(Array.isArray(o.iceServers) && { iceServers: o.iceServers }),
    onLink: (link, id) => state.helpers.add(link, id),
    onClose: (id) => state.helpers.remove(id),
  });
  await pairing.start();
  remember('evolve-llm-pairing', '1');
  $('pair-link').value = `${location.origin}${location.pathname}#pair=${code}`;
  $('pair-qr').innerHTML = (await import('./qr.mjs')).qrSvg($('pair-link').value, { title: 'Pairing link' });
  renderDevices();
}

function stopPairing() {
  pairing?.stop();
  pairing = null;
  try { localStorage.removeItem('evolve-llm-pairing'); } catch { /* storage unavailable */ }
  renderDevices();
}

function renderDevices() {
  $('pair-info').hidden = !pairing;
  $('pair-start').hidden = Boolean(pairing);
  const list = $('pair-list');
  list.replaceChildren();
  const hs = state.helpers?.list ?? [];
  if (pairing && !hs.length) list.append(el('li', {}, el('span', { text: 'No other device yet' }), el('span', { text: 'Waiting for the link to be opened' })));
  for (const h of hs) {
    const p = h.progress?.total ? ` ${Math.round((100 * h.progress.loaded) / h.progress.total)}%` : '';
    const layers = h.stage ? `layers ${h.stage.first + 1}–${h.stage.end}` : '';
    const what = h.state === 'connecting' ? 'Connecting'
      : h.state === 'ready' ? `Ready · ${formatBytes(h.capacity)} for the model`
      : h.state === 'loading' ? `Loading ${layers}${p}`
      : h.state === 'loaded' ? `Running ${layers}` : h.state;
    list.append(el('li', {}, el('span', { text: h.name }), el('span', { text: what })));
  }
  if (state.phase === 'loading' && state.split?.pending) {
    const busy = state.split.helpers.filter((h) => h.state === 'loading');
    if (busy.length) $('loading-step').textContent = `Loading on your other devices: ${busy.map((h) => `${h.name}${h.progress?.total ? ` ${Math.round((100 * h.progress.loaded) / h.progress.total)}%` : ''}`).join(', ')}`;
  }
}

function onSplitPlan({ helpers: plan, ctx, mine }) {
  const sp = state.split;
  if (!sp) return;
  sp.plan = { mine, helpers: plan.map(({ helper, stage }) => ({ h: sp.helpers[helper], stage })) };
  sp.pending = Promise.all(sp.plan.helpers.map(({ h, stage }) => state.helpers.load(h, sp.source, stage, ctx)));
  // A device that cannot load its layers ends the load now rather than after this one's.
  sp.pending.catch((e) => {
    if (state.split !== sp || state.phase !== 'loading') return;
    worker.postMessage({ type: 'cancel' });
    state.helpers.unloadAll();
    onError({ during: 'load', message: e.message });
  });
}

function onRemoteForward({ id, hidden, n, pos0 }) {
  state.helpers.forward(hidden, n, pos0).then(
    (out) => worker.postMessage({ type: 'remote-result', id, hidden: out }, [out]),
    (e) => worker.postMessage({ type: 'remote-result', id, error: e.message || String(e) }),
  );
}

/** Where the loaded model's layers run, for the status line. */
function splitSummary() {
  const plan = state.split?.plan;
  if (!plan?.helpers.length) return '';
  const here = plan.mine?.end ? `layers 1–${plan.mine.end} here` : 'the input and output here';
  return ` · ${here}, ${plan.helpers.map(({ h, stage }) => `${stage.first + 1}–${stage.end} on ${h.name}`).join(', ')}`;
}

// A page opened from a pairing link lends this device to the chat device.
let helping = null;
async function startHelping(code) {
  state.helping = true;
  const dlg = $('helping');
  $('helping-code').textContent = code;
  dlg.addEventListener('cancel', (e) => e.preventDefault());
  $('helping-stop').onclick = () => { location.replace(location.pathname); };
  dlg.showModal();
  while (!state.adapter) await new Promise((r) => setTimeout(r, 200));
  if (!state.adapter.ok) { $('helping-status').textContent = 'This browser cannot run models (WebGPU is unavailable), so it cannot help.'; return; }
  const { Helping, Pairing } = await import('./devices.mjs');
  const { capacityOf } = await import('./split.mjs');
  const o = networkOverrides();
  const about = { name: deviceName(), capacity: capacityOf(state.settings.gpuMemory, IS_MOBILE), gpu: state.adapter.description || '' };
  keepAwake(true);
  const show = (s) => { $('helping-status').textContent = s; };
  pairing = new Pairing({
    code,
    role: 'helper',
    ...(Array.isArray(o.relays) && { relays: o.relays }),
    ...(Array.isArray(o.iceServers) && { iceServers: o.iceServers }),
    onLink: (link) => {
      helping?.stop();
      show('Connected. Waiting for your other device to load a model.');
      helping = new Helping({
        link, worker, about, settings: () => state.settings,
        onState: (st) => {
          if (st.state === 'loading') show(`Loading layers ${st.stage.first + 1}–${st.stage.end} from Hugging Face${st.progress?.total ? `: ${Math.round((100 * st.progress.loaded) / st.progress.total)}%` : '…'}`);
          else if (st.state === 'loaded') show(`Running layers ${st.stage.first + 1}–${st.stage.end}${st.served ? ` · ${st.served} ${st.served === 1 ? 'pass' : 'passes'}` : ''}. Keep this tab open.`);
          else if (st.state === 'error') show(`Could not load: ${st.message}`);
          else if (st.state === 'idle') show('Connected. Waiting for your other device to load a model.');
        },
      });
    },
    onClose: () => show('Your other device disconnected. Waiting for it to come back…'),
  });
  await pairing.start();
}

// ---- A phone using this computer, and this phone using a computer ----

let hostPairing = null;
let hosting = null;
async function startHosting() {
  if (hostPairing) return;
  const { Hosting, Pairing, newCode } = await import('./devices.mjs');
  const code = recall('evolve-llm-host-code') || newCode();
  remember('evolve-llm-host-code', code);
  const o = networkOverrides();
  const links = new Map();
  hostPairing = new Pairing({
    code,
    role: 'host',
    ...(Array.isArray(o.relays) && { relays: o.relays }),
    ...(Array.isArray(o.iceServers) && { iceServers: o.iceServers }),
    onLink: (link, id) => {
      links.set(id, link);
      // The phone drives this page's worker from now on, starting with the model loaded here.
      if (!hosting) {
        if (['generating', 'agent', 'preparing'].includes(state.phase)) worker.postMessage({ type: 'stop' });
        stopSharing();
        hosting = new Hosting({ worker, onState: renderHosting });
        hosting.loaded = state.loaded ? state.lastLoaded : null;
        state.hosting = true;
        const dlg = $('hosting');
        dlg.addEventListener('cancel', (e) => e.preventDefault());
        dlg.showModal();
        keepAwake(true);
      }
      hosting.attach(link);
    },
    onClose: (id) => { hosting?.detach(links.get(id)); links.delete(id); },
  });
  await hostPairing.start();
  remember('evolve-llm-hosting', '1');
  const link = `${location.origin}${location.pathname}#remote=${code}`;
  $('host-link').value = link;
  $('host-qr').innerHTML = (await import('./qr.mjs')).qrSvg(link, { title: 'Link for your phone' });
  $('host-info').hidden = false;
  $('host-start').hidden = true;
}

function stopHosting() {
  hostPairing?.stop();
  hostPairing = null;
  try { localStorage.removeItem('evolve-llm-hosting'); } catch { /* storage unavailable */ }
  $('host-info').hidden = true;
  $('host-start').hidden = false;
  if (hosting) {
    hosting.stop();
    hosting = null;
    state.hosting = false;
    $('hosting').close();
    keepAwake(false);
    restartEngine('Your phone no longer uses this computer.');
  }
}

function renderHosting({ state: s, loaded, tokens }) {
  const what = loaded ? `${loaded.chat?.name || 'A model'} is loaded` : 'No model is loaded yet';
  $('hosting-status').textContent = s === 'connected'
    ? `Connected to your phone. ${what}${tokens ? ` · ${tokens.toLocaleString()} tokens written` : ''}.`
    : `Waiting for your phone to reconnect. ${what}.`;
}

// The phone's side of the connection.
let remoteNotice = null;
async function startRemote() {
  const { Pairing } = await import('./devices.mjs');
  const o = networkOverrides();
  const links = new Map();
  pairing = new Pairing({
    code: REMOTE_CODE,
    role: 'remote',
    ...(Array.isArray(o.relays) && { relays: o.relays }),
    ...(Array.isArray(o.iceServers) && { iceServers: o.iceServers }),
    onLink: (link, id) => { links.set(id, link); remoteWorker.attach(link); },
    onClose: (id) => { remoteWorker.detach(links.get(id)); links.delete(id); },
  });
  await pairing.start();
  remember('evolve-llm-remote-code', REMOTE_CODE);
  $('gpu-name').textContent = 'Connecting to your computer…';
}

async function onRemoteConnection(s, info) {
  if (s === 'connected') {
    remoteNotice = null;
    $('status-line').textContent = 'Connected to your computer: models run on its GPU, and this phone shows the conversation.';
    // Show the model the computer already has loaded, once its GPU is described.
    for (let i = 0; i < 50 && !state.adapter; i++) await new Promise((r) => setTimeout(r, 100));
    if (info.loaded && !state.loaded && ['idle', 'starting'].includes(state.phase)) onLoaded(info.loaded);
  } else {
    remoteNotice = 'Lost the connection to your computer. Reconnecting…';
    if (['generating', 'agent', 'preparing', 'loading'].includes(state.phase)) {
      finishAgent(new Error('The connection to your computer was lost.'));
      onError({ during: state.phase === 'loading' ? 'load' : 'generate', message: 'The connection to your computer was lost. Keep this page open and the screen on while it works.' });
    }
    $('status-line').textContent = remoteNotice;
  }
}

// Screen wake lock while this device works for another (or a phone waits on its computer).
let wakeLock = null;
let wantAwake = false;
async function keepAwake(on) {
  wantAwake = on;
  try {
    if (on && !wakeLock && navigator.wakeLock && document.visibilityState === 'visible') {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });
    } else if (!on && wakeLock) {
      await wakeLock.release();
      wakeLock = null;
    }
  } catch { /* not allowed now (battery saver, or the page is hidden) */ }
}
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && wantAwake) keepAwake(true); });

function deviceName() {
  const ua = navigator.userAgent;
  const os = /Android/.test(ua) ? 'Android device' : /iPhone|iPad/.test(ua) ? 'iPhone or iPad' : /Mac OS X/.test(ua) ? 'Mac' : /Windows/.test(ua) ? 'Windows PC' : /CrOS/.test(ua) ? 'Chromebook' : /Linux/.test(ua) ? 'Linux computer' : 'Device';
  return os;
}

// ---- Sharing model pieces with other visitors (opt-in) ----

let swarmSession = null;
let swarmToken = 0;
async function startSharing(source) {
  const token = ++swarmToken;
  try {
    const { startSwarm } = await import('./swarm-client.mjs');
    // Advanced: other Nostr relays, manifests or ICE servers (self-hosting, tests), as JSON in
    // localStorage['evolve-llm-swarm'] = {"relays": [...], "manifests": "https://...", "iceServers": [...]}.
    let custom = {};
    try { custom = JSON.parse(localStorage.getItem('evolve-llm-swarm') || '{}') || {}; } catch { /* default */ }
    const swarmOptions = {};
    if (custom.manifests) swarmOptions.manifests = new (await import('./swarm-protocol.mjs')).Manifests(custom.manifests);
    if (Array.isArray(custom.relays)) swarmOptions.signal = new (await import('./swarm-signal.mjs')).Signal({ relays: custom.relays });
    if (Array.isArray(custom.iceServers)) swarmOptions.iceServers = custom.iceServers;
    const session = startSwarm({
      swarmOptions,
      parts: source.parts.map((p) => ({ sha256: p.sha256, bytes: p.bytes })),
      cacheKey: source.key,
      worker,
      uploadBytesPerSec: Number(state.settings.shareUpload) * 1e6,
      onStats: (s) => { state.swarmStats = s; renderSwarm(); },
    });
    if (token !== swarmToken) { session.stop(); return; }
    swarmSession = session;
    // The worker's first reads already wait on the session; this only reports how it started.
    session.ready.then((shared) => {
      if (token !== swarmToken) return;
      state.swarmStats = shared ? { peers: 0, fromPeers: 0, toPeers: 0 } : null;
      if (!shared) $('status-line').textContent = 'No piece manifest is published for this model yet, so it is read from Hugging Face only.';
      renderSwarm();
    });
  } catch (e) {
    if (token === swarmToken) $('status-line').textContent = `Sharing could not start: ${e.message || e}`;
  }
}
function stopSharing() {
  swarmToken++;
  swarmSession?.stop();
  swarmSession = null;
  state.swarmStats = null;
}
function renderSwarm() {
  const s = state.swarmStats;
  if (!s || state.phase === 'generating') return;
  $('status-line').textContent = `Sharing pieces with ${s.peers} ${s.peers === 1 ? 'visitor' : 'visitors'} · ${formatBytes(s.fromPeers)} received, ${formatBytes(s.toPeers)} sent`;
}

function setStep(step, skipDownload = false) {
  const order = ['download', 'upload', 'prepare'];
  for (const li of $('loading-steps').children) {
    const i = order.indexOf(li.dataset.step);
    const at = order.indexOf(step);
    li.dataset.state = i < at ? (li.dataset.step === 'download' && skipDownload ? 'skip' : 'done') : i === at ? 'active' : '';
  }
}

function setProgress(loaded, total, detail) {
  const pct = total ? Math.min(100, (loaded / total) * 100) : 0;
  const bar = $('loading-bar');
  bar.firstElementChild.style.setProperty('width', `${pct.toFixed(1)}%`);
  bar.setAttribute('aria-valuenow', String(Math.round(pct)));
  $('loading-detail').textContent = detail;
}

function onProgress({ phase, loaded, total, rate, part, parts }) {
  const bar = $('loading-bar');
  bar.classList.toggle('indeterminate', phase === 'prepare');
  if (phase === 'download') {
    $('loading-step').textContent = parts > 1 ? `Downloading part ${part} of ${parts} from Hugging Face` : 'Downloading from Hugging Face';
    setStep('download');
    const left = rate > 0 ? (total - loaded) / rate : 0;
    setProgress(loaded, total, `${formatBytes(loaded)} of ${formatBytes(total)} · ${formatBytes(rate)}/s${left > 1 ? ` · about ${formatDuration(left)} left` : ''}`);
  } else if (phase === 'connect') {
    $('loading-step').textContent = 'Connecting to Hugging Face';
    setProgress(0, 1, 'This model is read from Hugging Face as it runs, so it is not downloaded first.');
  } else if (phase === 'upload') {
    $('loading-step').textContent = state.loading?.stream ? 'Reading the GPU-resident weights from Hugging Face' : 'Moving weights to the GPU';
    setStep('upload', state.loading?.kind === 'file' || $('loading-steps').firstElementChild.dataset.state === 'skip');
    setProgress(loaded, total, `${formatBytes(loaded)} of ${formatBytes(total)} · ${formatBytes(rate)}/s`);
  } else {
    $('loading-step').textContent = 'Preparing the context';
    setStep('prepare', $('loading-steps').firstElementChild.dataset.state === 'skip');
    setProgress(1, 1, 'Allocating the key-value cache and compiling GPU kernels');
  }
}

// A model read from Hugging Face can take minutes per token: show that it is working.
function onStreamProgress({ fetched, cacheHit, fromPeers, rate }) {
  const text = `Reading weights: ${formatBytes(fetched)} from Hugging Face${fromPeers ? `, ${formatBytes(fromPeers)} from other visitors` : ''} at ${formatBytes(rate)}/s${cacheHit ? `, ${formatBytes(cacheHit)} from this device` : ''}`;
  if (state.phase === 'loading') setProgress(0, 1, text);
  else $('status-line').textContent = text;
}

function formatDuration(s) {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function onLoaded(msg) {
  const { model, chat, source } = msg;
  state.lastLoaded = msg;
  const label = state.loading?.label || chat.name;
  state.loaded = { ...source, key: state.loading?.key ?? source.key, label, info: model, chat, tools: Boolean(msg.tools), stream: Boolean(state.loading?.stream) };
  state.loading = null;
  $('loading-bar').classList.remove('indeterminate');
  const secs = (performance.now() - loadStarted) / 1000;
  const cfg = model.config;
  const types = Object.keys(model.weight_types).join(', ');
  $('ready-title').textContent = state.split?.plan?.helpers.length ? `${label} is running on your devices` : `${label} is running on your GPU`;
  const active = model.active_parameters < model.parameters * 0.9 ? `, ${formatParams(model.active_parameters)} active per token` : '';
  const pl = chat.placement || {};
  const where = pl.streaming
    ? ` · ${formatBytes(chat.weight_bytes + chat.state_bytes)} on the GPU, ${formatBytes(pl.streamed_bytes)} read from storage as needed${pl.expert_slots ? ` (an expert cache of ${pl.expert_slots.toLocaleString()} of ${pl.experts_total.toLocaleString()} experts)` : ` (${pl.resident_layers} of ${cfg.n_layer} layers on the GPU)`}`
    : ` · ${formatBytes(chat.weight_bytes + chat.state_bytes)} of GPU memory`;
  $('ready-meta').textContent = `${formatParams(model.parameters)} parameters${active} · ${types} · ${cfg.n_layer} layers · ${chat.context.toLocaleString()}-token context${where}${splitSummary()} · ready in ${secs.toFixed(1)} s`;
  $('suggestions').replaceChildren(...SUGGESTIONS.map((s) => el('button', { type: 'button', text: s, onclick: () => send(s) })));
  $('status-line').textContent = 'Replies are generated on this device. Responses can be wrong; check anything important.';
  setPhase('ready');
  $('prompt').focus();
}

function formatParams(n) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  return n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 0 : 1)}B` : `${Math.round(n / 1e6)}M`;
}

// ---- Conversation ----

function renderMessages() {
  const list = $('messages');
  const nodes=state.messages.map((m,i)=>messageNode(m,i));
  for(const child of [...list.children])if(!nodes.includes(child))child.remove();
  nodes.forEach((node,i)=>{if(list.children[i]!==node)list.insertBefore(node,list.children[i]||null);});
  $('ready').hidden = !(state.loaded && !state.messages.length && state.phase !== 'loading');
}

function copyText(text, button) {
  const done = () => {
    if (!button) return;
    const old = button.textContent;
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = old; }, 1200);
  };
  navigator.clipboard?.writeText(text).then(done, () => {});
}

function safeHref(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'https:' || u.protocol === 'http:' ? u.href : null;
  } catch {
    return null;
  }
}

function stepNode(step) {
  const what = step.name === 'web_search'
    ? `Searched the web for “${step.args?.query || ''}”`
    : `Read ${String(step.args?.url || '').replace(/^https?:\/\//, '').slice(0, 80)}`;
  if (!step.done) return el('li', { class: 'running', text: what });
  const links = (step.sources || []).map((s) => {
    const href = safeHref(s.url);
    return el('li', {}, href ? el('a', { href, target: '_blank', rel: 'noopener noreferrer', text: s.title || href }) : el('span', { text: s.title || '' }));
  });
  if (!links.length) return el('li', { text: `${what} · nothing found` });
  return el('li', {}, el('details', {}, el('summary', { text: `${what} · ${links.length} ${links.length === 1 ? 'source' : 'sources'}` }), el('ol', {}, ...links)));
}

function fillAssistant(bubble, m) {
  const parts = splitThinking(stripToolCalls(m.content));
  const nodes = [];
  if (m.steps?.length) nodes.push(el('ul', { class: 'web-steps', 'aria-label': 'Web activity' }, ...m.steps.map(stepNode)));
  for (const p of parts) {
    if (p.kind === 'think') {
      const words = p.text.trim().split(/\s+/).filter(Boolean).length;
      const details = el('details', { class: 'think', open: !p.done ? true : null },
        el('summary', { text: p.done ? `Reasoning · ${words} words` : 'Thinking…' }),
        el('div', { class: 'think-body', text: p.text.trim() }));
      nodes.push(details);
    } else {
      const div = el('div', { class: 'answer' });
      div.append(renderMarkdown(document, p.text.replace(/^\s+/, ''), copyText));
      nodes.push(div);
    }
  }
  if (m === state.live) {
    const last = nodes[nodes.length - 1];
    const target = last?.classList.contains('answer') ? last.lastElementChild || last : null;
    if (target && !target.matches('figure')) target.classList.add('cursor');
    else nodes.push(el('span', { class: 'cursor' }));
  }
  bubble.replaceChildren(...nodes);
}

function messageNode(m, i) {
  if (m.error) {
    return m.element ||= el('li', { class: 'msg error' }, el('div', { class: 'bubble', text: m.content }));
  }
  if (m.role === 'user') {
    return m.element ||= el('li', { class: 'msg user' }, el('div', { class: 'bubble', text: m.content }));
  }
  const bubble=m.element?.querySelector('.bubble')||el('div',{class:'bubble'});
  fillAssistant(bubble,m);
  const li=m.element ||= el('li',{class:'msg assistant','data-index':i},el('span',{class:'who',text:state.loaded?.label||'Model'}),bubble);
  li.querySelector(':scope > .msg-foot')?.remove();li.querySelector(':scope > .trimmed')?.remove();
  if(m.workView&&m.workView.root.parentElement!==li)li.append(m.workView.root);
  if (m !== state.live) {
    const last = i === state.messages.length - 1;
    li.append(el('div', { class: 'msg-foot' },
      m.stats ? el('span', { class: 'msg-stats', text: m.stats }) : null,
      el('span', { class: 'msg-actions' },
        el('button', { type: 'button', text: 'Copy', onclick: (e) => copyText(splitThinking(m.content).filter((p) => p.kind === 'answer').map((p) => p.text.trim()).join('\n\n'), e.currentTarget) }),
        last && state.phase === 'ready' ? el('button', { type: 'button', text: 'Regenerate', onclick: regenerate }) : null)));
    if (m.dropped) li.append(el('span', { class: 'trimmed', text: `Earlier messages (${m.dropped}) were left out to fit the context.` }));
  }
  return li;
}

let frame = 0;
function scheduleLiveRender() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    const node = $('messages').lastElementChild;
    if (!node || !state.live) return;
    const scroller = $('scroll');
    const stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
    fillAssistant(node.querySelector('.bubble'), state.live);
    if (stick) scroller.scrollTop = scroller.scrollHeight;
  });
}

function conversation() {
  const msgs = [];
  const system = state.settings.system.trim();
  if (system) msgs.push({ role: 'system', content: system });
  // Earlier answers go back without their thinking, as reasoning models' own templates do.
  for (const m of state.messages) if (!m.error && (m.role === 'user' || m.role === 'assistant')) msgs.push({ role: m.role, content: m.role === 'assistant' ? stripToolCalls(m.content).replace(/<think>[\s\S]*?<\/think>\s*/g, '') : m.content });
  return msgs;
}

function syncWebToggle() {
  const on = Boolean(state.settings.internet);
  const b = $('web-toggle');
  b.setAttribute('aria-pressed', String(on));
  b.title = on ? 'Web search is on: search terms go to Wikipedia, DuckDuckGo and Jina' : 'Let the model search the web';
}

function send(text) {
  text = text.trim();
  if (!text || state.phase !== 'ready') return;
  state.messages.push({ role: 'user', content: text });
  runResponse();
}

function regenerate() {
  if (state.phase !== 'ready') return;
  while (state.messages.length && state.messages[state.messages.length - 1].role !== 'user') state.messages.pop().workView?.dispose();
  if (state.messages.length) runResponse();
}

// One chat: a model whose template takes tools answers through the assistant (files, git, the
// web, app previews), deciding itself whether a message needs them; other models chat plainly.
function runResponse(){
  const question=state.messages.at(-1)?.content||'';
  if(localComputer.connected&&state.loaded?.tools)runLocalResponse(question);
  else if(state.loaded?.tools)runAssistant(question);
  else runGeneration();
}

// With a paired computer connected (web/local-computer.mjs), the same chat works on that computer.
async function runLocalResponse(question) {
  const context=state.messages.slice(0,-1).filter(m=>!m.error).slice(-6).map(m=>`${m.role}: ${stripToolCalls(m.content)}`).join('\n').slice(-6000);
  const m={role:'assistant',content:'',assistant:true,localComputer:true,started:performance.now()};
  m.workView=createWorkView();
  state.live=m;state.messages.push(m);setPhase('preparing');renderMessages();
  $('scroll').scrollTop=$('scroll').scrollHeight;
  const t0=performance.now();let calls=0,error=null;
  trace({type:'run',chat:state.chatId,question,model:state.loaded?.label,computer:true});
  try {
    const result=await localComputer.runTask(question,{context,onEvent:e=>{
      m.workView.event(e);
      if(e.type==='tool')trace({type:'tool',chat:state.chatId,name:e.name,args:JSON.stringify(e.args)});
      else if(e.type==='tool-result')trace({type:'result',chat:state.chatId,name:e.name,error:Boolean(e.isError),output:(e.result?.content||[]).map(c=>c.text||'').join('\n')});
      else if(e.type==='answer'){const t=stripToolCalls(e.text||'').trim();if(t){m.content=t;scheduleLiveRender();}}
    }});
    calls=result.calls;
    const last=result.messages.filter(m=>m.role==='assistant').at(-1);
    m.content=stripToolCalls((last?.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n')).trim()||m.content||'Done.';
    m.stats=`On your computer${result.calls?` · ${result.calls} step${result.calls===1?'':'s'}`:''}`;
  } catch(e) {error=String(e.message||e);m.content=(m.content?m.content+'\n\n':'')+(m.cancelRequested?'Stopped.':`I stopped before finishing: ${error}`);}
  finally {
    m.workView.finish();state.live=null;setPhase(state.loaded?'ready':'idle');renderMessages();$('scroll').scrollTop=$('scroll').scrollHeight;
    $('status-line').textContent=localComputer.connected?'Computer connected · the chat works on this computer.':'Local computer disconnected.';
    trace({type:'done',chat:state.chatId,ms:Math.round(performance.now()-t0),calls,error,stopped:Boolean(m.cancelRequested),answer:m.content});
    saveCurrentChat();
  }
}

let assistant=null;
async function getAssistant(){
  if(!assistant){
    const {createAssistant}=await import('./assistant.mjs');
    assistant||=createAssistant({
      turn:async(messages,tools,opts)=>{
        const t0=performance.now();
        try{
          const r=await agentTurn(messages,tools,{...opts,options:{enable_thinking:Boolean(state.settings.thinking)}});
          trace({type:'turn',chat:state.chatId,ms:Math.round(performance.now()-t0),messages:messages.length,promptTokens:r.promptTokens,tokens:r.n,reason:r.reason,text:r.text});
          return r;
        }catch(error){trace({type:'turn-error',chat:state.chatId,ms:Math.round(performance.now()-t0),error:String(error.message||error)});throw error;}
      },
      getNetwork:()=>({enabled:Boolean(state.settings.internet),jinaKey:state.settings.jinaKey||''}),
      sandbox:async()=>(await ensureWorkbench()).sandbox(),
    });
  }
  return assistant;
}

async function runAssistant(question){
  // Earlier turns as plain text: their answers, without reasoning or tool markup.
  const history=state.messages.slice(0,-1).filter(m=>!m.error&&(m.role==='user'||m.role==='assistant')).slice(-8)
    .map(m=>({role:m.role,content:m.role==='assistant'?stripToolCalls(m.content).replace(/<think>[\s\S]*?<\/think>\s*/g,'').trim():m.content})).filter(m=>m.content);
  const m={role:'assistant',content:'',assistant:true,started:performance.now()};
  m.workView=createWorkView();
  state.live=m;state.messages.push(m);setPhase('agent');renderMessages();
  $('scroll').scrollTop=$('scroll').scrollHeight;
  // A short follow-up to a website task ("I logged in") keeps the browser.
  const browsing=wantsBrowser(question)||(state.lastTaskBrowsing&&question.length<240&&!/\b(make|build|create|write|code|game|app)\b/i.test(question));state.lastTaskBrowsing=browsing;
  const s=state.settings,t0=performance.now();
  trace({type:'run',chat:state.chatId,question,model:state.loaded?.label,ctx:Number(s.ctx),thinking:Boolean(s.thinking),web:Boolean(s.internet),autonomy:Number(s.autonomy??1),history:history.length,browser:Boolean(browsing)});
  let calls=0,error=null;
  try{
    const a=await getAssistant();
    if(m.cancelRequested)throw new Error('Stopped.');
    const result=await a.run(question,{history,view:m.workView,modelName:state.loaded?.label||'Local model',maxTokens:Math.max(2048,Number(state.settings.maxTokens)||0),
      starter:wantsSnakeStarter(question),
      browser:browsing?()=>import('./browser-session.mjs').then(b=>b.ensureBrowser({show:true})):undefined,
      confirmBrowserAction:t=>Promise.resolve(confirm(t+'\n\nAllow this?')),
      onText:t=>{m.content=t;scheduleLiveRender();},
      onEvent:e=>{
        if(e.type==='tool')trace({type:'tool',chat:state.chatId,name:e.name,args:JSON.stringify(e.args)});
        else if(e.type==='tool-result')trace({type:'result',chat:state.chatId,name:e.name,error:Boolean(e.isError),output:(e.result?.content||[]).map(c=>c.text||'').join('\n')});
        else if(e.type==='retry')trace({type:'retry',chat:state.chatId,message:e.message,output:e.output});
      }});
    calls=result.calls;error=result.error;
    // Never claim success the model did not state: the steps show what actually ran.
    m.content=result.answer||m.content||(result.calls?'I stopped without saying what I did. The steps above show what actually ran.':'');
    if(result.error)m.content=(m.content?m.content+'\n\n':'')+(m.cancelRequested?'Stopped.':`I stopped before finishing: ${result.error}`);
    if(!m.content.trim())m.content='(no reply)';
    m.stats=result.calls?`${result.calls} step${result.calls===1?'':'s'}`:'';
  }catch(e){error=String(e.message||e);m.content=(m.content?m.content+'\n\n':'')+(m.cancelRequested?'Stopped.':`I stopped before finishing: ${error}`);}
  finally{
    m.workView.finish();state.live=null;setPhase(state.loaded?'ready':'idle');renderMessages();$('scroll').scrollTop=$('scroll').scrollHeight;
    trace({type:'done',chat:state.chatId,ms:Math.round(performance.now()-t0),calls,error,stopped:Boolean(m.cancelRequested),answer:m.content});
    saveCurrentChat();
  }
}

// ---- Saved chats: every finished reply saves the conversation in this browser ----
function saveCurrentChat(){
  const messages=state.messages.filter(m=>m!==state.live&&(m.role==='user'||m.role==='assistant'));
  if(!messages.length)return;
  const title=(messages.find(m=>m.role==='user')?.content||'Chat').replace(/\s+/g,' ').slice(0,80);
  saveChat({id:state.chatId,title,updated:Date.now(),model:state.loaded?.label||'',
    messages:messages.map(m=>({role:m.role,content:m.content,...(m.error&&{error:true}),...(m.stats&&{stats:m.stats}),...(m.workView&&m.workView.record)}))}).catch(()=>{});
}
async function openChat(id){
  if(!['ready','idle'].includes(state.phase))return;
  const chat=await getChat(id);if(!chat)return;
  for(const m of state.messages)m.workView?.dispose();
  const openFile=async path=>(await getAssistant()).read(path);
  state.messages=chat.messages.map(m=>({...m,steps:undefined,preview:undefined,...(m.role==='assistant'&&(m.steps?.length||m.preview)&&{workView:createWorkView({restore:{steps:m.steps,preview:m.preview},openFile})})}));
  state.chatId=chat.id;assistant?.reset();
  setPhase(state.phase);renderMessages();$('scroll').scrollTop=$('scroll').scrollHeight;
}
async function showChats(){
  const list=$('chat-list');
  const chats=await listChats().catch(()=>[]);
  const when=t=>new Date(t).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'});
  list.replaceChildren(...chats.map(c=>el('li',{class:c.id===state.chatId?'current':''},
    el('button',{type:'button',class:'chat-open',onclick:()=>{$('chats').close();openChat(c.id);}},el('strong',{text:c.title}),el('small',{text:`${when(c.updated)}${c.model?' · '+c.model:''}`})),
    el('button',{type:'button',class:'ghost chat-delete','aria-label':`Delete “${c.title}”`,text:'Delete',onclick:async()=>{await deleteChat(c.id).catch(()=>{});showChats();}}))));
  $('chats-empty').hidden=chats.length>0;
  if(!$('chats').open)$('chats').showModal();
}
$('open-chats').addEventListener('click',()=>{showChats();});

function runGeneration() {
  const now = new Date();
  const s = state.settings;
  const history = conversation();
  trace({ type: 'chat', chat: state.chatId, question: state.messages.at(-1)?.content || '', model: state.loaded?.label, ctx: Number(s.ctx) });
  state.live = { role: 'assistant', content: '', started: performance.now() };
  state.messages.push(state.live);
  setPhase('generating');
  renderMessages();
  $('scroll').scrollTop = $('scroll').scrollHeight;
  worker.postMessage({
    type: 'generate',
    messages: history,
    // A model read from Hugging Face takes minutes per token: keep its thinking short.
    options: { date: [now.getFullYear(), now.getMonth() + 1, now.getDate()], enable_thinking: Boolean(s.thinking), ...(state.loaded?.stream && { thinking_effort: 'low' }) },
    sampler: { temperature: Number(s.temperature), top_p: Number(s.top_p), top_k: Number(s.top_k), min_p: Number(s.min_p), repeat_penalty: Number(s.repeat_penalty), repeat_last_n: 64, seed: (Math.random() * 2 ** 32) >>> 0 },
    maxTokens: Number(s.maxTokens),
    internet: { enabled: Boolean(s.internet), jinaKey: s.jinaKey || '' },
  });
}

function onPrefill(p) {
  if (!state.live) return;
  state.live.prefill ??= p;
  if (p.dropped) state.live.dropped = p.dropped;
}

function onTool(msg, done) {
  const m = state.live;
  if (!m) return;
  m.steps ??= [];
  if (done) {
    const step = [...m.steps].reverse().find((s) => !s.done && s.name === msg.name) || m.steps[m.steps.length - 1];
    if (step) Object.assign(step, { done: true, sources: msg.sources });
  } else {
    m.steps.push({ name: msg.name, args: msg.args, done: false });
  }
  scheduleLiveRender();
}

function onDone(d) {
  const m = state.live;
  if (!m) return;
  const p = m.prefill || {};
  const rate = d.tokens > 0 && d.ms > 0 ? (d.tokens / (d.ms / 1000)).toFixed(1) : '0';
  const prompt = p.computed ? ` · read ${p.computed} prompt tokens in ${(p.ms / 1000).toFixed(2)} s${p.reused ? ` (${p.reused} reused)` : ''}` : '';
  const why = d.reason === 'stopped' ? ' · stopped' : d.reason === 'length' ? ' · reached the reply limit' : '';
  const st = d.stream || {};
  const stream = st.bytes_read ? ` · read ${formatBytes(st.bytes_read)} from storage${st.expert_requests ? `, ${Math.round((100 * st.expert_hits) / st.expert_requests)}% of experts already on the GPU` : ''}` : '';
  const n = d.net;
  const net = n && (n.fetched || n.cacheHit || n.fromPeers) ? ` · ${formatBytes(n.fetched)} from Hugging Face${n.fromPeers ? `, ${formatBytes(n.fromPeers)} from other visitors` : ''}${n.cacheHit ? `, ${formatBytes(n.cacheHit)} from this device's cache` : ''}` : '';
  const slow = d.tokens > 0 && d.ms / d.tokens > 10000 ? `${d.tokens} tokens · ${formatDuration(d.ms / 1000 / d.tokens)} per token` : `${d.tokens} tokens · ${rate} tokens/s`;
  m.stats = `${slow}${prompt}${why}${stream}${net}`;
  if (!m.content.trim()) m.content = d.reason === 'stopped' ? '' : '(no reply)';
  state.live = null;
  trace({ type: 'reply', chat: state.chatId, tokens: d.tokens, ms: Math.round(d.ms), reason: d.reason, text: m.content });
  saveCurrentChat();
  $('status-line').textContent = `Context: ${d.used.toLocaleString()} of ${d.context.toLocaleString()} tokens used`;
  setPhase('ready');
  renderMessages();
}

function onError({ during, message }) {
  if (during === 'load') {
    state.loading = null;
    state.loaded = null;
    $('loading-bar').classList.remove('indeterminate');
    setPhase('idle');
    const notice = $('gpu-notice');
    notice.className = 'notice error';
    notice.replaceChildren(el('h2', { text: 'The model could not be loaded' }), el('p', { text: message }));
    notice.hidden = false;
    $('welcome').hidden = false;
    worker.postMessage({ type: 'listCache' });
    return;
  }
  if (during === 'generate') {
    if (state.live) {
      state.messages.pop();
      state.live = null;
    }
    state.messages.push({ role: 'assistant', error: true, content: message });
    setPhase(state.loaded ? 'ready' : 'idle');
    renderMessages();
    return;
  }
  $('status-line').textContent = message;
}

function onWorkerMessage(event) {
  const msg = event.data;
  // While lending this device to another, or letting a phone use it, its worker answers that
  // device (see devices.mjs).
  if ((state.helping || state.hosting) && msg.type !== 'adapter') return;
  if (msg.requestId) {
    if (!agentRequest || msg.requestId !== agentRequest.id) return;
    if (msg.type === 'agent_event' && msg.event.type === 'token') agentRequest.onToken?.(msg.event.text);
    else if (msg.type === 'agent_done') finishAgent(agentRequest.aborted ? new Error(agentRequest.timedOut ? `One model turn took more than ${agentRequest.timedOut} minutes, so it was stopped. A smaller model, a faster GPU or a shorter request helps.` : 'Stopped.') : null, msg);
    else if (msg.type === 'error') finishAgent(new Error(msg.message));
    return;
  }
  switch (msg.type) {
    case 'adapter': showGpu(msg.adapter); break;
    case 'cache': state.cache = msg; renderModels(); break;
    case 'progress': onProgress(msg); break;
    case 'stream-progress': onStreamProgress(msg); break;
    case 'inspected': break;
    case 'loaded':
      // With other devices running part of the model, it is ready when they are.
      if (state.split?.pending) {
        const sp = state.split;
        sp.pending.then(() => { if (state.split === sp && state.phase === 'loading') onLoaded(msg); }, () => {});
      } else onLoaded(msg);
      break;
    case 'split-plan': onSplitPlan(msg); break;
    case 'remote-forward': onRemoteForward(msg); break;
    case 'remote-control': state.helpers?.control(msg.op); break;
    case 'prefill': onPrefill(msg); break;
    case 'token':
      if (state.live) {
        state.live.content += msg.text;
        scheduleLiveRender();
      }
      break;
    case 'tool': onTool(msg, false); break;
    case 'tool-result': onTool(msg, true); break;
    case 'round':
      // The model's next turn after a tool result replaces the tool-call text.
      if (state.live) { state.live.content = ''; scheduleLiveRender(); }
      break;
    case 'done': onDone(msg); break;
    case 'error': onError(msg); break;
    case 'unloaded': break;
  }
};
function onWorkerError(e) {
  if (agentRequest) finishAgent(new Error(e.message || 'Model worker failed.'));
  state.loaded = null; setPhase('idle');
  showGpu({ ok: false, reason: 'no-adapter' });
  $('status-line').textContent = `The engine could not start: ${e.message || 'worker error'}`;
};

// ---- Controls ----

const prompt = $('prompt');
function autosize() {
  prompt.style.setProperty('height', 'auto');
  prompt.style.setProperty('height', `${Math.min(prompt.scrollHeight, 220)}px`);
}
prompt.addEventListener('input', autosize);
prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $('composer').requestSubmit();
  }
});
$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  if (state.phase !== 'ready') return;
  const text = prompt.value;
  if (!text.trim()) return;
  prompt.value = '';
  autosize();
  send(text);
});
function stopCurrent(){if(state.live?.assistant){state.live.cancelRequested=true;if(state.live.localComputer)localComputer.cancel();else assistant?.cancel();}else worker.postMessage({type:'stop'});}
$('stop').addEventListener('click',stopCurrent);
$('web-toggle').addEventListener('click', () => {
  state.settings.internet = !state.settings.internet;
  networkChanged();
  saveSettings();
  syncWebToggle();
  $('status-line').textContent = state.settings.internet
    ? 'Web search is on. Search terms and pages the model opens go to Wikipedia, DuckDuckGo and Jina Reader; the conversation stays here.'
    : 'Web search is off. Nothing leaves this device.';
});
document.addEventListener('keydown', (e) => {
  if(e.key==='Escape'&&['generating','agent','preparing'].includes(state.phase))stopCurrent();
});
$('cancel-load').addEventListener('click', () => worker.postMessage({ type: 'cancel' }));
$('new-chat').addEventListener('click', () => {
  if (state.phase !== 'ready') return;
  for(const m of state.messages)m.workView?.dispose();
  state.messages = [];state.chatId=newChatId();assistant?.reset();
  worker.postMessage({ type: 'reset' });
  renderMessages();
  setPhase('ready');
  $('status-line').textContent = 'Started a new conversation.';
  prompt.focus();
});
$('start-recommended').addEventListener('click', () => {
  const last = ALL.find((m) => m.id === recall('evolve-llm-last'));
  startLoad(catalogEntry(last || MODELS.find((m) => m.recommended)));
});
const HF_HELP = $('hf-help').textContent;

function hfSource(ref, file) {
  const first = splitParts(file)?.[0] || file;
  const key = cacheKey(ref.repo, first, ref.revision);
  return { kind: 'hf', id: key, key, repo: ref.repo, file: first, revision: ref.revision, url: hfUrl(ref.repo, first, ref.revision), bytes: 0,
    name: first.split('/').pop().replace(/(-\d{5}-of-\d{5})?\.gguf$/i, ''), resolve: true, stream: $('hf-stream').checked };
}

$('hf-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const help = $('hf-help');
  const results = $('hf-results');
  results.hidden = true;
  try {
    const ref = parseHfReference($('hf-input').value);
    help.classList.remove('error');
    help.textContent = HF_HELP;
    if (ref.file) {
      startLoad(hfSource(ref, ref.file));
      return;
    }
    help.textContent = `Looking up ${ref.repo}…`;
    const groups = await listRepo(ref.repo, ref.revision);
    if (!groups.length) throw new Error('That repository has no GGUF files.');
    help.textContent = `${groups.length} GGUF ${groups.length === 1 ? 'model' : 'models'} in ${ref.repo}. Choose one:`;
    results.replaceChildren(...groups.map((g) => el('li', {}, el('button', { type: 'button', onclick: () => startLoad(hfSource(ref, g.file)) },
      el('span', { class: 'name', text: g.file.split('/').pop().replace(/(-\d{5}-of-\d{5})?\.gguf$/i, '') }),
      el('span', { class: 'size', text: `${formatBytes(g.bytes)}${g.parts.length > 1 ? ` · ${g.parts.length} files` : ''}` })))));
    results.hidden = false;
  } catch (err) {
    help.classList.add('error');
    help.textContent = err.message;
  }
});
$('file-input').addEventListener('change', (e) => {
  const files = [...(e.target.files || [])].sort((a, b) => a.name.localeCompare(b.name));
  e.target.value = '';
  if (files.length) startLoad({ kind: 'file', files, key: null, id: null });
});

// Drawer on narrow screens.
function openDrawer() { app.dataset.drawer = 'open'; $('open-models').setAttribute('aria-expanded', 'true'); $('scrim').hidden = false; }
function closeDrawer() { delete app.dataset.drawer; $('open-models').setAttribute('aria-expanded', 'false'); $('scrim').hidden = true; }
$('open-models').addEventListener('click', () => (app.dataset.drawer ? closeDrawer() : openDrawer()));
$('scrim').addEventListener('click', closeDrawer);
const topBar = document.querySelector('.top');
const syncTop = () => document.documentElement.style.setProperty('--top-h', `${topBar.offsetHeight}px`);
new ResizeObserver(syncTop).observe(topBar);

// Settings dialog.
const dialog = $('settings');
const form = $('settings-form');
const RANGES = ['temperature', 'top_p', 'top_k', 'min_p', 'repeat_penalty', 'maxTokens'];
function fillSettings() {
  const s = state.settings;
  form.elements.system.value = s.system;
  for (const k of RANGES) {
    form.elements[k].value = s[k];
    form.elements[`${k}Out`].value = s[k];
  }
  form.elements.ctx.value = String(s.ctx);
  form.elements.gpu.value = s.gpu;
  form.elements.gpuMemory.value = String(s.gpuMemory);
  form.elements.cacheGb.value = String(s.cacheGb ?? 10);
  form.elements.share.checked = Boolean(s.share);
  form.elements.shareUpload.value = String(s.shareUpload ?? 2);
  form.elements.thinking.checked = Boolean(s.thinking);
  form.elements.internet.checked = Boolean(s.internet);
  form.elements.jinaKey.value = s.jinaKey || '';
  form.elements.githubToken.value = s.githubToken || '';
  form.elements.autonomy.value = String(s.autonomy ?? 1);
}
form.addEventListener('input', (e) => {
  const t = e.target;
  if (!t.name) return;
  if (t.type === 'checkbox') {
    state.settings[t.name] = t.checked;
    if (t.name === 'internet') { syncWebToggle(); networkChanged(); }
  }
  else if (t.type === 'range') {
    state.settings[t.name] = Number(t.value);
    form.elements[`${t.name}Out`].value = t.value;
  } else if (t.name === 'ctx') state.settings.ctx = Number(t.value);
  else if (t.name === 'gpu') {
    state.settings.gpu = t.value;
    worker.postMessage({ type: 'adapter', gpu: t.value });
  }
  else state.settings[t.name] = t.value;
  saveSettings();
  if (t.name === 'githubToken' || t.name === 'autonomy') configureVm();
});
form.addEventListener('change', (e) => {
  if (['ctx', 'gpu', 'gpuMemory', 'cacheGb', 'share', 'shareUpload'].includes(e.target.name) && state.loaded) $('status-line').textContent = 'This setting applies the next time a model loads.';
});
$('open-settings').addEventListener('click', () => { fillSettings(); dialog.showModal(); });
$('reset-settings').addEventListener('click', () => {
  state.settings = { ...DEFAULTS };
  saveSettings();
  fillSettings();
  syncWebToggle();
  networkChanged();
  worker.postMessage({ type: 'adapter', gpu: state.settings.gpu });
});

setPhase('starting');
syncWebToggle();

// ---- Terminal and Pi: a lease on the existing model, not a second GPU copy ----
let workbench=null, workbenchPromise=null, agentRequest=null, agentSequence=0;
worker.onmessage=onWorkerMessage;worker.onerror=onWorkerError;
const terminalButton=el('button',{class:'ghost',type:'button',id:'open-terminal',text:'Terminal',title:'A Linux terminal on the same files the assistant works on'});
$('open-settings').before(terminalButton);
function finishAgent(error,value) {
  const request=agentRequest;if(!request)return;agentRequest=null;clearTimeout(request.timer);clearTimeout(request.cancelTimer);request.signal?.removeEventListener('abort',request.abort);
  error?request.reject(error):request.resolve(value);
}
function restartEngine(reason) {
  stopSharing();
  state.helpers?.unloadAll();
  worker.terminate();finishAgent(new Error(reason));state.loaded=null;state.loading=null;
  worker=newWorker();worker.onmessage=onWorkerMessage;worker.onerror=onWorkerError;
  setPhase('starting');worker.postMessage({type:'hello',gpu:state.settings.gpu});$('status-line').textContent=reason+' Choose a saved model to reload it.';
}
// Time limits for Pi, in ms (0: none). A model read from Hugging Face can take minutes to hours
// per token, and one that streams from this device's storage seconds, so they get longer ones;
// `stop` is how long a cancelled turn may take to wind down before the worker is restarted.
function piLimits() {
  if (state.loaded?.stream) return { turn: 0, task: 0, stop: 120000 };
  if (state.loaded?.chat?.placement?.streaming) return { turn: 1800000, task: 7200000, stop: 60000 };
  // A 4B model on an integrated GPU reads a long prompt and writes a long reply in minutes, not seconds.
  return { turn: 600000, task: 180000, stop: 10000 };
}
function agentTurn(messages,tools,{signal,onToken,maxTokens=2048,options}={}) {
  if(state.phase!=='agent'||agentRequest)return Promise.reject(new Error('The model is busy.'));
  if(signal?.aborted)return Promise.reject(new Error('Stopped.'));
  return new Promise((resolve,reject)=>{
    const id=++agentSequence;
    const limits=piLimits();
    const abort=()=>{if(agentRequest?.id!==id)return;agentRequest.aborted=true;worker.postMessage({type:'stop'});agentRequest.cancelTimer=setTimeout(()=>restartEngine('Stopping took too long, so the model was unloaded.'),limits.stop)};
    // A turn past its limit is stopped like Stop does; the model stays loaded unless stopping hangs.
    agentRequest={id,resolve,reject,onToken,signal,abort,timer:limits.turn?setTimeout(()=>{if(agentRequest?.id===id){agentRequest.timedOut=Math.round(limits.turn/60000);abort();}},limits.turn):null};
    signal?.addEventListener('abort',abort,{once:true});
    worker.postMessage({type:'agent_turn',requestId:id,messages,tools,options,maxTokens:Math.max(64,Math.min(4096,Number(maxTokens)||2048)),sampler:{temperature:0,top_p:.95,top_k:40,min_p:.05,repeat_penalty:1.1,repeat_last_n:64,seed:1}});
  });
}
function networkChanged() {
  worker.postMessage({type:'network_policy',enabled:Boolean(state.settings.internet)});workbench?.setNetwork();
}
async function ensureWorkbench(){
  workbenchPromise ||= import('./workbench.mjs').then(({mountWorkbench})=>mountWorkbench({
    ready:()=>['ready','preparing'].includes(state.phase)&&Boolean(state.loaded),turn:agentTurn,modelName:()=>state.loaded?.label||'Local model',taskTimeLimit:()=>piLimits().task,
    lease:()=>{if(state.phase!=='ready'||!state.loaded)throw new Error('The model is busy or not loaded.');setPhase('agent');let released=false;return()=>{if(!released){released=true;setPhase(state.loaded?'ready':'idle');}};},
    getNetwork:()=>({enabled:Boolean(state.settings.internet),jinaKey:state.settings.jinaKey||''}),
    setNetwork:enabled=>{state.settings.internet=enabled;saveSettings();syncWebToggle();networkChanged();},
  })).catch(error=>{workbenchPromise=null;throw error;});
  workbench=await workbenchPromise;return workbench;
}
terminalButton.addEventListener('click',async()=>{
  terminalButton.disabled=true;
  try{await(await ensureWorkbench()).open();}catch(error){$('status-line').textContent=String(error.message||error);}
  finally{terminalButton.disabled=false;}
});
// ---- Browser: a web engine compiled to WebAssembly, shared by the person and Pi ----
const browserButton=el('button',{class:'ghost',type:'button',id:'open-browser',text:'Browser',title:'A web browser in this page, shared with the assistant'});
$('open-settings').before(browserButton);
browserButton.addEventListener('click',async()=>{
  browserButton.disabled=true;
  try{const {openBrowser,isBrowserOpen,closeBrowser}=await import('./browser-session.mjs');if(isBrowserOpen())closeBrowser();else await openBrowser();}
  catch(error){$('status-line').textContent=String(error.message||error);}
  finally{browserButton.disabled=false;}
});

worker.postMessage({ type: 'hello', gpu: state.settings.gpu });
configureVm();
{
  const code = new URLSearchParams(location.hash.slice(1)).get('pair');
  if (REMOTE_CODE) {
    // This page is a screen for the computer: pairing and sharing are the computer's business.
    document.querySelector('.settings fieldset.devices').hidden = true;
    $('models-intro').textContent = 'These run on your computer: each downloads once from Hugging Face to it, and the list shows what it has saved. This phone shows the conversation.';
    startRemote().catch((e) => { $('status-line').textContent = `Could not reach your computer: ${e.message || e}`; });
  } else if (code) startHelping(code.toUpperCase().replace(/[^A-Z0-9-]/g, '').slice(0, 24));
  else {
    if (recall('evolve-llm-pairing') === '1') startPairing().catch(() => {});
    if (recall('evolve-llm-hosting') === '1') startHosting().catch(() => {});
  }
  $('host-start').addEventListener('click', () => startHosting().catch((e) => { $('status-line').textContent = `Could not start: ${e.message || e}`; }));
  $('host-stop').addEventListener('click', stopHosting);
  $('hosting-stop').addEventListener('click', stopHosting);
  $('host-copy').addEventListener('click', () => navigator.clipboard?.writeText($('host-link').value).catch(() => {}));
  $('pair-start').addEventListener('click', () => startPairing().catch((e) => { $('status-line').textContent = `Pairing could not start: ${e.message || e}`; }));
  $('pair-stop').addEventListener('click', stopPairing);
  $('pair-copy').addEventListener('click', () => navigator.clipboard?.writeText($('pair-link').value).catch(() => {}));
}

// Local tools are opt-in and use the existing loaded model.
import { mountLocalComputer } from './local-computer.mjs';
const localComputer = mountLocalComputer({
  ready:()=>['ready','preparing'].includes(state.phase)&&Boolean(state.loaded), turn:agentTurn,
  onConnectionChange:connected=>{$('status-line').textContent=connected?'Computer connected · the chat works on this computer.':'Local computer disconnected.';},
  modelName:()=>state.loaded?.label||'Local model',
  lease:()=>{if(!['ready','preparing'].includes(state.phase)||!state.loaded)throw new Error('The model is not ready.');setPhase('agent');let released=false;return()=>{if(!released){released=true;setPhase(state.loaded?(state.live?.assistant?'preparing':'ready'):'idle');}};}
});
