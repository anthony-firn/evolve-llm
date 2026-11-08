import { MODELS, cacheKey, formatBytes, hfUrl, parseHfReference } from './models.js';
import { renderMarkdown, splitThinking } from './markdown.js';

const $ = (id) => document.getElementById(id);
const app = $('app');
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

const DEFAULTS = { system: '', temperature: 0.7, top_p: 0.95, top_k: 40, min_p: 0.05, repeat_penalty: 1.1, maxTokens: 1024, ctx: 4096, thinking: true };
const SUGGESTIONS = [
  'Explain WebGPU to a curious 12-year-old.',
  'Write a haiku about running AI on my own laptop.',
  'Give me three tips for writing clear Rust code.',
  'Plan a relaxed weekend in Lisbon in five bullet points.',
];

const state = {
  phase: 'starting', // starting | idle | loading | ready | generating
  adapter: null,
  cache: { models: [], supported: true },
  loading: null, // { label, id, key }
  loaded: null, // { id, key, label, info, chat }
  messages: [], // { role, content, stats?, error? }
  live: null, // the assistant message being written
  settings: loadSettings(),
};

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
  const ready = phase === 'ready';
  const generating = phase === 'generating';
  $('prompt').disabled = !(ready || generating);
  $('prompt').placeholder = state.loaded ? `Message ${state.loaded.label}` : 'Choose a model to start';
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
  $('gpu-name').textContent = describeGpu(adapter);
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
  return { kind: 'hf', id: m.id, key, repo: m.repo, file: m.file, revision: 'main', url: hfUrl(m.repo, m.file), bytes: m.bytes, name: m.name };
}

function renderModels() {
  const list = $('model-list');
  const busy = state.phase === 'loading' || state.phase === 'generating' || !state.adapter?.ok;
  const items = MODELS.map((m) => {
    const key = cacheKey(m.repo, m.file);
    const cached = cachedEntry(key);
    const saved = cached?.complete && cached.stored === cached.bytes;
    const partial = cached && !saved && cached.stored > 0;
    const running = state.loaded?.key === key;
    const tags = [
      el('span', { class: 'tag', text: m.quant }),
      running ? el('span', { class: 'tag run', text: 'Running' }) : null,
      saved && !running ? el('span', { class: 'tag ok', text: 'Saved' }) : null,
      partial ? el('span', { class: 'tag', text: `Paused at ${Math.round((cached.stored / cached.bytes) * 100)}%` }) : null,
      m.recommended && !saved && !running ? el('span', { class: 'tag rec', text: 'Recommended' }) : null,
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
  });
  // Downloads that are not in the catalog.
  for (const c of state.cache.models) {
    if (MODELS.some((m) => cacheKey(m.repo, m.file) === c.key)) continue;
    const running = state.loaded?.key === c.key;
    const entry = { kind: 'hf', id: c.key, key: c.key, repo: c.repo, file: c.file, revision: c.revision || 'main', url: c.url, bytes: c.bytes, name: c.file };
    items.push(el('li', {},
      el('button', { class: 'model', type: 'button', disabled: busy || running, 'aria-current': running ? 'true' : null, onclick: () => startLoad(entry) },
        el('span', { class: 'row' }, el('span', { class: 'name', text: c.file.replace(/\.gguf$/i, '') }), el('span', { class: 'size', text: formatBytes(c.bytes) })),
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

function startLoad(source) {
  if (!state.adapter?.ok) return;
  closeDrawer();
  if (state.phase === 'generating') return;
  const label = source.kind === 'file' ? source.file.name.replace(/\.gguf$/i, '') : source.name;
  state.loading = { label, key: source.key, id: source.id, kind: source.kind };
  state.loaded = null;
  state.messages = [];
  renderMessages();
  $('loading-title').textContent = label;
  for (const li of $('loading-steps').children) li.dataset.state = '';
  const saved = source.kind === 'hf' && cachedEntry(source.key)?.complete;
  setStep(source.kind === 'file' || saved ? 'upload' : 'download', source.kind === 'file' || saved);
  setProgress(0, 1, '');
  $('loading-step').textContent = saved ? 'Saved on this device' : source.kind === 'file' ? 'From this device' : 'Downloading';
  loadStarted = performance.now();
  setPhase('loading');
  if (source.kind === 'hf') remember('evolve-llm-last', source.id);
  worker.postMessage({ type: 'load', source, ctx: Number(state.settings.ctx) });
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

function onProgress({ phase, loaded, total, rate }) {
  const bar = $('loading-bar');
  bar.classList.toggle('indeterminate', phase === 'prepare');
  if (phase === 'download') {
    $('loading-step').textContent = 'Downloading from Hugging Face';
    setStep('download');
    const left = rate > 0 ? (total - loaded) / rate : 0;
    setProgress(loaded, total, `${formatBytes(loaded)} of ${formatBytes(total)} · ${formatBytes(rate)}/s${left > 1 ? ` · about ${formatDuration(left)} left` : ''}`);
  } else if (phase === 'upload') {
    $('loading-step').textContent = 'Moving weights to the GPU';
    setStep('upload', state.loading?.kind === 'file' || $('loading-steps').firstElementChild.dataset.state === 'skip');
    setProgress(loaded, total, `${formatBytes(loaded)} of ${formatBytes(total)} · ${formatBytes(rate)}/s`);
  } else {
    $('loading-step').textContent = 'Preparing the context';
    setStep('prepare', $('loading-steps').firstElementChild.dataset.state === 'skip');
    setProgress(1, 1, 'Allocating the key-value cache and compiling GPU kernels');
  }
}

function formatDuration(s) {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function onLoaded({ model, chat, source }) {
  const label = state.loading?.label || chat.name;
  state.loaded = { ...source, label, info: model, chat };
  state.loading = null;
  $('loading-bar').classList.remove('indeterminate');
  const secs = (performance.now() - loadStarted) / 1000;
  const cfg = model.config;
  const types = Object.keys(model.weight_types).join(', ');
  $('ready-title').textContent = `${label} is running on your GPU`;
  $('ready-meta').textContent = `${formatParams(model.parameters)} parameters · ${types} · ${cfg.n_layer} layers · ${chat.context.toLocaleString()}-token context · ${formatBytes(chat.weight_bytes + chat.state_bytes)} of GPU memory · ready in ${secs.toFixed(1)} s`;
  $('suggestions').replaceChildren(...SUGGESTIONS.map((s) => el('button', { type: 'button', text: s, onclick: () => send(s) })));
  $('status-line').textContent = 'Replies are generated on this device. Responses can be wrong; check anything important.';
  setPhase('ready');
  $('prompt').focus();
}

function formatParams(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${Math.round(n / 1e6)}M`;
}

// ---- Conversation ----

function renderMessages() {
  const list = $('messages');
  list.replaceChildren(...state.messages.map((m, i) => messageNode(m, i)));
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

function fillAssistant(bubble, m) {
  const parts = splitThinking(m.content);
  const nodes = [];
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
    return el('li', { class: 'msg error' }, el('div', { class: 'bubble', text: m.content }));
  }
  if (m.role === 'user') {
    return el('li', { class: 'msg user' }, el('div', { class: 'bubble', text: m.content }));
  }
  const bubble = el('div', { class: 'bubble' });
  fillAssistant(bubble, m);
  const li = el('li', { class: 'msg assistant', 'data-index': i }, el('span', { class: 'who', text: state.loaded?.label || 'Model' }), bubble);
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
  for (const m of state.messages) if (!m.error && (m.role === 'user' || m.role === 'assistant')) msgs.push({ role: m.role, content: m.content });
  return msgs;
}

function send(text) {
  text = text.trim();
  if (!text || state.phase !== 'ready') return;
  state.messages.push({ role: 'user', content: text });
  runGeneration();
}

function regenerate() {
  if (state.phase !== 'ready') return;
  while (state.messages.length && state.messages[state.messages.length - 1].role !== 'user') state.messages.pop();
  if (state.messages.length) runGeneration();
}

function runGeneration() {
  const now = new Date();
  const s = state.settings;
  const history = conversation();
  state.live = { role: 'assistant', content: '', started: performance.now() };
  state.messages.push(state.live);
  setPhase('generating');
  renderMessages();
  $('scroll').scrollTop = $('scroll').scrollHeight;
  worker.postMessage({
    type: 'generate',
    messages: history,
    options: { date: [now.getFullYear(), now.getMonth() + 1, now.getDate()], enable_thinking: Boolean(s.thinking) },
    sampler: { temperature: Number(s.temperature), top_p: Number(s.top_p), top_k: Number(s.top_k), min_p: Number(s.min_p), repeat_penalty: Number(s.repeat_penalty), repeat_last_n: 64, seed: (Math.random() * 2 ** 32) >>> 0 },
    maxTokens: Number(s.maxTokens),
  });
}

function onPrefill(p) {
  if (!state.live) return;
  state.live.prefill = p;
  if (p.dropped) state.live.dropped = p.dropped;
}

function onDone(d) {
  const m = state.live;
  if (!m) return;
  const p = m.prefill || {};
  const rate = d.tokens > 0 && d.ms > 0 ? (d.tokens / (d.ms / 1000)).toFixed(1) : '0';
  const prompt = p.computed ? ` · read ${p.computed} prompt tokens in ${(p.ms / 1000).toFixed(2)} s${p.reused ? ` (${p.reused} reused)` : ''}` : '';
  const why = d.reason === 'stopped' ? ' · stopped' : d.reason === 'length' ? ' · reached the reply limit' : '';
  m.stats = `${d.tokens} tokens · ${rate} tokens/s${prompt}${why}`;
  if (!m.content.trim()) m.content = d.reason === 'stopped' ? '' : '(no reply)';
  state.live = null;
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

worker.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'adapter': showGpu(msg.adapter); break;
    case 'cache': state.cache = msg; renderModels(); break;
    case 'progress': onProgress(msg); break;
    case 'inspected': break;
    case 'loaded': onLoaded(msg); break;
    case 'prefill': onPrefill(msg); break;
    case 'token':
      if (state.live) {
        state.live.content += msg.text;
        scheduleLiveRender();
      }
      break;
    case 'done': onDone(msg); break;
    case 'error': onError(msg); break;
    case 'unloaded': break;
  }
};
worker.onerror = (e) => {
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
$('stop').addEventListener('click', () => worker.postMessage({ type: 'stop' }));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.phase === 'generating') worker.postMessage({ type: 'stop' });
});
$('cancel-load').addEventListener('click', () => worker.postMessage({ type: 'cancel' }));
$('new-chat').addEventListener('click', () => {
  if (state.phase !== 'ready') return;
  state.messages = [];
  worker.postMessage({ type: 'reset' });
  renderMessages();
  setPhase('ready');
  $('status-line').textContent = 'Started a new conversation.';
  prompt.focus();
});
$('start-recommended').addEventListener('click', () => {
  const last = MODELS.find((m) => m.id === recall('evolve-llm-last'));
  startLoad(catalogEntry(last || MODELS.find((m) => m.recommended)));
});
$('hf-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const help = $('hf-help');
  try {
    const ref = parseHfReference($('hf-input').value);
    help.classList.remove('error');
    help.textContent = 'Llama, Mistral, Qwen2, Qwen3 and SmolLM layouts. Q4, Q5, Q6, Q8, IQ4 and F16 weights.';
    const key = cacheKey(ref.repo, ref.file, ref.revision);
    startLoad({ kind: 'hf', id: key, key, ...ref, url: hfUrl(ref.repo, ref.file, ref.revision), bytes: 0, name: ref.file.split('/').pop().replace(/\.gguf$/i, '') });
  } catch (err) {
    help.classList.add('error');
    help.textContent = err.message;
  }
});
$('file-input').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (file) startLoad({ kind: 'file', file, key: null, id: null });
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
  form.elements.thinking.checked = Boolean(s.thinking);
}
form.addEventListener('input', (e) => {
  const t = e.target;
  if (!t.name) return;
  if (t.type === 'checkbox') state.settings[t.name] = t.checked;
  else if (t.type === 'range') {
    state.settings[t.name] = Number(t.value);
    form.elements[`${t.name}Out`].value = t.value;
  } else if (t.name === 'ctx') state.settings.ctx = Number(t.value);
  else state.settings[t.name] = t.value;
  saveSettings();
});
form.addEventListener('change', (e) => {
  if (e.target.name === 'ctx' && state.loaded) $('status-line').textContent = 'The new context length applies the next time a model loads.';
});
$('open-settings').addEventListener('click', () => { fillSettings(); dialog.showModal(); });
$('reset-settings').addEventListener('click', () => { state.settings = { ...DEFAULTS }; saveSettings(); fillSettings(); });

setPhase('starting');
worker.postMessage({ type: 'hello' });
