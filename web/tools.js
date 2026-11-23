import { publicUrl, requestOptions, boundedText } from './network.js';
// Internet access for the model, straight from the browser: no server of ours sees the
// conversation. Search terms go to Wikipedia and DuckDuckGo (and Jina, with the person's own
// key); pages the model opens are read through Jina Reader, or Wikipedia's API for articles.

export const TOOL_SCHEMAS = [
  { type: 'function', function: { name: 'web_search', description: 'Search the web. Returns numbered results with links.',
    parameters: { type: 'object', properties: { query: { type: 'string', description: 'What to search for' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'open_url', description: 'Read the text of a web page, such as a search result.',
    parameters: { type: 'object', properties: { url: { type: 'string', description: 'The http or https address' } }, required: ['url'] } } },
];

const NAMES = new Set(TOOL_SCHEMAS.map((t) => t.function.name));
const PAGE_CHARS = 6000;

function call(obj, allowed = NAMES) {
  if (!obj || typeof obj !== 'object') return null;
  const name = obj.name || obj.function?.name;
  let args = obj.arguments ?? obj.parameters ?? obj.function?.arguments ?? {};
  if (typeof args === 'string') {
    try { args = JSON.parse(args); } catch { args = {}; }
  }
  if (!args || typeof args !== 'object') return null;
  if (allowed.has(name)) return { name, args };
  // A small model may call a command as if it were a tool: {"name": "git", "arguments":
  // {"command": "commit -m 'x'"}}. With a bash tool, that is the command line "git commit -m 'x'".
  const rest = typeof args.command === 'string' ? args.command : typeof args.args === 'string' ? args.args : Array.isArray(args.args) ? args.args.join(' ') : null;
  if (allowed.has('bash') && typeof name === 'string' && /^[a-z][\w.+-]*$/.test(name) && rest !== null && Object.keys(args).length === 1)
    return { name: 'bash', args: { command: rest.trim().startsWith(name + ' ') || rest.trim() === name ? rest.trim() : `${name} ${rest}`.trim() } };
  return null;
}

function parseJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

/**
 * Control tokens that carry reply markup (thinking and response channels, tool calls). Decoding
 * hides control tokens; the worker keeps these in the raw reply so calls can be read from it.
 * Kimi K2's GGUF names its tool markers as reserved tokens, so those are mapped to their names.
 */
export const MARKUP = new Set([
  '<|open|>', '<|sep|>', '<|close|>', '<|end_of_msg|>',
  '<｜tool▁calls▁begin｜>', '<｜tool▁call▁begin｜>', '<｜tool▁sep｜>', '<｜tool▁call▁end｜>', '<｜tool▁calls▁end｜>',
  '<|tool_calls_section_begin|>', '<|tool_calls_section_end|>', '<|tool_call_begin|>', '<|tool_call_argument_begin|>', '<|tool_call_end|>',
  '<|python_tag|>', '[TOOL_CALLS]',
]);
const RESERVED = {
  '<|reserved_token_163595|>': '<|tool_calls_section_begin|>',
  '<|reserved_token_163596|>': '<|tool_calls_section_end|>',
  '<|reserved_token_163597|>': '<|tool_call_begin|>',
  '<|reserved_token_163598|>': '<|tool_call_argument_begin|>',
  '<|reserved_token_163599|>': '<|tool_call_end|>',
};

/** The raw-reply text of one generated token: its markup if it is a markup control token. */
export function rawPiece(controlText, piece) {
  const t = RESERVED[controlText] || controlText;
  return t && MARKUP.has(t) ? t : piece;
}

const unescapeAttr = (s) => s.replace(/&quot;/g, '"').replace(/&amp;/g, '&');

/** A parameter value as its schema type wants it (integers, booleans, objects…). */
function coerce(value, type) {
  const v = String(value);
  if (type === 'integer' || type === 'number') return Number.isFinite(Number(v.trim())) ? Number(v.trim()) : v;
  if (type === 'boolean') return v.trim() === 'true' ? true : v.trim() === 'false' ? false : v;
  if (type === 'object' || type === 'array' || type === 'null') return parseJson(v) ?? v;
  return v;
}

// Each family's call syntax, as (name, arguments object or JSON text) pairs.
function familyCalls(text, types) {
  // Kimi K3: <|open|>call tool="name" index="1"<|sep|> typed arguments or a JSON block <|close|>call<|sep|>
  const k3 = [...text.matchAll(/<\|open\|>call tool="([^"]*)"[^<]*<\|sep\|>([\s\S]*?)<\|close\|>call<\|sep\|>/g)];
  if (k3.length) return k3.map((m) => {
    const json = /<\|open\|>json[^<]*<\|sep\|>([\s\S]*?)<\|close\|>json<\|sep\|>/.exec(m[2]);
    if (json) return { name: unescapeAttr(m[1]), arguments: json[1] };
    const args = {};
    for (const a of m[2].matchAll(/<\|open\|>argument key="([^"]*)" type="([^"]*)"<\|sep\|>([\s\S]*?)<\|close\|>argument<\|sep\|>/g)) {
      args[unescapeAttr(a[1])] = coerce(a[3], a[2]);
    }
    return { name: unescapeAttr(m[1]), arguments: args };
  });
  // DeepSeek V3.1: <｜tool▁call▁begin｜>name<｜tool▁sep｜>{json}<｜tool▁call▁end｜>; V3 puts
  // `function` first and the name and a fenced block after the separator.
  const ds = [...text.matchAll(/<｜tool▁call▁begin｜>([\s\S]*?)<｜tool▁sep｜>([\s\S]*?)<｜tool▁call▁end｜>/g)];
  if (ds.length) return ds.map((m) => {
    if (m[1].trim() === 'function') {
      const v3 = /^\s*([^\n]+)\n```(?:json)?\n([\s\S]*?)\n```/.exec(m[2]);
      return v3 ? { name: v3[1].trim(), arguments: v3[2] } : { name: '', arguments: null };
    }
    return { name: m[1].trim(), arguments: m[2] };
  });
  // Kimi K2: <|tool_call_begin|>functions.name:0<|tool_call_argument_begin|>{json}<|tool_call_end|>
  const k2 = [...text.matchAll(/<\|tool_call_begin\|>\s*([^<]*?)\s*<\|tool_call_argument_begin\|>([\s\S]*?)<\|tool_call_end\|>/g)];
  if (k2.length) return k2.map((m) => ({ name: m[1].replace(/^functions\./, '').replace(/:\d+$/, ''), arguments: m[2] }));
  // Qwen3 Coder: <function=name><parameter=key>\nvalue\n</parameter></function>
  const qc = [...text.matchAll(/<function=([^>\n]+)>([\s\S]*?)<\/function>/g)];
  if (qc.length) return qc.map((m) => {
    const name = m[1].trim();
    const args = {};
    for (const a of m[2].matchAll(/<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g)) {
      const key = a[1].trim();
      args[key] = coerce(a[2], types?.[name]?.[key]);
    }
    return { name, arguments: args };
  });
  // Mistral: [TOOL_CALLS] [{"name": …, "arguments": {…}}]
  const ms = /\[TOOL_CALLS\]\s*(\[[\s\S]*\])/.exec(text);
  if (ms) {
    const list = parseJson(ms[1]);
    if (Array.isArray(list)) return list.map((c) => ({ name: c?.name, arguments: c?.arguments ?? c?.parameters ?? {} }));
  }
  return null;
}

/**
 * Tool calls in a model's raw reply, in whichever syntax its template taught it: Kimi K3 and
 * K2, DeepSeek V3 and V3.1, Qwen3 Coder, Mistral, Qwen/Hermes `<tool_call>{json}</tool_call>`
 * blocks, or a reply that is one JSON call object (bare, fenced, or after `<|python_tag|>`, as
 * Llama 3 writes it). `types` maps tool name -> parameter -> JSON-schema type, for syntaxes
 * whose values are untyped text. A reply whose calls are all invalid yields none.
 */
// A shell command written where the tool name goes, with its quotes unescaped so the JSON does
// not parse ({"name": "gh pr create --title "Fix" --body "..."}): small models do this. When
// the name starts with a common command and bash is a tool, it is that bash command.
const SHELL_WORDS = /^(git|gh|ls|cat|cd|grep|sed|echo|mkdir|touch|rm|mv|cp|python3|node|npm|npx|cargo|head|tail|wc|find|diff|edit)\s/;
function commandAsName(body, allowed) {
  const m = /^\{\s*"name"\s*:\s*"([\s\S]+)"\s*\}$/.exec(String(body).trim());
  if (!m || !SHELL_WORDS.test(m[1]) || ![...allowed].includes('bash')) return null;
  // The JSON string's closing quote often also closes the last argument ("--body "Done."}).
  const quotes = (m[1].match(/(?<!\\)"/g) || []).length;
  return call({ name: 'bash', arguments: { command: quotes % 2 ? `${m[1]}"` : m[1] } }, allowed);
}

export function parseToolCalls(text, allowed = NAMES, types = null) {
  text = String(text);
  const native = familyCalls(text, types);
  if (native) {
    const parsed = native.map((c) => call({ name: c.name, arguments: c.arguments }, allowed));
    return parsed.every(Boolean) ? parsed : [];
  }
  const calls = [];
  for (const m of text.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)) {
    const c = call(parseJson(m[1]), allowed) || commandAsName(m[1], allowed);
    if (c) calls.push(c);
  }
  // Small models often end the reply right after the JSON, without </tool_call>. What follows
  // the last opening tag still counts when it is complete JSON (a cut-off call does not parse).
  const open = text.lastIndexOf('<tool_call>');
  if (open >= 0 && !text.slice(open).includes('</tool_call>')) {
    const c = call(parseJson(text.slice(open + '<tool_call>'.length).trim()), allowed);
    if (c) calls.push(c);
  }
  if (calls.length) return calls;
  const bare = text.replace(/^\s*<\|python_tag\|>/, '').trim();
  // A reply consisting entirely of complete JSON blocks is another call
  // envelope. Prose, truncated blocks, and invalid calls reject the whole reply.
  const fencePattern = /```(?:json)?\s*\n([\s\S]*?)\n```/g;
  const fenced = [...bare.matchAll(fencePattern)];
  if (fenced.length && !bare.replace(fencePattern, '').trim()) {
    const parsed = fenced.map(m => call(parseJson(m[1]), allowed));
    return parsed.every(Boolean) ? parsed : [];
  }
  if (bare.startsWith('{') && bare.endsWith('}')) {
    const c = call(parseJson(bare), allowed);
    if (c) calls.push(c);
  }
  return calls;
}

/** True when a raw reply tried to call a tool in any known syntax (valid or not). */
export function attemptedToolCall(text) {
  return /<tool_call>|"name"\s*:|<\|open\|>tools<\|sep\|>|<｜tool▁call|<\|tool_call_begin\|>|<function=|\[TOOL_CALLS\]/.test(String(text));
}

/**
 * The raw reply as the person should see it: tool-call markup removed (an unfinished call too),
 * Kimi K3's thinking channel as <think>…</think> and its other channel tags removed, and a
 * markup tag still being written held back, so the text only grows while a reply streams.
 */
export function displayText(raw) {
  let t = String(raw)
    .replace(/<\|open\|>tools<\|sep\|>[\s\S]*?(<\|close\|>tools<\|sep\|>|$)/g, '')
    .replace(/<\|open\|>think<\|sep\|>/g, '<think>')
    .replace(/<\|close\|>think<\|sep\|>/g, '</think>')
    .replace(/<\|(?:open|close)\|>[^<]*?<\|sep\|>/g, '')
    .replace(/<\|end_of_msg\|>/g, '')
    .replace(/<｜tool▁calls▁begin｜>[\s\S]*?(<｜tool▁calls▁end｜>|$)/g, '')
    .replace(/<\|tool_calls_section_begin\|>[\s\S]*?(<\|tool_calls_section_end\|>|$)/g, '')
    .replace(/\[TOOL_CALLS\][\s\S]*$/, '')
    .replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, '')
    .replace(/<\|python_tag\|>/g, '');
  return t.replace(/<\|(?:open|close)\|>[^<]*$/, '');
}

/** The reply without tool-call markup, for display (an unfinished call is hidden too). */
export function stripToolCalls(text) {
  let t = displayText(text);
  const bare = t.trim();
  if (bare.startsWith('{') && /"(name|function)"\s*:/.test(bare) && /"(web_search|open_url)"/.test(bare)) t = '';
  return t;
}

/** Parameter types of tool schemas (OpenAI style), for `parseToolCalls`. */
export function schemaTypes(schemas) {
  const out = {};
  for (const s of schemas || []) {
    const f = s.function || s;
    out[f.name] = Object.fromEntries(Object.entries(f.parameters?.properties || {}).map(([k, v]) => [k, v.type]));
  }
  return out;
}

const strip = (html) => String(html || '').replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&#039;/g, "'").replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');

function language() {
  const l = (globalThis.navigator?.language || 'en').split('-')[0].toLowerCase();
  return /^[a-z]{2,3}$/.test(l) ? l : 'en';
}

async function getJson(fetchFn, url, init) {
  const res = await fetchFn(url, init);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  if (!res.body && !res.text) return res.json();
  return JSON.parse(await boundedText(res, 256 * 1024));
}

async function wikipediaSearch(query, { fetchFn, lang }) {
  const url = `https://${lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&srlimit=5&format=json&utf8=1&origin=*`;
  const d = await getJson(fetchFn, url);
  return (d.query?.search || []).map((r) => ({
    title: `${r.title} (Wikipedia)`,
    url: `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g, '_'))}`,
    snippet: strip(r.snippet),
  }));
}

async function duckduckgo(query, { fetchFn }) {
  const d = await getJson(fetchFn, `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1&t=evolve-llm`);
  const out = [];
  if (d.AbstractText && d.AbstractURL) out.push({ title: d.Heading || d.AbstractSource || 'DuckDuckGo', url: d.AbstractURL, snippet: d.AbstractText });
  const topics = (d.RelatedTopics || []).flatMap((t) => (t.Topics ? t.Topics : [t]));
  for (const t of topics.slice(0, 4)) {
    if (t.FirstURL && t.Text) out.push({ title: t.Text.split(' - ')[0].slice(0, 80), url: t.FirstURL, snippet: t.Text });
  }
  return out;
}

async function jinaSearch(query, { fetchFn, jinaKey }) {
  const d = await getJson(fetchFn, `https://s.jina.ai/?q=${encodeURIComponent(query)}`, {
    headers: { Accept: 'application/json', Authorization: `Bearer ${jinaKey}`, 'X-Respond-With': 'no-content' },
  });
  return (d.data || []).slice(0, 6).map((r) => ({ title: r.title || r.url, url: r.url, snippet: r.description || '' }));
}

/** Numbered results for the model, and the same results as links for the page. */
export async function webSearch(query, opts = {}) {
  const baseFetch = opts.fetchFn || fetch;
  const o = { fetchFn: (url, init) => { const policy=requestOptions(opts); return baseFetch(url, { ...policy, ...init, signal:policy.signal }); }, lang: opts.lang || language(), jinaKey: opts.jinaKey || '' };
  const q = String(query || '').trim().slice(0, 200);
  if (!q) return { text: 'The search query was empty.', sources: [] };
  opts.signal?.throwIfAborted();
  const jobs = [wikipediaSearch(q, o), duckduckgo(q, o)];
  if (o.jinaKey) jobs.unshift(jinaSearch(q, o));
  const settled = await Promise.allSettled(jobs);
  opts.signal?.throwIfAborted();
  const seen = new Set();
  const sources = [];
  for (const s of settled) {
    if (s.status !== 'fulfilled') continue;
    for (const r of s.value) {
      if (!r.url || seen.has(r.url)) continue;
      try { r.url = publicUrl(r.url); } catch { continue; }
      seen.add(r.url);
      sources.push(r);
    }
  }
  if (!sources.length) {
    const failed = settled.every((s) => s.status === 'rejected');
    return { text: failed ? 'The search services could not be reached.' : `No results for "${q}".`, sources: [] };
  }
  const text = sources.slice(0, 8).map((r, i) => `[${i + 1}] ${r.title}\n${r.url}\n${r.snippet.slice(0, 300)}`).join('\n\n');
  return { text, sources: sources.slice(0, 8) };
}

/** The readable text of one page. */
export async function openUrl(url, opts = {}) {
  const fetchFn = opts.fetchFn || fetch;
  let u;
  try { u = new URL(String(url || '').trim()); } catch { return { text: 'That is not a web address.', sources: [] }; }
  if (!['https:','http:'].includes(u.protocol)) return { text: 'Only http and https pages can be opened.', sources: [] };
  try { u = new URL(publicUrl(u.href)); } catch { return { text: 'Private, local and credential-bearing URLs are unavailable.', sources: [] }; }
  if (u.protocol !== 'https:' && u.protocol !== 'http:') return { text: 'Only http and https pages can be opened.', sources: [] };
  const wiki = /^([a-z-]+)\.(m\.)?wikipedia\.org$/.exec(u.hostname);
  try {
    if (wiki && u.pathname.startsWith('/wiki/')) {
      const title = decodeURIComponent(u.pathname.slice(6));
      const d = await getJson((url,init) => { const policy=requestOptions(opts); return fetchFn(url,{...policy,...init,signal:policy.signal}); }, `https://${wiki[1]}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&origin=*&titles=${encodeURIComponent(title)}`);
      const page = Object.values(d.query?.pages || {})[0];
      if (!page?.extract) return { text: `Wikipedia has no article "${title}".`, sources: [] };
      return { text: page.extract.slice(0, PAGE_CHARS), sources: [{ title: `${page.title} (Wikipedia)`, url: u.href }] };
    }
    const headers = { Accept: 'text/plain' };
    if (opts.jinaKey) headers.Authorization = `Bearer ${opts.jinaKey}`;
    const res = await fetchFn(`https://r.jina.ai/${u.href}`, { ...requestOptions(opts), headers });
    if (!res.ok) return { text: `The page could not be read (HTTP ${res.status}).`, sources: [] };
    const body = await boundedText(res, 65536, true);
    const title = /^Title:\s*(.+)$/m.exec(body)?.[1] || u.hostname;
    return { text: body.slice(0, PAGE_CHARS), sources: [{ title, url: u.href }] };
  } catch {
    opts.signal?.throwIfAborted();
    return { text: 'The page could not be reached.', sources: [] };
  }
}

export async function runTool(c, opts = {}) {
  if (opts.enabled === false) return { text: 'Internet access is off.', sources: [] };
  opts.signal?.throwIfAborted();
  if (c.name === 'web_search') return webSearch(c.args.query, opts);
  if (c.name === 'open_url') return openUrl(c.args.url, opts);
  return { text: `Unknown tool ${c.name}.`, sources: [] };
}

// Short-lived, per-worker cache. No persisted keys or private request data.
export function createWebTools(fetchFn = globalThis.fetch) {
  const cache = new Map();
  return async (name, args, opts = {}) => {
    if (!opts.enabled) throw new Error('Web access is off. Enable Web to use network commands.');
    opts.signal?.throwIfAborted();
    const key = JSON.stringify([name,args,opts.jinaKey || '',opts.lang || '']);
    const old = cache.get(key);
    if (old && Date.now() - old.time < 60000) return { ...old.result, cached: true };
    const result = await runTool({name,args},{...opts,fetchFn});
    opts.signal?.throwIfAborted();
    if(result.sources.length) { cache.set(key,{time:Date.now(),result}); if(cache.size>32)cache.delete(cache.keys().next().value); }
    return result;
  };
}
