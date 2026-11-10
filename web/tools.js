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
  return allowed.has(name) && args && typeof args === 'object' ? { name, args } : null;
}

function parseJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

/**
 * Tool calls in a model's reply: Qwen/Hermes `<tool_call>{json}</tool_call>` blocks, or a reply
 * that is one JSON call object as Llama 3 writes it (optionally after `<|python_tag|>`).
 */
export function parseToolCalls(text, allowed = NAMES) {
  const calls = [];
  for (const m of String(text).matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)) {
    const c = call(parseJson(m[1]), allowed);
    if (c) calls.push(c);
  }
  if (calls.length) return calls;
  const bare = String(text).replace(/^\s*<\|python_tag\|>/, '').trim();
  if (bare.startsWith('{') && bare.endsWith('}')) {
    const c = call(parseJson(bare), allowed);
    if (c) calls.push(c);
  }
  return calls;
}

/** The reply without tool-call markup, for display (an unfinished call is hidden too). */
export function stripToolCalls(text) {
  let t = String(text).replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, '');
  const bare = t.replace(/^\s*<\|python_tag\|>/, '').trim();
  if (bare.startsWith('{') && /"(name|function)"\s*:/.test(bare) && /"(web_search|open_url)"/.test(bare)) t = '';
  return t;
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
