// Unit tests for the browser modules that do not need a GPU: node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseBlocks, parseInline, splitThinking } from '../web/markdown.js';
import { LARGE, MODELS, cacheKey, formatBytes, hfUrl, listRepo, parseHfReference, resolveParts, splitParts } from '../web/models.js';

test('splits reasoning from the answer, including an unfinished thought', () => {
  assert.deepEqual(splitThinking('<think>a\nb</think>\n\nAnswer'), [
    { kind: 'think', text: 'a\nb', done: true },
    { kind: 'answer', text: '\n\nAnswer' },
  ]);
  assert.deepEqual(splitThinking('<think>still going'), [{ kind: 'think', text: 'still going', done: false }]);
  assert.deepEqual(splitThinking('<think>\n\n</think>\n\nHi'), [{ kind: 'think', text: '\n\n', done: true }, { kind: 'answer', text: '\n\nHi' }]);
});

test('parses the block structure models produce', () => {
  const blocks = parseBlocks('# Title\n\nText **bold**\nnext line\n\n1. one\n2. two\n\n> quote\n\n---\n\n```rust\nfn main() {}\n```\n\n```py\nunfinished\n');
  assert.deepEqual(blocks.map((b) => b.type), ['heading', 'paragraph', 'list', 'quote', 'rule', 'code', 'code']);
  assert.equal(blocks[2].ordered, true);
  assert.deepEqual(blocks[2].items, ['one', 'two']);
  assert.equal(blocks[5].text, 'fn main() {}');
  assert.equal(blocks[5].closed, true);
  assert.equal(blocks[6].text, 'unfinished');
  assert.equal(blocks[6].closed, false);
});

test('inline markup never produces raw HTML and only links to web addresses', () => {
  const spans = parseInline('<img src=x onerror=alert(1)> **b** `c` [x](javascript:alert(1)) [ok](https://example.com)');
  assert.equal(spans[0].type, 'text');
  assert.ok(spans[0].text.includes('<img'));
  const links = spans.filter((s) => s.type === 'link');
  assert.deepEqual(links.map((l) => l.href), ['https://example.com']);
  assert.ok(spans.some((s) => s.type === 'strong'));
  assert.ok(spans.some((s) => s.type === 'code' && s.text === 'c'));
});

test('reads Hugging Face references in the forms people paste', () => {
  assert.deepEqual(parseHfReference('Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf'), { repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: 'main' });
  assert.deepEqual(parseHfReference('https://huggingface.co/bartowski/x-GGUF/blob/main/sub/m-Q4_K_M.gguf?download=true'), { repo: 'bartowski/x-GGUF', file: 'sub/m-Q4_K_M.gguf', revision: 'main' });
  assert.throws(() => parseHfReference('https://example.com/a/b/c.gguf'), /huggingface\.co/);
  assert.throws(() => parseHfReference('a/b/model.bin'), /\.gguf/);
  assert.deepEqual(parseHfReference('a/b/model-00002-of-00003.gguf'), { repo: 'a/b', file: 'model-00002-of-00003.gguf', revision: 'main' });
  assert.deepEqual(parseHfReference('unsloth/Kimi-K2-Instruct-0905-GGUF'), { repo: 'unsloth/Kimi-K2-Instruct-0905-GGUF', file: null, revision: 'main' });
  assert.throws(() => parseHfReference('a/b/../c.gguf'), /characters/);
  assert.equal(hfUrl('a/b', 'dir/m 1.gguf'), 'https://huggingface.co/a/b/resolve/main/dir/m%201.gguf');
});

test('split models are found and listed part by part', async () => {
  assert.deepEqual(splitParts('Q2/m-00002-of-00003.gguf'), ['Q2/m-00001-of-00003.gguf', 'Q2/m-00002-of-00003.gguf', 'Q2/m-00003-of-00003.gguf']);
  assert.equal(splitParts('m.gguf'), null);
  const tree = [
    { type: 'file', path: 'Q2/m-00002-of-00002.gguf', size: 0, lfs: { size: 20 } },
    { type: 'file', path: 'Q2/m-00001-of-00002.gguf', size: 0, lfs: { size: 10 } },
    { type: 'file', path: 'small.gguf', size: 5 },
    { type: 'file', path: 'mmproj-f16.gguf', size: 1 },
    { type: 'file', path: 'README.md', size: 1 },
    { type: 'directory', path: 'Q2' },
  ];
  const fake = async () => ({ ok: true, status: 200, json: async () => tree });
  const groups = await listRepo('a/b', 'main', fake);
  assert.deepEqual(groups.map((g) => [g.file, g.bytes, g.parts.length]), [['small.gguf', 5, 1], ['Q2/m-00001-of-00002.gguf', 30, 2]]);
  const r = await resolveParts('a/b', 'Q2/m-00002-of-00002.gguf', 'main', fake);
  assert.deepEqual(r.parts.map((p) => p.file), ['Q2/m-00001-of-00002.gguf', 'Q2/m-00002-of-00002.gguf']);
  assert.equal(r.parts[1].url, 'https://huggingface.co/a/b/resolve/main/Q2/m-00002-of-00002.gguf');
});

test('catalog entries are complete and storage keys are stable', () => {
  const ids = new Set();
  for (const m of [...MODELS, ...LARGE]) {
    assert.ok(m.repo && m.file.endsWith('.gguf') && m.bytes > 1e8 && m.license && m.licenseUrl, m.id);
    assert.ok(!ids.has(m.id), `duplicate ${m.id}`);
    ids.add(m.id);
  }
  assert.equal(MODELS.filter((m) => m.recommended).length, 1);
  assert.equal(cacheKey('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf'), 'Qwen__Qwen3-0.6B-GGUF__main__Qwen3-0.6B-Q8_0.gguf');
  assert.equal(formatBytes(491400032), '491 MB');
  assert.equal(formatBytes(2497280256), '2.50 GB');
});

import { openUrl, parseToolCalls, stripToolCalls, webSearch } from '../web/tools.js';

test('reads tool calls in the forms models write them', () => {
  assert.deepEqual(parseToolCalls('Let me look.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "WebGPU"}}\n</tool_call>'),
    [{ name: 'web_search', args: { query: 'WebGPU' } }]);
  assert.deepEqual(parseToolCalls('{"name": "open_url", "parameters": {"url": "https://example.com"}}'),
    [{ name: 'open_url', args: { url: 'https://example.com' } }]);
  assert.deepEqual(parseToolCalls('<|python_tag|>{"name": "web_search", "parameters": {"query": "x"}}'), [{ name: 'web_search', args: { query: 'x' } }]);
  assert.deepEqual(parseToolCalls('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'), []);
  assert.deepEqual(parseToolCalls('The answer is {"name": "web_search"} in prose.'), []);
  const fenced = '```json\n{"name":"web_search","arguments":{"query":"x"}}\n```';
  assert.deepEqual(parseToolCalls(fenced), [{name:'web_search',args:{query:'x'}}]);
  assert.deepEqual(parseToolCalls('For example:\n' + fenced), []);
  assert.deepEqual(parseToolCalls(fenced + '\n' + fenced), [{name:'web_search',args:{query:'x'}},{name:'web_search',args:{query:'x'}}]);
  assert.deepEqual(parseToolCalls(fenced + '\n```json\n{"name":"web_search"\n```'), []);
  assert.deepEqual(parseToolCalls('```json\n{"name":"web_search","arguments":{\n```'), []);
  assert.equal(stripToolCalls('Sure.<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>'), 'Sure.');
  assert.equal(stripToolCalls('Thinking <tool_call>{"name": "web'), 'Thinking ');
  assert.equal(stripToolCalls('{"name": "web_search", "parameters": {"query": "a"}}'), '');
  assert.equal(stripToolCalls('Plain answer.'), 'Plain answer.');
});

function fakeFetch(routes) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    for (const [prefix, body] of routes) {
      if (url.startsWith(prefix)) {
        return { ok: true, status: 200, json: async () => body, text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
      }
    }
    return { ok: false, status: 503, json: async () => ({}), text: async () => '' };
  };
  fn.calls = calls;
  return fn;
}

test('web search merges Wikipedia and DuckDuckGo results with numbered links', async () => {
  const fetchFn = fakeFetch([
    ['https://en.wikipedia.org/w/api.php?action=query&list=search', { query: { search: [{ title: 'WebGPU', snippet: 'A <span class="searchmatch">graphics</span> API &amp; more' }] } }],
    ['https://api.duckduckgo.com/', { Heading: 'WebGPU', AbstractText: 'WebGPU is a web API.', AbstractURL: 'https://en.wikipedia.org/wiki/WebGPU',
      RelatedTopics: [{ Text: 'WGSL - shading language', FirstURL: 'https://duckduckgo.com/WGSL' }, { Topics: [{ Text: 'Vulkan - API', FirstURL: 'https://duckduckgo.com/Vulkan' }] }] }],
  ]);
  const r = await webSearch('webgpu', { fetchFn, lang: 'en' });
  assert.deepEqual(r.sources.map((s) => s.url), ['https://en.wikipedia.org/wiki/WebGPU', 'https://duckduckgo.com/WGSL', 'https://duckduckgo.com/Vulkan']);
  assert.match(r.text, /^\[1\] WebGPU \(Wikipedia\)\nhttps:\/\/en\.wikipedia\.org\/wiki\/WebGPU\nA graphics API & more/);
  assert.match(r.text, /\[3\] Vulkan/);
  assert.ok(fetchFn.calls.every((c) => !c.init?.headers?.Authorization), 'no key is sent without one');
  const none = await webSearch('x', { fetchFn: fakeFetch([]), lang: 'en' });
  assert.equal(none.text, 'The search services could not be reached.');
});

test('pages open through Wikipedia or Jina Reader, and only web addresses open', async () => {
  const fetchFn = fakeFetch([
    ['https://de.wikipedia.org/w/api.php?action=query&prop=extracts', { query: { pages: { 1: { title: 'WebGPU', extract: 'Text '.repeat(3000) } } } }],
    ['https://r.jina.ai/https://example.com/a', 'Title: Example\n\nBody'],
  ]);
  const w = await openUrl('https://de.wikipedia.org/wiki/WebGPU', { fetchFn });
  assert.equal(w.text.length, 6000);
  assert.equal(w.sources[0].title, 'WebGPU (Wikipedia)');
  const j = await openUrl('https://example.com/a', { fetchFn });
  assert.equal(j.sources[0].title, 'Example');
  assert.match((await openUrl('javascript:alert(1)', { fetchFn })).text, /Only http/);
  assert.match((await openUrl('not a url', { fetchFn })).text, /not a web address/);
});
