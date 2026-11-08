// Unit tests for the browser modules that do not need a GPU: node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseBlocks, parseInline, splitThinking } from '../web/markdown.js';
import { MODELS, cacheKey, formatBytes, hfUrl, parseHfReference } from '../web/models.js';

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
  assert.throws(() => parseHfReference('a/b/model-00001-of-00002.gguf'), /Split/);
  assert.throws(() => parseHfReference('a/b/../c.gguf'), /characters/);
  assert.equal(hfUrl('a/b', 'dir/m 1.gguf'), 'https://huggingface.co/a/b/resolve/main/dir/m%201.gguf');
});

test('catalog entries are complete and storage keys are stable', () => {
  const ids = new Set();
  for (const m of MODELS) {
    assert.ok(m.repo && m.file.endsWith('.gguf') && m.bytes > 1e8 && m.license && m.licenseUrl, m.id);
    assert.ok(!ids.has(m.id), `duplicate ${m.id}`);
    ids.add(m.id);
  }
  assert.equal(MODELS.filter((m) => m.recommended).length, 1);
  assert.equal(cacheKey('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf'), 'Qwen__Qwen3-0.6B-GGUF__main__Qwen3-0.6B-Q8_0.gguf');
  assert.equal(formatBytes(491400032), '491 MB');
  assert.equal(formatBytes(2497280256), '2.50 GB');
});
