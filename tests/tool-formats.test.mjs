import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { attemptedToolCall, displayText, parseToolCalls, rawPiece, stripToolCalls } from '../web/tools.js';
import { assistantWithCalls, toolResult, toolTransport } from '../web/tool-transport.mjs';

// Tool calls as each family's own chat template writes them (tests/tool_templates.rs).
const renders = JSON.parse(readFileSync(new URL('./fixtures/tool-call-renders.json', import.meta.url), 'utf8'));
const allowed = new Set(['web_search']);
const types = { web_search: { query: 'string', limit: 'integer' } };
// Generation stops at the end-of-turn token, which is not part of the reply.
const reply = (s) => s.replace(/(<\|im_end\|>|<\|eot_id\|>|<｜end▁of▁sentence｜>)\s*$/, '');

for (const [family, text] of Object.entries(renders)) {
  test(`${family}: the call its template writes parses back and is hidden from the person`, () => {
    const raw = reply(text);
    assert.deepEqual(parseToolCalls(raw, allowed, types), [{ name: 'web_search', args: { query: 'WebGPU spec', limit: 3 } }]);
    assert.ok(attemptedToolCall(raw));
    assert.equal(stripToolCalls(raw).trim(), '');
  });
}

test('Kimi K3.1 channels: thinking becomes a think block, tags never show, and streaming only grows', () => {
  const marker = /(<\|open\|>|<\|sep\|>|<\|close\|>|<\|end_of_msg\|>)/;
  const raw = '<|open|>think<|sep|>Look it up.<|close|>think<|sep|><|open|>response<|sep|>Searching now.<|close|>response<|sep|>'
    + reply(renders['kimi-k3.1']).slice('<|close|>response<|sep|>'.length);
  assert.equal(displayText(raw), '<think>Look it up.</think>Searching now.');
  assert.deepEqual(parseToolCalls(raw, allowed, types), [{ name: 'web_search', args: { query: 'WebGPU spec', limit: 3 } }]);
  // Token by token: markers arrive whole (they are single control tokens), text a few characters at a time.
  const tokens = raw.split(marker).filter(Boolean).flatMap((t) => (marker.test(t) ? [t] : t.match(/[\s\S]{1,3}/g)));
  let sofar = '';
  let shown = '';
  for (const t of tokens) {
    sofar += t;
    const d = displayText(sofar);
    assert.ok(d.startsWith(shown), `display shrank at ${JSON.stringify(sofar.slice(-40))}`);
    shown = d;
  }
  assert.equal(shown, '<think>Look it up.</think>Searching now.');
});

test('Kimi K2 files that name tool markers as reserved tokens still parse', () => {
  const pieces = [['<|reserved_token_163595|>', ''], ['<|reserved_token_163597|>', ''], ['', 'functions.web_search:0'], ['<|reserved_token_163598|>', ''],
    ['', '{"query": "x"}'], ['<|reserved_token_163599|>', ''], ['<|reserved_token_163596|>', '']];
  const raw = pieces.map(([c, p]) => rawPiece(c, p)).join('');
  assert.deepEqual(parseToolCalls(raw, allowed), [{ name: 'web_search', args: { query: 'x' } }]);
  assert.equal(rawPiece('<|some_other_control|>', 'shown'), 'shown');
});

test('DeepSeek V3 (fenced arguments) and Mistral calls parse; unknown tools reject the reply', () => {
  const v3 = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>web_search\n```json\n{"query": "x"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>';
  assert.deepEqual(parseToolCalls(v3, allowed), [{ name: 'web_search', args: { query: 'x' } }]);
  assert.deepEqual(parseToolCalls('[TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "y"}}]', allowed), [{ name: 'web_search', args: { query: 'y' } }]);
  assert.deepEqual(parseToolCalls('<|open|>tools<|sep|><|open|>call tool="rm" index="1"<|sep|><|close|>call<|sep|><|close|>tools<|sep|>', allowed), []);
});

test('structured history reaches native templates, and text reaches the rest', () => {
  const tools = [{ type: 'function', function: { name: 'web_search', parameters: {} } }];
  const msgs = [{ role: 'user', content: 'q' }, assistantWithCalls('Let me check.', [{ id: 'functions.web_search:0', name: 'web_search', args: { query: 'x' } }]), toolResult('functions.web_search:0', 'web_search', 'R')];
  assert.deepEqual(toolTransport(msgs, tools, true).messages, msgs);
  assert.equal(toolTransport(msgs, tools, true, { argsAsString: true }).messages[1].tool_calls[0].function.arguments, '{"query":"x"}');
  const text = toolTransport(msgs, tools, false).messages;
  assert.match(text[2].content, /^Let me check\.\n<tool_call>\n\{"name":"web_search","arguments":\{"query":"x"\}\}\n<\/tool_call>$/);
  assert.equal(text[3].role, 'user');
  assert.match(text[3].content, /<tool_response>\nR\n<\/tool_response>/);
});

test('a call whose reply ends after the JSON, without </tool_call>, still counts; a cut-off one does not', () => {
  const call = '<tool_call>\n{"name": "bash", "arguments": {"command": "cd app && grep -n \\"x\\" README.md"}}';
  const bash = new Set(['bash']);
  assert.deepEqual(parseToolCalls(call, bash).map((c) => [c.name, c.args.command]), [['bash', 'cd app && grep -n "x" README.md']]);
  assert.equal(parseToolCalls('<tool_call>\n{"name": "bash", "arguments": {"command": "ls', bash).length, 0);
});

test('a command called as if it were a tool runs as that command line with bash', () => {
  const names = new Set(['bash', 'read']);
  assert.deepEqual(parseToolCalls('<tool_call>\n{"name": "git", "arguments": {"command": "commit -m \\"Fix add\\""}}\n</tool_call>', names), [{ name: 'bash', args: { command: 'git commit -m "Fix add"' } }]);
  assert.deepEqual(parseToolCalls('<tool_call>{"name": "gh", "arguments": {"command": "gh pr create --title x"}}</tool_call>', names), [{ name: 'bash', args: { command: 'gh pr create --title x' } }]);
  assert.deepEqual(parseToolCalls('<tool_call>{"name": "python3", "arguments": {"args": ["test_calc.py"]}}</tool_call>', names), [{ name: 'bash', args: { command: 'python3 test_calc.py' } }]);
  assert.deepEqual(parseToolCalls('<tool_call>{"name": "git", "arguments": {"command": "status"}}</tool_call>', new Set(['read'])), []);
  assert.deepEqual(parseToolCalls('<tool_call>{"name": "git", "arguments": {"command": "status", "cwd": "x"}}</tool_call>', names), []);
});

test('a shell command written as the tool name, with unescaped quotes, becomes a bash call', async () => {
  const { parseToolCalls } = await import('../web/tools.js');
  const names = new Set(['bash', 'read', 'write']);
  const text = '<think>\nNow the pull request.\n</think>\n\n<tool_call>\n{"name": "gh pr create --title "Add a line" --body "Added `x` to README."}\n</tool_call>';
  assert.deepEqual(parseToolCalls(text, names), [{ name: 'bash', args: { command: 'gh pr create --title "Add a line" --body "Added `x` to README."' } }]);
  // Not for names that are not commands, nor without a bash tool.
  assert.deepEqual(parseToolCalls('<tool_call>\n{"name": "make "it" so"}\n</tool_call>', names), []);
  assert.deepEqual(parseToolCalls(text, new Set(['read', 'write'])), []);
});

test('GUI models\' actions become browser tools', async () => {
  const { parseToolCalls } = await import('../web/tools.js');
  const names = new Set(['browser_open', 'browser_click_at', 'browser_type', 'browser_key', 'browser_scroll']);
  const one = (j) => parseToolCalls('<tool_call>\n' + JSON.stringify(j) + '\n</tool_call>', names);
  assert.deepEqual(one({ name: 'web', arguments: { url: 'http://x/look', browser_open: true } }), [{ name: 'browser_open', args: { url: 'http://x/look' } }]);
  assert.deepEqual(one({ name: 'computer_use', arguments: { action: 'left_click', coordinate: [512.4, 300] } }), [{ name: 'browser_click_at', args: { x: 512, y: 300 } }]);
  assert.deepEqual(one({ name: 'computer_use', arguments: { action: 'type', text: 'kettle' } }), [{ name: 'browser_type', args: { text: 'kettle', submit: false } }]);
  assert.deepEqual(one({ name: 'computer_use', arguments: { action: 'key', keys: ['enter'] } }), [{ name: 'browser_key', args: { key: 'Enter' } }]);
  assert.deepEqual(one({ name: 'computer_use', arguments: { action: 'scroll', pixels: -300 } }), [{ name: 'browser_scroll', args: { direction: 'down' } }]);
  assert.deepEqual(one({ name: 'computer_use', arguments: { action: 'key', keys: ['ctrl+alt+del'] } }), []);
  // Without browser tools nothing is mapped.
  assert.deepEqual(parseToolCalls('<tool_call>\n{"name": "web", "arguments": {"url": "http://x/"}}\n</tool_call>', new Set(['bash'])), []);
});
