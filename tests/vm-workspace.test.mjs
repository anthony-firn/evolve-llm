import test from 'node:test';
import assert from 'node:assert/strict';
import { looseReplace, around, VmWorkspace } from '../web/vm/workspace.mjs';

const CALC = 'def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n';

test('an edit that differs from the file only in blank lines changes just the differing line', () => {
  const old = 'def add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a - b';
  const out = looseReplace(CALC, old, old.replace('a - b', 'a + b'));
  assert.equal(out, CALC.replace('a - b', 'a + b'));
});

test('spaces at line ends and a missing blank line still match; a line is inserted after another', () => {
  assert.equal(looseReplace('# App\n\nRun:\n\nnpm start\n', 'Run:  \nnpm start', 'Run:\nnpm start\nnpm test'), '# App\n\nRun:\n\nnpm start\nnpm test\n');
});

test('no single place, partial lines, and blank-line-only changes are left to the exact edit', () => {
  assert.equal(looseReplace(CALC, '    return a - b', '    return a + b'), null);
  assert.equal(looseReplace(CALC, 'a - b', 'a + b'), null);
  assert.equal(looseReplace(CALC, 'def add(a, b):\n\n    return a - b', 'def add(a, b):\n    return a - b'), null);
  assert.equal(looseReplace(CALC, 'def mul(a, b):', 'def mul(a, c):'), null);
});

test('an edit answers with the edited lines as they are now, so a left-over old line shows', async () => {
  let file = CALC;
  const vm = { read: async () => ({ ok: true, data: file }), write: async (_p, d) => { file = d; } };
  const ws = new VmWorkspace(vm, { cwd: '/home/user/app' });
  const r = await ws.request('edit', { path: 'calc.py', oldText: 'def add(a, b):', newText: 'def add(a, b):\n    return a + b' });
  assert.equal(r.now, '1: def add(a, b):\n2:     return a + b\n3:     return a - b\n4: ');
  // Long edits are not echoed.
  assert.deepEqual(around('x\n'.repeat(40), 0, 60), {});
});
