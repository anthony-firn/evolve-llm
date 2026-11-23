import test from 'node:test';
import assert from 'node:assert/strict';
import { jsonError, jsError, htmlErrors, pythonError, pythonCheckCommand, checkKind, syntaxNote } from '../web/syntax-check.mjs';

test('JSON, JavaScript and inline scripts are checked, with the line of the error', () => {
  assert.equal(jsonError('{"a": 1}'), null);
  assert.match(jsonError('{"a": 1,}'), /JSON/);
  assert.equal(jsError('const a = 1;\nfunction f() { return a; }\n'), null);
  assert.deepEqual(jsError('const a = 1;\nlet b = ;\nconst c = 2;\n'), { line: 2, message: "Unexpected token ';'" });
  // Modules: imports and exports are fine, and lines are kept.
  assert.equal(jsError("import { x } from './x.js';\nexport function f() { return x; }\nexport default 3;\n"), null);
  assert.equal(jsError("import { x } from './x.js';\n\nexport function f( { return x; }\n").line, 3);
  // A brace left open is found at the end.
  assert.deepEqual(jsError('function f() {\n  return 1;\n'), { line: 3, message: 'the file ends with a brace { not closed' });
  assert.deepEqual(jsError('console.log(f(1,\n  2)\n'), { line: 3, message: 'the file ends with a parenthesis ( not closed' });
  // The engine's own message, where it has one.
  assert.deepEqual(jsError('console.log(f(1, 2);\n'), { line: 1, message: 'missing ) after argument list' });
  // A template literal across lines does not confuse the line search.
  assert.equal(jsError('const s = `a\nb`;\nconst t = ;\n').line, 3);
  const page = '<!doctype html>\n<p>Hi</p>\n<script>\nlet n = 0;\nbutton.onclick = () => { n++; \n</script>\n<script src="x.js"></script>\n<script type="application/json">{oops}</script>\n';
  const errs = htmlErrors(page);
  assert.equal(errs.length, 1); assert.equal(errs[0].line, 6); // the script's end, where the brace is still open
  assert.deepEqual(htmlErrors('<script type="module">import a from "./a.js";\nexport const b = a;\n</script>'), []);
});

test('Python errors are read from RustPython\'s traceback, and the check command quotes the path', () => {
  const out = 'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n  File "/w/bad.py", line 4\n    def sub(a, b)\n                 ^\nSyntaxError: expected \':\'\n';
  assert.deepEqual(pythonError(out), { line: 4, message: "SyntaxError: expected ':'" });
  assert.equal(pythonError(''), null);
  assert.equal(pythonCheckCommand("it's.py"), `python3 -c "import sys;compile(open(sys.argv[1]).read(),sys.argv[1],'exec')" 'it'\\''s.py'`);
  assert.deepEqual(['a.json', 'b.mjs', 'c.HTML', 'd.py', 'e.rs'].map(checkKind), ['json', 'js', 'html', 'python', null]);
});

test('the note says what is wrong and where, and nothing when the file is fine', async () => {
  assert.equal(await syntaxNote('app.js', 'let a = 1;\n'), null);
  assert.equal(await syntaxNote('app.js', 'let a = ;\n'), "Check: app.js has a syntax error at line 1: Unexpected token ';'. Fix it before going on.");
  assert.equal(await syntaxNote('calc.py', 'x', { python: async () => 'File "calc.py", line 2\nSyntaxError: invalid syntax' }), 'Check: calc.py has a syntax error at line 2: SyntaxError: invalid syntax. Fix it before going on.');
  assert.equal(await syntaxNote('calc.py', 'x', { python: async () => '' }), null);
  assert.equal(await syntaxNote('notes.md', '# hi'), null);
});
