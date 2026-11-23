/**
 * Edit-time checks: a file's syntax right after it is written or edited, so a model fixes a slip
 * at once instead of finding it at test time (or never). Quiet when the file is fine; a check
 * that cannot run (a page whose policy blocks the JavaScript parser) is skipped.
 */

/** The JSON error in `text`, or null. */
export function jsonError(text) {
  try { JSON.parse(text); return null; } catch (e) { return e.message; }
}

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

/**
 * The JavaScript syntax error in `code`, as { line, message }, or null. Parsed by the page's own
 * engine (compiled, never run); modules have their import and export words blanked out first,
 * keeping line numbers. The engine gives no line, so the first prefix of lines that fails with the
 * same message is the line (an unfinished prefix fails differently, as "end of input" or similar).
 */
export function jsError(code, { module = /^\s*(import|export)\s/m.test(code) } = {}) {
  const body = module
    ? code.replace(/^(\s*)import\s[^;\n]*(;|$)/gm, (m) => m.replace(/[^\n]/g, ' ')).replace(/^(\s*)export\s+(default\s+)?/gm, '$1')
    : code;
  const parse = (src) => {
    try { new AsyncFunction(src); return null; } catch (e) {
      if (e instanceof SyntaxError) return e.message;
      throw e;
    }
  };
  let message;
  try { message = parse(body); } catch { return null; }
  if (!message) return null;
  const lines = body.split('\n');
  // Something left open (a brace, bracket or string): one closing character at the end makes it
  // parse. (The engine wraps the code in a function, so this shows as some unexpected token.)
  for (const close of ['}', ')', ']', '`', "'", '"']) {
    if (parse(`${body}\n${close}`) === null || parse(`${body}${close}`) === null)
      return { line: lines.length, message: `the file ends with a ${close === '}' ? 'brace {' : close === ')' ? 'parenthesis (' : close === ']' ? 'bracket [' : 'string'} not closed` };
  }
  for (let i = 1; i <= lines.length; i++) {
    if (parse(lines.slice(0, i).join('\n')) === message) return { line: i, message };
  }
  return { line: null, message };
}

/** Syntax errors in an HTML page's inline scripts, as [{ line, message }] (page lines). */
export function htmlErrors(html) {
  const out = [];
  for (const m of String(html).matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
    const attrs = m[1];
    if (/\bsrc\s*=/i.test(attrs)) continue;
    const type = (/\btype\s*=\s*["']?([^"'\s>]+)/i.exec(attrs) || [])[1]?.toLowerCase();
    if (type && !['module', 'text/javascript', 'application/javascript'].includes(type)) continue;
    const err = jsError(m[2], { module: type === 'module' });
    if (!err) continue;
    const before = html.slice(0, m.index + m[0].indexOf('>') + 1).split('\n').length - 1;
    out.push({ line: err.line == null ? null : before + err.line, message: err.message });
  }
  return out;
}

/** A Python syntax error from compile()'s traceback (RustPython, as CPython): { line, message }. */
export function pythonError(output) {
  const text = String(output || '');
  const message = [...text.matchAll(/^(\w*(?:Error|Exception)\b.*)$/gm)].at(-1)?.[1];
  if (!message) return null;
  const line = [...text.matchAll(/File "[^"]*", line (\d+)/g)].at(-1)?.[1];
  return { line: line ? Number(line) : null, message };
}

/** The command that compiles a Python file without running it (for the VM's python3). */
export function pythonCheckCommand(path) {
  const q = `'${String(path).replace(/'/g, `'\\''`)}'`;
  return `python3 -c "import sys;compile(open(sys.argv[1]).read(),sys.argv[1],'exec')" ${q}`;
}

/** Which check a path gets: 'json', 'js', 'html', 'python' or null. */
export function checkKind(path) {
  const ext = String(path).toLowerCase().split('.').pop();
  return { json: 'json', js: 'js', mjs: 'js', cjs: 'js', html: 'html', htm: 'html', py: 'python' }[ext] || null;
}

/**
 * The note for a file just written: null when it is fine (or cannot be checked here). `text` is
 * its content; `python(path)` runs the Python check and resolves to its output, or null.
 */
export async function syntaxNote(path, text, { python } = {}) {
  const kind = checkKind(path);
  const at = (e) => (e.line ? ` at line ${e.line}` : '');
  if (kind === 'json') {
    const e = jsonError(text);
    return e ? `Check: ${path} is not valid JSON: ${e}. Fix it before going on.` : null;
  }
  if (kind === 'js') {
    const e = jsError(text);
    return e ? `Check: ${path} has a syntax error${at(e)}: ${e.message}. Fix it before going on.` : null;
  }
  if (kind === 'html') {
    const errs = htmlErrors(text);
    return errs.length ? `Check: a script in ${path} has a syntax error${at(errs[0])}: ${errs[0].message}. Fix it before going on.` : null;
  }
  if (kind === 'python' && python) {
    const out = await python(path);
    const e = out == null ? null : pythonError(out);
    return e ? `Check: ${path} has a syntax error${at(e)}: ${e.message}. Fix it before going on.` : null;
  }
  return null;
}
