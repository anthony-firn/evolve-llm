import init, { Shell } from './shell-pkg/shell.js';
import { createWebTools } from './tools.js';
const web = createWebTools();
let shell, running = false, controller = null;
const ready = init().then(() => shell = new Shell());
const json = text => { const value = JSON.parse(text); if (value.error) throw new Error(value.error); return value; };
const emit = (id, result, changed = false) => self.postMessage({ id, result, snapshot: changed ? shell.snapshot() : null });
async function execute(command, network, signal) {
  const jobs = json(shell.plan(command)); let code = 0, stdout = '', stderr = '', changed = false;
  for (const job of jobs) {
    signal.throwIfAborted();
    if (job.condition === 'success' && code !== 0 || job.condition === 'failure' && code === 0) continue;
    let input = '';
    for (const cmd of job.pipeline) {
      signal.throwIfAborted();
      if (cmd.input) input = shell.read(cmd.input);
      let result;
      if (['curl', 'search'].includes(cmd.args[0])) {
        if (cmd.args.length !== 2) throw new Error('Use curl URL or search "query". Flags are unsupported.');
        const read = await web(cmd.args[0] === 'curl' ? 'open_url' : 'web_search', cmd.args[0] === 'curl' ? { url: cmd.args[1] } : { query: cmd.args[1] }, { ...network, signal });
        result = { stdout: read.text + '\n', stderr: '', code: read.sources.length ? 0 : 1 };
      } else result = json(shell.run(JSON.stringify(cmd.args), input));
      code = result.code; stderr += result.stderr ? result.stderr + '\n' : ''; changed ||= Boolean(result.changed);
      input = result.stdout || '';
      if (cmd.output && code === 0) { shell.write(cmd.output, input, cmd.append); input = ''; changed = true; }
      if (code !== 0) break;
    }
    stdout += input;
    if (stdout.length > 131072) stdout = stdout.slice(0, 131072) + '\n[output truncated]';
  }
  return { result: { stdout, stderr, code, cwd: JSON.parse(shell.snapshot()).cwd }, changed };
}
self.onmessage = async ({ data }) => {
  if (data.type === 'cancel') { controller?.abort(); return; }
  await ready;
  if (running) { self.postMessage({ id: data.id, error: 'Workspace is busy.' }); return; }
  running = true; controller = new AbortController();
  const before = shell.snapshot();
  try {
    if (data.type === 'restore') { shell.restore(data.snapshot); emit(data.id, { ok: true }); }
    else if (data.type === 'exec') { const { result, changed } = await execute(data.command, data.network || {}, controller.signal); emit(data.id, result, changed); }
    else if (data.type === 'read') emit(data.id, { text: shell.read(data.path) });
    else if (data.type === 'write') { shell.write(data.path, data.text, false); emit(data.id, { ok: true }, true); }
    else if (data.type === 'edit') {
      if (typeof data.oldText !== 'string' || !data.oldText || typeof data.newText !== 'string') throw new Error('Specify nonempty oldText and a newText string.');
      const text = shell.read(data.path), first = text.indexOf(data.oldText);
      if (first < 0 || text.indexOf(data.oldText, first + data.oldText.length) >= 0) throw new Error('oldText must match exactly once.');
      shell.write(data.path, text.slice(0, first) + data.newText + text.slice(first + data.oldText.length), false); emit(data.id, { ok: true }, true);
    } else if (data.type === 'snapshot') emit(data.id, { snapshot: shell.snapshot() });
    else throw new Error('Unknown workspace operation.');
  } catch (error) {
    shell.restore(before); self.postMessage({ id: data.id, error: controller.signal.aborted ? 'Workspace operation cancelled.' : String(error.message || error) });
  } finally { running = false; controller = null; }
};
