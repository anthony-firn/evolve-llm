// Drives the VM worker from tests (tests/vm_browser_test.py): window.vmExec(command, options).
const w = new Worker(new URL('./worker.mjs' + location.search, import.meta.url), { type: 'module' });
let seq = 0;
const pending = new Map();
w.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'ready') window.vmReady = true;
  if (m.type === 'error') window.vmError = m.message;
  if (m.type === 'ask') {
    // Tests answer programs' questions with window.vmAnswer (default yes) and see them asked.
    (window.vmQuestions ??= []).push(m.question);
    w.postMessage({ type: 'answer', id: m.id, yes: window.vmAnswer ?? true });
  }
  if (m.type === 'result') {
    pending.get(m.id)?.(m);
    pending.delete(m.id);
  }
};
const call = (msg) => new Promise((r) => {
  const id = ++seq;
  pending.set(id, r);
  w.postMessage({ ...msg, id });
});
window.vmExec = (command, opts = {}) => call({ type: 'exec', command, ...opts });
window.vmRead = (path) => call({ type: 'read', path });
window.vmWrite = (path, data) => call({ type: 'write', path, data });
window.vmConfig = (config) => w.postMessage({ type: 'config', ...config });
