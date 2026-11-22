// The page's connection to the Evolve VM worker: terminal sessions, and commands and files for
// Pi. One VM serves the whole page.

import { trace } from '../trace.mjs';

let shared = null;
let config = () => ({});
let computer = null;

/** The paired computer for builds: { call({type, data}), run(req, write, signal) -> status },
 *  or null. Programs in the VM (native, cargo, npm, npx, node) reach it through these. */
export function provideComputer(c) {
  computer = c;
}

/** Whether a computer is paired for building, and which repository: { project } or null. */
export function pairedComputer() {
  return computer?.project ? { project: computer.project } : null;
}

/** The page's one VM (started on first use), configured from the page's settings. */
export function sharedVm() {
  if (!shared) {
    shared = new VmClient();
    shared.configure(config());
  }
  return shared;
}

/** Where the VM's settings come from (GitHub token, autonomy); applied now and on change. */
export function provideConfig(fn) {
  config = fn;
  shared?.configure(fn());
}

export class VmClient {
  constructor(url = new URL('./worker.mjs', import.meta.url)) {
    this.worker = new Worker(url, { type: 'module' });
    this.seq = 0;
    this.pending = new Map();
    this.outputs = new Map();
    this.exits = new Map();
    this.agentAborts = new Map();
    this.ready = new Promise((resolve, reject) => {
      this.worker.addEventListener('error', () => reject(new Error('The Linux VM could not start.')), { once: true });
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') resolve();
        else if (m.type === 'error') reject(new Error(m.message));
        else if (m.type === 'output') this.outputs.get(m.tty)?.(m.data);
        else if (m.type === 'session-exit') this.exits.get(m.tty)?.(m.code);
        else if (m.type === 'agent-cancel') this.agentAborts.get(m.id)?.abort();
        else if (m.type === 'agent') {
          const write = (text) => this.worker.postMessage({ type: 'agent-output', id: m.id, text });
          const abort = new AbortController();
          this.agentAborts.set(m.id, abort);
          Promise.resolve(this.onAgent(m.task, m.cwd, write, abort.signal)).finally(() => this.agentAborts.delete(m.id)).then((code) => this.worker.postMessage({ type: 'agent-done', id: m.id, code: code | 0 }), (e) => {
            write(`pi: ${e?.message || e}\n`);
            this.worker.postMessage({ type: 'agent-done', id: m.id, code: 1 });
          });
        } else if (m.type === 'computer-run') {
          const write = (text) => this.worker.postMessage({ type: 'agent-output', id: m.id, text });
          const abort = new AbortController();
          this.agentAborts.set(m.id, abort);
          Promise.resolve().then(() => this.onComputerRun(m.req, write, abort.signal)).finally(() => this.agentAborts.delete(m.id)).then((code) => this.worker.postMessage({ type: 'agent-done', id: m.id, code: code | 0 }), (e) => {
            write(`${e?.message || e}\n`);
            this.worker.postMessage({ type: 'agent-done', id: m.id, code: 1 });
          });
        } else if (m.type === 'computer') Promise.resolve().then(() => this.onComputer(m.req)).then(
          (answer) => this.worker.postMessage({ type: 'computer-answer', id: m.id, answer }),
          (e) => this.worker.postMessage({ type: 'computer-answer', id: m.id, answer: { error: e?.message || String(e) } }));
        else if (m.type === 'ask') Promise.resolve(this.onAsk(m.question)).then((yes) => this.worker.postMessage({ type: 'answer', id: m.id, yes }), () => this.worker.postMessage({ type: 'answer', id: m.id, yes: false }));
        else if (m.type === 'session' || m.type === 'result') {
          this.pending.get(m.id)?.(m);
          this.pending.delete(m.id);
        }
      };
    });
    this.ready.catch(() => {}); // a build without the VM: callers that await it see the failure
  }

  /** `pi TASK` in a terminal: run Pi in `cwd`, writing progress with `write`; resolves the status. */
  onAgent(task, cwd, write) {
    write('pi: open Terminal in Evolve LLM with a tool-capable model loaded, then run pi again.\n');
    return 1;
  }

  /** A request to the paired computer: { type, data } -> its answer. */
  onComputer(req) {
    if (!computer) return { error: 'no computer is paired for building' };
    trace({ type: 'computer', request: req.type });
    return computer.call(req);
  }

  /** A command on the paired computer, its output written as it comes: -> exit status. */
  onComputerRun(req, write, signal) {
    if (!computer) return 127;
    trace({ type: 'computer-run', command: req.command, cwd: req.cwd });
    return computer.run(req, write, signal);
  }

  /** A program asks to do something that changes GitHub (push, open a pull request, merge). */
  onAsk(question) {
    const allowed = window.confirm(`Allow the assistant to do this?\n\n${question.replace(/\?$/, '')}`);
    trace({ type: 'ask', question, allowed });
    return allowed;
  }

  configure({ github } = {}) {
    if (github) this.worker.postMessage({ type: 'config', github });
  }

  call(msg) {
    return new Promise((resolve) => {
      const id = ++this.seq;
      this.pending.set(id, resolve);
      this.worker.postMessage({ ...msg, id });
    });
  }

  /** A new shell on its own terminal: resolves the terminal's number. */
  async session({ onOutput, onExit }) {
    await this.ready;
    const { tty } = await this.call({ type: 'session' });
    this.outputs.set(tty, onOutput);
    this.exits.set(tty, (code) => {
      this.outputs.delete(tty);
      this.exits.delete(tty);
      onExit?.(code);
    });
    return tty;
  }

  input(tty, data) {
    this.worker.postMessage({ type: 'input', tty, data });
  }

  resize(tty, cols, rows) {
    this.worker.postMessage({ type: 'resize', tty, cols, rows });
  }

  close(tty) {
    this.worker.postMessage({ type: 'close', tty });
  }

  /** Run a command with sh -c: {code, stdout, stderr, timedOut}. */
  async exec(command, options = {}) {
    await this.ready;
    return this.call({ type: 'exec', command, ...options });
  }

  async read(path) {
    await this.ready;
    return this.call({ type: 'read', path });
  }

  async write(path, data) {
    await this.ready;
    return this.call({ type: 'write', path, data });
  }

  async list(path) {
    await this.ready;
    return (await this.call({ type: 'list', path })).entries;
  }
}
