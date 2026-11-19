// The page's connection to the Evolve VM worker: terminal sessions, and commands and files for
// Pi. One VM serves the whole page.

let shared = null;

/** The page's one VM (started on first use). */
export function sharedVm() {
  shared ??= new VmClient();
  return shared;
}

export class VmClient {
  constructor(url = new URL('./worker.mjs', import.meta.url)) {
    this.worker = new Worker(url, { type: 'module' });
    this.seq = 0;
    this.pending = new Map();
    this.outputs = new Map();
    this.exits = new Map();
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') resolve();
        else if (m.type === 'error') reject(new Error(m.message));
        else if (m.type === 'output') this.outputs.get(m.tty)?.(m.data);
        else if (m.type === 'session-exit') this.exits.get(m.tty)?.(m.code);
        else if (m.type === 'session' || m.type === 'result') {
          this.pending.get(m.id)?.(m);
          this.pending.delete(m.id);
        }
      };
    });
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
