// The Evolve VM in a worker: the kernel, its programs, terminal sessions, and commands run on
// behalf of the page (Pi's tools). /home is saved to this browser's storage (OPFS) and restored
// when the VM starts.
//
// Messages in:  {type:'session', id}  {type:'input', tty, data}  {type:'resize', tty, cols, rows}
//               {type:'exec', id, command, cwd?, timeoutMs?, stdin?}  {type:'read', id, path}
//               {type:'write', id, path, data}  {type:'list', id, path}  {type:'save'}
// Messages out: {type:'ready'}  {type:'session', id, tty}  {type:'output', tty, data}
//               {type:'session-exit', tty, code}  {type:'result', id, ...}

import init, { Vm } from './kernel/evolve_vm.js';
globalThis.VM_TRACE = new URL(location.href).searchParams.has('trace');
import { Host } from './host.mjs';

const enc = new TextEncoder();
const dec = new TextDecoder();
const SNAPSHOT = 'evolve-vm-home.snap';
const ENV = ['HOME=/home/user', 'USER=user', 'PATH=/bin:/usr/bin', 'TERM=xterm-256color', 'LANG=C.UTF-8', 'SHELL=/bin/sh'];

let vm;
let host;
let askSeq = 0;
const asks = new Map();
const agents = new Map();
let init_pid;
let dirty = false;
const sessions = new Map(); // tty -> shell pid

async function boot() {
  await init();
  vm = new Vm();
  vm.seed(crypto.getRandomValues(new Uint32Array(1))[0] * 4294967296 + Date.now());
  vm.set_time(Date.now() * 1e6);
  host = new Host(vm, {
    onOutput: (tty, data) => postMessage({ type: 'output', tty, data }, [data.buffer]),
    onExit: (pid, code) => {
      for (const [tty, shell] of sessions) {
        if (shell === pid) {
          sessions.delete(tty);
          postMessage({ type: 'session-exit', tty, code });
        }
      }
    },
    onChange: () => { dirty = true; },
  });
  // `pi` tasks run on the page (where the model is); its progress comes back as output.
  host.agent = (task, cwd, write, pid) => new Promise((resolve) => {
    const id = ++askSeq;
    // Ctrl+C on `pi` stops Pi itself.
    const watch = setInterval(() => {
      if (vm.killed(pid) >= 0 || !vm.running(pid)) postMessage({ type: 'agent-cancel', id });
    }, 250);
    agents.set(id, { resolve: (code) => { clearInterval(watch); resolve(code); }, write, pid });
    postMessage({ type: 'agent', id, task, cwd });
  });
  // Questions from programs (push, pull requests) go to the page and wait for the answer.
  host.ask = (question) => new Promise((resolve) => {
    const id = ++askSeq;
    asks.set(id, resolve);
    postMessage({ type: 'ask', id, question });
  });
  // Programs: the shell, and the utilities under each of their names.
  const [sh, utils, gh, pi, list] = await Promise.all([
    fetch(new URL('./bin/sh.wasm', import.meta.url)).then((r) => r.arrayBuffer()),
    fetch(new URL('./bin/utils.wasm', import.meta.url)).then((r) => r.arrayBuffer()),
    fetch(new URL('./bin/gh.wasm', import.meta.url)).then((r) => r.arrayBuffer()),
    fetch(new URL('./bin/pi.wasm', import.meta.url)).then((r) => r.arrayBuffer()),
    fetch(new URL('./programs.json', import.meta.url)).then((r) => r.json()),
  ]);
  vm.mkdir_p('/usr/lib/evolve');
  vm.write_file('/bin/sh', new Uint8Array(sh));
  vm.write_file('/usr/lib/evolve/utils', new Uint8Array(utils));
  vm.write_file('/usr/lib/evolve/gh', new Uint8Array(gh));
  for (const name of list.utils) vm.symlink('/usr/lib/evolve/utils', `/bin/${name}`);
  for (const name of ['gh', 'git']) vm.symlink('/usr/lib/evolve/gh', `/bin/${name}`);
  vm.write_file('/bin/pi', new Uint8Array(pi));
  await restore();
  // A process that owns the pipes of commands run for the page.
  const t = vm.create_tty();
  init_pid = vm.spawn_on_tty(['init'], ENV, '/', t);
  setInterval(save, 3000);
  postMessage({ type: 'ready' });
}

async function opfs() {
  return navigator.storage.getDirectory();
}

async function restore() {
  try {
    const f = await (await (await opfs()).getFileHandle(SNAPSHOT)).getFile();
    vm.restore(new Uint8Array(await f.arrayBuffer()));
  } catch {
    vm.write_file('/home/user/README', enc.encode('This is your home folder in the Evolve VM. It is kept in this browser.\nRun `help` for the shell, or `ls /bin` for the programs.\n'));
  }
}

async function save() {
  if (!dirty) return;
  dirty = false;
  try {
    const h = await (await opfs()).getFileHandle(SNAPSHOT, { create: true });
    const w = await h.createWritable();
    await w.write(vm.snapshot('/home'));
    await w.close();
  } catch {
    dirty = true;
  }
}

function newSession() {
  const tty = vm.create_tty();
  host.ttys.add(tty);
  const pid = vm.spawn_on_tty(['sh'], [...ENV, 'PWD=/home/user'], '/home/user', tty);
  sessions.set(tty, pid);
  host.run(pid, '/bin/sh');
  return tty;
}

/** Run `command` with sh -c: its output, errors and status, ended after `timeoutMs`. */
async function exec({ command, cwd = '/home/user', timeoutMs = 120000, stdin = '' }) {
  const [inR, inW] = vm.pipe(init_pid);
  const [outR, outW] = vm.pipe(init_pid);
  const [errR, errW] = vm.pipe(init_pid);
  const env = [...ENV, `PWD=${cwd}`];
  const pid = vm.spawn(init_pid, ['sh', '-c', command], env, cwd, new Uint32Array([0, inR, 1, outW, 2, errW]), 0);
  for (const fd of [inR, outW, errW]) vm.fd_close(init_pid, fd);
  if (pid < 0) {
    for (const fd of [inW, outR, errR]) vm.fd_close(init_pid, fd);
    return { code: 127, stdout: '', stderr: `cannot start the shell in ${cwd} (error ${vm.errno()})` };
  }
  if (stdin) vm.fd_write(init_pid, inW, enc.encode(stdin));
  vm.fd_close(init_pid, inW);
  const run = host.run(pid, '/bin/sh');
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    vm.kill_group(vm.pgid(pid), 124);
    host.changed();
  }, timeoutMs);
  const drain = async (fd) => {
    const parts = [];
    let size = 0;
    for (;;) {
      const d = vm.fd_read(init_pid, fd, 65536);
      if (d) {
        if (!d.length) break;
        if (size < 1 << 20) parts.push(d);
        size += d.length;
        host.changed();
        continue;
      }
      await host.wait();
    }
    vm.fd_close(init_pid, fd);
    const all = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
    let at = 0;
    for (const p of parts) {
      all.set(p, at);
      at += p.length;
    }
    return { text: dec.decode(all), truncated: size > all.length };
  };
  const [out, err] = await Promise.all([drain(outR), drain(errR)]);
  await run;
  clearTimeout(timer);
  const code = vm.wait(init_pid, pid);
  return { code: timedOut ? 124 : code, stdout: out.text, stderr: err.text, truncated: out.truncated || err.truncated, timedOut };
}

const handlers = {
  /** {github: {token, autonomy (0 local only, 1 ask before push, 2 push freely), api?, raw?}} */
  config: ({ github }) => {
    if (github) host.github = { ...host.github, ...github };
  },
  'agent-output': ({ id, text }) => agents.get(id)?.write(text),
  'agent-done': ({ id, code }) => {
    agents.get(id)?.resolve(code);
    agents.delete(id);
  },
  answer: ({ id, yes }) => {
    asks.get(id)?.(Boolean(yes));
    asks.delete(id);
  },
  session: ({ id }) => postMessage({ type: 'session', id, tty: newSession() }),
  input: ({ tty, data }) => {
    vm.tty_input(tty, typeof data === 'string' ? enc.encode(data) : new Uint8Array(data));
    host.changed();
  },
  resize: ({ tty, cols, rows }) => vm.tty_resize(tty, cols, rows),
  close: ({ tty }) => {
    // A closed tab ends its shell and everything it runs.
    const pid = sessions.get(tty);
    if (pid !== undefined) {
      sessions.delete(tty);
      vm.kill_group(vm.pgid(pid), 129);
      vm.set_ignore_interrupt(pid, false);
      vm.kill_group(vm.pgid(pid), 129);
      host.ttys.delete(tty);
      host.changed();
    }
  },
  exec: async (m) => postMessage({ type: 'result', id: m.id, ...(await exec(m)) }),
  read: ({ id, path }) => {
    const d = vm.read_file(path);
    postMessage({ type: 'result', id, ok: Boolean(d), data: d ? dec.decode(d) : null, kind: vm.kind(path) });
  },
  write: ({ id, path, data }) => {
    vm.set_time(Date.now() * 1e6);
    const parent = path.slice(0, path.lastIndexOf('/')) || '/';
    vm.mkdir_p(parent);
    const ok = vm.write_file(path, typeof data === 'string' ? enc.encode(data) : new Uint8Array(data));
    dirty = true;
    postMessage({ type: 'result', id, ok });
  },
  list: ({ id, path }) => postMessage({ type: 'result', id, entries: JSON.parse(vm.list(path) || 'null') }),
  save: async ({ id }) => {
    dirty = true;
    await save();
    postMessage({ type: 'result', id, ok: true });
  },
};

const booted = boot().catch((e) => postMessage({ type: 'error', message: e?.message || String(e) }));
onmessage = async (e) => {
  await booted;
  const h = handlers[e.data?.type];
  if (h) await h(e.data);
};
