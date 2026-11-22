// The Evolve VM host: runs WASI programs as processes of the kernel (vm/src/lib.rs).
//
// Each program is a WebAssembly module the browser compiles to machine code. Its WASI calls come
// here and go to the kernel; a call that has to wait (a read with no input yet, a full pipe, a
// child still running, a sleep) suspends that program with JSPI (WebAssembly.Suspending and
// WebAssembly.promising) while the others keep running. The `evolve` imports give the shell
// what WASI lacks: starting programs, waiting for them, pipes and the terminal's foreground.

/** The start of a program file that is fetched the first time it runs (see module()). */
export const LAZY = '#!evolve-lazy '; // `#!` so the kernel counts it as a program

const E = { SUCCESS: 0, AGAIN: 6, BADF: 8, CHILD: 12, INVAL: 28, NOENT: 44, NOSYS: 52, NOTSUP: 58, FAULT: 21 };
const ALL_RIGHTS = 0x1fffffffn;
const enc = new TextEncoder();
const dec = new TextDecoder();

export class Exit {
  constructor(code) {
    this.code = code;
  }
}

export class Host {
  /** `onOutput(tty, bytes)` receives terminal output; `onExit(pid, code)` each ended program. */
  constructor(vm, { onOutput = () => {}, onExit = () => {}, onChange = () => {} } = {}) {
    this.vm = vm;
    this.onOutput = onOutput;
    this.onExit = onExit;
    this.onChange = onChange;
    this.waiters = [];
    this.modules = new Map();
    this.ttys = new Set();
    this.pending = false;
    this.running = new Map();
    this.http = new Map();
    // GitHub access for programs (gh, git): the token stays here; programs never see it.
    this.github = { token: '', autonomy: 1, api: 'https://api.github.com', raw: 'https://raw.githubusercontent.com' };
    this.ask = async () => false;
    // Pi on the page runs `pi` tasks: (task, cwd, write(text)) -> exit status.
    this.agent = async (task, cwd, write) => {
      write('pi: the agent is not available here\n');
      return 1;
    };
  }

  /** One request from a program: only GitHub's API and raw files, with the token for the API. */
  async fetch(r) {
    try {
      let url = new URL(r.url);
      const api = url.origin === 'https://api.github.com';
      if (!api && url.origin !== 'https://raw.githubusercontent.com') return { error: `${url.origin} is not reachable from the VM` };
      // Tests point GitHub at a local fake.
      const base = api ? this.github.api : this.github.raw;
      url = new URL(base.replace(/\/$/, '') + url.pathname + url.search);
      const headers = {};
      for (const [k, v] of Object.entries(r.headers || {})) if (/^(accept|content-type|x-github-api-version)$/i.test(k)) headers[k] = v;
      if (api && this.github.token) headers.Authorization = `Bearer ${this.github.token}`;
      const body = r.body ? Uint8Array.from(atob(r.body), (c) => c.charCodeAt(0)) : undefined;
      const res = await fetch(url, { method: r.method || 'GET', headers, body, redirect: 'follow' });
      const data = new Uint8Array(await res.arrayBuffer());
      let bin = '';
      for (let i = 0; i < data.length; i += 0x8000) bin += String.fromCharCode(...data.subarray(i, i + 0x8000));
      return { status: res.status, body: btoa(bin) };
    } catch (e) {
      return { error: `network request failed: ${e?.message || e}` };
    }
  }

  tick() {
    this.vm.set_time(Date.now() * 1e6);
  }

  /** Something changed in the kernel: let waiting programs retry, and flush terminals. */
  changed() {
    if (this.pending) return;
    this.pending = true;
    queueMicrotask(() => {
      this.pending = false;
      const w = this.waiters;
      this.waiters = [];
      for (const r of w) r();
      for (const t of this.ttys) {
        const out = this.vm.tty_take_output(t);
        if (out.length) this.onOutput(t, out);
      }
    });
  }

  wait() {
    return new Promise((r) => this.waiters.push(r));
  }

  async module(path) {
    let bytes = this.vm.read_file(path);
    if (!bytes) throw new Error(`${path}: not found`);
    // A program loaded on first use (python3): the file names it in bin/ next to this script.
    const lazy = dec.decode(bytes.subarray(0, 64));
    if (lazy.startsWith(LAZY)) {
      const name = lazy.slice(LAZY.length).trim();
      const r = await fetch(new URL(`./bin/${name}`, import.meta.url));
      if (!r.ok) throw new Error(`${name} is not part of this build (HTTP ${r.status})`);
      bytes = new Uint8Array(await r.arrayBuffer());
      this.vm.write_file(path, bytes);
    }
    const key = `${path}:${bytes.length}:${bytes[bytes.length - 1]}`;
    let m = this.modules.get(key);
    if (!m) {
      m = await WebAssembly.compile(bytes);
      this.modules.set(key, m);
    }
    return m;
  }

  /**
   * The program file to run for `argv` (PATH lookup, `#!` scripts through their interpreter):
   * [path, argv], or null.
   */
  resolve(pid, argv, env) {
    const pathVar = (env.find((e) => e.startsWith('PATH=')) || 'PATH=/bin:/usr/bin').slice(5);
    const path = this.vm.which(pid, argv[0], pathVar);
    if (!path) return null;
    const head = this.vm.read_file(path).subarray(0, 256);
    if (dec.decode(head).startsWith(LAZY)) return [path, argv]; // fetched when it runs (module())
    if (head[0] === 0x23 && head[1] === 0x21) {
      const line = dec.decode(head).split('\n')[0].slice(2).trim().split(/\s+/);
      let interp = line[0];
      const extra = line.slice(1);
      // `#!/usr/bin/env python3` runs python3 from PATH.
      if (interp.endsWith('/env') && extra.length) interp = this.vm.which(pid, extra.shift(), pathVar) || interp;
      const ip = this.vm.which(pid, interp, pathVar);
      if (!ip) return null;
      return [ip, [interp, ...extra, path, ...argv.slice(1)]];
    }
    return [path, argv];
  }

  /** Run process `pid` (already in the kernel) from program file `path`; resolves its status. */
  run(pid, path) {
    const p = this.#run(pid, path);
    this.running.set(pid, p);
    return p;
  }

  async #run(pid, path) {
    let code;
    try {
      const module = await this.module(path);
      let memory = null;
      const instance = await WebAssembly.instantiate(module, this.imports(pid, () => memory));
      memory = instance.exports.memory;
      if (!instance.exports._start) throw new Error('not a WASI command (no _start)');
      await WebAssembly.promising(instance.exports._start)();
      code = 0;
    } catch (e) {
      if (e instanceof Exit) code = e.code;
      else {
        code = 134;
        this.vm.fd_write(pid, 2, enc.encode(`${this.vm.argv(pid)[0] || path}: ${e?.message || e}\n`));
      }
    }
    this.vm.exit(pid, code);
    this.running.delete(pid);
    this.changed();
    this.onExit(pid, code);
    return code;
  }

  imports(pid, getMemory) {
    const vm = this.vm;
    const host = this;
    const view = () => new DataView(getMemory().buffer);
    const bytes = () => new Uint8Array(getMemory().buffer);
    const str = (ptr, len) => dec.decode(bytes().slice(ptr, ptr + len));
    const check = () => {
      const k = vm.killed(pid);
      if (k >= 0) throw new Exit(k);
    };
    const iovecs = (ptr, n) => {
      const v = view();
      const out = [];
      for (let i = 0; i < n; i++) out.push([v.getUint32(ptr + i * 8, true), v.getUint32(ptr + i * 8 + 4, true)]);
      return out;
    };
    const gather = (ptr, n) => {
      const parts = iovecs(ptr, n);
      const total = parts.reduce((s, [, l]) => s + l, 0);
      const out = new Uint8Array(total);
      const b = bytes();
      let at = 0;
      for (const [p, l] of parts) {
        out.set(b.subarray(p, p + l), at);
        at += l;
      }
      return out;
    };
    const scatter = (ptr, n, data) => {
      const b = bytes();
      let at = 0;
      for (const [p, l] of iovecs(ptr, n)) {
        const k = Math.min(l, data.length - at);
        b.set(data.subarray(at, at + k), p);
        at += k;
        if (at >= data.length) break;
      }
      return at;
    };
    const capacity = (ptr, n) => iovecs(ptr, n).reduce((s, [, l]) => s + l, 0);
    const errno = () => vm.errno();
    const setStat = (ptr, s) => {
      const v = view();
      v.setBigUint64(ptr, s[0], true);
      v.setBigUint64(ptr + 8, s[1], true);
      v.setUint8(ptr + 16, Number(s[2]));
      v.setBigUint64(ptr + 24, s[3], true);
      v.setBigUint64(ptr + 32, s[4], true);
      v.setBigUint64(ptr + 40, s[5], true);
      v.setBigUint64(ptr + 48, s[6], true);
      v.setBigUint64(ptr + 56, s[7], true);
    };
    // A call that may have to wait: return at once when it can, or a promise (JSPI suspends).
    // `attempt(state, ...args)` returns undefined to wait; `state` lasts across its retries.
    const blocking = (attempt) => new WebAssembly.Suspending((...args) => {
      check();
      const state = {};
      const r = attempt(state, ...args);
      if (r !== undefined) return r;
      return (async () => {
        for (;;) {
          await host.wait();
          check();
          const again = attempt(state, ...args);
          if (again !== undefined) return again;
        }
      })();
    });

    const wasi = {
      args_sizes_get(argc, size) {
        const a = vm.argv(pid);
        view().setUint32(argc, a.length, true);
        view().setUint32(size, a.reduce((s, x) => s + enc.encode(x).length + 1, 0), true);
        return 0;
      },
      args_get(argv, buf) {
        let at = buf;
        vm.argv(pid).forEach((a, i) => {
          const e = enc.encode(`${a}\0`);
          bytes().set(e, at);
          view().setUint32(argv + i * 4, at, true);
          at += e.length;
        });
        return 0;
      },
      environ_sizes_get(count, size) {
        const env = vm.env(pid);
        view().setUint32(count, env.length, true);
        view().setUint32(size, env.reduce((s, x) => s + enc.encode(x).length + 1, 0), true);
        return 0;
      },
      environ_get(environ, buf) {
        let at = buf;
        vm.env(pid).forEach((a, i) => {
          const e = enc.encode(`${a}\0`);
          bytes().set(e, at);
          view().setUint32(environ + i * 4, at, true);
          at += e.length;
        });
        return 0;
      },
      clock_res_get(id, ptr) {
        view().setBigUint64(ptr, 1000n, true);
        return 0;
      },
      clock_time_get(id, precision, ptr) {
        const ns = id === 0 ? BigInt(Date.now()) * 1000000n : BigInt(Math.round(performance.now() * 1e6));
        view().setBigUint64(ptr, ns, true);
        return 0;
      },
      fd_advise: () => 0,
      fd_allocate: () => 0,
      fd_datasync: () => 0,
      fd_sync: () => 0,
      fd_close(fd) {
        const ok = vm.fd_close(pid, fd);
        host.changed();
        return ok ? 0 : errno();
      },
      fd_fdstat_get(fd, ptr) {
        const s = vm.fd_fdstat(pid, fd);
        if (!s.length) return errno();
        const v = view();
        v.setUint8(ptr, s[0]);
        v.setUint16(ptr + 2, s[1], true);
        v.setBigUint64(ptr + 8, ALL_RIGHTS, true);
        v.setBigUint64(ptr + 16, ALL_RIGHTS, true);
        return 0;
      },
      fd_fdstat_set_flags(fd, flags) {
        return vm.fd_set_flags(pid, fd, flags) ? 0 : errno();
      },
      fd_fdstat_set_rights: () => 0,
      fd_filestat_get(fd, ptr) {
        const s = vm.fd_filestat(pid, fd);
        if (!s) return errno();
        setStat(ptr, s);
        return 0;
      },
      fd_filestat_set_size(fd, size) {
        host.tick();
        return vm.fd_set_size(pid, fd, size) ? 0 : errno();
      },
      fd_filestat_set_times: () => 0,
      fd_pread(fd, iovs, n, offset, nread) {
        const d = vm.fd_pread(pid, fd, capacity(iovs, n), offset);
        if (!d) return errno();
        view().setUint32(nread, scatter(iovs, n, d), true);
        return 0;
      },
      fd_pwrite(fd, iovs, n, offset, nwritten) {
        host.tick();
        const w = vm.fd_pwrite(pid, fd, gather(iovs, n), offset);
        if (w < 0) return errno();
        view().setUint32(nwritten, w, true);
        host.onChange();
        return 0;
      },
      fd_prestat_get(fd, ptr) {
        const name = vm.fd_prestat(pid, fd);
        if (name === undefined) return errno();
        view().setUint8(ptr, 0);
        view().setUint32(ptr + 4, enc.encode(name).length, true);
        return 0;
      },
      fd_prestat_dir_name(fd, ptr, len) {
        const name = vm.fd_prestat(pid, fd);
        if (name === undefined) return errno();
        bytes().set(enc.encode(name).subarray(0, len), ptr);
        return 0;
      },
      fd_read: blocking((st, fd, iovs, n, nread) => {
        const d = vm.fd_read(pid, fd, capacity(iovs, n));
        if (!d) {
          const e = errno();
          return e === E.AGAIN ? undefined : e;
        }
        view().setUint32(nread, scatter(iovs, n, d), true);
        host.changed();
        return 0;
      }),
      fd_write: blocking((st, fd, iovs, n, nwritten) => {
        const data = gather(iovs, n);
        if (!data.length) {
          view().setUint32(nwritten, 0, true);
          return 0;
        }
        host.tick();
        const w = vm.fd_write(pid, fd, data);
        if (w < 0) {
          const e = errno();
          return e === E.AGAIN ? undefined : e;
        }
        view().setUint32(nwritten, w, true);
        host.changed();
        host.onChange();
        return 0;
      }),
      fd_readdir(fd, buf, len, cookie, used) {
        const d = vm.fd_readdir(pid, fd, cookie, len);
        if (!d) return errno();
        bytes().set(d, buf);
        view().setUint32(used, d.length, true);
        return 0;
      },
      fd_renumber(from, to) {
        return vm.fd_renumber(pid, from, to) ? 0 : errno();
      },
      fd_seek(fd, offset, whence, ptr) {
        const r = vm.fd_seek(pid, fd, offset, whence);
        if (r === undefined) return errno();
        view().setBigUint64(ptr, r, true);
        return 0;
      },
      fd_tell(fd, ptr) {
        const r = vm.fd_seek(pid, fd, 0n, 1);
        if (r === undefined) return errno();
        view().setBigUint64(ptr, r, true);
        return 0;
      },
      path_create_directory(fd, p, l) {
        host.tick();
        const ok = vm.path_mkdir(pid, fd, str(p, l));
        host.onChange();
        return ok ? 0 : errno();
      },
      path_filestat_get(fd, flags, p, l, ptr) {
        const s = vm.path_stat(pid, fd, str(p, l), (flags & 1) !== 0);
        if (!s) return errno();
        setStat(ptr, s);
        return 0;
      },
      path_filestat_set_times(fd, flags, p, l, atim, mtim, fst) {
        const when = fst & 2 ? mtim : BigInt(Date.now()) * 1000000n;
        return vm.path_set_times(pid, fd, str(p, l), when) ? 0 : errno();
      },
      path_link: () => E.NOTSUP,
      path_open(dirfd, dirflags, p, l, oflags, rights, inheriting, fdflags, fdPtr) {
        host.tick();
        const fd = vm.path_open(pid, dirfd, str(p, l), (dirflags & 1) !== 0, oflags, rights, fdflags);
        if (fd < 0) return errno();
        view().setUint32(fdPtr, fd, true);
        if (oflags) host.onChange();
        return 0;
      },
      path_readlink(fd, p, l, buf, bufLen, used) {
        const t = vm.path_readlink(pid, fd, str(p, l));
        if (t === undefined) return errno();
        const e = enc.encode(t).subarray(0, bufLen);
        bytes().set(e, buf);
        view().setUint32(used, e.length, true);
        return 0;
      },
      path_remove_directory(fd, p, l) {
        const ok = vm.path_remove(pid, fd, str(p, l), true);
        host.onChange();
        return ok ? 0 : errno();
      },
      path_rename(fd1, p1, l1, fd2, p2, l2) {
        const ok = vm.path_rename(pid, fd1, str(p1, l1), fd2, str(p2, l2));
        host.onChange();
        return ok ? 0 : errno();
      },
      path_symlink(p1, l1, fd, p2, l2) {
        const ok = vm.path_symlink(pid, str(p1, l1), fd, str(p2, l2));
        host.onChange();
        return ok ? 0 : errno();
      },
      path_unlink_file(fd, p, l) {
        const ok = vm.path_remove(pid, fd, str(p, l), false);
        host.onChange();
        return ok ? 0 : errno();
      },
      poll_oneoff: blocking((st, inPtr, outPtr, nsubs, neventsPtr) => {
        const v = view();
        const now = performance.now();
        st.start ??= now;
        let deadline = null;
        const events = [];
        for (let i = 0; i < nsubs; i++) {
          const s = inPtr + i * 48;
          const tag = v.getUint8(s + 8);
          const userdata = v.getBigUint64(s, true);
          if (tag === 0) {
            // A clock: relative to when this call began, or an absolute time on its clock.
            const clock = v.getUint32(s + 16, true);
            const t = Number(v.getBigUint64(s + 24, true)) / 1e6;
            const abs = (v.getUint16(s + 40, true) & 1) !== 0;
            const due = !abs ? st.start + t : clock === 0 ? now + (t - Date.now()) : t;
            deadline = deadline === null ? due : Math.min(deadline, due);
            if (now >= due) events.push([userdata, 0]);
          } else if (vm.fd_ready(pid, v.getUint32(s + 16, true), tag === 2)) {
            events.push([userdata, tag]);
          }
        }
        if (!events.length) {
          // Nothing ready: wait for a change, or wake at the earliest deadline.
          if (deadline !== null && !st.timer) st.timer = setTimeout(() => host.changed(), Math.max(0, deadline - now) + 1);
          return undefined;
        }
        clearTimeout(st.timer);
        events.forEach(([userdata, type], i) => {
          const o = outPtr + i * 32;
          v.setBigUint64(o, userdata, true);
          v.setUint16(o + 8, 0, true);
          v.setUint8(o + 10, type);
          v.setBigUint64(o + 16, 0n, true);
          v.setUint16(o + 24, 0, true);
        });
        v.setUint32(neventsPtr, events.length, true);
        return 0;
      }),
      proc_exit(code) {
        throw new Exit(code);
      },
      proc_raise: () => E.NOSYS,
      sched_yield: new WebAssembly.Suspending(() => new Promise((r) => setTimeout(() => r(0), 0))),
      random_get(ptr, len) {
        const b = bytes();
        for (let at = 0; at < len; at += 65536) crypto.getRandomValues(b.subarray(ptr + at, ptr + Math.min(len, at + 65536)));
        return 0;
      },
      sock_accept: () => E.NOTSUP,
      sock_recv: () => E.NOTSUP,
      sock_send: () => E.NOTSUP,
      sock_shutdown: () => E.NOTSUP,
    };

    const evolve = {
      http_fetch: new WebAssembly.Suspending(async (reqPtr, reqLen, outLenPtr) => {
        check();
        let reqs = JSON.parse(str(reqPtr, reqLen));
        if (!Array.isArray(reqs)) reqs = [reqs];
        const out = enc.encode(JSON.stringify(await Promise.all(reqs.map((r) => host.fetch(r)))));
        check();
        host.http.set(pid, out);
        view().setUint32(outLenPtr, out.length, true);
        return 0;
      }),
      http_read(bufPtr, len) {
        const b = host.http.get(pid) || new Uint8Array();
        bytes().set(b.subarray(0, len), bufPtr);
        host.http.delete(pid);
        return 0;
      },
      ask_user: new WebAssembly.Suspending(async (qPtr, qLen, answerPtr) => {
        check();
        const yes = await host.ask(str(qPtr, qLen), pid);
        check();
        view().setUint32(answerPtr, yes ? 1 : 0, true);
        return 0;
      }),
      agent_run: new WebAssembly.Suspending(async (taskPtr, taskLen, cwdPtr, cwdLen, statusPtr) => {
        check();
        // Pi's progress goes to this program's standard output, wherever that leads.
        const write = (text) => {
          if (!vm.running(pid)) return;
          vm.fd_write(pid, 1, enc.encode(text));
          host.changed();
        };
        const code = await host.agent(str(taskPtr, taskLen), str(cwdPtr, cwdLen), write, pid);
        view().setInt32(statusPtr, code, true);
        return 0;
      }),
      policy(levelPtr) {
        view().setUint32(levelPtr, host.github.autonomy, true);
        return 0;
      },
      proc_spawn(argvPtr, argvLen, envPtr, envLen, cwdPtr, cwdLen, mapPtr, mapLen, pgid, pidOut) {
        const argv = str(argvPtr, argvLen).split('\0');
        const env = envLen ? str(envPtr, envLen).split('\0') : [];
        const cwd = str(cwdPtr, cwdLen);
        const map = new Uint32Array(getMemory().buffer.slice(mapPtr, mapPtr + mapLen * 4));
        const found = host.resolve(pid, argv, env);
        if (!found) return E.NOENT;
        const [path, args] = found;
        const child = vm.spawn(pid, args, env, cwd, map, pgid);
        if (child < 0) return errno();
        view().setUint32(pidOut, child, true);
        host.run(child, path);
        return 0;
      },
      proc_wait: blocking((st, child, statusPtr) => {
        const code = vm.wait(pid, child);
        if (code === -1 && errno() === E.AGAIN) return undefined;
        if (code === -1 && errno() === E.CHILD) return E.CHILD;
        view().setInt32(statusPtr, code, true);
        return 0;
      }),
      fd_pipe(rPtr, wPtr) {
        const p = vm.pipe(pid);
        if (!p.length) return errno();
        view().setUint32(rPtr, p[0], true);
        view().setUint32(wPtr, p[1], true);
        return 0;
      },
      tty_set_foreground(pgid) {
        const t = vm.tty_of(pid);
        if (t >= 0) vm.set_foreground(t, pgid === 0 ? vm.pgid(pid) : pgid);
        return 0;
      },
      proc_ignore_interrupt(on) {
        vm.set_ignore_interrupt(pid, on !== 0);
        return 0;
      },
      tty_set_raw(on) {
        const t = vm.tty_of(pid);
        if (t >= 0) vm.tty_set_raw(t, on !== 0);
        return 0;
      },
      tty_size(colsPtr, rowsPtr) {
        const t = vm.tty_of(pid);
        const [c, r] = vm.tty_size(Math.max(0, t));
        view().setUint32(colsPtr, c, true);
        view().setUint32(rowsPtr, r, true);
        return 0;
      },
    };
    // Debugging: log every WASI call of every program (globalThis.VM_TRACE = true).
    if (globalThis.VM_TRACE) {
      for (const [name, f] of Object.entries(wasi)) {
        if (typeof f !== 'function') continue;
        wasi[name] = (...args) => {
          const r = f(...args);
          console.log(`[${pid}] ${name}(${args.join(', ')}) -> ${r}${name.startsWith('path_') ? ` ${JSON.stringify(str(args[name === 'path_symlink' ? 3 : name === 'path_open' ? 2 : name === 'path_filestat_get' ? 2 : 1], args[name === 'path_open' ? 3 : name === 'path_filestat_get' ? 3 : 2]))}` : ''}`);
          return r;
        };
      }
    }
    return { wasi_snapshot_preview1: wasi, evolve };
  }
}
