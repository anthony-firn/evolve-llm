// Pi's workspace on the Evolve VM: the same requests as the browser sandbox (web/shell-client.mjs)
// — exec, read, write, edit — answered by the VM, in a working directory.

export class VmWorkspace {
  constructor(vm, { cwd = '/home/user', timeoutMs = 600000 } = {}) {
    this.vm = vm;
    this.cwd = cwd;
    this.timeoutMs = timeoutMs;
    this.checkpoint = null;
  }

  path(p) {
    const abs = String(p || '').startsWith('/') ? String(p) : `${this.cwd.replace(/\/$/, '')}/${p}`;
    const out = [];
    for (const part of abs.split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') out.pop();
      else out.push(part);
    }
    return `/${out.join('/')}`;
  }

  async request(type, data = {}, { signal } = {}) {
    signal?.throwIfAborted();
    if (type === 'exec') {
      const r = await this.vm.exec(data.command, { cwd: this.cwd, timeoutMs: this.timeoutMs });
      signal?.throwIfAborted();
      // A `cd` carries over to the next command and to file paths, as in a terminal.
      if (r.cwd) this.cwd = r.cwd;
      return { code: r.code, stdout: r.stdout, stderr: r.stderr, cwd: this.cwd, ...(r.timedOut && { timedOut: true }), ...(r.truncated && { truncated: true }) };
    }
    if (type === 'read') {
      const r = await this.vm.read(this.path(data.path));
      if (!r.ok) throw new Error(r.kind === 3 ? `${data.path} is a directory; run ls ${data.path} to see its files` : `${data.path}: no such file in ${this.cwd}; run ls to see the files here`);
      return { path: this.path(data.path), text: r.data };
    }
    if (type === 'write') {
      await this.vm.write(this.path(data.path), data.text ?? '');
      return { path: this.path(data.path), bytes: new TextEncoder().encode(data.text ?? '').length };
    }
    if (type === 'edit') {
      const path = this.path(data.path);
      const r = await this.vm.read(path);
      if (!r.ok) throw new Error(`${data.path}: no such file in ${this.cwd}; run ls to see the files here`);
      if (data.oldText && data.oldText === (data.newText ?? '')) throw new Error('oldText and newText are the same, so nothing would change. Put the corrected text in newText.');
      const count = data.oldText ? r.data.split(data.oldText).length - 1 : 0;
      if (count !== 1) throw new Error(count ? `oldText matches ${count} places in ${data.path}; include more context` : `oldText is not in ${data.path}; read the file again and copy the exact text`);
      await this.vm.write(path, r.data.replace(data.oldText, () => data.newText ?? ''));
      return { path, replaced: 1 };
    }
    if (type === 'snapshot') return { snapshot: JSON.stringify({ files: {}, dirs: [], cwd: this.cwd }) };
    throw new Error(`${type} is not available in the Linux workspace`);
  }

  dispose() {}
}
