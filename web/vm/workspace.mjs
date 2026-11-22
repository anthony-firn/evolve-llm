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
      if (count === 0) {
        const loose = looseReplace(r.data, data.oldText || '', data.newText ?? '');
        if (loose != null) {
          await this.vm.write(path, loose);
          return { path, replaced: 1, note: 'matched with blank lines and spaces at line ends ignored; only the changed lines were replaced' };
        }
      }
      if (count !== 1) throw new Error(count ? `oldText matches ${count} places in ${data.path}; include more context` : `oldText is not in ${data.path}; read the file again and copy the exact text`);
      await this.vm.write(path, r.data.replace(data.oldText, () => data.newText ?? ''));
      return { path, replaced: 1 };
    }
    if (type === 'snapshot') return { snapshot: JSON.stringify({ files: {}, dirs: [], cwd: this.cwd }) };
    throw new Error(`${type} is not available in the Linux workspace`);
  }

  dispose() {}
}

/** An edit whose oldText differs from the file only in blank lines or spaces at line ends (small
 *  models rarely copy blank lines exactly): its whole lines are found once in the file, and only
 *  the lines that differ between oldText and newText are replaced, so the file's own blank lines
 *  stay. Returns the new text, or null when there is no single such place. */
export function looseReplace(text, oldText, newText) {
  const norm = l => l.replace(/\s+$/, '');
  const fileL = text.split('\n'), oldL = oldText.split('\n'), newL = newText.split('\n');
  const want = oldL.map(norm).filter(Boolean);
  if (!want.length) return null;
  const lines = fileL.map((l, i) => [norm(l), i]).filter(([l]) => l);
  const hits = [];
  for (let i = 0; i + want.length <= lines.length; i++) if (want.every((w, j) => lines[i + j][0] === w)) hits.push(i);
  if (hits.length !== 1) return null;
  const at = lines.slice(hits[0], hits[0] + want.length).map(([, i]) => i);
  let p = 0;
  while (p < oldL.length && p < newL.length && norm(oldL[p]) === norm(newL[p])) p++;
  let s = 0;
  while (s < oldL.length - p && s < newL.length - p && norm(oldL[oldL.length - 1 - s]) === norm(newL[newL.length - 1 - s])) s++;
  const oldMid = oldL.slice(p, oldL.length - s), newMid = newL.slice(p, newL.length - s);
  const before = oldL.slice(0, p).map(norm).filter(Boolean).length;
  const changed = oldMid.map(norm).filter(Boolean).length;
  let from, to;
  if (changed) [from, to] = [at[before], at[before + changed - 1] + 1];
  else if (oldMid.length) return null; // only blank lines differ: leave that to an exact edit
  else from = to = before ? at[before - 1] + 1 : at[0];
  const out = [...fileL.slice(0, from), ...newMid, ...fileL.slice(to)].join('\n');
  return out === text ? null : out;
}
