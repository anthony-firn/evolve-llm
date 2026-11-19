//! The kernel as the JavaScript host sees it. Calls that fail return a sentinel (-1, None or
//! false) and leave the WASI errno in `errno()`.

use crate::{errno, Kernel};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Vm {
    k: Kernel,
}

fn pairs(flat: &[u32]) -> Vec<(u32, u32)> {
    flat.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

#[wasm_bindgen]
impl Vm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Vm {
        Vm { k: Kernel::new() }
    }

    pub fn errno(&self) -> u16 {
        self.k.errno
    }

    pub fn set_time(&mut self, ns: f64) {
        self.k.now = ns as u64;
    }

    pub fn seed(&mut self, s: f64) {
        self.k.seed(s as u64);
    }

    // ---- files, for the host ----

    pub fn write_file(&mut self, path: &str, data: &[u8]) -> bool {
        self.k.write_file(path, data)
    }

    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        self.k.read_file(path)
    }

    pub fn mkdir_p(&mut self, path: &str) {
        self.k.mkdir_p(path);
    }

    pub fn set_mode(&mut self, path: &str, mode: u32) -> bool {
        self.k.set_mode(path, mode)
    }

    pub fn symlink(&mut self, target: &str, path: &str) -> bool {
        let root = self.k.root;
        self.k.symlink(target, root, path).is_ok()
    }

    /// `ls -l`-style JSON listing of a directory: [{name, kind, size, mtime}].
    pub fn list(&self, path: &str) -> Option<String> {
        let ino = self.k.lookup(self.k.root, path, true).ok()?;
        let entries = self.k.list(ino).ok()?;
        let rows: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| {
                let s = self.k.stat_ino(e.ino);
                serde_json::json!({ "name": e.name, "kind": e.kind, "size": s[4], "mtime": s[6] })
            })
            .collect();
        Some(serde_json::to_string(&rows).unwrap())
    }

    /// 0 none, 3 directory, 4 file, 7 symlink, 2 device.
    pub fn kind(&self, path: &str) -> u8 {
        self.k.lookup(self.k.root, path, false).ok().and_then(|i| self.k.inode(i)).map(|n| n.filetype()).unwrap_or(0)
    }

    pub fn snapshot(&self, prefix: &str) -> Vec<u8> {
        self.k.snapshot(prefix)
    }

    pub fn restore(&mut self, bytes: &[u8]) -> bool {
        self.k.restore(bytes)
    }

    // ---- processes and terminals ----

    pub fn create_tty(&mut self) -> u32 {
        self.k.create_tty()
    }

    pub fn spawn_on_tty(&mut self, argv: Vec<String>, env: Vec<String>, cwd: &str, tty: u32) -> i32 {
        match self.k.spawn_on_tty(argv, env, cwd, tty) {
            Ok(pid) => pid as i32,
            Err(e) => {
                self.k.errno = e;
                -1
            }
        }
    }

    /// `fd_map`: flattened (child fd, parent fd) pairs. `pgid` 0 starts a new group.
    pub fn spawn(&mut self, parent: u32, argv: Vec<String>, env: Vec<String>, cwd: &str, fd_map: &[u32], pgid: u32) -> i32 {
        match self.k.spawn(parent, argv, env, cwd, &pairs(fd_map), pgid) {
            Ok(pid) => pid as i32,
            Err(e) => {
                self.k.errno = e;
                -1
            }
        }
    }

    pub fn argv(&self, pid: u32) -> Vec<String> {
        self.k.proc(pid).map(|p| p.argv.clone()).unwrap_or_default()
    }

    pub fn env(&self, pid: u32) -> Vec<String> {
        self.k.proc(pid).map(|p| p.env.clone()).unwrap_or_default()
    }

    pub fn cwd(&self, pid: u32) -> String {
        self.k.proc(pid).map(|p| self.k.path_of(p.cwd)).unwrap_or_else(|| "/".into())
    }

    pub fn exit(&mut self, pid: u32, code: i32) {
        self.k.exit(pid, code);
    }

    /// Exit status, or -1 with errno AGAIN while the child runs (CHILD if it is not one).
    pub fn wait(&mut self, parent: u32, child: u32) -> i32 {
        match self.k.wait(parent, child) {
            Ok(c) => c,
            Err(e) => {
                self.k.errno = e;
                -1
            }
        }
    }

    /// The exit status a signal asked for, or -1.
    pub fn killed(&self, pid: u32) -> i32 {
        self.k.proc(pid).and_then(|p| p.killed).unwrap_or(-1)
    }

    pub fn running(&self, pid: u32) -> bool {
        self.k.proc(pid).is_some_and(|p| p.exit.is_none())
    }

    pub fn set_ignore_interrupt(&mut self, pid: u32, on: bool) {
        self.k.set_ignore_interrupt(pid, on);
    }

    pub fn kill_group(&mut self, pgid: u32, status: i32) {
        self.k.kill_group(pgid, status);
    }

    pub fn pgid(&self, pid: u32) -> u32 {
        self.k.proc(pid).map(|p| p.pgid).unwrap_or(0)
    }

    pub fn set_foreground(&mut self, tty: u32, pgid: u32) {
        self.k.set_foreground(tty, pgid);
    }

    pub fn tty_input(&mut self, id: u32, bytes: &[u8]) {
        self.k.tty_input(id, bytes);
    }

    pub fn tty_take_output(&mut self, id: u32) -> Vec<u8> {
        self.k.tty_take_output(id)
    }

    pub fn tty_resize(&mut self, id: u32, cols: u16, rows: u16) {
        self.k.tty_resize(id, cols, rows);
    }

    pub fn tty_set_raw(&mut self, id: u32, raw: bool) {
        self.k.tty_set_raw(id, raw);
    }

    /// The terminal on a process's standard input, or -1.
    pub fn tty_of(&self, pid: u32) -> i32 {
        self.k.tty_of(pid).map(|t| t as i32).unwrap_or(-1)
    }

    /// [cols, rows] of a terminal.
    pub fn tty_size(&self, id: u32) -> Vec<u32> {
        self.k.tty(id).map(|t| vec![t.cols as u32, t.rows as u32]).unwrap_or_else(|| vec![80, 24])
    }

    pub fn which(&self, pid: u32, name: &str, path: &str) -> Option<String> {
        self.k.which(pid, name, path)
    }

    // ---- WASI calls ----

    /// Bytes taken, or -1 (AGAIN: wait until the pipe has room).
    pub fn fd_write(&mut self, pid: u32, fd: u32, data: &[u8]) -> i32 {
        self.k.fd_write(pid, fd, data).map(|n| n as i32).unwrap_or(-1)
    }

    /// None with AGAIN means wait for input.
    pub fn fd_read(&mut self, pid: u32, fd: u32, max: u32) -> Option<Vec<u8>> {
        self.k.fd_read(pid, fd, max)
    }

    pub fn fd_pread(&mut self, pid: u32, fd: u32, max: u32, offset: u64) -> Option<Vec<u8>> {
        self.k.fd_pread(pid, fd, max, offset)
    }

    pub fn fd_pwrite(&mut self, pid: u32, fd: u32, data: &[u8], offset: u64) -> i32 {
        self.k.fd_pwrite(pid, fd, data, offset).map(|n| n as i32).unwrap_or(-1)
    }

    pub fn fd_seek(&mut self, pid: u32, fd: u32, offset: i64, whence: u8) -> Option<u64> {
        self.k.fd_seek(pid, fd, offset, whence)
    }

    pub fn fd_close(&mut self, pid: u32, fd: u32) -> bool {
        self.k.fd_close(pid, fd)
    }

    /// [file type, flags], or empty.
    pub fn fd_fdstat(&mut self, pid: u32, fd: u32) -> Vec<u32> {
        self.k.fd_fdstat(pid, fd).map(|(t, f)| vec![t as u32, f as u32]).unwrap_or_default()
    }

    pub fn fd_set_flags(&mut self, pid: u32, fd: u32, flags: u16) -> bool {
        self.k.fd_set_flags(pid, fd, flags)
    }

    /// [dev, ino, filetype, nlink, size, atim, mtim, ctim].
    pub fn fd_filestat(&mut self, pid: u32, fd: u32) -> Option<Vec<u64>> {
        self.k.fd_filestat(pid, fd).map(|s| s.to_vec())
    }

    pub fn fd_set_size(&mut self, pid: u32, fd: u32, size: u64) -> bool {
        self.k.fd_set_size(pid, fd, size)
    }

    pub fn fd_prestat(&mut self, pid: u32, fd: u32) -> Option<String> {
        self.k.fd_prestat(pid, fd)
    }

    /// WASI dirents from `cookie`, packed as fd_readdir returns them, at most `buf_len` bytes.
    pub fn fd_readdir(&mut self, pid: u32, fd: u32, cookie: u64, buf_len: u32) -> Option<Vec<u8>> {
        let entries = self.k.fd_readdir(pid, fd, cookie)?;
        let mut out = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            out.extend((cookie + i as u64 + 1).to_le_bytes());
            out.extend(e.ino.to_le_bytes());
            out.extend((e.name.len() as u32).to_le_bytes());
            out.push(e.kind);
            out.extend([0u8; 3]);
            out.extend(e.name.as_bytes());
            if out.len() >= buf_len as usize {
                break;
            }
        }
        out.truncate(buf_len as usize);
        Some(out)
    }

    pub fn fd_renumber(&mut self, pid: u32, from: u32, to: u32) -> bool {
        self.k.fd_renumber(pid, from, to)
    }

    pub fn fd_dup2(&mut self, pid: u32, from: u32, to: u32) -> bool {
        self.k.fd_dup2(pid, from, to)
    }

    /// [read fd, write fd], or empty.
    pub fn pipe(&mut self, pid: u32) -> Vec<u32> {
        self.k.pipe(pid).map(|(r, w)| vec![r, w]).unwrap_or_default()
    }

    pub fn fd_ready(&self, pid: u32, fd: u32, write: bool) -> bool {
        self.k.fd_ready(pid, fd, write)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn path_open(&mut self, pid: u32, dirfd: u32, path: &str, follow: bool, oflags: u16, rights: u64, fdflags: u16) -> i32 {
        self.k.path_open(pid, dirfd, path, follow, oflags, rights, fdflags).map(|f| f as i32).unwrap_or(-1)
    }

    pub fn path_stat(&mut self, pid: u32, dirfd: u32, path: &str, follow: bool) -> Option<Vec<u64>> {
        self.k.path_stat(pid, dirfd, path, follow).map(|s| s.to_vec())
    }

    pub fn path_mkdir(&mut self, pid: u32, dirfd: u32, path: &str) -> bool {
        self.k.path_mkdir(pid, dirfd, path)
    }

    pub fn path_remove(&mut self, pid: u32, dirfd: u32, path: &str, dir: bool) -> bool {
        self.k.path_remove(pid, dirfd, path, dir)
    }

    pub fn path_rename(&mut self, pid: u32, fd1: u32, from: &str, fd2: u32, to: &str) -> bool {
        self.k.path_rename(pid, fd1, from, fd2, to)
    }

    pub fn path_symlink(&mut self, pid: u32, target: &str, dirfd: u32, path: &str) -> bool {
        self.k.path_symlink(pid, target, dirfd, path)
    }

    pub fn path_readlink(&mut self, pid: u32, dirfd: u32, path: &str) -> Option<String> {
        self.k.path_readlink(pid, dirfd, path)
    }

    pub fn path_set_times(&mut self, pid: u32, dirfd: u32, path: &str, mtime: u64) -> bool {
        self.k.path_set_times(pid, dirfd, path, mtime)
    }
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
const _: u16 = errno::SUCCESS;
