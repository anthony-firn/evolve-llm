//! The Evolve VM kernel: an operating system for WebAssembly programs, compiled to WebAssembly.
//!
//! Programs are WASI (preview 1) modules that the browser compiles to native machine code, so
//! nothing is emulated: the kernel provides what an operating system provides around them. A
//! file system (directories, files, symlinks, devices), open file descriptions shared between
//! processes, pipes, terminals with a line discipline (editing, echo, Ctrl+C, Ctrl+D), and a
//! process table. The JavaScript host (web/vm/host.mjs) instantiates each program, passes its
//! WASI calls here, and suspends a program whose call would block (JSPI) until the kernel says
//! it can continue.
//!
//! Every call that can fail sets `errno` (WASI numbering); the host reads it after the call.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
mod bindings;

/// WASI errno values.
pub mod errno {
    pub const SUCCESS: u16 = 0;
    pub const AGAIN: u16 = 6;
    pub const BADF: u16 = 8;
    pub const CHILD: u16 = 12;
    pub const EXIST: u16 = 20;
    pub const INVAL: u16 = 28;
    pub const IO: u16 = 29;
    pub const ISDIR: u16 = 31;
    pub const LOOP: u16 = 32;
    pub const NAMETOOLONG: u16 = 37;
    pub const NOENT: u16 = 44;
    pub const NOSYS: u16 = 52;
    pub const NOTDIR: u16 = 54;
    pub const NOTEMPTY: u16 = 55;
    pub const NOTSUP: u16 = 58;
    pub const PERM: u16 = 63;
    pub const PIPE: u16 = 64;
    pub const SPIPE: u16 = 70;
    pub const SRCH: u16 = 71;
}

/// WASI file types.
pub mod filetype {
    pub const UNKNOWN: u8 = 0;
    pub const CHARACTER_DEVICE: u8 = 2;
    pub const DIRECTORY: u8 = 3;
    pub const REGULAR_FILE: u8 = 4;
    pub const SYMBOLIC_LINK: u8 = 7;
}

const OFLAG_CREAT: u16 = 1;
const OFLAG_DIRECTORY: u16 = 2;
const OFLAG_EXCL: u16 = 4;
const OFLAG_TRUNC: u16 = 8;
const FDFLAG_APPEND: u16 = 1;
const RIGHT_FD_READ: u64 = 1 << 1;
const RIGHT_FD_WRITE: u64 = 1 << 6;
const PIPE_CAPACITY: usize = 1 << 20;
const MAX_SYMLINKS: u32 = 40;

pub type Ino = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dev {
    Null,
    Zero,
    Random,
    Tty(u32),
}

#[derive(Clone, Debug)]
pub enum Node {
    Dir(BTreeMap<String, Ino>),
    File(Vec<u8>),
    Symlink(String),
    Dev(Dev),
}

#[derive(Clone, Debug)]
pub struct Inode {
    pub node: Node,
    pub parent: Ino,
    pub mode: u32,
    pub mtime: u64,
    pub ctime: u64,
}

impl Inode {
    pub fn filetype(&self) -> u8 {
        match self.node {
            Node::Dir(_) => filetype::DIRECTORY,
            Node::File(_) => filetype::REGULAR_FILE,
            Node::Symlink(_) => filetype::SYMBOLIC_LINK,
            Node::Dev(_) => filetype::CHARACTER_DEVICE,
        }
    }

    fn size(&self) -> u64 {
        match &self.node {
            Node::File(d) => d.len() as u64,
            Node::Symlink(t) => t.len() as u64,
            Node::Dir(e) => e.len() as u64,
            Node::Dev(_) => 0,
        }
    }
}

/// What an open file description refers to.
#[derive(Clone, Debug, PartialEq)]
pub enum Obj {
    File(Ino),
    Dir(Ino),
    PipeRead(u32),
    PipeWrite(u32),
    Tty(u32),
    Null,
    Zero,
    Random,
}

/// An open file description: shared by every descriptor duplicated or inherited from it.
#[derive(Debug)]
pub struct Desc {
    pub obj: Obj,
    pub offset: u64,
    pub read: bool,
    pub write: bool,
    pub flags: u16,
    /// Set for the preopened directories WASI programs find at startup ("/" and ".").
    pub preopen: Option<String>,
}

type DescRef = Rc<RefCell<Desc>>;

#[derive(Default, Debug)]
pub struct Pipe {
    buf: VecDeque<u8>,
    readers: u32,
    writers: u32,
}

#[derive(Debug)]
pub struct Tty {
    pub input: VecDeque<u8>,
    line: Vec<u8>,
    pub output: Vec<u8>,
    pub raw: bool,
    eof: u32,
    escape: u8,
    pub fg: u32,
    pub cols: u16,
    pub rows: u16,
}

#[derive(Debug)]
pub struct Proc {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub cwd: Ino,
    pub fds: BTreeMap<u32, DescRef>,
    pub argv: Vec<String>,
    pub env: Vec<String>,
    pub exit: Option<i32>,
    /// Exit status a signal (Ctrl+C) asked for; the host ends the program at its next call.
    pub killed: Option<i32>,
    /// Shells ignore Ctrl+C (it only ends what they run).
    pub ignore_interrupt: bool,
}

/// One directory entry as `fd_readdir` and listings report it.
#[derive(Debug, Clone, PartialEq)]
pub struct DirEntry {
    pub name: String,
    pub ino: Ino,
    pub kind: u8,
}

pub struct Kernel {
    inodes: HashMap<Ino, Inode>,
    next_ino: Ino,
    pub root: Ino,
    pipes: HashMap<u32, Pipe>,
    next_pipe: u32,
    ttys: HashMap<u32, Tty>,
    procs: BTreeMap<u32, Proc>,
    next_pid: u32,
    pub errno: u16,
    pub now: u64,
    rng: u64,
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}

fn split_path(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|c| !c.is_empty() && *c != ".")
}

impl Kernel {
    /// A fresh system: /bin, /usr/bin, /home/user, /tmp, /etc and /dev.
    pub fn new() -> Kernel {
        let mut k = Kernel {
            inodes: HashMap::new(),
            next_ino: 1,
            root: 1,
            pipes: HashMap::new(),
            next_pipe: 1,
            ttys: HashMap::new(),
            procs: BTreeMap::new(),
            next_pid: 1,
            errno: 0,
            now: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
        };
        k.inodes.insert(1, Inode { node: Node::Dir(BTreeMap::new()), parent: 1, mode: 0o755, mtime: 0, ctime: 0 });
        k.next_ino = 2;
        for d in ["/bin", "/usr/bin", "/usr/lib", "/home/user", "/tmp", "/etc", "/dev", "/mnt"] {
            k.mkdir_p(d);
        }
        for (name, dev) in [("null", Dev::Null), ("zero", Dev::Zero), ("urandom", Dev::Random), ("random", Dev::Random)] {
            let dev_dir = k.lookup(k.root, "/dev", true).unwrap();
            k.insert(dev_dir, name, Node::Dev(dev), 0o666);
        }
        k.write_file("/etc/hostname", b"evolve\n");
        k.write_file("/etc/passwd", b"user:x:1000:1000:user:/home/user:/bin/sh\n");
        k
    }

    // ---- inodes and paths ----

    fn alloc(&mut self, node: Node, parent: Ino, mode: u32) -> Ino {
        let ino = self.next_ino;
        self.next_ino += 1;
        self.inodes.insert(ino, Inode { node, parent, mode, mtime: self.now, ctime: self.now });
        ino
    }

    fn insert(&mut self, dir: Ino, name: &str, node: Node, mode: u32) -> Ino {
        let ino = self.alloc(node, dir, mode);
        let now = self.now;
        if let Some(Inode { node: Node::Dir(entries), mtime, .. }) = self.inodes.get_mut(&dir) {
            entries.insert(name.to_string(), ino);
            *mtime = now;
        }
        ino
    }

    pub fn inode(&self, ino: Ino) -> Option<&Inode> {
        self.inodes.get(&ino)
    }

    fn entries(&self, dir: Ino) -> Result<&BTreeMap<String, Ino>, u16> {
        match self.inodes.get(&dir).map(|i| &i.node) {
            Some(Node::Dir(e)) => Ok(e),
            Some(_) => Err(errno::NOTDIR),
            None => Err(errno::NOENT),
        }
    }

    /// Resolve `path` from directory `start` (absolute paths start at the root), following
    /// symlinks along the way and, with `follow`, at the end.
    pub fn lookup(&self, start: Ino, path: &str, follow: bool) -> Result<Ino, u16> {
        let mut depth = 0;
        self.walk(start, path, follow, &mut depth)
    }

    fn walk(&self, start: Ino, path: &str, follow: bool, depth: &mut u32) -> Result<Ino, u16> {
        if path.len() > 4096 {
            return Err(errno::NAMETOOLONG);
        }
        let mut cur = if path.starts_with('/') { self.root } else { start };
        let parts: Vec<&str> = split_path(path).collect();
        for (i, part) in parts.iter().enumerate() {
            let last = i + 1 == parts.len();
            if *part == ".." {
                cur = self.inodes.get(&cur).map(|n| n.parent).ok_or(errno::NOENT)?;
                continue;
            }
            let next = *self.entries(cur)?.get(*part).ok_or(errno::NOENT)?;
            if let Some(Node::Symlink(target)) = self.inodes.get(&next).map(|n| &n.node) {
                if !last || follow {
                    *depth += 1;
                    if *depth > MAX_SYMLINKS {
                        return Err(errno::LOOP);
                    }
                    cur = self.walk(cur, &target.clone(), true, depth)?;
                    continue;
                }
            }
            cur = next;
        }
        // A trailing slash requires a directory.
        if path.ends_with('/') && !matches!(self.inodes.get(&cur).map(|n| &n.node), Some(Node::Dir(_))) {
            return Err(errno::NOTDIR);
        }
        Ok(cur)
    }

    /// The directory that would hold `path`, and the name in it.
    fn parent_of(&self, start: Ino, path: &str) -> Result<(Ino, String), u16> {
        let trimmed = path.trim_end_matches('/');
        let (dir, name) = match trimmed.rfind('/') {
            Some(i) => (&trimmed[..i.max(if trimmed.starts_with('/') && i == 0 { 1 } else { i })], &trimmed[i + 1..]),
            None => (".", trimmed),
        };
        if name.is_empty() || name == "." || name == ".." {
            return Err(if trimmed.is_empty() { errno::EXIST } else { errno::INVAL });
        }
        let dir_ino = if dir.is_empty() { self.root } else { self.lookup(if trimmed.starts_with('/') { self.root } else { start }, dir, true)? };
        self.entries(dir_ino)?;
        Ok((dir_ino, name.to_string()))
    }

    /// The absolute path of a directory (for prompts and getcwd).
    pub fn path_of(&self, mut ino: Ino) -> String {
        let mut parts = Vec::new();
        while ino != self.root {
            let parent = match self.inodes.get(&ino) {
                Some(n) => n.parent,
                None => break,
            };
            let name = self.entries(parent).ok().and_then(|e| e.iter().find(|(_, &i)| i == ino).map(|(n, _)| n.clone()));
            match name {
                Some(n) => parts.push(n),
                None => break,
            }
            ino = parent;
        }
        parts.reverse();
        format!("/{}", parts.join("/"))
    }

    // ---- file system operations (also used directly by the host) ----

    pub fn mkdir_p(&mut self, path: &str) -> Ino {
        let mut cur = self.root;
        for part in split_path(path) {
            cur = match self.entries(cur).ok().and_then(|e| e.get(part).copied()) {
                Some(i) => i,
                None => self.insert(cur, part, Node::Dir(BTreeMap::new()), 0o755),
            };
        }
        cur
    }

    pub fn write_file(&mut self, path: &str, data: &[u8]) -> bool {
        match self.parent_of(self.root, path) {
            Ok((dir, name)) => {
                let existing = self.entries(dir).ok().and_then(|e| e.get(&name).copied());
                match existing {
                    Some(ino) => match self.inodes.get_mut(&ino) {
                        Some(Inode { node: Node::File(d), mtime, .. }) => {
                            *d = data.to_vec();
                            *mtime = self.now;
                            true
                        }
                        _ => false,
                    },
                    None => {
                        self.insert(dir, &name, Node::File(data.to_vec()), 0o644);
                        true
                    }
                }
            }
            Err(_) => false,
        }
    }

    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        let ino = self.lookup(self.root, path, true).ok()?;
        match &self.inodes.get(&ino)?.node {
            Node::File(d) => Some(d.clone()),
            _ => None,
        }
    }

    pub fn set_mode(&mut self, path: &str, mode: u32) -> bool {
        match self.lookup(self.root, path, true).ok().and_then(|i| self.inodes.get_mut(&i)) {
            Some(n) => {
                n.mode = mode;
                true
            }
            None => false,
        }
    }

    pub fn symlink(&mut self, target: &str, start: Ino, path: &str) -> Result<(), u16> {
        let (dir, name) = self.parent_of(start, path)?;
        if self.entries(dir)?.contains_key(&name) {
            return Err(errno::EXIST);
        }
        self.insert(dir, &name, Node::Symlink(target.to_string()), 0o777);
        Ok(())
    }

    pub fn list(&self, dir: Ino) -> Result<Vec<DirEntry>, u16> {
        Ok(self
            .entries(dir)?
            .iter()
            .map(|(n, &i)| DirEntry { name: n.clone(), ino: i, kind: self.inodes[&i].filetype() })
            .collect())
    }

    fn mkdir_at(&mut self, start: Ino, path: &str) -> Result<(), u16> {
        let (dir, name) = self.parent_of(start, path)?;
        if self.entries(dir)?.contains_key(&name) {
            return Err(errno::EXIST);
        }
        self.insert(dir, &name, Node::Dir(BTreeMap::new()), 0o755);
        Ok(())
    }

    fn remove_at(&mut self, start: Ino, path: &str, want_dir: bool) -> Result<(), u16> {
        let (dir, name) = self.parent_of(start, path)?;
        let ino = *self.entries(dir)?.get(&name).ok_or(errno::NOENT)?;
        match (&self.inodes[&ino].node, want_dir) {
            (Node::Dir(e), true) if !e.is_empty() => return Err(errno::NOTEMPTY),
            (Node::Dir(_), false) => return Err(errno::ISDIR),
            (Node::Dir(_), true) => {}
            (_, true) => return Err(errno::NOTDIR),
            _ => {}
        }
        if let Some(Node::Dir(e)) = self.inodes.get_mut(&dir).map(|n| &mut n.node) {
            e.remove(&name);
        }
        // Open descriptions keep the data; drop the inode only when nothing refers to it.
        if !self.inode_in_use(ino) {
            self.inodes.remove(&ino);
        }
        Ok(())
    }

    fn inode_in_use(&self, ino: Ino) -> bool {
        self.procs.values().any(|p| p.fds.values().any(|d| matches!(d.borrow().obj, Obj::File(i) | Obj::Dir(i) if i == ino)))
    }

    fn rename_at(&mut self, from_start: Ino, from: &str, to_start: Ino, to: &str) -> Result<(), u16> {
        let (fdir, fname) = self.parent_of(from_start, from)?;
        let ino = *self.entries(fdir)?.get(&fname).ok_or(errno::NOENT)?;
        let (tdir, tname) = self.parent_of(to_start, to)?;
        // A directory cannot move into itself.
        let mut d = tdir;
        loop {
            if d == ino {
                return Err(errno::INVAL);
            }
            if d == self.root {
                break;
            }
            d = self.inodes[&d].parent;
        }
        if let Some(&existing) = self.entries(tdir)?.get(&tname) {
            if existing == ino {
                return Ok(());
            }
            match (&self.inodes[&existing].node, &self.inodes[&ino].node) {
                (Node::Dir(e), Node::Dir(_)) if !e.is_empty() => return Err(errno::NOTEMPTY),
                (Node::Dir(_), Node::Dir(_)) => {}
                (Node::Dir(_), _) => return Err(errno::ISDIR),
                (_, Node::Dir(_)) => return Err(errno::NOTDIR),
                _ => {}
            }
            if !self.inode_in_use(existing) {
                self.inodes.remove(&existing);
            }
        }
        if let Some(Node::Dir(e)) = self.inodes.get_mut(&fdir).map(|n| &mut n.node) {
            e.remove(&fname);
        }
        if let Some(Node::Dir(e)) = self.inodes.get_mut(&tdir).map(|n| &mut n.node) {
            e.insert(tname, ino);
        }
        if let Some(n) = self.inodes.get_mut(&ino) {
            n.parent = tdir;
            n.ctime = self.now;
        }
        Ok(())
    }

    // ---- processes ----

    fn new_desc(&mut self, obj: Obj, read: bool, write: bool, flags: u16) -> DescRef {
        match obj {
            Obj::PipeRead(p) => self.pipes.get_mut(&p).unwrap().readers += 1,
            Obj::PipeWrite(p) => self.pipes.get_mut(&p).unwrap().writers += 1,
            _ => {}
        }
        Rc::new(RefCell::new(Desc { obj, offset: 0, read, write, flags, preopen: None }))
    }

    /// Drop a descriptor's reference to its description; the last one closes a pipe end.
    fn release(&mut self, d: DescRef) {
        if Rc::strong_count(&d) == 1 {
            match d.borrow().obj {
                Obj::PipeRead(p) => {
                    if let Some(pipe) = self.pipes.get_mut(&p) {
                        pipe.readers -= 1;
                    }
                }
                Obj::PipeWrite(p) => {
                    if let Some(pipe) = self.pipes.get_mut(&p) {
                        pipe.writers -= 1;
                    }
                }
                _ => {}
            }
            if let Obj::PipeRead(p) | Obj::PipeWrite(p) = d.borrow().obj {
                if self.pipes.get(&p).is_some_and(|x| x.readers == 0 && x.writers == 0) {
                    self.pipes.remove(&p);
                }
            }
        }
    }

    pub fn create_tty(&mut self) -> u32 {
        let id = self.ttys.len() as u32;
        self.ttys.insert(id, Tty { input: VecDeque::new(), line: Vec::new(), output: Vec::new(), raw: false, eof: 0, escape: 0, fg: 0, cols: 80, rows: 24 });
        let dev = self.lookup(self.root, "/dev", true).unwrap();
        self.insert(dev, &format!("tty{id}"), Node::Dev(Dev::Tty(id)), 0o620);
        id
    }

    /// A process with standard input, output and error on `tty` (for a terminal's shell).
    pub fn spawn_on_tty(&mut self, argv: Vec<String>, env: Vec<String>, cwd: &str, tty: u32) -> Result<u32, u16> {
        let pid = self.new_proc(0, 0, argv, env, cwd)?;
        let d = self.new_desc(Obj::Tty(tty), true, true, 0);
        let p = self.procs.get_mut(&pid).unwrap();
        for fd in 0..3 {
            p.fds.insert(fd, d.clone());
        }
        if let Some(t) = self.ttys.get_mut(&tty) {
            t.fg = pid;
        }
        Ok(pid)
    }

    /// A child of `parent`: `fd_map` lists (child fd, parent fd) pairs that share descriptions.
    /// `pgid` 0 starts a new process group.
    pub fn spawn(&mut self, parent: u32, argv: Vec<String>, env: Vec<String>, cwd: &str, fd_map: &[(u32, u32)], pgid: u32) -> Result<u32, u16> {
        let pp = self.procs.get(&parent).ok_or(errno::SRCH)?;
        let mut inherited = Vec::new();
        for &(child_fd, parent_fd) in fd_map {
            inherited.push((child_fd, pp.fds.get(&parent_fd).ok_or(errno::BADF)?.clone()));
        }
        let start = pp.cwd;
        let cwd_path = if cwd.is_empty() { self.path_of(start) } else { cwd.to_string() };
        let pid = self.new_proc(parent, pgid, argv, env, &cwd_path)?;
        let p = self.procs.get_mut(&pid).unwrap();
        for (fd, d) in inherited {
            p.fds.insert(fd, d);
        }
        Ok(pid)
    }

    fn new_proc(&mut self, ppid: u32, pgid: u32, argv: Vec<String>, env: Vec<String>, cwd: &str) -> Result<u32, u16> {
        let cwd_ino = self.lookup(self.root, cwd, true)?;
        self.entries(cwd_ino)?;
        let pid = self.next_pid;
        self.next_pid += 1;
        let mut fds = BTreeMap::new();
        // The one preopened directory, "/": WASI's C library maps every path, absolute or relative
        // to its working directory, onto it. (A second preopen named "." would take over every
        // path, since the library strips both names to the same empty prefix.)
        for (fd, ino, name) in [(3, self.root, "/")] {
            let d = self.new_desc(Obj::Dir(ino), true, false, 0);
            d.borrow_mut().preopen = Some(name.to_string());
            fds.insert(fd, d);
        }
        self.procs.insert(pid, Proc { pid, ppid, pgid: if pgid == 0 { pid } else { pgid }, cwd: cwd_ino, fds, argv, env, exit: None, killed: None, ignore_interrupt: false });
        Ok(pid)
    }

    pub fn proc(&self, pid: u32) -> Option<&Proc> {
        self.procs.get(&pid)
    }

    /// The program ended: close its descriptors (pipe readers see end of file).
    pub fn exit(&mut self, pid: u32, code: i32) {
        let fds = match self.procs.get_mut(&pid) {
            Some(p) => {
                p.exit = Some(code);
                std::mem::take(&mut p.fds)
            }
            None => return,
        };
        for (_, d) in fds {
            self.release(d);
        }
        for t in self.ttys.values_mut() {
            if t.fg == pid {
                t.raw = false;
            }
        }
    }

    /// The exit status of a child, removing it once collected; AGAIN while it runs.
    pub fn wait(&mut self, parent: u32, child: u32) -> Result<i32, u16> {
        let c = self.procs.get(&child).ok_or(errno::CHILD)?;
        if c.ppid != parent {
            return Err(errno::CHILD);
        }
        match c.exit {
            Some(code) => {
                self.procs.remove(&child);
                Ok(code)
            }
            None => Err(errno::AGAIN),
        }
    }

    /// Ask every process in a group to stop (the host ends each at its next call).
    pub fn kill_group(&mut self, pgid: u32, status: i32) {
        for p in self.procs.values_mut() {
            if p.pgid == pgid && p.exit.is_none() && !(status == 130 && p.ignore_interrupt) {
                p.killed = Some(status);
            }
        }
    }

    pub fn set_ignore_interrupt(&mut self, pid: u32, on: bool) {
        if let Some(p) = self.procs.get_mut(&pid) {
            p.ignore_interrupt = on;
        }
    }

    pub fn set_foreground(&mut self, tty: u32, pgid: u32) {
        if let Some(t) = self.ttys.get_mut(&tty) {
            t.fg = pgid;
        }
    }

    // ---- terminals ----

    pub fn tty(&self, id: u32) -> Option<&Tty> {
        self.ttys.get(&id)
    }

    /// Keystrokes from the terminal: cooked mode edits a line (echoing it) until Enter.
    pub fn tty_input(&mut self, id: u32, bytes: &[u8]) {
        let mut signal = None;
        let t = match self.ttys.get_mut(&id) {
            Some(t) => t,
            None => return,
        };
        if t.raw {
            t.input.extend(bytes);
            return;
        }
        for &b in bytes {
            // Skip escape sequences (arrow keys and the like) in cooked mode.
            if t.escape == 1 {
                t.escape = if b == b'[' || b == b'O' { 2 } else { 0 };
                continue;
            }
            if t.escape == 2 {
                if (0x40..=0x7e).contains(&b) {
                    t.escape = 0;
                }
                continue;
            }
            match b {
                0x1b => t.escape = 1,
                b'\r' | b'\n' => {
                    t.line.push(b'\n');
                    t.input.extend(t.line.drain(..));
                    t.output.extend_from_slice(b"\r\n");
                }
                0x7f | 0x08 => {
                    // Erase one character (all bytes of a UTF-8 sequence).
                    while let Some(c) = t.line.pop() {
                        if c & 0xC0 != 0x80 {
                            t.output.extend_from_slice(b"\x08 \x08");
                            break;
                        }
                    }
                }
                0x03 => {
                    t.line.clear();
                    t.output.extend_from_slice(b"^C\r\n");
                    signal = Some(t.fg);
                }
                0x04 => {
                    if t.line.is_empty() {
                        t.eof += 1;
                    } else {
                        t.input.extend(t.line.drain(..));
                    }
                }
                0x15 => {
                    for _ in 0..String::from_utf8_lossy(&t.line).chars().count() {
                        t.output.extend_from_slice(b"\x08 \x08");
                    }
                    t.line.clear();
                }
                0x17 => {
                    while t.line.last() == Some(&b' ') {
                        t.line.pop();
                        t.output.extend_from_slice(b"\x08 \x08");
                    }
                    while t.line.last().is_some_and(|&c| c != b' ') {
                        let c = t.line.pop().unwrap();
                        if c & 0xC0 != 0x80 {
                            t.output.extend_from_slice(b"\x08 \x08");
                        }
                    }
                }
                b'\t' => {
                    t.line.push(b);
                    t.output.push(b);
                }
                0..=0x1f => {}
                _ => {
                    t.line.push(b);
                    t.output.push(b);
                }
            }
        }
        if let Some(fg) = signal {
            self.kill_group(fg, 130);
        }
    }

    /// What programs wrote to the terminal since the last call.
    pub fn tty_take_output(&mut self, id: u32) -> Vec<u8> {
        self.ttys.get_mut(&id).map(|t| std::mem::take(&mut t.output)).unwrap_or_default()
    }

    pub fn tty_resize(&mut self, id: u32, cols: u16, rows: u16) {
        if let Some(t) = self.ttys.get_mut(&id) {
            t.cols = cols;
            t.rows = rows;
        }
    }

    pub fn tty_set_raw(&mut self, id: u32, raw: bool) {
        if let Some(t) = self.ttys.get_mut(&id) {
            t.raw = raw;
            if raw {
                let line = std::mem::take(&mut t.line);
                t.input.extend(line);
            }
        }
    }

    /// The terminal a process's standard input is, if any.
    pub fn tty_of(&self, pid: u32) -> Option<u32> {
        match self.procs.get(&pid)?.fds.get(&0)?.borrow().obj {
            Obj::Tty(t) => Some(t),
            _ => None,
        }
    }

    // ---- WASI calls ----

    fn fail<T>(&mut self, e: u16) -> Option<T> {
        self.errno = e;
        None
    }

    fn desc(&self, pid: u32, fd: u32) -> Result<DescRef, u16> {
        self.procs.get(&pid).ok_or(errno::SRCH)?.fds.get(&fd).cloned().ok_or(errno::BADF)
    }

    fn base(&self, pid: u32, dirfd: u32) -> Result<Ino, u16> {
        match self.desc(pid, dirfd)?.borrow().obj {
            Obj::Dir(i) => Ok(i),
            _ => Err(errno::NOTDIR),
        }
    }

    fn rng(&mut self) -> u8 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 24) as u8
    }

    pub fn seed(&mut self, s: u64) {
        self.rng = s | 1;
    }

    /// Write bytes; returns how many were taken (a full pipe takes some, or blocks with AGAIN).
    pub fn fd_write(&mut self, pid: u32, fd: u32, data: &[u8]) -> Option<u32> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let mut d = d.borrow_mut();
        if !d.write {
            return self.fail(errno::BADF);
        }
        match d.obj.clone() {
            Obj::File(ino) => {
                let now = self.now;
                let append = d.flags & FDFLAG_APPEND != 0;
                let Some(Inode { node: Node::File(buf), mtime, .. }) = self.inodes.get_mut(&ino) else { return self.fail(errno::IO) };
                let at = if append { buf.len() } else { d.offset as usize };
                if buf.len() < at + data.len() {
                    buf.resize(at + data.len(), 0);
                }
                buf[at..at + data.len()].copy_from_slice(data);
                *mtime = now;
                d.offset = (at + data.len()) as u64;
                Some(data.len() as u32)
            }
            Obj::PipeWrite(p) => {
                let pipe = self.pipes.get_mut(&p).unwrap();
                if pipe.readers == 0 {
                    return self.fail(errno::PIPE);
                }
                let room = PIPE_CAPACITY.saturating_sub(pipe.buf.len());
                if room == 0 {
                    return self.fail(errno::AGAIN);
                }
                let n = room.min(data.len());
                pipe.buf.extend(&data[..n]);
                Some(n as u32)
            }
            Obj::Tty(t) => {
                if let Some(t) = self.ttys.get_mut(&t) {
                    for &b in data {
                        if b == b'\n' {
                            t.output.push(b'\r');
                        }
                        t.output.push(b);
                    }
                }
                Some(data.len() as u32)
            }
            Obj::Null | Obj::Zero | Obj::Random => Some(data.len() as u32),
            Obj::Dir(_) => self.fail(errno::ISDIR),
            Obj::PipeRead(_) => self.fail(errno::BADF),
        }
    }

    /// Read up to `max` bytes (empty at end of file); AGAIN when the caller must wait.
    pub fn fd_read(&mut self, pid: u32, fd: u32, max: u32) -> Option<Vec<u8>> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let mut d = d.borrow_mut();
        if !d.read {
            return self.fail(errno::BADF);
        }
        let max = max as usize;
        match d.obj.clone() {
            Obj::File(ino) => {
                let Some(Inode { node: Node::File(buf), .. }) = self.inodes.get(&ino) else { return self.fail(errno::IO) };
                let at = (d.offset as usize).min(buf.len());
                let out = buf[at..(at + max).min(buf.len())].to_vec();
                d.offset += out.len() as u64;
                Some(out)
            }
            Obj::PipeRead(p) => {
                let pipe = self.pipes.get_mut(&p).unwrap();
                if pipe.buf.is_empty() {
                    if pipe.writers == 0 {
                        return Some(Vec::new());
                    }
                    return self.fail(errno::AGAIN);
                }
                let n = max.min(pipe.buf.len());
                Some(pipe.buf.drain(..n).collect())
            }
            Obj::Tty(t) => {
                let Some(t) = self.ttys.get_mut(&t) else { return self.fail(errno::IO) };
                if !t.input.is_empty() {
                    // Cooked mode hands over one line at a time.
                    let mut n = max.min(t.input.len());
                    if !t.raw {
                        if let Some(nl) = t.input.iter().position(|&c| c == b'\n') {
                            n = n.min(nl + 1);
                        }
                    }
                    return Some(t.input.drain(..n).collect());
                }
                if t.eof > 0 {
                    t.eof -= 1;
                    return Some(Vec::new());
                }
                self.fail(errno::AGAIN)
            }
            Obj::Null => Some(Vec::new()),
            Obj::Zero => Some(vec![0; max]),
            Obj::Random => Some((0..max).map(|_| self.rng()).collect()),
            Obj::Dir(_) => self.fail(errno::ISDIR),
            Obj::PipeWrite(_) => self.fail(errno::BADF),
        }
    }

    /// Positional read and write for regular files (descriptor offset unchanged).
    pub fn fd_pread(&mut self, pid: u32, fd: u32, max: u32, offset: u64) -> Option<Vec<u8>> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let saved = d.borrow().offset;
        if !matches!(d.borrow().obj, Obj::File(_)) {
            return self.fail(errno::SPIPE);
        }
        d.borrow_mut().offset = offset;
        let out = self.fd_read(pid, fd, max);
        d.borrow_mut().offset = saved;
        out
    }

    pub fn fd_pwrite(&mut self, pid: u32, fd: u32, data: &[u8], offset: u64) -> Option<u32> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let (saved, flags) = (d.borrow().offset, d.borrow().flags);
        if !matches!(d.borrow().obj, Obj::File(_)) {
            return self.fail(errno::SPIPE);
        }
        d.borrow_mut().offset = offset;
        d.borrow_mut().flags = flags & !FDFLAG_APPEND;
        let out = self.fd_write(pid, fd, data);
        d.borrow_mut().offset = saved;
        d.borrow_mut().flags = flags;
        out
    }

    pub fn fd_seek(&mut self, pid: u32, fd: u32, offset: i64, whence: u8) -> Option<u64> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let mut d = d.borrow_mut();
        let size = match d.obj {
            Obj::File(ino) => self.inodes.get(&ino).map(|n| n.size()).unwrap_or(0),
            Obj::Dir(_) => 0,
            _ => return self.fail(errno::SPIPE),
        };
        let base = match whence {
            0 => 0i64,
            1 => d.offset as i64,
            2 => size as i64,
            _ => return self.fail(errno::INVAL),
        };
        let pos = base + offset;
        if pos < 0 {
            return self.fail(errno::INVAL);
        }
        d.offset = pos as u64;
        Some(d.offset)
    }

    pub fn fd_close(&mut self, pid: u32, fd: u32) -> bool {
        self.errno = 0;
        let removed = self.procs.get_mut(&pid).and_then(|p| p.fds.remove(&fd));
        match removed {
            Some(d) => {
                self.release(d);
                true
            }
            None => {
                self.errno = errno::BADF;
                false
            }
        }
    }

    /// [file type, flags] of a descriptor.
    pub fn fd_fdstat(&mut self, pid: u32, fd: u32) -> Option<(u8, u16)> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let d = d.borrow();
        let ft = match d.obj {
            Obj::File(_) => filetype::REGULAR_FILE,
            Obj::Dir(_) => filetype::DIRECTORY,
            Obj::Tty(_) | Obj::Null | Obj::Zero | Obj::Random => filetype::CHARACTER_DEVICE,
            Obj::PipeRead(_) | Obj::PipeWrite(_) => filetype::UNKNOWN,
        };
        Some((ft, d.flags))
    }

    pub fn fd_set_flags(&mut self, pid: u32, fd: u32, flags: u16) -> bool {
        match self.desc(pid, fd) {
            Ok(d) => {
                d.borrow_mut().flags = flags;
                true
            }
            Err(e) => {
                self.errno = e;
                false
            }
        }
    }

    /// [dev, ino, filetype, nlink, size, atime, mtime, ctime].
    pub fn stat_ino(&self, ino: Ino) -> [u64; 8] {
        let n = &self.inodes[&ino];
        [1, ino, n.filetype() as u64, 1, n.size(), n.mtime, n.mtime, n.ctime]
    }

    pub fn fd_filestat(&mut self, pid: u32, fd: u32) -> Option<[u64; 8]> {
        self.errno = 0;
        let d = match self.desc(pid, fd) {
            Ok(d) => d,
            Err(e) => return self.fail(e),
        };
        let obj = d.borrow().obj.clone();
        match obj {
            Obj::File(i) | Obj::Dir(i) if self.inodes.contains_key(&i) => Some(self.stat_ino(i)),
            Obj::File(_) | Obj::Dir(_) => self.fail(errno::NOENT),
            Obj::Tty(_) | Obj::Null | Obj::Zero | Obj::Random => Some([1, 0, filetype::CHARACTER_DEVICE as u64, 1, 0, self.now, self.now, self.now]),
            _ => Some([1, 0, filetype::UNKNOWN as u64, 1, 0, self.now, self.now, self.now]),
        }
    }

    pub fn fd_set_size(&mut self, pid: u32, fd: u32, size: u64) -> bool {
        self.errno = 0;
        let ino = match self.desc(pid, fd).map(|d| d.borrow().obj.clone()) {
            Ok(Obj::File(i)) => i,
            Ok(_) => {
                self.errno = errno::INVAL;
                return false;
            }
            Err(e) => {
                self.errno = e;
                return false;
            }
        };
        if let Some(Inode { node: Node::File(d), mtime, .. }) = self.inodes.get_mut(&ino) {
            d.resize(size as usize, 0);
            *mtime = self.now;
        }
        true
    }

    /// The name of a preopened directory (none: the program has seen them all).
    pub fn fd_prestat(&mut self, pid: u32, fd: u32) -> Option<String> {
        self.errno = 0;
        match self.desc(pid, fd) {
            Ok(d) => match d.borrow().preopen.clone() {
                Some(n) => Some(n),
                None => self.fail(errno::BADF),
            },
            Err(e) => self.fail(e),
        }
    }

    /// Directory entries from position `cookie`, including "." and "..".
    pub fn fd_readdir(&mut self, pid: u32, fd: u32, cookie: u64) -> Option<Vec<DirEntry>> {
        self.errno = 0;
        let ino = match self.desc(pid, fd).map(|d| d.borrow().obj.clone()) {
            Ok(Obj::Dir(i)) => i,
            Ok(_) => return self.fail(errno::NOTDIR),
            Err(e) => return self.fail(e),
        };
        let parent = self.inodes.get(&ino).map(|n| n.parent).unwrap_or(ino);
        let mut all = vec![
            DirEntry { name: ".".into(), ino, kind: filetype::DIRECTORY },
            DirEntry { name: "..".into(), ino: parent, kind: filetype::DIRECTORY },
        ];
        match self.list(ino) {
            Ok(l) => all.extend(l),
            Err(e) => return self.fail(e),
        }
        Some(all.into_iter().skip(cookie as usize).collect())
    }

    /// Move descriptor `from` to `to` (closing what `to` was).
    pub fn fd_renumber(&mut self, pid: u32, from: u32, to: u32) -> bool {
        self.errno = 0;
        let Some(p) = self.procs.get_mut(&pid) else {
            self.errno = errno::SRCH;
            return false;
        };
        let Some(d) = p.fds.remove(&from) else {
            self.errno = errno::BADF;
            return false;
        };
        let old = p.fds.insert(to, d);
        if let Some(old) = old {
            self.release(old);
        }
        true
    }

    /// Duplicate `from` as `to` (both refer to one description); returns `to`.
    pub fn fd_dup2(&mut self, pid: u32, from: u32, to: u32) -> bool {
        self.errno = 0;
        let d = match self.desc(pid, from) {
            Ok(d) => d,
            Err(e) => {
                self.errno = e;
                return false;
            }
        };
        if from == to {
            return true;
        }
        let old = self.procs.get_mut(&pid).unwrap().fds.insert(to, d);
        if let Some(old) = old {
            self.release(old);
        }
        true
    }

    fn free_fd(&self, pid: u32) -> u32 {
        let fds = &self.procs[&pid].fds;
        (4..).find(|n| !fds.contains_key(n)).unwrap()
    }

    /// A pipe: (read end, write end).
    pub fn pipe(&mut self, pid: u32) -> Option<(u32, u32)> {
        self.errno = 0;
        if !self.procs.contains_key(&pid) {
            return self.fail(errno::SRCH);
        }
        let id = self.next_pipe;
        self.next_pipe += 1;
        self.pipes.insert(id, Pipe::default());
        let r = self.new_desc(Obj::PipeRead(id), true, false, 0);
        let w = self.new_desc(Obj::PipeWrite(id), false, true, 0);
        let rfd = self.free_fd(pid);
        self.procs.get_mut(&pid).unwrap().fds.insert(rfd, r);
        let wfd = self.free_fd(pid);
        self.procs.get_mut(&pid).unwrap().fds.insert(wfd, w);
        Some((rfd, wfd))
    }

    /// Whether a read (or write) on `fd` would return without waiting.
    pub fn fd_ready(&self, pid: u32, fd: u32, write: bool) -> bool {
        let Ok(d) = self.desc(pid, fd) else { return true };
        let obj = d.borrow().obj.clone();
        match (obj, write) {
            (Obj::PipeRead(p), false) => self.pipes.get(&p).is_none_or(|x| !x.buf.is_empty() || x.writers == 0),
            (Obj::PipeWrite(p), true) => self.pipes.get(&p).is_none_or(|x| x.buf.len() < PIPE_CAPACITY || x.readers == 0),
            (Obj::Tty(t), false) => self.ttys.get(&t).is_none_or(|x| !x.input.is_empty() || x.eof > 0),
            _ => true,
        }
    }

    pub fn path_open(&mut self, pid: u32, dirfd: u32, path: &str, follow: bool, oflags: u16, rights: u64, fdflags: u16) -> Option<u32> {
        self.errno = 0;
        let base = match self.base(pid, dirfd) {
            Ok(b) => b,
            Err(e) => return self.fail(e),
        };
        let found = self.lookup(base, path, follow || oflags & OFLAG_CREAT == 0);
        let ino = match found {
            Ok(ino) => {
                if oflags & OFLAG_CREAT != 0 && oflags & OFLAG_EXCL != 0 {
                    return self.fail(errno::EXIST);
                }
                ino
            }
            Err(errno::NOENT) if oflags & OFLAG_CREAT != 0 => {
                let (dir, name) = match self.parent_of(base, path) {
                    Ok(x) => x,
                    Err(e) => return self.fail(e),
                };
                self.insert(dir, &name, Node::File(Vec::new()), 0o644)
            }
            Err(e) => return self.fail(e),
        };
        let read = rights & RIGHT_FD_READ != 0 || rights == 0;
        let write = rights & RIGHT_FD_WRITE != 0;
        let obj = match &self.inodes[&ino].node {
            Node::Dir(_) => {
                if write && oflags & OFLAG_DIRECTORY == 0 && rights & RIGHT_FD_WRITE != 0 && rights & RIGHT_FD_READ == 0 {
                    return self.fail(errno::ISDIR);
                }
                Obj::Dir(ino)
            }
            _ if oflags & OFLAG_DIRECTORY != 0 => return self.fail(errno::NOTDIR),
            Node::File(_) => Obj::File(ino),
            Node::Dev(Dev::Null) => Obj::Null,
            Node::Dev(Dev::Zero) => Obj::Zero,
            Node::Dev(Dev::Random) => Obj::Random,
            Node::Dev(Dev::Tty(t)) => Obj::Tty(*t),
            Node::Symlink(_) => return self.fail(errno::LOOP),
        };
        if oflags & OFLAG_TRUNC != 0 {
            if let Some(Inode { node: Node::File(d), mtime, .. }) = self.inodes.get_mut(&ino) {
                d.clear();
                *mtime = self.now;
            }
        }
        let is_dir = matches!(obj, Obj::Dir(_));
        let d = self.new_desc(obj, read, write && !is_dir, fdflags);
        let fd = self.free_fd(pid);
        self.procs.get_mut(&pid).unwrap().fds.insert(fd, d);
        Some(fd)
    }

    pub fn path_stat(&mut self, pid: u32, dirfd: u32, path: &str, follow: bool) -> Option<[u64; 8]> {
        self.errno = 0;
        match self.base(pid, dirfd).and_then(|b| self.lookup(b, path, follow)) {
            Ok(ino) => Some(self.stat_ino(ino)),
            Err(e) => self.fail(e),
        }
    }

    pub fn path_mkdir(&mut self, pid: u32, dirfd: u32, path: &str) -> bool {
        self.errno = match self.base(pid, dirfd).and_then(|b| self.mkdir_at(b, path)) {
            Ok(()) => 0,
            Err(e) => e,
        };
        self.errno == 0
    }

    pub fn path_remove(&mut self, pid: u32, dirfd: u32, path: &str, dir: bool) -> bool {
        self.errno = match self.base(pid, dirfd).and_then(|b| self.remove_at(b, path, dir)) {
            Ok(()) => 0,
            Err(e) => e,
        };
        self.errno == 0
    }

    pub fn path_rename(&mut self, pid: u32, fd1: u32, from: &str, fd2: u32, to: &str) -> bool {
        self.errno = match self.base(pid, fd1).and_then(|a| self.base(pid, fd2).map(|b| (a, b))).and_then(|(a, b)| self.rename_at(a, from, b, to)) {
            Ok(()) => 0,
            Err(e) => e,
        };
        self.errno == 0
    }

    pub fn path_symlink(&mut self, pid: u32, target: &str, dirfd: u32, path: &str) -> bool {
        self.errno = match self.base(pid, dirfd).and_then(|b| self.symlink(target, b, path)) {
            Ok(()) => 0,
            Err(e) => e,
        };
        self.errno == 0
    }

    pub fn path_readlink(&mut self, pid: u32, dirfd: u32, path: &str) -> Option<String> {
        self.errno = 0;
        match self.base(pid, dirfd).and_then(|b| self.lookup(b, path, false)) {
            Ok(ino) => match &self.inodes[&ino].node {
                Node::Symlink(t) => Some(t.clone()),
                _ => self.fail(errno::INVAL),
            },
            Err(e) => self.fail(e),
        }
    }

    pub fn path_set_times(&mut self, pid: u32, dirfd: u32, path: &str, mtime: u64) -> bool {
        self.errno = 0;
        match self.base(pid, dirfd).and_then(|b| self.lookup(b, path, true)) {
            Ok(ino) => {
                if let Some(n) = self.inodes.get_mut(&ino) {
                    n.mtime = mtime;
                }
                true
            }
            Err(e) => {
                self.errno = e;
                false
            }
        }
    }

    /// Resolve a command name through PATH (or a path) to an executable file.
    pub fn which(&self, pid: u32, name: &str, path_var: &str) -> Option<String> {
        let cwd = self.procs.get(&pid).map(|p| p.cwd).unwrap_or(self.root);
        let is_exe = |ino: Ino| matches!(&self.inodes[&ino].node, Node::File(d) if d.starts_with(b"\0asm") || d.starts_with(b"#!"));
        if name.contains('/') {
            let ino = self.lookup(cwd, name, true).ok()?;
            return is_exe(ino).then(|| if name.starts_with('/') { name.to_string() } else { format!("{}/{}", self.path_of(cwd).trim_end_matches('/'), name.trim_start_matches("./")) });
        }
        path_var.split(':').filter(|d| !d.is_empty()).find_map(|dir| {
            let p = format!("{}/{}", dir.trim_end_matches('/'), name);
            self.lookup(self.root, &p, true).ok().filter(|&i| is_exe(i)).map(|_| p)
        })
    }

    // ---- persistence ----

    /// Every file, directory and symlink under `prefix`, for saving to browser storage:
    /// records of (kind, mode, mtime, path, data).
    pub fn snapshot(&self, prefix: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let Ok(start) = self.lookup(self.root, prefix, false) else { return out };
        let mut stack = vec![(start, prefix.trim_end_matches('/').to_string())];
        while let Some((ino, path)) = stack.pop() {
            let n = &self.inodes[&ino];
            let (kind, data): (u8, &[u8]) = match &n.node {
                Node::Dir(e) => {
                    for (name, &child) in e.iter().rev() {
                        stack.push((child, format!("{path}/{name}")));
                    }
                    (b'd', &[])
                }
                Node::File(d) => (b'f', d),
                Node::Symlink(t) => (b'l', t.as_bytes()),
                Node::Dev(_) => continue,
            };
            out.push(kind);
            out.extend(n.mode.to_le_bytes());
            out.extend(n.mtime.to_le_bytes());
            out.extend((path.len() as u32).to_le_bytes());
            out.extend(path.as_bytes());
            out.extend((data.len() as u64).to_le_bytes());
            out.extend(data);
        }
        out
    }

    /// Recreate what `snapshot` saved (over what is there).
    pub fn restore(&mut self, bytes: &[u8]) -> bool {
        let mut at = 0usize;
        let take = |at: &mut usize, n: usize| -> Option<&[u8]> {
            let s = bytes.get(*at..*at + n)?;
            *at += n;
            Some(s)
        };
        while at < bytes.len() {
            let Some(rec) = (|| {
                let kind = take(&mut at, 1)?[0];
                let mode = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?);
                let mtime = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?);
                let plen = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?) as usize;
                let path = String::from_utf8(take(&mut at, plen)?.to_vec()).ok()?;
                let dlen = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?) as usize;
                let data = take(&mut at, dlen)?.to_vec();
                Some((kind, mode, mtime, path, data))
            })() else {
                return false;
            };
            let (kind, mode, mtime, path, data) = rec;
            match kind {
                b'd' => {
                    let ino = self.mkdir_p(&path);
                    if let Some(n) = self.inodes.get_mut(&ino) {
                        n.mode = mode;
                    }
                }
                b'f' => {
                    self.write_file(&path, &data);
                    self.set_mode(&path, mode);
                }
                b'l' => {
                    let _ = self.remove_at(self.root, &path, false);
                    let _ = self.symlink(&String::from_utf8_lossy(&data), self.root, &path);
                }
                _ => return false,
            }
            if let Ok(ino) = self.lookup(self.root, &path, false) {
                if let Some(n) = self.inodes.get_mut(&ino) {
                    n.mtime = mtime;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(k: &mut Kernel) -> u32 {
        let t = k.create_tty();
        k.spawn_on_tty(vec!["sh".into()], vec![], "/home/user", t).unwrap()
    }

    #[test]
    fn paths_symlinks_and_files() {
        let mut k = Kernel::new();
        let pid = shell(&mut k);
        assert!(k.path_mkdir(pid, 3, "home/user/src"));
        let fd = k.path_open(pid, 3, "home/user/src/main.rs", true, OFLAG_CREAT, RIGHT_FD_WRITE, 0).unwrap();
        assert_eq!(k.fd_write(pid, fd, b"fn main() {}\n"), Some(13));
        assert!(k.path_symlink(pid, "src/main.rs", 3, "home/user/link.rs"));
        assert_eq!(k.read_file("/home/user/link.rs").unwrap(), b"fn main() {}\n");
        assert_eq!(k.path_readlink(pid, 3, "home/user/link.rs").as_deref(), Some("src/main.rs"));
        // Relative to "/" (fd 3), with "..".
        assert!(k.path_stat(pid, 3, "home/user/src/../src/main.rs", true).is_some());
        assert_eq!(k.path_stat(pid, 3, "home/user/missing", true), None);
        assert_eq!(k.errno, errno::NOENT);
        assert!(!k.path_remove(pid, 3, "home/user/src", true));
        assert_eq!(k.errno, errno::NOTEMPTY);
        assert!(k.path_rename(pid, 3, "home/user/src", 3, "tmp/moved"));
        assert_eq!(k.read_file("/tmp/moved/main.rs").unwrap(), b"fn main() {}\n");
        assert_eq!(k.path_of(k.lookup(k.root, "/tmp/moved", true).unwrap()), "/tmp/moved");
        assert!(!k.path_rename(pid, 3, "tmp", 3, "tmp/moved/inside"), "a directory cannot move into itself");
        let fd = k.path_open(pid, 3, "tmp/moved/main.rs", true, 0, RIGHT_FD_READ, 0).unwrap();
        assert_eq!(k.fd_seek(pid, fd, 3, 0), Some(3));
        assert_eq!(k.fd_read(pid, fd, 4).unwrap(), b"main");
        assert_eq!(k.fd_pread(pid, fd, 2, 0).unwrap(), b"fn");
    }

    #[test]
    fn pipes_end_when_every_writer_closes() {
        let mut k = Kernel::new();
        let sh = shell(&mut k);
        let (r, w) = k.pipe(sh).unwrap();
        // Two children share the write end; the reader sees EOF only after both close it.
        let a = k.spawn(sh, vec!["a".into()], vec![], "", &[(1, w)], 0).unwrap();
        let b = k.spawn(sh, vec!["b".into()], vec![], "", &[(0, r)], 0).unwrap();
        k.fd_close(sh, w);
        k.fd_close(sh, r);
        assert_eq!(k.fd_read(b, 0, 10), None);
        assert_eq!(k.errno, errno::AGAIN);
        assert!(!k.fd_ready(b, 0, false));
        assert_eq!(k.fd_write(a, 1, b"hello"), Some(5));
        assert_eq!(k.fd_read(b, 0, 10).unwrap(), b"hello");
        k.exit(a, 0);
        assert_eq!(k.fd_read(b, 0, 10).unwrap(), b"", "end of file once the writer exits");
        k.exit(b, 0);
        assert_eq!(k.wait(sh, a), Ok(0));
        assert_eq!(k.wait(sh, a), Err(errno::CHILD));
        assert!(k.pipes.is_empty());
    }

    #[test]
    fn terminal_line_editing_and_interrupt() {
        let mut k = Kernel::new();
        let sh = shell(&mut k);
        k.tty_input(0, b"lsx\x7f -l\x1b[A\r");
        assert_eq!(k.fd_read(sh, 0, 100).unwrap(), b"ls -l\n");
        assert_eq!(k.tty_take_output(0), b"lsx\x08 \x08 -l\r\n");
        k.tty_input(0, b"\x04");
        assert_eq!(k.fd_read(sh, 0, 100).unwrap(), b"", "Ctrl+D on an empty line is end of file");
        let child = k.spawn(sh, vec!["sleep".into()], vec![], "", &[(0, 0), (1, 1)], 0).unwrap();
        k.set_foreground(0, child);
        k.tty_input(0, b"abc\x03");
        assert_eq!(k.proc(child).unwrap().killed, Some(130));
        assert_eq!(k.proc(sh).unwrap().killed, None);
        // Ctrl+C at the shell's own prompt leaves the shell running.
        k.set_ignore_interrupt(sh, true);
        k.set_foreground(0, k.proc(sh).unwrap().pgid);
        k.tty_input(0, b"\x03");
        assert_eq!(k.proc(sh).unwrap().killed, None);
        assert_eq!(k.fd_write(child, 1, b"a\nb"), Some(3));
        assert!(k.tty_take_output(0).ends_with(b"a\r\nb"));
    }

    #[test]
    fn readdir_and_snapshot_round_trip() {
        let mut k = Kernel::new();
        let pid = shell(&mut k);
        k.write_file("/home/user/a.txt", b"one");
        k.mkdir_p("/home/user/d/e");
        k.symlink("a.txt", k.root, "/home/user/l").unwrap();
        let fd = k.path_open(pid, 3, "home/user", true, OFLAG_DIRECTORY, RIGHT_FD_READ, 0).unwrap();
        let names: Vec<String> = k.fd_readdir(pid, fd, 0).unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names, [".", "..", "a.txt", "d", "l"]);
        assert_eq!(k.fd_readdir(pid, fd, 3).unwrap().len(), 2);
        let snap = k.snapshot("/home/user");
        let mut fresh = Kernel::new();
        assert!(fresh.restore(&snap));
        assert_eq!(fresh.read_file("/home/user/l").unwrap(), b"one");
        assert!(fresh.lookup(fresh.root, "/home/user/d/e", true).is_ok());
    }
}
