//! The Evolve VM shell: a POSIX-style command language for WASI programs.
//!
//! Quoting ('...', "...", \), variables ($X, ${X}, $?, $$, $0-$9, $@, $#), ~, wildcards (* ? [..]),
//! pipelines, redirections (<, >, >>, 2>, 2>>, &>, 2>&1, >&2), `;`, `&&`, `||`, `&` and comments.
//! Other programs start through the kernel (`evolve` imports); the shell waits for them in the
//! terminal's foreground, so Ctrl+C ends them and not the shell.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::os::fd::{FromRawFd, IntoRawFd};

#[link(wasm_import_module = "evolve")]
extern "C" {
    fn proc_spawn(argv: *const u8, argv_len: usize, env: *const u8, env_len: usize, cwd: *const u8, cwd_len: usize, fdmap: *const u32, fdmap_len: usize, pgid: u32, pid_out: *mut u32) -> u32;
    fn proc_wait(pid: u32, status: *mut i32) -> u32;
    fn fd_pipe(r: *mut u32, w: *mut u32) -> u32;
    fn tty_set_foreground(pgid: u32) -> u32;
    fn proc_ignore_interrupt(on: u32) -> u32;
}

const ENOENT: u32 = 44;

#[derive(Debug, Clone)]
struct Seg {
    text: String,
    var: bool,
    quoted: bool,
}

type Word = Vec<Seg>;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(Vec<(String, bool, bool)>),
    Op(&'static str),
}

#[derive(Debug, Clone)]
enum RedirKind {
    In,
    Out,
    Append,
    Dup(u32),
}

#[derive(Debug, Clone)]
struct Redir {
    fd: u32,
    kind: RedirKind,
    target: Word,
}

#[derive(Debug, Clone, Default)]
struct Cmd {
    assigns: Vec<(String, Word)>,
    args: Vec<Word>,
    redirs: Vec<Redir>,
}

/// Split a line into words and operators. Each word keeps its pieces with (text, is a
/// variable reference, was quoted).
fn lex(line: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = line.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    let mut word: Vec<(String, bool, bool)> = Vec::new();
    let mut lit = String::new();
    let mut in_word = false;
    macro_rules! flush_lit {
        ($q:expr) => {
            if !lit.is_empty() {
                word.push((std::mem::take(&mut lit), false, $q));
            }
        };
    }
    macro_rules! end_word {
        () => {
            flush_lit!(false);
            if in_word {
                toks.push(Tok::Word(std::mem::take(&mut word)));
                in_word = false;
            }
        };
    }
    // A $NAME or ${NAME} reference starting at c[i] == '$'; returns the end index.
    let var = |i: usize| -> Option<(String, usize)> {
        let n = c.get(i + 1)?;
        if *n == '{' {
            let end = (i + 2..c.len()).find(|&j| c[j] == '}')?;
            return Some((c[i + 2..end].iter().collect(), end + 1));
        }
        if n.is_ascii_digit() || matches!(n, '?' | '$' | '@' | '#' | '*') {
            return Some((n.to_string(), i + 2));
        }
        let mut j = i + 1;
        while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '_') {
            j += 1;
        }
        (j > i + 1).then(|| (c[i + 1..j].iter().collect(), j))
    };
    while i < c.len() {
        let ch = c[i];
        match ch {
            ' ' | '\t' => {
                end_word!();
                i += 1;
            }
            '\n' => {
                // An unquoted newline ends a command, as `;` does, unless an operator continues it.
                end_word!();
                if !matches!(toks.last(), None | Some(Tok::Op("|" | "&&" | "||" | ";" | "&"))) {
                    toks.push(Tok::Op(";"));
                }
                i += 1;
            }
            // A comment runs to the end of its line.
            '#' if !in_word => {
                while i < c.len() && c[i] != '\n' {
                    i += 1;
                }
            }
            '\'' => {
                in_word = true;
                flush_lit!(false);
                let end = (i + 1..c.len()).find(|&j| c[j] == '\'').ok_or("unterminated quote")?;
                word.push((c[i + 1..end].iter().collect(), false, true));
                i = end + 1;
            }
            '"' => {
                in_word = true;
                flush_lit!(false);
                i += 1;
                let mut s = String::new();
                loop {
                    let ch = *c.get(i).ok_or("unterminated quote")?;
                    match ch {
                        '"' => break,
                        '\\' if matches!(c.get(i + 1), Some('"' | '\\' | '$' | '`')) => {
                            s.push(c[i + 1]);
                            i += 2;
                        }
                        '$' => match var(i) {
                            Some((name, end)) => {
                                if !s.is_empty() {
                                    word.push((std::mem::take(&mut s), false, true));
                                }
                                word.push((name, true, true));
                                i = end;
                            }
                            None => {
                                s.push('$');
                                i += 1;
                            }
                        },
                        _ => {
                            s.push(ch);
                            i += 1;
                        }
                    }
                }
                word.push((s, false, true));
                i += 1;
            }
            '\\' => {
                in_word = true;
                flush_lit!(false);
                if let Some(&n) = c.get(i + 1) {
                    word.push((n.to_string(), false, true));
                }
                i += 2;
            }
            '$' => match var(i) {
                Some((name, end)) => {
                    in_word = true;
                    flush_lit!(false);
                    word.push((name, true, false));
                    i = end;
                }
                None => {
                    in_word = true;
                    lit.push('$');
                    i += 1;
                }
            },
            '|' | '&' | ';' | '<' | '>' => {
                // `2>`, `2>>` and `2>&1` keep the descriptor number with the operator.
                if in_word && lit.chars().all(|d| d.is_ascii_digit()) && !lit.is_empty() && word.is_empty() && (ch == '>' || ch == '<') {
                    let fd = std::mem::take(&mut lit);
                    in_word = false;
                    let op = if c.get(i + 1) == Some(&'>') && ch == '>' { ">>" } else if ch == '>' { ">" } else { "<" };
                    i += op.len();
                    toks.push(Tok::Op(if fd == "2" {
                        if op == ">>" { "2>>" } else { "2>" }
                    } else {
                        op_static(op)
                    }));
                    if fd != "2" && fd != "1" && fd != "0" {
                        return Err(format!("redirecting descriptor {fd} is not supported"));
                    }
                    continue;
                }
                end_word!();
                let two: String = c[i..(i + 2).min(c.len())].iter().collect();
                let op = match two.as_str() {
                    "&&" | "||" | ">>" | "&>" => two.clone(),
                    _ => ch.to_string(),
                };
                i += op.len();
                toks.push(Tok::Op(op_static(&op)));
            }
            _ => {
                in_word = true;
                lit.push(ch);
                i += 1;
            }
        }
    }
    end_word!();
    Ok(toks)
}

fn op_static(op: &str) -> &'static str {
    match op {
        "&&" => "&&",
        "||" => "||",
        ">>" => ">>",
        "&>" => "&>",
        "|" => "|",
        "&" => "&",
        ";" => ";",
        "<" => "<",
        _ => ">",
    }
}

fn to_word(parts: Vec<(String, bool, bool)>) -> Word {
    parts.into_iter().map(|(text, var, quoted)| Seg { text, var, quoted }).collect()
}

/// Commands joined by `;`, `&&`, `||` and `&`: each pipeline with the operator after it.
fn parse(toks: Vec<Tok>) -> Result<Vec<(Vec<Cmd>, &'static str)>, String> {
    let mut list = Vec::new();
    let mut pipeline = Vec::new();
    let mut cmd = Cmd::default();
    let mut it = toks.into_iter().peekable();
    while let Some(t) = it.next() {
        match t {
            Tok::Word(parts) => {
                let w = to_word(parts);
                // NAME=value before the command name.
                if cmd.args.is_empty() {
                    if let Some(first) = w.first().filter(|s| !s.var && !s.quoted) {
                        if let Some(eq) = first.text.find('=') {
                            let name = &first.text[..eq];
                            if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !name.starts_with(|c: char| c.is_ascii_digit()) {
                                let mut value = vec![Seg { text: first.text[eq + 1..].to_string(), var: false, quoted: true }];
                                value.extend(w[1..].iter().cloned());
                                cmd.assigns.push((name.to_string(), value));
                                continue;
                            }
                        }
                    }
                }
                cmd.args.push(w);
            }
            Tok::Op(op @ ("<" | ">" | ">>" | "2>" | "2>>" | "&>")) => {
                // `2>&1` and `>&2` duplicate a descriptor instead of naming a file.
                if let (Some(Tok::Op("&")), true) = (it.peek(), op == "2>" || op == ">") {
                    it.next();
                    let target = match it.next() {
                        Some(Tok::Word(p)) => to_word(p),
                        _ => return Err("expected a descriptor after >&".into()),
                    };
                    let to: u32 = target.iter().map(|s| s.text.as_str()).collect::<String>().parse().map_err(|_| "expected a descriptor after >&")?;
                    cmd.redirs.push(Redir { fd: if op == "2>" { 2 } else { 1 }, kind: RedirKind::Dup(to), target: Vec::new() });
                    continue;
                }
                let target = match it.next() {
                    Some(Tok::Word(p)) => to_word(p),
                    _ => return Err(format!("expected a file after {op}")),
                };
                let (fd, kind) = match op {
                    "<" => (0, RedirKind::In),
                    ">" => (1, RedirKind::Out),
                    ">>" => (1, RedirKind::Append),
                    "2>" => (2, RedirKind::Out),
                    "2>>" => (2, RedirKind::Append),
                    _ => {
                        cmd.redirs.push(Redir { fd: 1, kind: RedirKind::Out, target });
                        cmd.redirs.push(Redir { fd: 2, kind: RedirKind::Dup(1), target: Vec::new() });
                        continue;
                    }
                };
                cmd.redirs.push(Redir { fd, kind, target });
            }
            Tok::Op("|") => {
                if cmd.args.is_empty() && cmd.assigns.is_empty() {
                    return Err("syntax error near |".into());
                }
                pipeline.push(std::mem::take(&mut cmd));
            }
            Tok::Op(op) => {
                if cmd.args.is_empty() && cmd.assigns.is_empty() && cmd.redirs.is_empty() {
                    if pipeline.is_empty() && op != ";" {
                        return Err(format!("syntax error near {op}"));
                    }
                } else {
                    pipeline.push(std::mem::take(&mut cmd));
                }
                if !pipeline.is_empty() {
                    list.push((std::mem::take(&mut pipeline), op));
                }
            }
        }
    }
    if !cmd.args.is_empty() || !cmd.assigns.is_empty() || !cmd.redirs.is_empty() {
        pipeline.push(cmd);
    }
    if !pipeline.is_empty() {
        list.push((pipeline, ";"));
    }
    Ok(list)
}

/// Match `name` against a wildcard pattern (* ? [abc] [a-z] [!x]).
fn glob_match(pat: &[char], name: &[char]) -> bool {
    match pat.first() {
        None => name.is_empty(),
        Some('*') => (0..=name.len()).any(|k| glob_match(&pat[1..], &name[k..])),
        Some('?') => !name.is_empty() && glob_match(&pat[1..], &name[1..]),
        Some('[') => {
            let Some(end) = pat.iter().skip(1).position(|&c| c == ']').map(|p| p + 1) else {
                return name.first() == Some(&'[') && glob_match(&pat[1..], &name[1..]);
            };
            let Some(&ch) = name.first() else { return false };
            let set = &pat[1..end];
            let (neg, set) = if matches!(set.first(), Some('!' | '^')) { (true, &set[1..]) } else { (false, set) };
            let mut hit = false;
            let mut k = 0;
            while k < set.len() {
                if k + 2 < set.len() && set[k + 1] == '-' {
                    hit |= set[k] <= ch && ch <= set[k + 2];
                    k += 3;
                } else {
                    hit |= set[k] == ch;
                    k += 1;
                }
            }
            hit != neg && glob_match(&pat[end + 1..], &name[1..])
        }
        Some(&c) => name.first() == Some(&c) && glob_match(&pat[1..], &name[1..]),
    }
}

/// Paths matching a pattern with wildcards in any component, sorted; empty if none.
fn glob(pattern: &str) -> Vec<String> {
    let absolute = pattern.starts_with('/');
    let parts: Vec<&str> = pattern.split('/').filter(|p| !p.is_empty()).collect();
    let mut paths = vec![if absolute { "/".to_string() } else { String::new() }];
    for (i, part) in parts.iter().enumerate() {
        let last = i + 1 == parts.len();
        let mut next = Vec::new();
        for base in &paths {
            let join = |n: &str| if base.is_empty() { n.to_string() } else if base.ends_with('/') { format!("{base}{n}") } else { format!("{base}/{n}") };
            if !part.contains(['*', '?', '[']) {
                let p = join(part);
                if last || fs::metadata(&p).map(|m| m.is_dir()).unwrap_or(false) {
                    next.push(p);
                }
                continue;
            }
            let dir = if base.is_empty() { "." } else { base.as_str() };
            let pat: Vec<char> = part.chars().collect();
            if let Ok(rd) = fs::read_dir(dir) {
                let mut names: Vec<String> = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
                names.sort();
                for n in names {
                    if n.starts_with('.') && !part.starts_with('.') {
                        continue;
                    }
                    if glob_match(&pat, &n.chars().collect::<Vec<_>>()) {
                        let p = join(&n);
                        if last || fs::metadata(&p).map(|m| m.is_dir()).unwrap_or(false) {
                            next.push(p);
                        }
                    }
                }
            }
        }
        paths = next;
    }
    if !parts.is_empty() && pattern.contains(['*', '?', '[']) {
        paths.into_iter().filter(|p| fs::symlink_metadata(p).is_ok()).collect()
    } else {
        Vec::new()
    }
}

struct Shell {
    vars: BTreeMap<String, String>,
    exported: std::collections::BTreeSet<String>,
    args: Vec<String>,
    status: i32,
    interactive: bool,
    exit: Option<i32>,
}

impl Shell {
    fn lookup(&self, name: &str) -> String {
        match name {
            "?" => self.status.to_string(),
            "$" => "1".into(),
            "#" => self.args.len().saturating_sub(1).to_string(),
            "@" | "*" => self.args.iter().skip(1).cloned().collect::<Vec<_>>().join(" "),
            n if n.chars().all(|c| c.is_ascii_digit()) => self.args.get(n.parse::<usize>().unwrap_or(99)).cloned().unwrap_or_default(),
            n => self.vars.get(n).cloned().unwrap_or_default(),
        }
    }

    /// A word's fields: variables and ~ substituted, unquoted variables split on spaces, and
    /// unquoted wildcards matched against files.
    fn expand(&self, w: &Word) -> Vec<String> {
        let mut fields: Vec<String> = vec![String::new()];
        let mut started = false;
        let mut wild = false;
        for (i, s) in w.iter().enumerate() {
            if s.var {
                let v = self.lookup(&s.text);
                if s.quoted {
                    fields.last_mut().unwrap().push_str(&v);
                    started = true;
                } else {
                    for (k, part) in v.split_whitespace().enumerate() {
                        if k > 0 {
                            fields.push(String::new());
                        }
                        fields.last_mut().unwrap().push_str(part);
                        started = true;
                    }
                }
            } else {
                let mut t = s.text.clone();
                if i == 0 && !s.quoted && (t == "~" || t.starts_with("~/")) {
                    t = format!("{}{}", self.lookup("HOME"), &t[1..]);
                }
                if !s.quoted && t.contains(['*', '?', '[']) {
                    wild = true;
                }
                fields.last_mut().unwrap().push_str(&t);
                started |= !t.is_empty() || s.quoted;
            }
        }
        if !started {
            return Vec::new();
        }
        if wild && fields.len() == 1 {
            let m = glob(&fields[0]);
            if !m.is_empty() {
                return m;
            }
        }
        fields
    }

    fn env_list(&self, extra: &[(String, String)]) -> Vec<String> {
        let mut env: BTreeMap<String, String> = self.exported.iter().filter_map(|k| self.vars.get(k).map(|v| (k.clone(), v.clone()))).collect();
        for (k, v) in extra {
            env.insert(k.clone(), v.clone());
        }
        env.insert("PWD".into(), cwd());
        env.into_iter().map(|(k, v)| format!("{k}={v}")).collect()
    }

    fn run_line(&mut self, line: &str) {
        let list = match lex(line).and_then(parse) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("sh: {e}");
                self.status = 2;
                return;
            }
        };
        let mut skip = false;
        for (pipeline, op) in list {
            if !skip {
                self.status = self.run_pipeline(&pipeline, op == "&");
                if self.exit.is_some() {
                    return;
                }
            }
            skip = match op {
                "&&" => self.status != 0,
                "||" => self.status == 0,
                _ => false,
            };
        }
    }

    fn run_pipeline(&mut self, cmds: &[Cmd], background: bool) -> i32 {
        if cmds.len() == 1 {
            let argv: Vec<String> = cmds[0].args.iter().flat_map(|w| self.expand(w)).collect();
            if argv.is_empty() {
                for (k, w) in &cmds[0].assigns {
                    let v = self.expand(w).join(" ");
                    self.vars.insert(k.clone(), v);
                }
                return 0;
            }
            if is_builtin(&argv[0]) {
                return self.run_builtin_redirected(&cmds[0], &argv);
            }
        }
        // Pipes between stages; each stage's standard descriptors.
        let n = cmds.len();
        let mut pipes = Vec::new();
        for _ in 1..n {
            let (mut r, mut w) = (0u32, 0u32);
            unsafe { fd_pipe(&mut r, &mut w) };
            pipes.push((r, w));
        }
        let mut pids = Vec::new();
        let mut pgid = 0;
        let mut last_status = 0;
        let mut opened: Vec<File> = Vec::new();
        for (i, cmd) in cmds.iter().enumerate() {
            let argv: Vec<String> = cmd.args.iter().flat_map(|w| self.expand(w)).collect();
            let mut fds = [if i > 0 { pipes[i - 1].0 } else { 0 }, if i + 1 < n { pipes[i].1 } else { 1 }, 2];
            if let Err(e) = self.apply_redirs(cmd, &mut fds, &mut opened) {
                eprintln!("sh: {e}");
                last_status = 1;
                continue;
            }
            if argv.is_empty() {
                continue;
            }
            if is_builtin(&argv[0]) {
                // A builtin inside a pipeline writes to the stage's output from the shell itself.
                last_status = self.run_builtin(&argv, fds[1], fds[2]);
                continue;
            }
            let extra: Vec<(String, String)> = cmd.assigns.iter().map(|(k, w)| (k.clone(), self.expand(w).join(" "))).collect();
            let env = self.env_list(&extra);
            let mut map = vec![0u32, fds[0], 1, fds[1], 2, fds[2]];
            // Close every other pipe end in the child.
            map.truncate(6);
            match spawn(&argv, &env, &cwd(), &map, pgid) {
                Ok(pid) => {
                    if pgid == 0 {
                        pgid = pid;
                    }
                    pids.push(pid);
                }
                Err(code) => {
                    if code == ENOENT {
                        eprintln!("sh: {}: command not found", argv[0]);
                        last_status = 127;
                    } else {
                        eprintln!("sh: {}: cannot run (error {code})", argv[0]);
                        last_status = 126;
                    }
                }
            }
        }
        drop(opened);
        // The children hold the pipe ends now.
        for (r, w) in pipes {
            drop(unsafe { File::from_raw_fd(r as i32) });
            drop(unsafe { File::from_raw_fd(w as i32) });
        }
        if background {
            if let Some(p) = pids.last() {
                if self.interactive {
                    eprintln!("[{p}]");
                }
            }
            return 0;
        }
        if pgid != 0 && self.interactive {
            unsafe { tty_set_foreground(pgid) };
        }
        let last = pids.last().copied();
        for pid in pids {
            let mut status = 0i32;
            unsafe { proc_wait(pid, &mut status) };
            if Some(pid) == last {
                last_status = status;
            }
        }
        if self.interactive {
            unsafe { tty_set_foreground(0) };
        }
        last_status
    }

    /// Open redirection targets: replaces the stage's descriptors.
    fn apply_redirs(&self, cmd: &Cmd, fds: &mut [u32; 3], opened: &mut Vec<File>) -> Result<(), String> {
        for r in &cmd.redirs {
            let fd = match &r.kind {
                RedirKind::Dup(to) => *fds.get(*to as usize).ok_or("bad descriptor")?,
                kind => {
                    let name = self.expand(&r.target).join(" ");
                    let f = match kind {
                        RedirKind::In => File::open(&name),
                        RedirKind::Out => File::create(&name),
                        _ => OpenOptions::new().append(true).create(true).open(&name),
                    }
                    .map_err(|e| format!("{name}: {}", describe(&e)))?;
                    let raw = f.into_raw_fd() as u32;
                    opened.push(unsafe { File::from_raw_fd(raw as i32) });
                    raw
                }
            };
            fds[r.fd as usize] = fd;
        }
        Ok(())
    }

    fn run_builtin_redirected(&mut self, cmd: &Cmd, argv: &[String]) -> i32 {
        let mut fds = [0u32, 1, 2];
        let mut opened = Vec::new();
        if let Err(e) = self.apply_redirs(cmd, &mut fds, &mut opened) {
            eprintln!("sh: {e}");
            return 1;
        }
        let saved: Vec<(String, Option<String>)> = cmd.assigns.iter().map(|(k, _)| (k.clone(), self.vars.get(k).cloned())).collect();
        for (k, w) in &cmd.assigns {
            let v = self.expand(w).join(" ");
            self.vars.insert(k.clone(), v);
        }
        let status = self.run_builtin(argv, fds[1], fds[2]);
        if argv[0] != "export" {
            for (k, v) in saved {
                match v {
                    Some(v) => self.vars.insert(k, v),
                    None => self.vars.remove(&k),
                };
            }
        }
        status
    }

    fn run_builtin(&mut self, argv: &[String], out_fd: u32, err_fd: u32) -> i32 {
        let mut out = FdOut(out_fd);
        let mut err = FdOut(err_fd);
        let args = &argv[1..];
        match argv[0].as_str() {
            "cd" => {
                let target = match args.first().map(|s| s.as_str()) {
                    None => self.lookup("HOME"),
                    Some("-") => self.lookup("OLDPWD"),
                    Some(p) => p.to_string(),
                };
                let old = cwd();
                // The C library keeps ".." in its working directory; resolve it here.
                let abs = if target.starts_with('/') { target.clone() } else { format!("{}/{target}", old.trim_end_matches('/')) };
                let mut parts: Vec<&str> = Vec::new();
                for c in abs.split('/') {
                    match c {
                        "" | "." => {}
                        ".." => {
                            parts.pop();
                        }
                        c => parts.push(c),
                    }
                }
                let normalized = format!("/{}", parts.join("/"));
                match env::set_current_dir(&normalized) {
                    Ok(()) => {
                        self.vars.insert("OLDPWD".into(), old);
                        self.vars.insert("PWD".into(), cwd());
                        0
                    }
                    Err(e) => {
                        let _ = writeln!(err, "cd: {target}: {}", describe(&e));
                        1
                    }
                }
            }
            "pwd" => {
                let _ = writeln!(out, "{}", cwd());
                0
            }
            // Backslash escapes are read, as by dash (sh on Debian and Ubuntu), which is what
            // `echo "a\nb"` is written for; -E reads them as plain text.
            "echo" => {
                let mut words = args;
                let (mut newline, mut escapes) = (true, true);
                while let Some(f) = words.first().filter(|f| f.len() > 1 && f.starts_with('-') && f[1..].chars().all(|c| "neE".contains(c))) {
                    newline &= !f.contains('n');
                    escapes = !f.contains('E');
                    words = &words[1..];
                }
                let text = words.join(" ");
                let (text, stop) = if escapes { echo_escapes(&text) } else { (text, false) };
                let _ = write!(out, "{text}{}", if newline && !stop { "\n" } else { "" });
                0
            }
            "export" => {
                for a in args {
                    match a.split_once('=') {
                        Some((k, v)) => {
                            self.vars.insert(k.into(), v.into());
                            self.exported.insert(k.into());
                        }
                        None => {
                            self.exported.insert(a.clone());
                        }
                    }
                }
                if args.is_empty() {
                    for k in &self.exported {
                        let _ = writeln!(out, "export {k}=\"{}\"", self.vars.get(k).cloned().unwrap_or_default());
                    }
                }
                0
            }
            "unset" => {
                for a in args {
                    self.vars.remove(a);
                    self.exported.remove(a);
                }
                0
            }
            "exit" => {
                self.exit = Some(args.first().and_then(|a| a.parse().ok()).unwrap_or(self.status));
                self.exit.unwrap()
            }
            "true" | ":" => 0,
            "false" => 1,
            "type" | "which" | "command" => {
                let names: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
                let mut status = 0;
                for n in names {
                    if is_builtin(n) && argv[0] != "which" {
                        let _ = writeln!(out, "{n} is a shell builtin");
                    } else if let Some(p) = which(n, &self.lookup("PATH")) {
                        let _ = writeln!(out, "{p}");
                    } else {
                        status = 1;
                        if argv[0] != "command" {
                            let _ = writeln!(err, "{n}: not found");
                        }
                    }
                }
                status
            }
            "source" | "." => match args.first().map(fs::read_to_string) {
                Some(Ok(text)) => {
                    for line in logical_lines(&text) {
                        self.run_line(&line);
                        if self.exit.is_some() {
                            break;
                        }
                    }
                    self.status
                }
                Some(Err(e)) => {
                    let _ = writeln!(err, "source: {}", describe(&e));
                    1
                }
                None => 2,
            },
            "read" => {
                // `read FILE` is a common slip for showing a file: say so instead of waiting for input.
                if let Some(bad) = args.iter().find(|n| !(n.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))) {
                    let _ = writeln!(err, "read: {bad} is not a variable name (read takes a line of input; to show a file, use: cat {bad})");
                    return 2;
                }
                let mut line = String::new();
                if io::stdin().lock().read_line(&mut line).unwrap_or(0) == 0 {
                    return 1;
                }
                let line = line.trim_end_matches('\n');
                let mut words = line.split_whitespace();
                for (i, name) in args.iter().enumerate() {
                    let v = if i + 1 == args.len() { words.by_ref().collect::<Vec<_>>().join(" ") } else { words.next().unwrap_or("").to_string() };
                    self.vars.insert(name.clone(), v);
                }
                0
            }
            "set" => {
                for (k, v) in &self.vars {
                    let _ = writeln!(out, "{k}={v}");
                }
                0
            }
            "help" => {
                let _ = writeln!(
                    out,
                    "Evolve VM shell. Builtins: cd pwd echo export unset exit true false type which source read set help.\n\
                     Programs in /bin and /usr/bin run as processes; `ls /bin` lists them.\n\
                     Pipes |, redirections < > >> 2> 2>&1 &>, and ; && || & work."
                );
                0
            }
            _ => 127,
        }
    }
}

struct FdOut(u32);

impl Write for FdOut {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut f = std::mem::ManuallyDrop::new(unsafe { File::from_raw_fd(self.0 as i32) });
        f.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// echo's backslash escapes: \n \t \\ \a \b \f \r \v, \0NNN (octal), and \c, which ends the
/// output there without a newline (the bool).
fn echo_escapes(s: &str) -> (String, bool) {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('a') => out.push('\u{7}'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('v') => out.push('\u{b}'),
            Some('\\') => out.push('\\'),
            Some('c') => return (out, true),
            Some('0') => {
                let mut v = 0u32;
                for _ in 0..3 {
                    match it.peek().and_then(|d| d.to_digit(8)) {
                        Some(d) => {
                            v = v * 8 + d;
                            it.next();
                        }
                        None => break,
                    }
                }
                out.push(char::from_u32(v).unwrap_or('?'));
            }
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    (out, false)
}

fn is_builtin(name: &str) -> bool {
    matches!(name, "cd" | "pwd" | "echo" | "export" | "unset" | "exit" | "true" | "false" | ":" | "type" | "which" | "command" | "source" | "." | "read" | "set" | "help")
}

fn which(name: &str, path: &str) -> Option<String> {
    if name.contains('/') {
        return fs::metadata(name).ok().map(|_| name.to_string());
    }
    path.split(':').map(|d| format!("{}/{name}", d.trim_end_matches('/'))).find(|p| fs::metadata(p).map(|m| m.is_file()).unwrap_or(false))
}

fn describe(e: &io::Error) -> String {
    match e.kind() {
        io::ErrorKind::NotFound => "No such file or directory".into(),
        io::ErrorKind::PermissionDenied => "Permission denied".into(),
        io::ErrorKind::AlreadyExists => "File exists".into(),
        _ => e.to_string(),
    }
}

fn cwd() -> String {
    env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| "/".into())
}

fn spawn(argv: &[String], env: &[String], cwd: &str, map: &[u32], pgid: u32) -> Result<u32, u32> {
    let a = argv.join("\0");
    let e = env.join("\0");
    let mut pid = 0u32;
    let r = unsafe { proc_spawn(a.as_ptr(), a.len(), e.as_ptr(), e.len(), cwd.as_ptr(), cwd.len(), map.as_ptr(), map.len(), pgid, &mut pid) };
    if r == 0 { Ok(pid) } else { Err(r) }
}

/// Lines with trailing-backslash continuations joined.
fn logical_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if let Some(stripped) = line.strip_suffix('\\') {
            cur.push_str(stripped);
            continue;
        }
        cur.push_str(line);
        out.push(std::mem::take(&mut cur));
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Ok(p) = env::var("PWD") {
        let _ = env::set_current_dir(p);
    }
    let mut vars: BTreeMap<String, String> = env::vars().collect();
    // Commands the page runs for Pi ask the shell where it ended up, so a `cd` carries over to the
    // next command. Not passed on: a nested shell's directory is not this one's.
    let cwd_file = vars.remove("EVOLVE_CWD_FILE");
    vars.entry("PATH".into()).or_insert_with(|| "/bin:/usr/bin".into());
    vars.entry("HOME".into()).or_insert_with(|| "/home/user".into());
    vars.insert("PWD".into(), cwd());
    let exported = vars.keys().cloned().collect();
    let mut sh = Shell { vars, exported, args: args.clone(), status: 0, interactive: false, exit: None };
    let code = match args.get(1).map(|s| s.as_str()) {
        Some("-c") => {
            sh.args = std::iter::once(args.get(3).cloned().unwrap_or("sh".into())).chain(args.iter().skip(4).cloned()).collect();
            sh.run_line(args.get(2).map(|s| s.as_str()).unwrap_or(""));
            if let Some(f) = &cwd_file {
                let _ = fs::write(f, cwd());
            }
            sh.exit.unwrap_or(sh.status)
        }
        Some(script) => match fs::read_to_string(script) {
            Ok(text) => {
                sh.args = args[1..].to_vec();
                for line in logical_lines(&text) {
                    sh.run_line(&line);
                    if sh.exit.is_some() {
                        break;
                    }
                }
                sh.exit.unwrap_or(sh.status)
            }
            Err(e) => {
                eprintln!("sh: {script}: {}", describe(&e));
                127
            }
        },
        None => {
            sh.interactive = true;
            unsafe { proc_ignore_interrupt(1) };
            let stdin = io::stdin();
            loop {
                let home = sh.lookup("HOME");
                let dir = cwd();
                let shown = if dir == home { "~".to_string() } else if let Some(rest) = dir.strip_prefix(&format!("{home}/")) { format!("~/{rest}") } else { dir };
                // The window title (OSC 0), then the prompt.
                print!("\x1b]0;user@evolve: {shown}\x07\x1b[1;92muser@evolve\x1b[0m:\x1b[1;94m{shown}\x1b[0m$ ");
                let _ = io::stdout().flush();
                let mut line = String::new();
                match stdin.lock().read_line(&mut line) {
                    Ok(0) => {
                        println!("exit");
                        break sh.status;
                    }
                    Ok(_) => {}
                    Err(_) => break 1,
                }
                sh.run_line(line.trim_end_matches('\n'));
                if let Some(code) = sh.exit {
                    break code;
                }
            }
        }
    };
    let _ = io::stdout().flush();
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexing_and_parsing() {
        let l = parse(lex(r#"FOO=1 echo "a $HOME b" 'c d' e\ f | grep -v x > out.txt 2>&1 && ls *.rs; exit 3"#).unwrap()).unwrap();
        assert_eq!(l.len(), 3);
        assert_eq!(l[0].1, "&&");
        let first = &l[0].0[0];
        assert_eq!(first.assigns[0].0, "FOO");
        assert_eq!(first.args.len(), 4);
        assert_eq!(l[0].0[1].redirs.len(), 2);
        assert!(lex("echo 'open").is_err());
        assert_eq!(echo_escapes("a\\nb\\tc\\\\d"), ("a\nb\tc\\d".to_string(), false));
        assert_eq!(echo_escapes("one\\ctwo"), ("one".to_string(), true));
        // Lines are commands; a newline after && continues; comments end at the line's end.
        let l = parse(lex("cd app # go in\ngit status &&\n  git diff\necho 'a\nb'").unwrap()).unwrap();
        assert_eq!(l.iter().map(|(p, op)| (p[0].args.len(), *op)).collect::<Vec<_>>(), vec![(2, ";"), (2, "&&"), (2, ";"), (2, ";")]);
        assert!(glob_match(&"*.r[sx]".chars().collect::<Vec<_>>(), &"main.rs".chars().collect::<Vec<_>>()));
        assert!(!glob_match(&"[!m]*".chars().collect::<Vec<_>>(), &"main".chars().collect::<Vec<_>>()));
    }
}
