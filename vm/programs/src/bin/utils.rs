//! Core utilities of the Evolve VM in one WASI program, chosen by the name it runs as
//! (/bin/ls, /bin/cat ... all link to it). Each takes the options people and agents use most.

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const NAMES: &[&str] = &[
    "ls", "cat", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "head", "tail", "wc", "sort", "uniq", "tee", "grep", "find", "env", "sleep",
    "basename", "dirname", "seq", "tr", "cut", "date", "yes", "stat", "ln", "readlink", "realpath", "clear", "uname", "whoami", "du", "diff",
    "test", "[", "printf", "nl", "rev", "xxd", "hostname", "id", "nproc", "df", "file", "edit", "sed",
];

type R = Result<i32, String>;

#[link(wasm_import_module = "wasi_snapshot_preview1")]
extern "C" {
    fn path_symlink(old: *const u8, old_len: usize, fd: u32, new: *const u8, new_len: usize) -> u16;
}

/// A symbolic link at `link` holding `target` (std has no stable call for this on WASI). Paths
/// resolve from the "/" (fd 3) or "." (fd 4) directory every program starts with.
fn symlink(target: &str, link: &Path) -> io::Result<()> {
    let l = link.to_string_lossy();
    let (fd, rel) = if let Some(abs) = l.strip_prefix('/') {
        (3, abs.to_string())
    } else {
        // Relative to the working directory, which programs here enter from $PWD at start.
        let cwd = env::current_dir()?;
        (3, format!("{}/{l}", cwd.to_string_lossy().trim_start_matches('/').trim_end_matches('/')).trim_start_matches('/').to_string())
    };
    match unsafe { path_symlink(target.as_ptr(), target.len(), fd, rel.as_ptr(), rel.len()) } {
        0 => Ok(()),
        20 => Err(io::Error::from(io::ErrorKind::AlreadyExists)),
        44 => Err(io::Error::from(io::ErrorKind::NotFound)),
        e => Err(io::Error::other(format!("error {e}"))),
    }
}

fn err(e: io::Error, what: &str) -> String {
    let msg = match e.kind() {
        io::ErrorKind::NotFound => "No such file or directory".to_string(),
        io::ErrorKind::AlreadyExists => "File exists".to_string(),
        io::ErrorKind::PermissionDenied => "Permission denied".to_string(),
        _ => e.to_string(),
    };
    format!("{what}: {msg}")
}

/// Options (short flags, with values for the listed ones) and operands.
struct Args {
    flags: BTreeSet<char>,
    values: Vec<(char, String)>,
    rest: Vec<String>,
}

impl Args {
    fn parse(args: &[String], with_value: &str) -> Args {
        let mut a = Args { flags: BTreeSet::new(), values: Vec::new(), rest: Vec::new() };
        let mut i = 0;
        let mut only_operands = false;
        while i < args.len() {
            let s = &args[i];
            if only_operands || !s.starts_with('-') || s == "-" || s.len() == 1 {
                a.rest.push(s.clone());
            } else if s == "--" {
                only_operands = true;
            } else if let Some(long) = s.strip_prefix("--") {
                a.values.push(('-', long.to_string()));
            } else {
                let chars: Vec<char> = s[1..].chars().collect();
                let mut k = 0;
                while k < chars.len() {
                    let c = chars[k];
                    if with_value.contains(c) {
                        let v: String = chars[k + 1..].iter().collect();
                        let v = if v.is_empty() {
                            i += 1;
                            args.get(i).cloned().unwrap_or_default()
                        } else {
                            v
                        };
                        a.values.push((c, v));
                        break;
                    }
                    a.flags.insert(c);
                    k += 1;
                }
            }
            i += 1;
        }
        a
    }
    fn has(&self, c: char) -> bool {
        self.flags.contains(&c)
    }
    fn value(&self, c: char) -> Option<&str> {
        self.values.iter().rev().find(|(k, _)| *k == c).map(|(_, v)| v.as_str())
    }
    fn long(&self, name: &str) -> bool {
        self.values.iter().any(|(k, v)| *k == '-' && v == name)
    }
}

fn read_input(path: &str) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    if path == "-" {
        io::stdin().read_to_end(&mut buf).map_err(|e| err(e, "stdin"))?;
    } else {
        File::open(path).and_then(|mut f| f.read_to_end(&mut buf)).map_err(|e| err(e, path))?;
    }
    Ok(buf)
}

/// The operands' lines, or standard input's.
fn lines_of(files: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let files = if files.is_empty() { vec!["-".to_string()] } else { files.to_vec() };
    for f in files {
        let text = String::from_utf8_lossy(&read_input(&f)?).into_owned();
        out.extend(text.lines().map(String::from));
    }
    Ok(out)
}

fn human(n: u64) -> String {
    let units = ["", "K", "M", "G", "T"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < units.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { n.to_string() } else if v < 10.0 { format!("{v:.1}{}", units[u]) } else { format!("{v:.0}{}", units[u]) }
}

fn mtime(m: &fs::Metadata) -> u64 {
    m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0)
}

/// UTC "YYYY-MM-DD HH:MM" for a Unix time.
fn date(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    let _ = s;
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}")
}

fn ls(args: &[String], out: &mut dyn Write) -> R {
    let a = Args::parse(args, "");
    let long = a.has('l');
    let all = a.has('a') || a.has('A');
    let tty = io::stdout().is_terminal();
    let one = a.has('1') || long || !tty;
    let targets = if a.rest.is_empty() { vec![".".to_string()] } else { a.rest.clone() };
    let mut status = 0;
    let show = |name: &str, path: &Path, out: &mut dyn Write| -> io::Result<()> {
        let meta = fs::symlink_metadata(path)?;
        let ft = meta.file_type();
        // -p and -F mark directories with a trailing slash.
        let marked = format!("{name}{}", if (a.has('p') || a.has('F')) && ft.is_dir() { "/" } else { "" });
        let name = marked.as_str();
        let colored = if !tty {
            name.to_string()
        } else if ft.is_dir() {
            format!("\x1b[1;94m{name}\x1b[0m")
        } else if ft.is_symlink() {
            format!("\x1b[1;96m{name}\x1b[0m")
        } else {
            name.to_string()
        };
        if long {
            let kind = if ft.is_dir() { 'd' } else if ft.is_symlink() { 'l' } else { '-' };
            let size = if a.has('h') { human(meta.len()) } else { meta.len().to_string() };
            let link = if ft.is_symlink() { format!(" -> {}", fs::read_link(path).map(|p| p.display().to_string()).unwrap_or_default()) } else { String::new() };
            writeln!(out, "{kind}{} user user {size:>8} {} {colored}{link}", if ft.is_dir() { "rwxr-xr-x" } else { "rw-r--r--" }, date(mtime(&meta)))
        } else if one {
            writeln!(out, "{colored}")
        } else {
            write!(out, "{colored}  ")
        }
    };
    for (ti, t) in targets.iter().enumerate() {
        let p = Path::new(t);
        match fs::metadata(p) {
            Ok(m) if m.is_dir() && !a.has('d') => {
                if targets.len() > 1 {
                    writeln!(out, "{}{t}:", if ti > 0 { "\n" } else { "" }).ok();
                }
                let mut names: Vec<String> = fs::read_dir(p).map_err(|e| err(e, t))?.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
                names.sort();
                for n in names.iter().filter(|n| all || !n.starts_with('.')) {
                    show(n, &p.join(n), out).map_err(|e| err(e, n))?;
                }
                if !one && !names.is_empty() {
                    writeln!(out).ok();
                }
            }
            Ok(_) => show(t, p, out).map_err(|e| err(e, t))?,
            Err(e) => {
                eprintln!("ls: {}", err(e, t));
                status = 2;
            }
        }
    }
    Ok(status)
}

fn copy_tree(from: &Path, to: &Path, recursive: bool) -> io::Result<()> {
    let m = fs::symlink_metadata(from)?;
    if m.is_dir() {
        if !recursive {
            return Err(io::Error::other(format!("-r not specified; omitting directory '{}'", from.display())));
        }
        fs::create_dir_all(to)?;
        for e in fs::read_dir(from)?.flatten() {
            copy_tree(&e.path(), &to.join(e.file_name()), true)?;
        }
        Ok(())
    } else {
        fs::copy(from, to).map(|_| ())
    }
}

/// Where operand `src` goes when the last operand is `dest` (into it, if it is a directory).
fn target(src: &str, dest: &str) -> PathBuf {
    let d = Path::new(dest);
    if d.is_dir() {
        d.join(Path::new(src).file_name().unwrap_or_default())
    } else {
        d.to_path_buf()
    }
}

fn walk(p: &Path, depth: usize, max: usize, f: &mut dyn FnMut(&Path, usize) -> bool) {
    if !f(p, depth) || depth >= max {
        return;
    }
    if fs::symlink_metadata(p).map(|m| m.is_dir()).unwrap_or(false) {
        let mut entries: Vec<PathBuf> = fs::read_dir(p).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        entries.sort();
        for e in entries {
            walk(&e, depth + 1, max, f);
        }
    }
}

fn glob_match(pat: &[char], name: &[char]) -> bool {
    match pat.first() {
        None => name.is_empty(),
        Some('*') => (0..=name.len()).any(|k| glob_match(&pat[1..], &name[k..])),
        Some('?') => !name.is_empty() && glob_match(&pat[1..], &name[1..]),
        Some(&c) => name.first() == Some(&c) && glob_match(&pat[1..], &name[1..]),
    }
}

// ---- sed: the forms models and people use most ----

enum SedAddr {
    Line(usize),
    Last,
    Re(regex::Regex),
}

enum SedOp {
    Sub(regex::Regex, String, bool, bool, usize),
    Del,
    Print,
    Quit,
    Append(String),
    Insert(String),
    Change(String),
}

struct SedCmd {
    a1: Option<SedAddr>,
    a2: Option<SedAddr>,
    neg: bool,
    op: SedOp,
    active: bool,
}

/// A basic regular expression (sed's default) as a Rust regex: \( \) \| \+ \? \{ \} are the
/// operators there, and ( ) | + ? { } are plain characters.
fn sed_regex(pat: &str, extended: bool, icase: bool) -> Result<regex::Regex, String> {
    if pat.is_empty() {
        return Err("sed: empty regular expression: write the text to find, as in s/old/new/".into());
    }
    let pat = if extended {
        pat.to_string()
    } else {
        let mut o = String::new();
        let mut it = pat.chars();
        while let Some(c) = it.next() {
            match c {
                '\\' => match it.next() {
                    Some(d) if "(){}|+?".contains(d) => o.push(d),
                    Some(d) => {
                        o.push('\\');
                        o.push(d);
                    }
                    None => o.push_str("\\\\"),
                },
                c if "(){}|+?".contains(c) => {
                    o.push('\\');
                    o.push(c);
                }
                c => o.push(c),
            }
        }
        o
    };
    regex::RegexBuilder::new(&pat).case_insensitive(icase).build().map_err(|e| format!("sed: {e}"))
}

fn sed_parse(script: &str, extended: bool) -> Result<Vec<SedCmd>, String> {
    let c: Vec<char> = script.chars().collect();
    let mut i = 0;
    let mut cmds = Vec::new();
    // Text up to an unescaped delimiter; \DELIM is the delimiter itself.
    let delimited = |i: &mut usize, d: char| -> Result<String, String> {
        let mut t = String::new();
        while *i < c.len() && c[*i] != d {
            if c[*i] == '\\' && *i + 1 < c.len() {
                if c[*i + 1] != d {
                    t.push('\\');
                }
                t.push(c[*i + 1]);
                *i += 2;
            } else {
                t.push(c[*i]);
                *i += 1;
            }
        }
        if *i >= c.len() {
            return Err(format!("sed: unterminated expression (missing {d}): write it as s/old/new/"));
        }
        *i += 1;
        Ok(t)
    };
    let address = |i: &mut usize| -> Result<Option<SedAddr>, String> {
        if *i < c.len() && c[*i].is_ascii_digit() {
            let mut n = 0usize;
            while *i < c.len() && c[*i].is_ascii_digit() {
                n = n * 10 + c[*i].to_digit(10).unwrap() as usize;
                *i += 1;
            }
            return Ok(Some(SedAddr::Line(n)));
        }
        if *i < c.len() && c[*i] == '$' {
            *i += 1;
            return Ok(Some(SedAddr::Last));
        }
        if *i < c.len() && c[*i] == '/' {
            *i += 1;
            return Ok(Some(SedAddr::Re(sed_regex(&delimited(i, '/')?, extended, false)?)));
        }
        Ok(None)
    };
    loop {
        while i < c.len() && (c[i].is_whitespace() || c[i] == ';') {
            i += 1;
        }
        if i >= c.len() {
            break;
        }
        let a1 = address(&mut i)?;
        let mut a2 = None;
        if a1.is_some() && i < c.len() && c[i] == ',' {
            i += 1;
            a2 = address(&mut i)?;
            if a2.is_none() {
                return Err("sed: a range needs an end, as in 2,5p".into());
            }
        }
        while i < c.len() && c[i] == ' ' {
            i += 1;
        }
        let neg = i < c.len() && c[i] == '!';
        if neg {
            i += 1;
        }
        let Some(&cmd) = c.get(i) else { return Err("sed: missing command after the address".into()) };
        i += 1;
        let op = match cmd {
            's' => {
                let d = *c.get(i).ok_or("sed: s needs /old/new/")?;
                i += 1;
                let pat = delimited(&mut i, d)?;
                let rep = delimited(&mut i, d)?;
                let (mut g, mut p, mut icase, mut nth) = (false, false, false, 0usize);
                while i < c.len() && !c[i].is_whitespace() && c[i] != ';' && c[i] != '}' {
                    match c[i] {
                        'g' => g = true,
                        'p' => p = true,
                        'i' | 'I' => icase = true,
                        d if d.is_ascii_digit() => nth = nth * 10 + d.to_digit(10).unwrap() as usize,
                        o => return Err(format!("sed: unknown s flag {o} (g, p, i or a number)")),
                    }
                    i += 1;
                }
                SedOp::Sub(sed_regex(&pat, extended, icase)?, rep, g, p, nth.max(1))
            }
            'd' => SedOp::Del,
            'p' => SedOp::Print,
            'q' => SedOp::Quit,
            'a' | 'i' | 'c' => {
                while i < c.len() && c[i] == ' ' {
                    i += 1;
                }
                if i < c.len() && c[i] == '\\' {
                    i += 1;
                    if i < c.len() && c[i] == '\n' {
                        i += 1;
                    }
                }
                let mut t = String::new();
                while i < c.len() && c[i] != '\n' {
                    t.push(c[i]);
                    i += 1;
                }
                let t = t.replace("\\n", "\n").replace("\\t", "\t");
                match cmd {
                    'a' => SedOp::Append(t),
                    'i' => SedOp::Insert(t),
                    _ => SedOp::Change(t),
                }
            }
            '{' | '}' => return Err("sed: { } groups are not available here; use several -e expressions".into()),
            o => return Err(format!("sed: unknown command {o} (s, d, p, q, a, i, c are available)")),
        };
        cmds.push(SedCmd { a1, a2, neg, op, active: false });
    }
    Ok(cmds)
}

fn sed_expand(rep: &str, caps: &regex::Captures) -> String {
    let mut s = String::new();
    let mut it = rep.chars();
    while let Some(c) = it.next() {
        match c {
            '&' => s.push_str(caps.get(0).map_or("", |m| m.as_str())),
            '\\' => match it.next() {
                Some(d @ '0'..='9') => s.push_str(caps.get(d as usize - '0' as usize).map_or("", |m| m.as_str())),
                Some('n') => s.push('\n'),
                Some('t') => s.push('\t'),
                Some(o) => s.push(o),
                None => s.push('\\'),
            },
            _ => s.push(c),
        }
    }
    s
}

fn sed_run(cmds: &mut [SedCmd], text: &str, quiet: bool) -> String {
    let had_nl = text.ends_with('\n');
    let lines: Vec<&str> = if text.is_empty() { vec![] } else { text.strip_suffix('\n').unwrap_or(text).split('\n').collect() };
    let mut out: Vec<String> = Vec::new();
    let matches = |a: &SedAddr, n: usize, ps: &str, last: bool| match a {
        SedAddr::Line(k) => n == *k,
        SedAddr::Last => last,
        SedAddr::Re(re) => re.is_match(ps),
    };
    'lines: for (idx, line) in lines.iter().enumerate() {
        let (n, last) = (idx + 1, idx + 1 == lines.len());
        let mut ps = line.to_string();
        let mut after = Vec::new();
        for cmd in cmds.iter_mut() {
            let mut sel = match (&cmd.a1, &cmd.a2) {
                (None, _) => true,
                (Some(a1), None) => matches(a1, n, &ps, last),
                (Some(a1), Some(a2)) => {
                    if cmd.active {
                        if match a2 { SedAddr::Line(k) => n >= *k, _ => matches(a2, n, &ps, last) } {
                            cmd.active = false;
                        }
                        true
                    } else if matches(a1, n, &ps, last) {
                        cmd.active = !matches!(a2, SedAddr::Line(k) if *k <= n);
                        true
                    } else {
                        false
                    }
                }
            };
            if cmd.neg {
                sel = !sel;
            }
            if !sel {
                continue;
            }
            match &cmd.op {
                SedOp::Sub(re, rep, g, p, nth) => {
                    let (mut res, mut lastpos, mut count, mut did) = (String::new(), 0, 0, false);
                    for caps in re.captures_iter(&ps) {
                        let m = caps.get(0).unwrap();
                        count += 1;
                        if count == *nth || (*g && count > *nth) {
                            res.push_str(&ps[lastpos..m.start()]);
                            res.push_str(&sed_expand(rep, &caps));
                            lastpos = m.end();
                            did = true;
                            if !*g {
                                break;
                            }
                        }
                    }
                    if did {
                        res.push_str(&ps[lastpos..]);
                        ps = res;
                        if *p {
                            out.push(ps.clone());
                        }
                    }
                }
                SedOp::Del => continue 'lines,
                SedOp::Print => out.push(ps.clone()),
                SedOp::Quit => {
                    if !quiet {
                        out.push(ps.clone());
                    }
                    out.extend(after);
                    break 'lines;
                }
                SedOp::Append(t) => after.push(t.clone()),
                SedOp::Insert(t) => out.push(t.clone()),
                SedOp::Change(t) => {
                    if !cmd.active {
                        out.push(t.clone());
                    }
                    continue 'lines;
                }
            }
        }
        if !quiet {
            out.push(ps);
        }
        out.extend(after);
    }
    let mut s = out.join("\n");
    if !out.is_empty() && (had_nl || lines.is_empty()) {
        s.push('\n');
    }
    s
}

/// sed [-n] [-i] [-E] [-e SCRIPT]... [SCRIPT] [FILE...]: s/RE/NEW/[g p i N], d, p, q, a/i/c TEXT,
/// with LINE, $ and /RE/ addresses, ranges and !. -i changes the files in place.
fn sed(args: &[String], out: &mut dyn Write) -> R {
    let (mut quiet, mut in_place, mut extended) = (false, false, false);
    let (mut scripts, mut files) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-n" | "--quiet" | "--silent" => quiet = true,
            "-E" | "-r" | "--regexp-extended" => extended = true,
            "-e" | "--expression" => {
                i += 1;
                scripts.push(args.get(i).ok_or("sed: -e needs a script")?.clone());
            }
            _ if a.starts_with("--in-place") || (a.starts_with("-i") && !a[2..].chars().all(|c| "nEr".contains(c))) => in_place = true,
            _ if a.len() > 1 && a.starts_with('-') && a[1..].chars().all(|c| "niEr".contains(c)) => {
                quiet |= a.contains('n');
                in_place |= a.contains('i');
                extended |= a.contains('E') || a.contains('r');
            }
            _ => files.push(a.to_string()),
        }
        i += 1;
    }
    if scripts.is_empty() {
        if files.is_empty() {
            return Err("usage: sed [-n] [-i] [-E] 's/old/new/' FILE   (also: '3d', '2,5p', '/text/a new line')".into());
        }
        scripts.push(files.remove(0));
    }
    let mut cmds = sed_parse(&scripts.join("\n"), extended)?;
    if in_place {
        if files.is_empty() {
            return Err("sed: -i needs a file to change".into());
        }
        for f in &files {
            let text = String::from_utf8_lossy(&read_input(f)?).into_owned();
            for c in cmds.iter_mut() {
                c.active = false;
            }
            fs::write(f, sed_run(&mut cmds, &text, quiet)).map_err(|e| err(e, f))?;
        }
        return Ok(0);
    }
    let mut text = String::new();
    for f in if files.is_empty() { vec!["-".to_string()] } else { files } {
        text.push_str(&String::from_utf8_lossy(&read_input(&f)?));
    }
    out.write_all(sed_run(&mut cmds, &text, quiet).as_bytes()).map_err(|e| err(e, "stdout"))?;
    Ok(0)
}

fn grep(args: &[String], out: &mut dyn Write) -> R {
    let a = Args::parse(args, "eAB");
    let mut rest = a.rest.clone();
    let pattern = match a.value('e') {
        Some(p) => p.to_string(),
        None if !rest.is_empty() => rest.remove(0),
        None => return Err("usage: grep [-inrvlcwFE] PATTERN [FILE...]".into()),
    };
    let mut pat = if a.has('F') { regex::escape(&pattern) } else { pattern.replace("\\|", "|").replace("\\(", "(").replace("\\)", ")") };
    if a.has('w') {
        pat = format!(r"\b(?:{pat})\b");
    }
    let re = regex::RegexBuilder::new(&pat).case_insensitive(a.has('i')).build().map_err(|e| format!("grep: {e}"))?;
    let mut files = Vec::new();
    for f in if rest.is_empty() { vec![if a.has('r') { ".".to_string() } else { "-".to_string() }] } else { rest } {
        if a.has('r') && Path::new(&f).is_dir() {
            walk(Path::new(&f), 0, usize::MAX, &mut |p, _| {
                let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if p.is_dir() {
                    return !(name.starts_with('.') && name.len() > 1) && name != "node_modules" && name != "target";
                }
                files.push(p.to_string_lossy().trim_start_matches("./").to_string());
                true
            });
        } else {
            files.push(f);
        }
    }
    let show_name = files.len() > 1 || a.has('r');
    let mut found = false;
    for f in &files {
        let data = match read_input(f) {
            Ok(d) => d,
            Err(e) => {
                if !a.has('s') {
                    eprintln!("grep: {e}");
                }
                continue;
            }
        };
        if data.iter().take(4096).any(|&b| b == 0) {
            continue; // binary
        }
        let text = String::from_utf8_lossy(&data);
        let mut count = 0;
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) != a.has('v') {
                found = true;
                count += 1;
                if a.has('l') {
                    writeln!(out, "{f}").ok();
                    break;
                }
                if !a.has('c') && !a.has('q') {
                    let prefix = format!("{}{}", if show_name { format!("{f}:") } else { String::new() }, if a.has('n') { format!("{}:", i + 1) } else { String::new() });
                    writeln!(out, "{prefix}{line}").ok();
                }
            }
        }
        if a.has('c') {
            writeln!(out, "{}{count}", if show_name { format!("{f}:") } else { String::new() }).ok();
        }
    }
    Ok(if found { 0 } else { 1 })
}

/// `test` / `[` expressions: -e -f -d -s -z -n, =, !=, -eq -ne -lt -le -gt -ge, !, -a, -o.
fn test(args: &[String]) -> bool {
    let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    if let Some(i) = a.iter().position(|&s| s == "-o") {
        return test(&args[..i]) || test(&args[i + 1..]);
    }
    if let Some(i) = a.iter().position(|&s| s == "-a") {
        return test(&args[..i]) && test(&args[i + 1..]);
    }
    match a.as_slice() {
        [] => false,
        ["!", rest @ ..] => !test(&rest.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
        [s] => !s.is_empty(),
        ["-e", p] => Path::new(p).exists(),
        ["-f", p] => Path::new(p).is_file(),
        ["-d", p] => Path::new(p).is_dir(),
        ["-s", p] => fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false),
        ["-L" | "-h", p] => fs::symlink_metadata(p).map(|m| m.file_type().is_symlink()).unwrap_or(false),
        ["-r" | "-w" | "-x", p] => Path::new(p).exists(),
        ["-z", s] => s.is_empty(),
        ["-n", s] => !s.is_empty(),
        [x, "=" | "==", y] => x == y,
        [x, "!=", y] => x != y,
        [x, op, y] => {
            let (x, y) = (x.parse::<i64>().unwrap_or(0), y.parse::<i64>().unwrap_or(0));
            match *op {
                "-eq" => x == y,
                "-ne" => x != y,
                "-lt" => x < y,
                "-le" => x <= y,
                "-gt" => x > y,
                "-ge" => x >= y,
                _ => false,
            }
        }
        _ => false,
    }
}

/// printf with %s %d %i %x %o %c %f %% and backslash escapes; reuses the format for extra args.
fn printf(args: &[String]) -> String {
    let Some(fmt) = args.first() else { return String::new() };
    let fmt = fmt.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\").replace("\\e", "\x1b");
    let mut vals = args[1..].iter();
    let mut out = String::new();
    loop {
        let mut used = false;
        let mut chars = fmt.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            let mut spec = String::new();
            while let Some(&n) = chars.peek() {
                if n.is_ascii_digit() || n == '-' || n == '.' {
                    spec.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            let conv = chars.next().unwrap_or('%');
            if conv == '%' {
                out.push('%');
                continue;
            }
            used = true;
            let v = vals.next().map(|s| s.as_str()).unwrap_or("");
            let width: usize = spec.trim_start_matches('-').split('.').next().and_then(|w| w.parse().ok()).unwrap_or(0);
            let s = match conv {
                'd' | 'i' => v.parse::<i64>().unwrap_or(0).to_string(),
                'x' => format!("{:x}", v.parse::<i64>().unwrap_or(0)),
                'o' => format!("{:o}", v.parse::<i64>().unwrap_or(0)),
                'c' => v.chars().next().map(String::from).unwrap_or_default(),
                'f' => {
                    let prec = spec.split('.').nth(1).and_then(|p| p.parse().ok()).unwrap_or(6);
                    format!("{:.*}", prec, v.parse::<f64>().unwrap_or(0.0))
                }
                _ => v.to_string(),
            };
            if spec.starts_with('-') {
                out.push_str(&format!("{s:<width$}"));
            } else {
                out.push_str(&format!("{s:>width$}"));
            }
        }
        if !used || vals.len() == 0 {
            break;
        }
    }
    out
}

/// Line diff (longest common subsequence) in unified style with the given context.
fn diff(a: &[String], b: &[String], names: (&str, &str), out: &mut dyn Write) -> i32 {
    let (n, m) = (a.len(), b.len());
    if n * m > 25_000_000 {
        writeln!(out, "Files {} and {} differ", names.0, names.1).ok();
        return 1;
    }
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            ops.push((' ', i, j));
            i += 1;
            j += 1;
        } else if i < n && (j == m || lcs[i + 1][j] >= lcs[i][j + 1]) {
            ops.push(('-', i, j));
            i += 1;
        } else {
            ops.push(('+', i, j));
            j += 1;
        }
    }
    if ops.iter().all(|o| o.0 == ' ') {
        return 0;
    }
    writeln!(out, "--- {}\n+++ {}", names.0, names.1).ok();
    let ctx = 3;
    let mut k = 0;
    while k < ops.len() {
        if ops[k].0 == ' ' {
            k += 1;
            continue;
        }
        let start = k.saturating_sub(ctx);
        let mut end = k;
        while end < ops.len() && (ops[end].0 != ' ' || ops[end..(end + 2 * ctx + 1).min(ops.len())].iter().any(|o| o.0 != ' ')) {
            end += 1;
        }
        let end = (end + ctx).min(ops.len());
        let (a0, b0) = (ops[start].1, ops[start].2);
        let alen = ops[start..end].iter().filter(|o| o.0 != '+').count();
        let blen = ops[start..end].iter().filter(|o| o.0 != '-').count();
        writeln!(out, "@@ -{},{alen} +{},{blen} @@", a0 + 1, b0 + 1).ok();
        for &(op, i, j) in &ops[start..end] {
            let line = if op == '+' { &b[j] } else { &a[i] };
            writeln!(out, "{op}{line}").ok();
        }
        k = end;
    }
    1
}

fn run(name: &str, args: &[String]) -> R {
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let out: &mut dyn Write = &mut out;
    match name {
        "ls" => ls(args, out),
        "cat" => {
            let a = Args::parse(args, "");
            let files = if a.rest.is_empty() { vec!["-".to_string()] } else { a.rest.clone() };
            let mut status = 0;
            let mut n = 0;
            for f in files {
                match read_input(&f) {
                    Ok(d) if a.has('n') => {
                        for line in String::from_utf8_lossy(&d).lines() {
                            n += 1;
                            writeln!(out, "{n:6}\t{line}").ok();
                        }
                    }
                    Ok(d) => {
                        out.write_all(&d).ok();
                    }
                    Err(e) => {
                        eprintln!("cat: {e}");
                        status = 1;
                    }
                }
                out.flush().ok();
            }
            Ok(status)
        }
        "cp" | "mv" => {
            let a = Args::parse(args, "");
            if a.rest.len() < 2 {
                return Err(format!("usage: {name} SOURCE... DEST"));
            }
            let (srcs, dest) = a.rest.split_at(a.rest.len() - 1);
            for s in srcs {
                let t = target(s, &dest[0]);
                let r = if name == "mv" { fs::rename(s, &t) } else { copy_tree(Path::new(s), &t, a.has('r') || a.has('R') || a.has('a')) };
                r.map_err(|e| err(e, s))?;
            }
            Ok(0)
        }
        "rm" => {
            let a = Args::parse(args, "");
            let mut status = 0;
            for f in &a.rest {
                let p = Path::new(f);
                let r = match fs::symlink_metadata(p) {
                    Ok(m) if m.is_dir() => {
                        if a.has('r') || a.has('R') {
                            fs::remove_dir_all(p)
                        } else {
                            Err(io::Error::other("Is a directory"))
                        }
                    }
                    Ok(_) => fs::remove_file(p),
                    Err(e) => Err(e),
                };
                if let Err(e) = r {
                    if !(a.has('f') && e.kind() == io::ErrorKind::NotFound) {
                        eprintln!("rm: {}", err(e, f));
                        status = 1;
                    }
                }
            }
            Ok(status)
        }
        "mkdir" => {
            let a = Args::parse(args, "m");
            for d in &a.rest {
                if a.has('p') { fs::create_dir_all(d) } else { fs::create_dir(d) }.map_err(|e| err(e, d))?;
            }
            Ok(0)
        }
        "rmdir" => {
            for d in Args::parse(args, "").rest {
                fs::remove_dir(&d).map_err(|e| err(e, &d))?;
            }
            Ok(0)
        }
        "touch" => {
            for f in Args::parse(args, "").rest {
                if Path::new(&f).exists() {
                    let data = fs::read(&f).map_err(|e| err(e, &f))?;
                    fs::write(&f, data).map_err(|e| err(e, &f))?;
                } else {
                    File::create(&f).map_err(|e| err(e, &f))?;
                }
            }
            Ok(0)
        }
        "head" | "tail" => {
            let fixed: Vec<String> = args.iter().map(|s| if s.len() > 1 && s.starts_with('-') && s[1..].chars().all(|c| c.is_ascii_digit()) { format!("-n{}", &s[1..]) } else { s.clone() }).collect();
            let a = Args::parse(&fixed, "nc");
            let n: usize = a.value('n').map(|v| v.trim_start_matches('+').parse().unwrap_or(10)).unwrap_or(10);
            if let Some(c) = a.value('c') {
                let c: usize = c.parse().unwrap_or(0);
                let d = read_input(a.rest.first().map(|s| s.as_str()).unwrap_or("-"))?;
                let part = if name == "head" { &d[..c.min(d.len())] } else { &d[d.len().saturating_sub(c)..] };
                out.write_all(part).ok();
                return Ok(0);
            }
            if name == "head" {
                // Stop after n lines: an endless producer (yes | head) then sees the pipe close.
                let files = if a.rest.is_empty() { vec!["-".to_string()] } else { a.rest.clone() };
                for f in files {
                    let reader: Box<dyn BufRead> = if f == "-" { Box::new(BufReader::new(io::stdin())) } else { Box::new(BufReader::new(File::open(&f).map_err(|e| err(e, &f))?)) };
                    for line in reader.lines().take(n) {
                        writeln!(out, "{}", line.map_err(|e| err(e, &f))?).ok();
                    }
                }
                return Ok(0);
            }
            let lines = lines_of(&a.rest)?;
            let take: Vec<&String> = if name == "head" {
                lines.iter().take(n).collect()
            } else if a.value('n').is_some_and(|v| v.starts_with('+')) {
                lines.iter().skip(n.saturating_sub(1)).collect()
            } else {
                lines.iter().skip(lines.len().saturating_sub(n)).collect()
            };
            for l in take {
                writeln!(out, "{l}").ok();
            }
            Ok(0)
        }
        "wc" => {
            let a = Args::parse(args, "");
            let files = if a.rest.is_empty() { vec!["-".to_string()] } else { a.rest.clone() };
            let all = !a.has('l') && !a.has('w') && !a.has('c') && !a.has('m');
            for f in &files {
                let d = read_input(f)?;
                let text = String::from_utf8_lossy(&d);
                let mut cols = Vec::new();
                if all || a.has('l') {
                    cols.push(d.iter().filter(|&&b| b == b'\n').count());
                }
                if all || a.has('w') {
                    cols.push(text.split_whitespace().count());
                }
                if all || a.has('c') || a.has('m') {
                    cols.push(if a.has('m') { text.chars().count() } else { d.len() });
                }
                let nums: Vec<String> = cols.iter().map(|c| format!("{c:>7}")).collect();
                writeln!(out, "{}{}", nums.join(" "), if f == "-" { String::new() } else { format!(" {f}") }).ok();
            }
            Ok(0)
        }
        "sort" => {
            let a = Args::parse(args, "kt");
            let mut lines = lines_of(&a.rest)?;
            let key = |l: &String| -> String {
                match a.value('k').and_then(|k| k.split(',').next()?.parse::<usize>().ok()) {
                    Some(k) => {
                        let fields: Vec<&str> = match a.value('t') { Some(t) => l.split(t).collect(), None => l.split_whitespace().collect() };
                        fields.get(k - 1).unwrap_or(&"").to_string()
                    }
                    None => l.clone(),
                }
            };
            if a.has('n') {
                lines.sort_by(|x, y| key(x).trim().parse::<f64>().unwrap_or(0.0).total_cmp(&key(y).trim().parse::<f64>().unwrap_or(0.0)));
            } else if a.has('f') {
                lines.sort_by_key(|l| key(l).to_lowercase());
            } else {
                lines.sort_by_key(key);
            }
            if a.has('r') {
                lines.reverse();
            }
            if a.has('u') {
                lines.dedup();
            }
            for l in lines {
                writeln!(out, "{l}").ok();
            }
            Ok(0)
        }
        "uniq" => {
            let a = Args::parse(args, "");
            let lines = lines_of(&a.rest)?;
            let mut i = 0;
            while i < lines.len() {
                let mut j = i;
                while j < lines.len() && lines[j] == lines[i] {
                    j += 1;
                }
                let count = j - i;
                if (!a.has('d') || count > 1) && (!a.has('u') || count == 1) {
                    if a.has('c') { writeln!(out, "{count:>7} {}", lines[i]) } else { writeln!(out, "{}", lines[i]) }.ok();
                }
                i = j;
            }
            Ok(0)
        }
        // Not a Unix tool, but what people and small models reach for to change one line:
        // OLD becomes NEW where it occurs exactly once; \n, \t and \\ in both are a new line,
        // a tab and a backslash.
        "edit" => {
            // `edit FILE LINE OLD NEW` (small models add the line number) works as without it.
            let picked: Vec<&String> = if args.len() == 4 && !args[1].is_empty() && args[1].chars().all(|c| c.is_ascii_digit()) { vec![&args[0], &args[2], &args[3]] } else { args.iter().collect() };
            let [file, old, new] = picked[..] else {
                return Err("edit: usage: edit FILE OLD NEW   (replaces OLD with NEW where it occurs exactly once; \\n is a new line)".into());
            };
            let unescape = |s: &str| s.replace("\\\\", "\u{0}").replace("\\n", "\n").replace("\\t", "\t").replace('\u{0}', "\\");
            let (old, new) = (unescape(old), unescape(new));
            if old.is_empty() {
                return Err("edit: OLD is empty: give the exact text to replace".into());
            }
            if old == new {
                return Err(format!("edit: OLD and NEW are the same, so nothing would change. Show the lines with grep -n \"WORDS\" -A3 {file}, then edit {file} 'exact old text' 'corrected text'"));
            }
            let text = fs::read_to_string(file).map_err(|e| err(e, file))?;
            match text.matches(old.as_str()).count() {
                1 => {
                    fs::write(file, text.replacen(&old, &new, 1)).map_err(|e| err(e, file))?;
                    writeln!(out, "edited {file}").ok();
                    Ok(0)
                }
                0 => Err(format!("edit: {file}: the text to replace is not there. Find the exact line with: grep -n \"SOME WORDS\" {file}")),
                n => Err(format!("edit: {file}: the text to replace is in {n} places; include more of the surrounding text")),
            }
        }
        "tee" => {
            let a = Args::parse(args, "");
            let mut files: Vec<File> = a.rest.iter().map(|f| OpenOptions::new().write(true).create(true).append(a.has('a')).truncate(!a.has('a')).open(f).map_err(|e| err(e, f))).collect::<Result<_, _>>()?;
            let mut buf = [0u8; 65536];
            let mut stdin = io::stdin();
            loop {
                let n = stdin.read(&mut buf).map_err(|e| err(e, "stdin"))?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n]).ok();
                out.flush().ok();
                for f in &mut files {
                    f.write_all(&buf[..n]).ok();
                }
            }
            Ok(0)
        }
        "grep" => grep(args, out),
        "sed" => sed(args, out),
        "find" => {
            let mut roots = Vec::new();
            let mut i = 0;
            while i < args.len() && !args[i].starts_with('-') {
                roots.push(args[i].clone());
                i += 1;
            }
            if roots.is_empty() {
                roots.push(".".into());
            }
            let (mut name, mut kind, mut maxdepth) = (None, None, usize::MAX);
            while i < args.len() {
                let v = args.get(i + 1).cloned().unwrap_or_default();
                match args[i].as_str() {
                    "-name" | "-iname" => name = Some((v, args[i] == "-iname")),
                    "-type" => kind = Some(v),
                    "-maxdepth" => maxdepth = v.parse().unwrap_or(usize::MAX),
                    _ => {}
                }
                i += 2;
            }
            for r in roots {
                walk(Path::new(&r), 0, maxdepth, &mut |p, _| {
                    let fname = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.to_string_lossy().into_owned());
                    let is_dir = p.is_dir();
                    let ok_kind = match kind.as_deref() {
                        Some("f") => p.is_file(),
                        Some("d") => is_dir,
                        Some("l") => p.is_symlink(),
                        _ => true,
                    };
                    let ok_name = match &name {
                        Some((pat, ci)) => {
                            let (pat, n) = if *ci { (pat.to_lowercase(), fname.to_lowercase()) } else { (pat.clone(), fname.clone()) };
                            glob_match(&pat.chars().collect::<Vec<_>>(), &n.chars().collect::<Vec<_>>())
                        }
                        None => true,
                    };
                    if ok_kind && ok_name {
                        writeln!(out, "{}", p.display()).ok();
                    }
                    true
                });
            }
            Ok(0)
        }
        "env" => {
            let mut vars: Vec<(String, String)> = env::vars().collect();
            vars.sort();
            for (k, v) in vars {
                writeln!(out, "{k}={v}").ok();
            }
            Ok(0)
        }
        "sleep" => {
            let secs: f64 = args.first().map(|s| s.trim_end_matches('s').parse().unwrap_or(0.0)).unwrap_or(0.0);
            std::thread::sleep(Duration::from_secs_f64(secs.max(0.0)));
            Ok(0)
        }
        "basename" => {
            let p = args.first().cloned().unwrap_or_default();
            let mut b = Path::new(&p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(p.clone());
            if let Some(suffix) = args.get(1) {
                if let Some(s) = b.strip_suffix(suffix.as_str()) {
                    b = s.to_string();
                }
            }
            writeln!(out, "{b}").ok();
            Ok(0)
        }
        "dirname" => {
            let p = args.first().cloned().unwrap_or_default();
            let d = Path::new(&p).parent().map(|d| d.to_string_lossy().into_owned()).filter(|d| !d.is_empty()).unwrap_or(".".into());
            writeln!(out, "{d}").ok();
            Ok(0)
        }
        "seq" => {
            let n: Vec<i64> = args.iter().filter_map(|s| s.parse().ok()).collect();
            let (first, step, last) = match n.as_slice() {
                [l] => (1, 1, *l),
                [f, l] => (*f, 1, *l),
                [f, s, l] => (*f, *s, *l),
                _ => return Err("usage: seq [FIRST [STEP]] LAST".into()),
            };
            let mut x = first;
            while (step > 0 && x <= last) || (step < 0 && x >= last) {
                writeln!(out, "{x}").ok();
                x += step;
            }
            Ok(0)
        }
        "tr" => {
            let a = Args::parse(args, "");
            let expand = |s: &str| -> Vec<char> {
                let c: Vec<char> = s.replace("\\n", "\n").replace("\\t", "\t").chars().collect();
                let mut out = Vec::new();
                let mut i = 0;
                while i < c.len() {
                    if i + 2 < c.len() && c[i + 1] == '-' {
                        out.extend(c[i]..=c[i + 2]);
                        i += 3;
                    } else {
                        out.push(c[i]);
                        i += 1;
                    }
                }
                out
            };
            let set1 = expand(a.rest.first().map(|s| s.as_str()).unwrap_or(""));
            let set2 = expand(a.rest.get(1).map(|s| s.as_str()).unwrap_or(""));
            let mut input = String::new();
            io::stdin().read_to_string(&mut input).ok();
            let res: String = input
                .chars()
                .filter_map(|ch| match set1.iter().position(|&s| s == ch) {
                    Some(_) if a.has('d') => None,
                    Some(i) => Some(*set2.get(i).or(set2.last()).unwrap_or(&ch)),
                    None => Some(ch),
                })
                .collect();
            write!(out, "{res}").ok();
            Ok(0)
        }
        "cut" => {
            let a = Args::parse(args, "dfc");
            let delim = a.value('d').unwrap_or("\t").to_string();
            let spec = a.value('f').or(a.value('c')).unwrap_or("1").to_string();
            let mut picks = Vec::new();
            for part in spec.split(',') {
                match part.split_once('-') {
                    Some((x, y)) => picks.push((x.parse().unwrap_or(1), y.parse().unwrap_or(usize::MAX))),
                    None => {
                        let v = part.parse().unwrap_or(1);
                        picks.push((v, v));
                    }
                }
            }
            let chosen = |i: usize| picks.iter().any(|&(x, y)| i >= x && i <= y);
            for line in lines_of(&a.rest)? {
                let res: Vec<String> = if a.value('c').is_some() {
                    line.chars().enumerate().filter(|(i, _)| chosen(i + 1)).map(|(_, c)| c.to_string()).collect()
                } else {
                    line.split(delim.as_str()).enumerate().filter(|(i, _)| chosen(i + 1)).map(|(_, f)| f.to_string()).collect()
                };
                writeln!(out, "{}", res.join(if a.value('c').is_some() { "" } else { &delim })).ok();
            }
            Ok(0)
        }
        "date" => {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            if args.first().map(|s| s.as_str()) == Some("+%s") {
                writeln!(out, "{now}").ok();
            } else {
                writeln!(out, "{} UTC", date(now)).ok();
            }
            Ok(0)
        }
        "yes" => {
            let word = if args.is_empty() { "y".to_string() } else { args.join(" ") };
            loop {
                if writeln!(out, "{word}").is_err() {
                    return Ok(0);
                }
            }
        }
        "stat" => {
            for f in Args::parse(args, "").rest {
                let m = fs::symlink_metadata(&f).map_err(|e| err(e, &f))?;
                let kind = if m.is_dir() { "directory" } else if m.file_type().is_symlink() { "symbolic link" } else { "regular file" };
                writeln!(out, "  File: {f}\n  Size: {}\t{kind}\nModify: {} UTC", m.len(), date(mtime(&m))).ok();
            }
            Ok(0)
        }
        "ln" => {
            let a = Args::parse(args, "");
            if !a.has('s') {
                return Err("ln: only symbolic links (-s) are supported".into());
            }
            let (t, l) = match a.rest.as_slice() {
                [t, l] => (t.clone(), target(t, l)),
                [t] => (t.clone(), PathBuf::from(Path::new(t).file_name().unwrap_or_default())),
                _ => return Err("usage: ln -s TARGET LINK".into()),
            };
            if a.has('f') {
                let _ = fs::remove_file(&l);
            }
            symlink(&t, &l).map_err(|e| err(e, &l.to_string_lossy()))?;
            Ok(0)
        }
        "readlink" | "realpath" => {
            let a = Args::parse(args, "");
            for f in &a.rest {
                let p = if name == "realpath" || a.has('f') {
                    let abs = if f.starts_with('/') { PathBuf::from(f) } else { env::current_dir().unwrap_or_default().join(f) };
                    fs::canonicalize(&abs).unwrap_or(abs)
                } else {
                    fs::read_link(f).map_err(|e| err(e, f))?
                };
                writeln!(out, "{}", p.display()).ok();
            }
            Ok(0)
        }
        "clear" => {
            write!(out, "\x1b[H\x1b[2J\x1b[3J").ok();
            Ok(0)
        }
        "uname" => {
            let a = Args::parse(args, "");
            writeln!(out, "{}", if a.has('a') { "Evolve evolve 0.1 wasm32 WASI" } else { "Evolve" }).ok();
            Ok(0)
        }
        "whoami" => {
            writeln!(out, "user").ok();
            Ok(0)
        }
        "id" => {
            writeln!(out, "uid=1000(user) gid=1000(user) groups=1000(user)").ok();
            Ok(0)
        }
        "hostname" => {
            writeln!(out, "evolve").ok();
            Ok(0)
        }
        "nproc" => {
            writeln!(out, "1").ok();
            Ok(0)
        }
        "df" => {
            writeln!(out, "Filesystem  Mounted on\nmemory      /").ok();
            Ok(0)
        }
        "du" => {
            let a = Args::parse(args, "");
            let roots = if a.rest.is_empty() { vec![".".to_string()] } else { a.rest.clone() };
            for r in roots {
                let mut total = 0u64;
                walk(Path::new(&r), 0, usize::MAX, &mut |p, _| {
                    if let Ok(m) = fs::symlink_metadata(p) {
                        if m.is_file() {
                            total += m.len();
                        }
                    }
                    true
                });
                let size = if a.has('h') { human(total) } else { total.div_ceil(1024).to_string() };
                writeln!(out, "{size}\t{r}").ok();
            }
            Ok(0)
        }
        "diff" => {
            let a = Args::parse(args, "");
            let [x, y] = a.rest.as_slice() else { return Err("usage: diff FILE1 FILE2".into()) };
            let (la, lb) = (lines_of(std::slice::from_ref(x))?, lines_of(std::slice::from_ref(y))?);
            Ok(diff(&la, &lb, (x, y), out))
        }
        "test" | "[" => {
            let mut v = args.to_vec();
            if name == "[" {
                if v.last().map(|s| s.as_str()) != Some("]") {
                    return Err("[: missing ']'".into());
                }
                v.pop();
            }
            Ok(if test(&v) { 0 } else { 1 })
        }
        "printf" => {
            write!(out, "{}", printf(args)).ok();
            Ok(0)
        }
        "nl" => {
            for (i, l) in lines_of(&Args::parse(args, "").rest)?.iter().enumerate() {
                writeln!(out, "{:6}\t{l}", i + 1).ok();
            }
            Ok(0)
        }
        "rev" => {
            for l in lines_of(&Args::parse(args, "").rest)? {
                writeln!(out, "{}", l.chars().rev().collect::<String>()).ok();
            }
            Ok(0)
        }
        "xxd" => {
            let d = read_input(args.first().map(|s| s.as_str()).unwrap_or("-"))?;
            for (i, chunk) in d.chunks(16).enumerate() {
                let hex: Vec<String> = chunk.chunks(2).map(|p| p.iter().map(|b| format!("{b:02x}")).collect()).collect();
                let text: String = chunk.iter().map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '.' }).collect();
                writeln!(out, "{:08x}: {:<40} {text}", i * 16, hex.join(" ")).ok();
            }
            Ok(0)
        }
        "file" => {
            for f in &Args::parse(args, "").rest {
                let d = read_input(f).unwrap_or_default();
                let kind = if Path::new(f).is_dir() {
                    "directory"
                } else if d.starts_with(b"\0asm") {
                    "WebAssembly (wasm) binary module"
                } else if d.starts_with(b"#!") {
                    "script, ASCII text executable"
                } else if d.iter().take(4096).any(|&b| b == 0) {
                    "data"
                } else if d.is_empty() {
                    "empty"
                } else {
                    "text"
                };
                writeln!(out, "{f}: {kind}").ok();
            }
            Ok(0)
        }
        _ => Err(format!("{name}: unknown utility")),
    }
}

fn main() {
    if let Ok(p) = env::var("PWD") {
        let _ = env::set_current_dir(p);
    }
    let argv: Vec<String> = env::args().collect();
    let name = argv.first().map(|a| Path::new(a).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
    // `utils NAME ARGS` also works, for the programs list.
    let (name, args) = if name == "utils" || name.ends_with(".wasm") {
        match argv.get(1) {
            Some(n) => (n.clone(), &argv[2..]),
            None => {
                println!("{}", NAMES.join(" "));
                return;
            }
        }
    } else {
        (name, &argv[1..])
    };
    let code = match run(&name, args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    };
    let _ = io::stdout().flush();
    std::process::exit(code);
}

