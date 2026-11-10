//! No WASI imports, processes, host filesystem, or sockets. All state is virtual.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
const MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_FILES: usize = 128;
const MAX_OUTPUT: usize = 128 * 1024;
#[derive(Serialize, Deserialize, Clone)]
struct State {
    files: BTreeMap<String, String>,
    dirs: BTreeSet<String>,
    cwd: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Command {
    pub args: Vec<String>,
    pub output: Option<String>,
    pub append: bool,
    pub input: Option<String>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Job {
    pub pipeline: Vec<Command>,
    pub condition: String,
}
#[derive(Serialize)]
struct Output {
    stdout: String,
    stderr: String,
    code: i32,
    cwd: String,
    changed: bool,
}
fn bounded(s: String) -> String {
    s.chars()
        .scan(0, |n, c| {
            *n += c.len_utf8();
            (*n <= MAX_OUTPUT).then_some(c)
        })
        .collect()
}
fn error(e: &str) -> String {
    serde_json::json!({"error":e}).to_string()
}
fn name(input: &str, cwd: &str) -> Result<String, String> {
    if input.contains('\0') || input.len() > 1024 {
        return Err("Invalid path".into());
    }
    let mut bits: Vec<&str> = if input.starts_with('/') {
        vec![]
    } else {
        cwd.split('/').filter(|s| !s.is_empty()).collect()
    };
    for s in input.split('/') {
        match s {
            "" | "." => {}
            ".." => {
                if bits.len() <= 1 {
                    return Err("Path escapes /workspace".into());
                }
                bits.pop();
            }
            _ => bits.push(s),
        }
    }
    if bits.first() != Some(&"workspace") {
        return Err("Only /workspace paths are available".into());
    }
    Ok(format!("/{}", bits.join("/")))
}
fn parent(p: &str) -> &str {
    p.rsplit_once('/').map(|(a, _)| a).unwrap_or("")
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Shell {
    state: State,
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl Shell {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Shell {
        Shell {
            state: State {
                files: BTreeMap::new(),
                dirs: BTreeSet::from(["/workspace".into()]),
                cwd: "/workspace".into(),
            },
        }
    }
    pub fn snapshot(&self) -> String {
        serde_json::to_string(&self.state).unwrap()
    }
    pub fn restore(&mut self, json: &str) -> Result<(), String> {
        if json.len() > MAX_BYTES * 6 + MAX_FILES * 8192 {
            return Err("Workspace too large".into());
        }
        let s: State = serde_json::from_str(json).map_err(|_| "Invalid workspace")?;
        Self::check(&s)?;
        self.state = s;
        Ok(())
    }
    pub fn read(&self, path: &str) -> Result<String, String> {
        let p = name(path, &self.state.cwd)?;
        self.state
            .files
            .get(&p)
            .cloned()
            .ok_or("File not found".into())
    }
    pub fn write(&mut self, path: &str, text: &str, append: bool) -> Result<(), String> {
        if text.len() > MAX_BYTES {
            return Err("File too large".into());
        }
        let p = name(path, &self.state.cwd)?;
        if !self.state.dirs.contains(parent(&p)) {
            return Err("Parent directory not found".into());
        }
        if self.state.dirs.contains(&p) {
            return Err("Path is a directory".into());
        }
        let previous = self.state.files.get(&p).cloned();
        let value = if append {
            previous.as_deref().unwrap_or("").to_owned() + text
        } else {
            text.into()
        };
        self.state.files.insert(p.clone(), value);
        if let Err(e) = Self::check(&self.state) {
            match previous {
                Some(v) => {
                    self.state.files.insert(p, v);
                }
                None => {
                    self.state.files.remove(&p);
                }
            }
            return Err(e);
        }
        Ok(())
    }
    pub fn plan(&self, line: &str) -> String {
        match parse(line) {
            Ok(p) => serde_json::to_string(&p).unwrap(),
            Err(e) => error(&e),
        }
    }
    pub fn run(&mut self, args: &str, stdin: &str) -> String {
        let args: Vec<String> = match serde_json::from_str(args) {
            Ok(v) => v,
            Err(_) => return error("Invalid arguments"),
        };
        if args.is_empty() {
            return serde_json::json!({"stdout":"","stderr":"","code":0,"cwd":self.state.cwd,"changed":false}).to_string();
        }
        if args.len() > 256 || args.iter().map(|a| a.len()).sum::<usize>() > 32768 {
            return error("Arguments exceed limits");
        }
        let mutating = matches!(
            args[0].as_str(),
            "cd" | "mkdir" | "touch" | "cp" | "mv" | "rm" | "tee"
        );
        let old = mutating.then(|| self.state.clone());
        let result = if stdin.len() > MAX_BYTES {
            Err("Input too large".into())
        } else {
            self.builtin(&args, stdin)
        };
        let (out, err, code) = match result {
            Ok(v) => {
                let code = if matches!(args[0].as_str(), "grep" | "rg") && v.is_empty() {
                    1
                } else {
                    0
                };
                (v, String::new(), code)
            }
            Err(e) => {
                if let Some(previous) = old {
                    self.state = previous
                };
                (String::new(), e, 1)
            }
        };
        serde_json::to_string(&Output {
            stdout: bounded(out),
            stderr: err,
            code,
            cwd: self.state.cwd.clone(),
            changed: mutating && code == 0,
        })
        .unwrap()
    }
}
impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}
impl Shell {
    fn check(s: &State) -> Result<(), String> {
        if s.files.len() > MAX_FILES || s.dirs.len() > MAX_FILES {
            return Err("Workspace entry limit exceeded".into());
        }
        if s.files
            .iter()
            .map(|(p, c)| p.len() + c.len())
            .sum::<usize>()
            > MAX_BYTES
        {
            return Err("Workspace exceeds 4 MiB".into());
        }
        if !s.dirs.contains("/workspace") || !s.dirs.contains(&s.cwd) {
            return Err("Invalid current directory".into());
        }
        for p in s.files.keys().chain(s.dirs.iter()) {
            if name(p, "/workspace")? != *p {
                return Err("Noncanonical workspace path".into());
            }
            if p != "/workspace" && !s.dirs.contains(parent(p)) {
                return Err("Missing parent directory".into());
            }
            if s.files.contains_key(p) && s.dirs.contains(p) {
                return Err("Conflicting workspace path".into());
            }
        }
        Ok(())
    }
    fn builtin(&mut self, a: &[String], stdin: &str) -> Result<String, String> {
        let cmd = a.first().map(|s| s.as_str()).unwrap_or("");
        let rest = &a[1..];
        let source = |paths: &[String], this: &Self| -> Result<String, String> {
            if paths.is_empty() {
                Ok(stdin.into())
            } else {
                let mut s = String::new();
                for p in paths {
                    let part = this.read(p)?;
                    if s.len() + part.len() > MAX_BYTES {
                        return Err("Combined input exceeds 4 MiB".into());
                    }
                    s.push_str(&part)
                }
                Ok(s)
            }
        };
        match cmd {
   "pwd"=>Ok(format!("{}\n",self.state.cwd)),
   "echo"=>Ok(rest.join(" ")+"\n"),
   "printf"=>{let fmt=rest.first().map(|s|s.replace("\\n","\n").replace("\\t","\t")).unwrap_or_default();let mut out=fmt;for value in rest.iter().skip(1){out=out.replacen("%s",value,1)}Ok(out)},
   "true"=>Ok(String::new()),"false"=>Err("Command returned false".into()),
   "cd"=>{let p=name(rest.first().map(|s|s.as_str()).unwrap_or("/workspace"),&self.state.cwd)?;if !self.state.dirs.contains(&p){return Err("Directory not found".into())}self.state.cwd=p;Ok(String::new())},
   "ls"|"find"=>{let p=name(rest.first().map(|s|s.as_str()).unwrap_or("."),&self.state.cwd)?;if self.state.files.contains_key(&p){return Ok(p+"\n")}
    if !self.state.dirs.contains(&p){return Err("Directory not found".into())}let prefix=p+"/";
    let list:BTreeSet<String>=self.state.files.keys().chain(self.state.dirs.iter()).filter(|s|s.starts_with(&prefix)).map(|s|if cmd=="find"{s.clone()}else{s[prefix.len()..].split('/').next().unwrap().into()}).collect();Ok(list.into_iter().collect::<Vec<_>>().join("\n")+"\n")},
   "mkdir"=>{let recursive=rest.first().map(|s|s=="-p").unwrap_or(false);for path in rest.iter().filter(|s|*s!="-p") {let p=name(path,&self.state.cwd)?;
    if recursive {let mut current=String::new();for bit in p.split('/').filter(|s|!s.is_empty()){current.push('/');current.push_str(bit);if self.state.files.contains_key(&current){return Err("Path is a file".into())}self.state.dirs.insert(current.clone());}}
    else {if !self.state.dirs.contains(parent(&p)){return Err("Parent directory not found".into())}if self.state.files.contains_key(&p){return Err("Path is a file".into())}self.state.dirs.insert(p);}}
    Self::check(&self.state)?;Ok(String::new())},
   "touch"=>{for p in rest{if self.read(p).is_err(){self.write(p,"",false)?}}Ok(String::new())},
   "cat"=>source(rest,self),
   "head"|"tail"=>{let mut n=10;let mut index=0;if rest.first().map(|s|s=="-n").unwrap_or(false){n=rest.get(1).ok_or("Expected line count")?.parse::<usize>().map_err(|_|"Invalid line count")?.min(10000);index=2}
    let text=source(&rest[index..],self)?;let lines:Vec<_>=text.lines().collect();let result=if cmd=="head"{lines.iter().take(n).copied().collect::<Vec<_>>()}else{lines.iter().skip(lines.len().saturating_sub(n)).copied().collect()};Ok(result.join("\n")+if result.is_empty(){""}else{"\n"})},
   "grep"|"rg"=>{let insensitive=rest.first().map(|s|s=="-i").unwrap_or(false);let index=if insensitive{1}else{0};let pattern=rest.get(index).ok_or("Expected a pattern")?;if pattern.len()>1024{return Err("Pattern too large".into())}
    let re=regex::RegexBuilder::new(pattern).case_insensitive(insensitive).size_limit(1024*1024).build().map_err(|_|"Invalid regular expression")?;
    let text=source(&rest[index+1..],self)?;let lines:Vec<_>=text.lines().filter(|line|re.is_match(line)).collect();Ok(lines.join("\n")+if lines.is_empty(){""}else{"\n"})},
   "wc"=>{let flags=rest.first().filter(|s|s.starts_with('-'));let text=source(&rest[usize::from(flags.is_some())..],self)?;Ok(match flags.map(|s|s.as_str()) {Some("-l")=>text.bytes().filter(|b|*b==b'\n').count().to_string(),Some("-w")=>text.split_whitespace().count().to_string(),Some("-c")=>text.len().to_string(),None=>format!("{} {} {}",text.bytes().filter(|b|*b==b'\n').count(),text.split_whitespace().count(),text.len()),_=>return Err("Use wc [-l|-w|-c] [file]".into())}+"\n")},
   "sort"|"uniq"=>{let text=source(rest,self)?;let mut lines:Vec<_>=text.lines().collect();if cmd=="sort"{lines.sort()}else{lines.dedup()}Ok(lines.join("\n")+if lines.is_empty(){""}else{"\n"})},
   "cp"|"mv"=>{if rest.len()!=2{return Err("Expected source and destination".into())}let text=self.read(&rest[0])?;self.write(&rest[1],&text,false)?;if cmd=="mv"{let p=name(&rest[0],&self.state.cwd)?;if p!=name(&rest[1],&self.state.cwd)?{self.state.files.remove(&p);}}Ok(String::new())},
   "rm"=>{for file in rest{let p=name(file,&self.state.cwd)?;if self.state.files.remove(&p).is_none(){return Err("rm only removes existing files; directories and flags are unsupported".into())}}Ok(String::new())},
   "tee"=>{for file in rest{self.write(file,stdin,false)?}Ok(stdin.into())},
   "help"=>Ok("Builtins: pwd cd ls find mkdir touch cat echo printf head tail grep rg wc sort uniq cp mv rm tee true false. Pipelines, quotes, ; && || and file redirection are supported. curl/search run through the browser's explicit Web setting. This is a bounded shell subset, not GNU Bash; no scripts, substitution, host processes, or environment access.\n".into()),
   _=>Err(format!("Unsupported command: {cmd}. Run help for supported commands.")),
  }
    }
}
// Tokenize without evaluating JavaScript, environment variables or command substitution.
pub fn parse(line: &str) -> Result<Vec<Job>, String> {
    if line.contains('\0') {
        return Err("NUL is unsupported".into());
    }
    if line.len() > 16384 {
        return Err("Command exceeds 16 KiB".into());
    }
    let mut tokens: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut quote = None;
    let mut word = false;
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        if let Some(q) = quote {
            if c == q {
                quote = None
            } else if c == '\\' && q == '"' {
                let next = it.next().ok_or("Unfinished escape")?;
                if next == '"' || next == '\\' {
                    buf.push(next)
                } else {
                    buf.push('\\');
                    buf.push(next)
                }
            } else {
                buf.push(c)
            }
            word = true;
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                word = true
            }
            '\\' => {
                buf.push(it.next().ok_or("Unfinished escape")?);
                word = true
            }
            '$' | '`' => return Err("Substitution and environment access are not supported".into()),
            c if c.is_whitespace() => {
                if word {
                    tokens.push(std::mem::take(&mut buf));
                    word = false
                }
            }
            '|' | '&' | ';' | '>' | '<' => {
                if word {
                    tokens.push(std::mem::take(&mut buf));
                    word = false
                }
                let mut op = c.to_string();
                if it.peek() == Some(&c) && c != ';' && c != '<' {
                    op.push(it.next().unwrap())
                }
                tokens.push(format!("\0{op}"));
            }
            _ => {
                buf.push(c);
                word = true
            }
        }
    }
    if quote.is_some() {
        return Err("Unclosed quote".into());
    }
    if word {
        tokens.push(buf)
    }
    if tokens
        .last()
        .map(|s| ["\0&&", "\0||", "\0|"].contains(&s.as_str()))
        .unwrap_or(false)
    {
        return Err("Missing final command".into());
    }
    let mut jobs = Vec::new();
    let mut pipeline = Vec::new();
    let mut command = Command {
        args: vec![],
        output: None,
        input: None,
        append: false,
    };
    let mut condition = "always".to_owned();
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if t.starts_with('\0') {
            match &t[1..] {
                ">" | ">>" | "<" => {
                    let path = tokens.get(i + 1).ok_or("Missing redirection path")?;
                    if path.starts_with('\0') {
                        return Err("Missing redirection path".into());
                    }
                    if t == "\0<" {
                        command.input = Some(path.clone())
                    } else {
                        command.output = Some(path.clone());
                        command.append = t == "\0>>"
                    }
                    i += 1
                }
                "|" | ";" | "&&" | "||" => {
                    if command.args.is_empty() {
                        return Err("Empty command".into());
                    }
                    pipeline.push(command);
                    command = Command {
                        args: vec![],
                        output: None,
                        input: None,
                        append: false,
                    };
                    if t != "\0|" {
                        jobs.push(Job {
                            pipeline: std::mem::take(&mut pipeline),
                            condition,
                        });
                        condition = match &t[1..] {
                            "&&" => "success",
                            "||" => "failure",
                            _ => "always",
                        }
                        .into();
                    }
                }
                _ => return Err("Background processes are unsupported".into()),
            }
        } else {
            command.args.push(t.clone())
        }
        i += 1;
    }
    if !command.args.is_empty() {
        pipeline.push(command)
    } else if !pipeline.is_empty() {
        return Err("Empty pipeline command".into());
    }
    if !pipeline.is_empty() {
        jobs.push(Job {
            pipeline,
            condition,
        })
    }
    if jobs.iter().map(|j| j.pipeline.len()).sum::<usize>() > 32 {
        return Err("At most 32 commands per invocation".into());
    }
    Ok(jobs)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quoting_and_pipelines() {
        let p = parse("printf 'a\\nb\\n' | grep b > result.txt && cat result.txt").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].pipeline.len(), 2);
        assert_eq!(p[0].pipeline[0].args[1], "a\\nb\\n");
        assert_eq!(p[0].pipeline[1].output.as_deref(), Some("result.txt"));
        assert_eq!(p[1].condition, "success")
    }
    #[test]
    fn paths_are_confined() {
        let mut s = Shell::new();
        assert!(s.write("../secret", "x", false).is_err());
        assert!(s.read("/etc/passwd").is_err());
        s.write("a", "hello", false).unwrap();
        assert_eq!(s.read("/workspace/a").unwrap(), "hello");
        assert!(s.restore(r#"{"files":{"/workspace/../etc/passwd":"x"},"dirs":["/workspace"],"cwd":"/workspace"}"#).is_err())
    }
    #[test]
    fn writes_are_atomic_at_quota() {
        let mut s = Shell::new();
        s.write("a", "before", false).unwrap();
        assert!(s.write("a", &"x".repeat(MAX_BYTES), false).is_err());
        assert_eq!(s.read("a").unwrap(), "before")
    }
    #[test]
    fn useful_utilities() {
        let mut s = Shell::new();
        let r: serde_json::Value =
            serde_json::from_str(&s.run(r#"["sort"]"#, "b\na\na\n")).unwrap();
        assert_eq!(r["stdout"], "a\na\nb\n");
        assert!(parse("echo $(whoami)").is_err());
        assert!(parse("cat a | ").is_err());
        assert!(parse("echo 'incomplete").is_err());
    }
}
