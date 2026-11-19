//! `gh` and `git` in the Evolve VM: work on GitHub repositories through GitHub's API.
//!
//! The VM has no sockets, so this is not Git's protocol: `clone` lists a commit's tree through
//! the API and downloads each file (checked against GitHub's own blob hash); `status`, `diff`
//! and `commit` work offline against the files as cloned; `push` creates the same blobs, trees
//! and commits through the Git Data API and moves a branch; `gh pr` and `gh run` open pull
//! requests and read CI results. Requests go through the host (evolve.http_fetch), which adds the
//! person's token for GitHub and enforces which hosts are reachable; pushing and opening pull
//! requests follow their autonomy setting (evolve.policy, evolve.ask_user).
//!
//! State lives in `.evolve-git/` at the top of a clone: config.json, head.json (path -> blob
//! hash of the last commit), objects/ (file contents by hash) and log.json (commits not pushed).

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[link(wasm_import_module = "evolve")]
extern "C" {
    fn http_fetch(req: *const u8, req_len: usize, resp_len: *mut u32) -> u32;
    fn http_read(buf: *mut u8, len: usize) -> u32;
    fn ask_user(q: *const u8, q_len: usize, answer: *mut u32) -> u32;
    fn policy(level: *mut u32) -> u32;
}

const API: &str = "https://api.github.com";
const STATE: &str = ".evolve-git";
const MAX_FILE: u64 = 20 << 20;

type R<T> = Result<T, String>;

// ---- the host bridge ----

/// Several requests at once (the host runs them in parallel): [(status, body)].
fn fetch_many(reqs: &[Value]) -> R<Vec<(u16, Vec<u8>)>> {
    let body = serde_json::to_vec(&Value::Array(reqs.to_vec())).unwrap();
    let mut len = 0u32;
    let e = unsafe { http_fetch(body.as_ptr(), body.len(), &mut len) };
    if e != 0 {
        return Err(format!("network request failed (error {e})"));
    }
    let mut buf = vec![0u8; len as usize];
    unsafe { http_read(buf.as_mut_ptr(), buf.len()) };
    let v: Value = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in v.as_array().ok_or("bad response")? {
        if let Some(err) = r["error"].as_str() {
            return Err(err.to_string());
        }
        let status = r["status"].as_u64().unwrap_or(0) as u16;
        let data = B64.decode(r["body"].as_str().unwrap_or("")).map_err(|e| e.to_string())?;
        out.push((status, data));
    }
    Ok(out)
}

fn request(method: &str, url: &str, body: Option<Value>) -> Value {
    let mut r = json!({ "method": method, "url": url, "headers": { "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28" } });
    if let Some(b) = body {
        r["body"] = Value::String(B64.encode(serde_json::to_vec(&b).unwrap()));
        r["headers"]["Content-Type"] = Value::String("application/json".into());
    }
    r
}

/// A GitHub API call; errors carry GitHub's message.
fn api(method: &str, path: &str, body: Option<Value>) -> R<Value> {
    let url = if path.starts_with("http") { path.to_string() } else { format!("{API}{path}") };
    let (status, data) = fetch_many(&[request(method, &url, body)])?.remove(0);
    let v: Value = if data.is_empty() { Value::Null } else { serde_json::from_slice(&data).unwrap_or(Value::String(String::from_utf8_lossy(&data).into_owned())) };
    if !(200..300).contains(&status) {
        let msg = v["message"].as_str().map(String::from).unwrap_or_else(|| format!("{v}"));
        let hint = match status {
            401 => " (add a GitHub token in Settings)",
            403 if msg.contains("rate limit") => " (GitHub's limit for requests without a token; add one in Settings)",
            404 => " (not found, or private without a token)",
            _ => "",
        };
        return Err(format!("GitHub {status}: {msg}{hint}"));
    }
    Ok(v)
}

fn level() -> u32 {
    let mut l = 1;
    unsafe { policy(&mut l) };
    l
}

/// Ask the person (on the page or their phone) to allow something.
fn ask(question: &str) -> bool {
    let mut a = 0u32;
    let e = unsafe { ask_user(question.as_ptr(), question.len(), &mut a) };
    e == 0 && a == 1
}

/// Whether an action that leaves this device may go ahead under the autonomy setting.
fn allowed(action: &str, to_default_branch: bool) -> R<()> {
    match level() {
        0 => Err(format!("{action}: not allowed (autonomy: local commits only; change it in Settings)")),
        2 if !to_default_branch => Ok(()),
        _ => {
            if ask(&format!("{action}?")) {
                Ok(())
            } else {
                Err(format!("{action}: declined"))
            }
        }
    }
}

// ---- local state ----

/// The first seven characters of a hash, as Git shows them.
fn short(s: &str) -> &str {
    &s[..s.len().min(7)]
}

fn blob_sha(data: &[u8]) -> String {
    let mut h = sha1_smol::Sha1::new();
    h.update(format!("blob {}\0", data.len()).as_bytes());
    h.update(data);
    h.digest().to_string()
}

struct Repo {
    root: PathBuf,
    config: Value,
}

impl Repo {
    /// The clone around the working directory.
    fn find() -> R<Repo> {
        let mut dir = env::current_dir().map_err(|e| e.to_string())?;
        loop {
            let c = dir.join(STATE).join("config.json");
            if let Ok(text) = fs::read_to_string(&c) {
                return Ok(Repo { root: dir, config: serde_json::from_str(&text).map_err(|e| e.to_string())? });
            }
            if !dir.pop() {
                return Err("not in a repository cloned with `gh repo clone` (see `gh help`)".into());
            }
        }
    }
    fn state(&self, name: &str) -> PathBuf {
        self.root.join(STATE).join(name)
    }
    fn owner_repo(&self) -> String {
        format!("{}/{}", self.config["owner"].as_str().unwrap_or(""), self.config["repo"].as_str().unwrap_or(""))
    }
    fn branch(&self) -> String {
        self.config["branch"].as_str().unwrap_or("main").to_string()
    }
    fn default_branch(&self) -> String {
        self.config["default_branch"].as_str().unwrap_or("main").to_string()
    }
    fn save_config(&self) -> R<()> {
        fs::write(self.state("config.json"), serde_json::to_string_pretty(&self.config).unwrap()).map_err(|e| e.to_string())
    }
    fn head(&self) -> BTreeMap<String, String> {
        fs::read_to_string(self.state("head.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }
    fn save_head(&self, h: &BTreeMap<String, String>) -> R<()> {
        fs::write(self.state("head.json"), serde_json::to_string(h).unwrap()).map_err(|e| e.to_string())
    }
    fn log(&self) -> Vec<Value> {
        fs::read_to_string(self.state("log.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }
    fn save_log(&self, l: &[Value]) -> R<()> {
        fs::write(self.state("log.json"), serde_json::to_string(l).unwrap()).map_err(|e| e.to_string())
    }
    fn object(&self, sha: &str) -> Option<Vec<u8>> {
        fs::read(self.state("objects").join(sha)).ok()
    }
    fn store(&self, data: &[u8]) -> R<String> {
        let sha = blob_sha(data);
        let p = self.state("objects").join(&sha);
        if !p.exists() {
            fs::write(p, data).map_err(|e| e.to_string())?;
        }
        Ok(sha)
    }
    /// Every file of the working tree: path -> content hash.
    fn work(&self) -> R<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        let mut stack = vec![self.root.clone()];
        while let Some(d) = stack.pop() {
            for e in fs::read_dir(&d).map_err(|e| e.to_string())?.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name == STATE || name == ".git" {
                    continue;
                }
                let p = e.path();
                let meta = fs::symlink_metadata(&p).map_err(|e| e.to_string())?;
                let rel = p.strip_prefix(&self.root).unwrap().to_string_lossy().into_owned();
                if meta.is_dir() {
                    stack.push(p);
                } else if meta.file_type().is_symlink() {
                    let t = fs::read_link(&p).map(|t| t.to_string_lossy().into_owned()).unwrap_or_default();
                    out.insert(rel, blob_sha(t.as_bytes()));
                } else {
                    out.insert(rel, blob_sha(&fs::read(&p).map_err(|e| e.to_string())?));
                }
            }
        }
        Ok(out)
    }
}

/// (added, modified, deleted) between two path -> hash maps.
fn changes(from: &BTreeMap<String, String>, to: &BTreeMap<String, String>) -> (Vec<String>, Vec<String>, Vec<String>) {
    let added = to.keys().filter(|k| !from.contains_key(*k)).cloned().collect();
    let modified = to.iter().filter(|(k, v)| from.get(*k).is_some_and(|f| f != *v)).map(|(k, _)| k.clone()).collect();
    let deleted = from.keys().filter(|k| !to.contains_key(*k)).cloned().collect();
    (added, modified, deleted)
}

fn url_path(p: &str) -> String {
    p.split('/').map(|s| s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect::<String>()).collect::<Vec<_>>().join("/")
}

/// Download blobs (verifying their hashes) into `root` and the object store.
fn download(owner: &str, repo: &str, commit: &str, private: bool, entries: &[(String, String, String)], root: &Path, objects: &Path) -> R<(usize, Vec<String>)> {
    let mut skipped = Vec::new();
    let mut done = 0;
    for chunk in entries.chunks(16) {
        let reqs: Vec<Value> = chunk
            .iter()
            .map(|(path, sha, _)| {
                if private {
                    request("GET", &format!("{API}/repos/{owner}/{repo}/git/blobs/{sha}"), None)
                } else {
                    json!({ "method": "GET", "url": format!("https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{}", url_path(path)) })
                }
            })
            .collect();
        let res = fetch_many(&reqs)?;
        for ((path, sha, mode), (status, body)) in chunk.iter().zip(res) {
            if status != 200 {
                skipped.push(format!("{path} (HTTP {status})"));
                continue;
            }
            let data = if private {
                let v: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
                B64.decode(v["content"].as_str().unwrap_or("").replace('\n', "")).map_err(|e| e.to_string())?
            } else {
                body
            };
            if blob_sha(&data) != *sha {
                return Err(format!("{path}: content does not match GitHub's hash; not cloned"));
            }
            fs::write(objects.join(sha), &data).map_err(|e| e.to_string())?;
            let target = root.join(path);
            if let Some(dir) = target.parent() {
                fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            let _ = fs::remove_file(&target);
            if mode == "120000" {
                symlink(&String::from_utf8_lossy(&data), &target)?;
            } else {
                fs::write(&target, &data).map_err(|e| e.to_string())?;
            }
            done += 1;
        }
        eprint!("\rReceiving files: {done}/{}", entries.len());
    }
    eprintln!();
    Ok((done, skipped))
}

#[link(wasm_import_module = "wasi_snapshot_preview1")]
extern "C" {
    fn path_symlink(old: *const u8, old_len: usize, fd: u32, new: *const u8, new_len: usize) -> u16;
}

fn symlink(target: &str, at: &Path) -> R<()> {
    let abs = if at.is_absolute() { at.to_path_buf() } else { env::current_dir().unwrap_or_default().join(at) };
    let rel = abs.to_string_lossy().trim_start_matches('/').to_string();
    match unsafe { path_symlink(target.as_ptr(), target.len(), 3, rel.as_ptr(), rel.len()) } {
        0 => Ok(()),
        e => Err(format!("cannot create symlink {} (error {e})", at.display())),
    }
}

/// A commit's tree: [(path, blob hash, mode)] of its files, and its tree hash.
fn tree_of(owner: &str, repo: &str, commit: &str) -> R<(Vec<(String, String, String)>, String, Vec<String>)> {
    let c = api("GET", &format!("/repos/{owner}/{repo}/git/commits/{commit}"), None)?;
    let tree = c["tree"]["sha"].as_str().ok_or("commit without a tree")?.to_string();
    let t = api("GET", &format!("/repos/{owner}/{repo}/git/trees/{tree}?recursive=1"), None)?;
    if t["truncated"].as_bool() == Some(true) {
        eprintln!("warning: the repository is too large for one listing; some files are missing");
    }
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    for e in t["tree"].as_array().cloned().unwrap_or_default() {
        let path = e["path"].as_str().unwrap_or("").to_string();
        match e["type"].as_str() {
            Some("blob") if e["size"].as_u64().unwrap_or(0) > MAX_FILE => skipped.push(format!("{path} (larger than 20 MB)")),
            Some("blob") => files.push((path, e["sha"].as_str().unwrap_or("").to_string(), e["mode"].as_str().unwrap_or("100644").to_string())),
            Some("commit") => skipped.push(format!("{path} (submodule)")),
            _ => {}
        }
    }
    Ok((files, tree, skipped))
}

// ---- commands ----

fn clone(args: &[String]) -> R<()> {
    let spec = args.iter().find(|a| !a.starts_with('-')).ok_or("usage: gh repo clone OWNER/REPO [DIR] [--branch NAME]")?;
    let spec = spec.trim_start_matches("https://github.com/").trim_end_matches(".git").to_string();
    let (owner, repo) = spec.split_once('/').ok_or("expected OWNER/REPO")?;
    let dir = args.iter().filter(|a| !a.starts_with('-')).nth(1).cloned().unwrap_or_else(|| repo.to_string());
    let info = api("GET", &format!("/repos/{owner}/{repo}"), None)?;
    let default = info["default_branch"].as_str().unwrap_or("main").to_string();
    let branch = args.iter().position(|a| a == "--branch" || a == "-b").and_then(|i| args.get(i + 1)).cloned().unwrap_or(default.clone());
    let private = info["private"].as_bool().unwrap_or(false);
    let b = api("GET", &format!("/repos/{owner}/{repo}/branches/{branch}"), None)?;
    let commit = b["commit"]["sha"].as_str().ok_or("branch without a commit")?.to_string();
    let root = PathBuf::from(&dir);
    if root.exists() && fs::read_dir(&root).map(|mut d| d.next().is_some()).unwrap_or(false) {
        return Err(format!("{dir} already exists and is not empty"));
    }
    eprintln!("Cloning {owner}/{repo} ({branch} at {})...", short(&commit));
    let (files, tree, mut skipped) = tree_of(owner, repo, &commit)?;
    let objects = root.join(STATE).join("objects");
    fs::create_dir_all(&objects).map_err(|e| e.to_string())?;
    let (n, more) = download(owner, repo, &commit, private, &files, &root, &objects)?;
    skipped.extend(more);
    let head: BTreeMap<String, String> = files.iter().filter(|(p, _, _)| root.join(p).exists() || fs::symlink_metadata(root.join(p)).is_ok()).map(|(p, s, _)| (p.clone(), s.clone())).collect();
    let modes: Map<String, Value> = files.iter().filter(|(_, _, m)| m != "100644").map(|(p, _, m)| (p.clone(), Value::String(m.clone()))).collect();
    let r = Repo { root: root.clone(), config: json!({ "owner": owner, "repo": repo, "private": private, "default_branch": default, "branch": branch, "upstream": { "branch": branch, "commit": commit, "tree": tree }, "modes": modes }) };
    r.save_config()?;
    r.save_head(&head)?;
    r.save_log(&[])?;
    // The tree as cloned: what the first local commit is compared with when pushing.
    fs::write(r.state("upstream.json"), serde_json::to_string(&head).unwrap()).map_err(|e| e.to_string())?;
    println!("Cloned {n} files into {dir}/");
    for s in skipped {
        println!("  skipped {s}");
    }
    Ok(())
}

fn status(r: &Repo) -> R<()> {
    let (a, m, d) = changes(&r.head(), &r.work()?);
    println!("On branch {}", r.branch());
    let ahead = r.log().len();
    if ahead > 0 {
        println!("Your branch is ahead of 'origin/{}' by {ahead} commit{} (not pushed).", r.config["upstream"]["branch"].as_str().unwrap_or(""), if ahead == 1 { "" } else { "s" });
    }
    if a.is_empty() && m.is_empty() && d.is_empty() {
        println!("nothing to commit, working tree clean");
        return Ok(());
    }
    println!("Changes not committed:");
    for p in &m {
        println!("\tmodified:   {p}");
    }
    for p in &a {
        println!("\tnew file:   {p}");
    }
    for p in &d {
        println!("\tdeleted:    {p}");
    }
    Ok(())
}

fn unified(a: &str, b: &str, name: &str, out: &mut dyn Write) {
    let x: Vec<&str> = a.lines().collect();
    let y: Vec<&str> = b.lines().collect();
    let (n, m) = (x.len(), y.len());
    if n * m > 16_000_000 {
        writeln!(out, "Binary or large file {name} differs").ok();
        return;
    }
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if x[i] == y[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && x[i] == y[j] {
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
    writeln!(out, "--- a/{name}\n+++ b/{name}").ok();
    let mut k = 0;
    while k < ops.len() {
        if ops[k].0 == ' ' {
            k += 1;
            continue;
        }
        let start = k.saturating_sub(3);
        let mut end = k;
        while end < ops.len() && (ops[end].0 != ' ' || ops[end..(end + 7).min(ops.len())].iter().any(|o| o.0 != ' ')) {
            end += 1;
        }
        let end = (end + 3).min(ops.len());
        let al = ops[start..end].iter().filter(|o| o.0 != '+').count();
        let bl = ops[start..end].iter().filter(|o| o.0 != '-').count();
        writeln!(out, "@@ -{},{al} +{},{bl} @@", ops[start].1 + 1, ops[start].2 + 1).ok();
        for &(op, i, j) in &ops[start..end] {
            writeln!(out, "{op}{}", if op == '+' { y[j] } else { x[i] }).ok();
        }
        k = end;
    }
}

fn diff(r: &Repo, paths: &[String]) -> R<()> {
    let head = r.head();
    let work = r.work()?;
    let (a, m, d) = changes(&head, &work);
    let out = io::stdout();
    let mut out = out.lock();
    let wanted = |p: &str| paths.is_empty() || paths.iter().any(|q| p == q || p.starts_with(&format!("{}/", q.trim_end_matches('/'))));
    for p in m.iter().chain(&a).chain(&d).filter(|p| wanted(p)) {
        let old = head.get(p).and_then(|s| r.object(s)).unwrap_or_default();
        let new = fs::read(r.root.join(p)).unwrap_or_default();
        unified(&String::from_utf8_lossy(&old), &String::from_utf8_lossy(&new), p, &mut out);
    }
    Ok(())
}

fn commit(r: &Repo, args: &[String]) -> R<()> {
    let msg = args.iter().position(|a| a == "-m" || a == "-am" || a == "--message").and_then(|i| args.get(i + 1)).cloned().ok_or("usage: git commit -m MESSAGE (every change is included)")?;
    let head = r.head();
    let work = r.work()?;
    let (a, m, d) = changes(&head, &work);
    if a.is_empty() && m.is_empty() && d.is_empty() {
        println!("nothing to commit, working tree clean");
        return Ok(());
    }
    for p in a.iter().chain(&m) {
        let full = r.root.join(p);
        let data = match fs::read_link(&full) {
            Ok(t) => t.to_string_lossy().into_owned().into_bytes(),
            Err(_) => fs::read(&full).map_err(|e| e.to_string())?,
        };
        r.store(&data)?;
    }
    let mut log = r.log();
    let id = blob_sha(format!("{msg}{}{}", log.len(), serde_json::to_string(&work).unwrap()).as_bytes());
    log.push(json!({ "id": id, "message": msg, "tree": work }));
    r.save_log(&log)?;
    r.save_head(&work)?;
    println!("[{} {}] {msg}", r.branch(), short(&id));
    println!(" {} file{} changed ({} new, {} modified, {} deleted)", a.len() + m.len() + d.len(), if a.len() + m.len() + d.len() == 1 { "" } else { "s" }, a.len(), m.len(), d.len());
    Ok(())
}

fn push(r: &mut Repo) -> R<()> {
    let log = r.log();
    if log.is_empty() {
        println!("Everything up-to-date");
        return Ok(());
    }
    let (owner, repo) = (r.config["owner"].as_str().unwrap().to_string(), r.config["repo"].as_str().unwrap().to_string());
    let branch = r.branch();
    let to_default = branch == r.default_branch();
    allowed(&format!("Push {} commit{} to {owner}/{repo} branch {branch}", log.len(), if log.len() == 1 { "" } else { "s" }), to_default)?;
    let mut parent = r.config["upstream"]["commit"].as_str().unwrap().to_string();
    let mut prev_tree_sha = r.config["upstream"]["tree"].as_str().unwrap().to_string();
    let mut prev: BTreeMap<String, String> = serde_json::from_value(r.config["upstream"]["files"].clone()).unwrap_or_default();
    if prev.is_empty() {
        // The tree as cloned: head minus the local commits = the first commit's starting point.
        prev = fs::read_to_string(r.state("upstream.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    }
    let modes: Map<String, Value> = r.config["modes"].as_object().cloned().unwrap_or_default();
    for c in &log {
        let tree: BTreeMap<String, String> = serde_json::from_value(c["tree"].clone()).map_err(|e| e.to_string())?;
        let (a, m, d) = changes(&prev, &tree);
        let mut entries = Vec::new();
        for p in a.iter().chain(&m) {
            let sha = &tree[p];
            let data = r.object(sha).ok_or(format!("{p}: committed content is missing"))?;
            let made = api("POST", &format!("/repos/{owner}/{repo}/git/blobs"), Some(json!({ "content": B64.encode(&data), "encoding": "base64" })))?;
            if made["sha"].as_str() != Some(sha.as_str()) {
                return Err(format!("{p}: GitHub stored a different hash"));
            }
            let mode = modes.get(p).and_then(|v| v.as_str()).unwrap_or("100644");
            entries.push(json!({ "path": p, "mode": mode, "type": "blob", "sha": sha }));
        }
        for p in &d {
            entries.push(json!({ "path": p, "mode": "100644", "type": "blob", "sha": Value::Null }));
        }
        let t = api("POST", &format!("/repos/{owner}/{repo}/git/trees"), Some(json!({ "base_tree": prev_tree_sha, "tree": entries })))?;
        prev_tree_sha = t["sha"].as_str().ok_or("no tree")?.to_string();
        let made = api("POST", &format!("/repos/{owner}/{repo}/git/commits"), Some(json!({ "message": c["message"], "tree": prev_tree_sha, "parents": [parent] })))?;
        parent = made["sha"].as_str().ok_or("no commit")?.to_string();
        prev = tree;
    }
    let exists = api("GET", &format!("/repos/{owner}/{repo}/git/ref/heads/{branch}"), None).is_ok();
    if exists {
        api("PATCH", &format!("/repos/{owner}/{repo}/git/refs/heads/{branch}"), Some(json!({ "sha": parent, "force": false })))?;
    } else {
        api("POST", &format!("/repos/{owner}/{repo}/git/refs"), Some(json!({ "ref": format!("refs/heads/{branch}"), "sha": parent })))?;
    }
    println!("To github.com/{owner}/{repo}\n   {} -> {branch}{}", short(&parent), if exists { "" } else { " (new branch)" });
    r.config["upstream"] = json!({ "branch": branch, "commit": parent, "tree": prev_tree_sha });
    fs::write(r.state("upstream.json"), serde_json::to_string(&prev).unwrap()).map_err(|e| e.to_string())?;
    r.save_config()?;
    r.save_log(&[])?;
    Ok(())
}

fn pr(r: &Repo, args: &[String]) -> R<()> {
    let or = r.owner_repo();
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    match args.first().map(|s| s.as_str()) {
        Some("create") => {
            let title = opt("--title").or(opt("-t")).ok_or("usage: gh pr create --title TITLE [--body TEXT] [--base BRANCH]")?;
            let base = opt("--base").unwrap_or(r.default_branch());
            let head = opt("--head").unwrap_or(r.branch());
            if !r.log().is_empty() {
                return Err("push your commits first (git push)".into());
            }
            allowed(&format!("Open a pull request on {or}: {head} into {base}, \"{title}\""), false)?;
            let p = api("POST", &format!("/repos/{or}/pulls"), Some(json!({ "title": title, "body": opt("--body").or(opt("-b")).unwrap_or_default(), "head": head, "base": base })))?;
            println!("{}", p["html_url"].as_str().unwrap_or(""));
        }
        Some("list") | None => {
            for p in api("GET", &format!("/repos/{or}/pulls?state=open&per_page=30"), None)?.as_array().cloned().unwrap_or_default() {
                println!("#{}\t{}\t{} -> {}", p["number"], p["title"].as_str().unwrap_or(""), p["head"]["ref"].as_str().unwrap_or(""), p["base"]["ref"].as_str().unwrap_or(""));
            }
        }
        Some("view") => {
            let n = args.get(1).ok_or("usage: gh pr view NUMBER")?;
            let p = api("GET", &format!("/repos/{or}/pulls/{n}"), None)?;
            println!("#{n} {}\nstate: {}{}\n{} -> {}\n{}\n\n{}", p["title"].as_str().unwrap_or(""), p["state"].as_str().unwrap_or(""), if p["merged"].as_bool() == Some(true) { " (merged)" } else { "" }, p["head"]["ref"].as_str().unwrap_or(""), p["base"]["ref"].as_str().unwrap_or(""), p["html_url"].as_str().unwrap_or(""), p["body"].as_str().unwrap_or(""));
            let sha = p["head"]["sha"].as_str().unwrap_or("");
            for c in api("GET", &format!("/repos/{or}/commits/{sha}/check-runs"), None)?["check_runs"].as_array().cloned().unwrap_or_default() {
                println!("check: {} — {}", c["name"].as_str().unwrap_or(""), c["conclusion"].as_str().or(c["status"].as_str()).unwrap_or(""));
            }
        }
        Some(other) => return Err(format!("gh pr {other}: not supported (create, list, view)")),
    }
    Ok(())
}

fn run_cmd(r: &Repo, args: &[String]) -> R<()> {
    let or = r.owner_repo();
    match args.first().map(|s| s.as_str()) {
        Some("list") | None => {
            let branch = args.iter().position(|a| a == "--branch" || a == "-b").and_then(|i| args.get(i + 1)).cloned();
            let q = branch.map(|b| format!("&branch={b}")).unwrap_or_default();
            for w in api("GET", &format!("/repos/{or}/actions/runs?per_page=10{q}"), None)?["workflow_runs"].as_array().cloned().unwrap_or_default() {
                println!("{}\t{}\t{}\t{}\t{}", w["id"], w["status"].as_str().unwrap_or(""), w["conclusion"].as_str().unwrap_or("-"), w["name"].as_str().unwrap_or(""), w["head_branch"].as_str().unwrap_or(""));
            }
        }
        Some("view") => {
            let id = args.get(1).ok_or("usage: gh run view RUN_ID")?;
            let w = api("GET", &format!("/repos/{or}/actions/runs/{id}"), None)?;
            println!("{} ({}): {} {}", w["name"].as_str().unwrap_or(""), w["head_branch"].as_str().unwrap_or(""), w["status"].as_str().unwrap_or(""), w["conclusion"].as_str().unwrap_or(""));
            for j in api("GET", &format!("/repos/{or}/actions/runs/{id}/jobs"), None)?["jobs"].as_array().cloned().unwrap_or_default() {
                println!("job {}: {}", j["name"].as_str().unwrap_or(""), j["conclusion"].as_str().or(j["status"].as_str()).unwrap_or(""));
                for s in j["steps"].as_array().cloned().unwrap_or_default() {
                    if s["conclusion"].as_str() != Some("success") && s["conclusion"].as_str() != Some("skipped") {
                        println!("  step {}: {}", s["name"].as_str().unwrap_or(""), s["conclusion"].as_str().or(s["status"].as_str()).unwrap_or(""));
                    }
                }
                if j["conclusion"].as_str() == Some("failure") {
                    for a in api("GET", &format!("/repos/{or}/check-runs/{}/annotations", j["id"]), None)?.as_array().cloned().unwrap_or_default() {
                        println!("  {}:{}: {}", a["path"].as_str().unwrap_or(""), a["start_line"], a["message"].as_str().unwrap_or(""));
                    }
                }
            }
        }
        Some(other) => return Err(format!("gh run {other}: not supported (list, view)")),
    }
    Ok(())
}

fn log_cmd(r: &Repo, args: &[String]) -> R<()> {
    let n: usize = args.iter().position(|a| a == "-n").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(10);
    let oneline = args.iter().any(|a| a == "--oneline");
    for c in r.log().iter().rev() {
        let id = c["id"].as_str().unwrap_or("");
        if oneline {
            println!("{} (not pushed) {}", short(&id), c["message"].as_str().unwrap_or(""));
        } else {
            println!("commit {id} (not pushed)\n\n    {}\n", c["message"].as_str().unwrap_or(""));
        }
    }
    let up = r.config["upstream"]["commit"].as_str().unwrap_or("");
    for c in api("GET", &format!("/repos/{}/commits?sha={up}&per_page={n}", r.owner_repo()), None)?.as_array().cloned().unwrap_or_default() {
        let sha = c["sha"].as_str().unwrap_or("");
        let msg = c["commit"]["message"].as_str().unwrap_or("");
        if oneline {
            println!("{} {}", short(&sha), msg.lines().next().unwrap_or(""));
        } else {
            println!("commit {sha}\nAuthor: {}\nDate:   {}\n\n    {}\n", c["commit"]["author"]["name"].as_str().unwrap_or(""), c["commit"]["author"]["date"].as_str().unwrap_or(""), msg.replace('\n', "\n    "));
        }
    }
    Ok(())
}

fn restore(r: &Repo, paths: &[String]) -> R<()> {
    let head = r.head();
    for p in paths {
        let sha = head.get(p).ok_or(format!("{p}: not in the last commit"))?;
        fs::write(r.root.join(p), r.object(sha).ok_or("content missing")?).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn git(args: &[String]) -> R<()> {
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rest = if args.is_empty() { &[][..] } else { &args[1..] };
    if cmd == "clone" {
        return clone(rest);
    }
    if matches!(cmd, "help" | "--help" | "-h") {
        println!("git (Evolve VM, through GitHub's API): clone status diff add commit log branch checkout -b switch -c restore rm push pull remote\nRun `gh help` for pull requests and CI.");
        return Ok(());
    }
    let mut r = Repo::find()?;
    match cmd {
        "status" => status(&r),
        "diff" => diff(&r, &rest.iter().filter(|a| !a.starts_with('-')).cloned().collect::<Vec<_>>()),
        "add" => Ok(()), // Every change is part of the next commit.
        "commit" => commit(&r, rest),
        "log" => log_cmd(&r, rest),
        "push" => push(&mut r),
        "branch" => {
            println!("* {}", r.branch());
            Ok(())
        }
        "checkout" | "switch" if rest.first().is_some_and(|a| a == "-b" || a == "-c") => {
            let name = rest.get(1).ok_or("usage: git checkout -b NAME")?;
            r.config["branch"] = Value::String(name.clone());
            r.save_config()?;
            println!("Switched to a new branch '{name}' (created on GitHub at the next push)");
            Ok(())
        }
        "checkout" | "restore" => restore(&r, &rest.iter().filter(|a| *a != "--" && !a.starts_with('-')).cloned().collect::<Vec<_>>()),
        "rm" => {
            for p in rest.iter().filter(|a| !a.starts_with('-')) {
                fs::remove_file(r.root.join(p)).map_err(|e| format!("{p}: {e}"))?;
            }
            Ok(())
        }
        "remote" => {
            println!("origin\thttps://github.com/{} (through the GitHub API)", r.owner_repo());
            Ok(())
        }
        "pull" => {
            let (a, m, d) = changes(&r.head(), &r.work()?);
            if !(a.is_empty() && m.is_empty() && d.is_empty() && r.log().is_empty()) {
                return Err("commit and push (or restore) your changes before pulling".into());
            }
            let (owner, repo) = (r.config["owner"].as_str().unwrap().to_string(), r.config["repo"].as_str().unwrap().to_string());
            let b = api("GET", &format!("/repos/{owner}/{repo}/branches/{}", r.branch()), None)?;
            let commit = b["commit"]["sha"].as_str().unwrap_or("").to_string();
            if Some(commit.as_str()) == r.config["upstream"]["commit"].as_str() {
                println!("Already up to date.");
                return Ok(());
            }
            let (files, tree, _) = tree_of(&owner, &repo, &commit)?;
            let new: BTreeMap<String, String> = files.iter().map(|(p, s, _)| (p.clone(), s.clone())).collect();
            let (a, m, d) = changes(&r.head(), &new);
            let fetch: Vec<_> = files.iter().filter(|(p, _, _)| a.contains(p) || m.contains(p)).cloned().collect();
            download(&owner, &repo, &commit, r.config["private"].as_bool().unwrap_or(false), &fetch, &r.root, &r.state("objects"))?;
            for p in &d {
                let _ = fs::remove_file(r.root.join(p));
            }
            r.save_head(&new)?;
            fs::write(r.state("upstream.json"), serde_json::to_string(&new).unwrap()).map_err(|e| e.to_string())?;
            r.config["upstream"] = json!({ "branch": r.branch(), "commit": commit, "tree": tree });
            r.save_config()?;
            println!("Updated to {} ({} new, {} changed, {} removed)", short(&commit), a.len(), m.len(), d.len());
            Ok(())
        }
        other => Err(format!("git {other}: not supported here (see `git help`)")),
    }
}

fn gh(args: &[String]) -> R<()> {
    match (args.first().map(|s| s.as_str()), args.get(1).map(|s| s.as_str())) {
        (Some("repo"), Some("clone")) => clone(&args[2..]),
        (Some("repo"), Some("list")) => {
            let path = match args.get(2) {
                Some(owner) => format!("/users/{owner}/repos?per_page=100&sort=pushed"),
                None => "/user/repos?per_page=100&sort=pushed&affiliation=owner".into(),
            };
            for r in api("GET", &path, None)?.as_array().cloned().unwrap_or_default() {
                println!("{}\t{}{}", r["full_name"].as_str().unwrap_or(""), r["description"].as_str().unwrap_or(""), if r["private"].as_bool() == Some(true) { " (private)" } else { "" });
            }
            Ok(())
        }
        (Some("auth"), _) => {
            match api("GET", "/user", None) {
                Ok(u) => println!("Signed in to github.com as {}", u["login"].as_str().unwrap_or("?")),
                Err(_) => println!("Not signed in: public repositories only (add a GitHub token in Settings)"),
            }
            let names = ["local commits only", "ask before pushing", "push and open pull requests freely"];
            println!("Autonomy: {}", names[level().min(2) as usize]);
            Ok(())
        }
        (Some("pr"), _) => pr(&Repo::find()?, &args[1..]),
        (Some("run"), _) => run_cmd(&Repo::find()?, &args[1..]),
        (Some("api"), Some(path)) => {
            let method = args.iter().position(|a| a == "-X").and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("GET");
            if method != "GET" {
                allowed(&format!("GitHub API {method} {path}"), true)?;
            }
            println!("{}", serde_json::to_string_pretty(&api(method, &format!("/{}", path.trim_start_matches('/')), None)?).unwrap());
            Ok(())
        }
        _ => {
            println!("gh (Evolve VM, through GitHub's API):\n  gh repo clone OWNER/REPO [DIR] [--branch B]   gh repo list [OWNER]   gh auth status\n  gh pr create --title T [--body B] [--base B]   gh pr list   gh pr view N\n  gh run list [--branch B]   gh run view ID   gh api PATH\nLocal work: git status, diff, commit -m, log, checkout -b, push, pull (see `git help`).");
            Ok(())
        }
    }
}

fn main() {
    if let Ok(p) = env::var("PWD") {
        let _ = env::set_current_dir(p);
    }
    let argv: Vec<String> = env::args().collect();
    let name = argv.first().map(|a| Path::new(a).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default();
    let r = if name == "git" { git(&argv[1..]) } else { gh(&argv[1..]) };
    if let Err(e) = r {
        eprintln!("{name}: {e}");
        std::process::exit(1);
    }
}
