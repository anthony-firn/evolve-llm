//! `gh` and `git` in the Evolve VM: work on GitHub repositories through GitHub's API.
//!
//! The VM has no sockets, so this is not Git's protocol: `clone` lists a commit's tree through
//! the API and downloads each file (checked against GitHub's own blob hash); `status`, `diff`
//! and `commit` work offline against the files as cloned; `push` creates the same blobs, trees
//! and commits through the Git Data API and moves a branch; `checkout BRANCH` and `pull` replace
//! the files with a branch's on GitHub; `merge` and `gh pr merge` merge on GitHub; `gh pr` and
//! `gh run` open pull requests and read CI results. Requests go through the host
//! (evolve.http_fetch), which adds the person's token for GitHub and enforces which hosts are
//! reachable; pushing and opening pull requests follow their autonomy setting (evolve.policy,
//! evolve.ask_user), and merging always asks them.
//!
//! Messages are written for a small model as much as for a person: every refusal says which
//! command to run instead.
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

/// Whether an action that changes GitHub may go ahead under the autonomy setting. `always_ask`
/// (merges, the default branch, raw API writes) asks even when pushing is allowed freely.
fn allowed(action: &str, always_ask: bool) -> R<()> {
    match level() {
        0 => Err(format!("{action}: not allowed (autonomy: local commits only; change it in Settings)")),
        2 if !always_ask => Ok(()),
        _ => {
            if ask(&format!("{action}?")) {
                Ok(())
            } else {
                Err(format!("{action}: declined by the person, so nothing changed on GitHub. Do not try again; tell the person what you did"))
            }
        }
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
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

/// Switching, pulling and merging replace files, so everything must be committed and pushed.
fn ready_for(r: &Repo, what: &str) -> R<()> {
    let (a, m, d) = changes(&r.head(), &r.work()?);
    let n = a.len() + m.len() + d.len();
    if n > 0 {
        return Err(format!("{what}: {n} changed file{} not committed yet. First run: git commit -m \"MESSAGE\" (or undo with: git restore FILE)", plural(n)));
    }
    let ahead = r.log().len();
    if ahead > 0 {
        return Err(format!("{what}: {ahead} commit{} on {} not pushed yet. First run: git push", plural(ahead), r.branch()));
    }
    Ok(())
}

/// The commit a branch on GitHub points to, or None when GitHub has no such branch.
fn branch_head(or: &str, name: &str) -> R<Option<String>> {
    match api("GET", &format!("/repos/{or}/branches/{}", url_path(name)), None) {
        Ok(b) => Ok(b["commit"]["sha"].as_str().map(String::from)),
        Err(e) if e.starts_with("GitHub 404") => Ok(None),
        Err(e) => Err(e),
    }
}

fn branch_names(or: &str) -> R<Vec<String>> {
    Ok(api("GET", &format!("/repos/{or}/branches?per_page=100"), None)?.as_array().cloned().unwrap_or_default().iter().filter_map(|b| b["name"].as_str().map(String::from)).collect())
}

fn no_branch(or: &str, name: &str) -> String {
    let names = branch_names(or).unwrap_or_default();
    format!("there is no branch {name} on GitHub (branches: {}). To start a new branch, run: git checkout -b {name}", names.join(", "))
}

/// Make the working tree a branch's commit on GitHub: download what changed, remove what went.
/// The tree must be clean (ready_for). Returns (added, modified, deleted).
fn sync_to(r: &mut Repo, branch: &str, commit: &str) -> R<(usize, usize, usize)> {
    let (owner, repo) = (r.config["owner"].as_str().unwrap_or("").to_string(), r.config["repo"].as_str().unwrap_or("").to_string());
    let (files, tree, _) = tree_of(&owner, &repo, commit)?;
    let new: BTreeMap<String, String> = files.iter().map(|(p, s, _)| (p.clone(), s.clone())).collect();
    let (a, m, d) = changes(&r.head(), &new);
    let fetch: Vec<_> = files.iter().filter(|(p, _, _)| a.contains(p) || m.contains(p)).cloned().collect();
    download(&owner, &repo, commit, r.config["private"].as_bool().unwrap_or(false), &fetch, &r.root, &r.state("objects"))?;
    for p in &d {
        let _ = fs::remove_file(r.root.join(p));
    }
    r.save_head(&new)?;
    fs::write(r.state("upstream.json"), serde_json::to_string(&new).unwrap()).map_err(|e| e.to_string())?;
    r.config["branch"] = Value::String(branch.to_string());
    r.config["upstream"] = json!({ "branch": branch, "commit": commit, "tree": tree });
    r.config["modes"] = Value::Object(files.iter().filter(|(_, _, m)| m != "100644").map(|(p, _, m)| (p.clone(), Value::String(m.clone()))).collect());
    r.save_config()?;
    Ok((a.len(), m.len(), d.len()))
}

/// Lines added and removed between two texts (by their longest common subsequence; for very long
/// files, only the change in length).
fn line_stats(a: &[u8], b: &[u8]) -> (usize, usize) {
    let (a, b) = (String::from_utf8_lossy(a), String::from_utf8_lossy(b));
    let x: Vec<&str> = a.lines().collect();
    let y: Vec<&str> = b.lines().collect();
    let (n, m) = (x.len(), y.len());
    if n * m > 16_000_000 {
        return (m.saturating_sub(n), n.saturating_sub(m));
    }
    let mut next = vec![0u32; m + 1];
    for i in (0..n).rev() {
        let mut row = vec![0u32; m + 1];
        for j in (0..m).rev() {
            row[j] = if x[i] == y[j] { next[j + 1] + 1 } else { next[j].max(row[j + 1]) };
        }
        next = row;
    }
    let same = next[0] as usize;
    (m - same, n - same)
}

/// "README.md +1 -311, new.txt +3": what changed between two trees, for a person deciding.
fn change_summary(r: &Repo, from: &BTreeMap<String, String>, to: &BTreeMap<String, String>) -> String {
    let (a, m, d) = changes(from, to);
    let mut parts: Vec<String> = Vec::new();
    for p in &m {
        let (plus, minus) = line_stats(&r.object(&from[p]).unwrap_or_default(), &r.object(&to[p]).unwrap_or_default());
        parts.push(format!("{p} +{plus} -{minus}"));
    }
    for p in &a {
        let n = String::from_utf8_lossy(&r.object(&to[p]).unwrap_or_default()).lines().count();
        parts.push(format!("{p} new, {n} line{}", plural(n)));
    }
    for p in &d {
        parts.push(format!("{p} deleted"));
    }
    let more = parts.len().saturating_sub(4);
    parts.truncate(4);
    if more > 0 {
        parts.push(format!("and {more} more"));
    }
    parts.join(", ")
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
        // Cloning again is harmless: a small model often repeats a clone it already made.
        let same = fs::read_to_string(root.join(STATE).join("config.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .is_some_and(|c| c["owner"].as_str() == Some(owner) && c["repo"].as_str() == Some(repo));
        if same {
            println!("{dir}/ already holds {owner}/{repo} (next: cd {dir})");
            return Ok(());
        }
        return Err(format!("{dir} already exists and is not empty; clone into another folder with: gh repo clone {owner}/{repo} NEW-FOLDER"));
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
    println!("Cloned {n} files into {dir}/ (next: cd {dir})");
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
        println!("Your branch is ahead of 'origin/{}' by {ahead} commit{} (not pushed; git push uploads them).", r.config["upstream"]["branch"].as_str().unwrap_or(""), plural(ahead));
    }
    if a.is_empty() && m.is_empty() && d.is_empty() {
        println!("nothing to commit, working tree clean");
        return Ok(());
    }
    println!("Changes not committed (git commit -m \"MESSAGE\" saves them all; git restore FILE undoes one):");
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
        // A failure, as in Git, so a chain such as `git commit -m X && git push` stops here.
        return Err("nothing to commit: no file has changed since the last commit. Change a file first, then commit".into());
    }
    for p in a.iter().chain(&m) {
        let full = r.root.join(p);
        let data = match fs::read_link(&full) {
            Ok(t) => t.to_string_lossy().into_owned().into_bytes(),
            Err(_) => fs::read(&full).map_err(|e| e.to_string())?,
        };
        r.store(&data)?;
    }
    // A file that lost most of its lines was most likely overwritten by mistake (`>` instead of
    // `>>`, or a write of one line): refuse, unless asked for.
    if !args.iter().any(|a| a == "--allow-shrink") {
        for p in &m {
            let (old, new) = (r.object(&head[p]).unwrap_or_default(), r.object(&work[p]).unwrap_or_default());
            let (before, after) = (String::from_utf8_lossy(&old).lines().count(), String::from_utf8_lossy(&new).lines().count());
            let (_, removed) = line_stats(&old, &new);
            if before >= 20 && after * 2 < before && removed * 5 >= before * 4 {
                return Err(format!("{p} would go from {before} lines to {after}, so nothing was committed. It looks overwritten (for example with > instead of >>). Undo it with: git restore {p}   (if you meant it: git commit -m \"MESSAGE\" --allow-shrink)"));
            }
            // A shorter file that lost a function or class it had: most likely rewritten from
            // memory (echo '...' > calc.py) rather than changed.
            let defs = |t: &[u8]| String::from_utf8_lossy(t).lines().filter_map(definition).collect::<Vec<_>>();
            let (had, has) = (defs(&old), defs(&new));
            let lost: Vec<&String> = had.iter().filter(|d| !has.contains(d)).collect();
            if after < before && !lost.is_empty() {
                let names = lost.iter().map(|d| d.as_str()).collect::<Vec<_>>().join(", ");
                return Err(format!("{p} would lose {names} ({before} lines to {after}), so nothing was committed. It looks rewritten instead of changed. Undo it with: git restore {p}, then change only the lines that need it: edit {p} 'old text' 'new text'   (if you meant to remove it: git commit -m \"MESSAGE\" --allow-shrink)"));
            }
        }
    }
    let summary = change_summary(r, &head, &work);
    let mut log = r.log();
    let id = blob_sha(format!("{msg}{}{}", log.len(), serde_json::to_string(&work).unwrap()).as_bytes());
    log.push(json!({ "id": id, "message": msg, "tree": work }));
    r.save_log(&log)?;
    r.save_head(&work)?;
    println!("[{} {}] {msg}", r.branch(), short(&id));
    println!(" {} file{} changed: {summary}", a.len() + m.len() + d.len(), plural(a.len() + m.len() + d.len()));
    Ok(())
}

fn push(r: &mut Repo) -> R<()> {
    let log = r.log();
    if log.is_empty() {
        if branch_head(&r.owner_repo(), &r.branch())?.is_none() {
            return Err(format!("branch {} has no commits to push yet. Change a file and run git commit -m \"MESSAGE\" first", r.branch()));
        }
        println!("Everything up-to-date");
        return Ok(());
    }
    let (owner, repo) = (r.config["owner"].as_str().unwrap().to_string(), r.config["repo"].as_str().unwrap().to_string());
    let branch = r.branch();
    let to_default = branch == r.default_branch();
    let base: BTreeMap<String, String> = fs::read_to_string(r.state("upstream.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let last: BTreeMap<String, String> = log.last().and_then(|c| serde_json::from_value(c["tree"].clone()).ok()).unwrap_or_default();
    allowed(&format!("Push {} commit{} to {owner}/{repo} branch {branch} ({})", log.len(), plural(log.len()), change_summary(r, &base, &last)), to_default)?;
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

/// A pull request's number from `12`, `#12`, its URL or a branch name; with none, the open pull
/// request of the current branch.
/// Flags of gh pr merge and gh pr view that take a value, so the value is not taken for the
/// pull request (`gh pr merge -b "Merge #1"`).
const PR_VALUE_FLAGS: &[&str] = &["-b", "--body", "-t", "--subject", "--title", "-F", "--body-file", "--match-head-commit", "-A", "--author-email", "--json", "-q", "--jq", "--template"];

fn pr_number(r: &Repo, args: &[String], verb: &str) -> R<String> {
    let mut given = None;
    let mut i = 0;
    while i < args.len() {
        if PR_VALUE_FLAGS.contains(&args[i].as_str()) {
            i += 2;
        } else if args[i].starts_with('-') {
            i += 1;
        } else {
            given = Some(&args[i]);
            break;
        }
    }
    if let Some(n) = given.map(|a| a.trim_start_matches('#').rsplit('/').next().unwrap_or("").to_string()).filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())) {
        return Ok(n);
    }
    let branch = given.cloned().unwrap_or_else(|| r.branch());
    let open = api("GET", &format!("/repos/{}/pulls?state=open&head={}:{}", r.owner_repo(), r.config["owner"].as_str().unwrap_or(""), url_path(&branch)), None)?;
    match open.as_array().and_then(|a| a.first()).and_then(|p| p["number"].as_u64()) {
        Some(n) => Ok(n.to_string()),
        // The open pull requests and their numbers are in the error, so the next command can be right.
        None => {
            let what = match given {
                Some(g) => format!("{g} is not a pull request number or a branch with an open pull request"),
                None => format!("no open pull request for branch {branch}"),
            };
            let open = api("GET", &format!("/repos/{}/pulls?state=open&per_page=10", r.owner_repo()), None)?.as_array().cloned().unwrap_or_default();
            let list: Vec<String> = open.iter().map(|p| format!("#{} \"{}\" ({} into {})", p["number"], p["title"].as_str().unwrap_or(""), p["head"]["ref"].as_str().unwrap_or(""), p["base"]["ref"].as_str().unwrap_or(""))).collect();
            match open.first() {
                Some(p) => Err(format!("{what}. Give its number, as in: gh pr {verb} {}. Open pull requests on {}: {}", p["number"], r.owner_repo(), list.join("; "))),
                None => Err(format!("{what}, and {} has no open pull requests. Open one with: gh pr create --title \"TITLE\"", r.owner_repo())),
            }
        }
    }
}

/// "def sub", "class App", "fn main", "function draw": a line that defines something by name.
fn definition(line: &str) -> Option<String> {
    let t = line.trim_start();
    for k in ["async def ", "def ", "class ", "pub fn ", "pub(crate) fn ", "fn ", "export default function ", "export async function ", "export function ", "async function ", "function ", "pub struct ", "struct ", "pub enum ", "enum "] {
        if let Some(rest) = t.strip_prefix(k) {
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            return (!name.is_empty()).then(|| format!("{k}{name}"));
        }
    }
    None
}

/// "2 passed, 1 failed" for the checks on a commit.
fn checks_summary(or: &str, sha: &str) -> R<String> {
    let runs = api("GET", &format!("/repos/{or}/commits/{sha}/check-runs"), None)?["check_runs"].as_array().cloned().unwrap_or_default();
    if runs.is_empty() {
        return Ok("no checks".into());
    }
    let count = |want: &[&str]| runs.iter().filter(|c| want.contains(&c["conclusion"].as_str().unwrap_or("pending"))).count();
    let parts = [(count(&["success", "neutral", "skipped"]), "passed"), (count(&["failure", "timed_out", "cancelled", "action_required", "startup_failure"]), "failed"), (count(&["pending"]), "still running")];
    Ok(parts.iter().filter(|(n, _)| *n > 0).map(|(n, w)| format!("{n} {w}")).collect::<Vec<_>>().join(", "))
}

/// gh pr merge: merge a pull request on GitHub. Always asks the person first.
fn pr_merge(r: &Repo, args: &[String]) -> R<()> {
    let or = r.owner_repo();
    let n = pr_number(r, args, "merge")?;
    let p = api("GET", &format!("/repos/{or}/pulls/{n}"), None)?;
    if p["merged"].as_bool() == Some(true) {
        println!("Pull request #{n} is already merged.");
        return Ok(());
    }
    if p["state"].as_str() != Some("open") {
        return Err(format!("pull request #{n} is closed, so it cannot be merged"));
    }
    let (head, base, title) = (p["head"]["ref"].as_str().unwrap_or(""), p["base"]["ref"].as_str().unwrap_or(""), p["title"].as_str().unwrap_or(""));
    if p["mergeable"].as_bool() == Some(false) {
        return Err(format!("pull request #{n} has merge conflicts with {base}, so it cannot be merged. Tell the person; they can resolve the conflicts on GitHub"));
    }
    let method = if args.iter().any(|a| a == "--squash" || a == "-s") {
        "squash"
    } else if args.iter().any(|a| a == "--rebase" || a == "-r") {
        "rebase"
    } else {
        "merge"
    };
    let checks = checks_summary(&or, p["head"]["sha"].as_str().unwrap_or(""))?;
    allowed(&format!("Merge pull request #{n} \"{title}\" ({head} into {base}) on {or}, checks: {checks}"), true)?;
    let done = api("PUT", &format!("/repos/{or}/pulls/{n}/merge"), Some(json!({ "merge_method": method }))).map_err(|e| format!("GitHub did not merge #{n}: {e}"))?;
    println!("Merged pull request #{n} into {base} ({}).", short(done["sha"].as_str().unwrap_or("")));
    if r.branch() == base {
        println!("Run git pull to get the merged files.");
    } else {
        println!("To work on {base} now, run: git checkout {base}");
    }
    if args.iter().any(|a| a == "--delete-branch" || a == "-d") {
        println!("The branch {head} was kept: deleting branches is not available here.");
    }
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
            // Checked before asking the person: a pull request needs commits on GitHub.
            if branch_head(&or, &head)?.is_none() {
                return Err(format!("branch {head} is not on GitHub yet. Commit a change, then git push, then gh pr create"));
            }
            let cmp = api("GET", &format!("/repos/{or}/compare/{}...{}", url_path(&base), url_path(&head)), None)?;
            if cmp["ahead_by"].as_u64() == Some(0) {
                return Err(format!("branch {head} has no commits that {base} does not have, so there is nothing to review. Commit a change and git push first"));
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
        Some(verb @ ("view" | "checks")) => {
            let n = pr_number(r, &args[1..], verb)?;
            let p = api("GET", &format!("/repos/{or}/pulls/{n}"), None)?;
            let conflicts = if p["mergeable"].as_bool() == Some(false) { " (has merge conflicts)" } else { "" };
            println!("#{n} {}\nstate: {}{}{conflicts}\n{} -> {}\n{}\n\n{}", p["title"].as_str().unwrap_or(""), p["state"].as_str().unwrap_or(""), if p["merged"].as_bool() == Some(true) { " (merged)" } else { "" }, p["head"]["ref"].as_str().unwrap_or(""), p["base"]["ref"].as_str().unwrap_or(""), p["html_url"].as_str().unwrap_or(""), p["body"].as_str().unwrap_or(""));
            let sha = p["head"]["sha"].as_str().unwrap_or("");
            for c in api("GET", &format!("/repos/{or}/commits/{sha}/check-runs"), None)?["check_runs"].as_array().cloned().unwrap_or_default() {
                println!("check: {} — {}", c["name"].as_str().unwrap_or(""), c["conclusion"].as_str().or(c["status"].as_str()).unwrap_or(""));
            }
        }
        Some("merge") => pr_merge(r, &args[1..])?,
        Some(other) => return Err(format!("gh pr {other}: not available here (create, list, view, checks, merge)")),
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
            run_summary(&or, id)?;
        }
        // Wait for a run to finish (the newest on a branch, by default this one), then show it:
        // one command instead of many model turns spent checking.
        Some("watch") => {
            let id = match args.iter().skip(1).find(|a| !a.starts_with('-')) {
                Some(id) => id.clone(),
                None => {
                    let branch = args.iter().position(|a| a == "--branch" || a == "-b").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| r.branch());
                    let mut found = None;
                    // A pushed branch's run can take a little while to appear.
                    for _ in 0..12 {
                        found = api("GET", &format!("/repos/{or}/actions/runs?per_page=1&branch={}", url_path(&branch)), None)?["workflow_runs"].get(0).map(|w| w["id"].to_string());
                        if found.is_some() {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_secs(5));
                    }
                    found.ok_or(format!("no CI run for branch {branch} after a minute (gh run list shows the runs)"))?
                }
            };
            // At most 8 minutes here: a command in the VM is stopped after 10, and CI can take longer.
            eprint!("Waiting for run {id}");
            let mut done = false;
            for _ in 0..48 {
                if api("GET", &format!("/repos/{or}/actions/runs/{id}"), None)?["status"].as_str() == Some("completed") {
                    done = true;
                    break;
                }
                eprint!(".");
                std::thread::sleep(std::time::Duration::from_secs(10));
            }
            eprintln!();
            run_summary(&or, &id)?;
            if !done {
                println!("Run {id} is still running after 8 minutes. To keep waiting, run: gh run watch {id}");
            }
        }
        Some(other) => return Err(format!("gh run {other}: not available here (list, view, watch)")),
    }
    Ok(())
}

/// A run's result: its status, each job, the steps that did not pass and the annotations
/// (file:line: message) of failed jobs.
fn run_summary(or: &str, id: &str) -> R<()> {
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

/// Put files back as they were in the last commit; "." means every changed file.
fn restore(r: &Repo, paths: &[String]) -> R<()> {
    let head = r.head();
    let paths: Vec<String> = if paths.iter().any(|p| p == ".") {
        let (_, m, d) = changes(&head, &r.work()?);
        m.into_iter().chain(d).collect()
    } else {
        paths.to_vec()
    };
    if paths.is_empty() {
        return Err("usage: git restore FILE (or git restore . for every changed file)".into());
    }
    for p in &paths {
        let sha = head.get(p).ok_or(format!("{p}: not in the last commit (a new file: remove it with rm {p})"))?;
        let at = r.root.join(p);
        if let Some(dir) = at.parent() {
            fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        fs::write(at, r.object(sha).ok_or("content missing")?).map_err(|e| e.to_string())?;
    }
    println!("Restored {} file{}", paths.len(), plural(paths.len()));
    Ok(())
}

/// git checkout BRANCH: the files of a branch on GitHub.
fn switch(r: &mut Repo, name: &str) -> R<()> {
    if name == r.branch() {
        println!("Already on '{name}'");
        return Ok(());
    }
    ready_for(r, &format!("git checkout {name}"))?;
    let or = r.owner_repo();
    let commit = branch_head(&or, name)?.ok_or_else(|| no_branch(&or, name))?;
    let (a, m, d) = sync_to(r, name, &commit)?;
    println!("Switched to branch '{name}' (at {}; {a} new, {m} changed, {d} removed files)", short(&commit));
    Ok(())
}

/// git merge BRANCH: merge BRANCH into the current branch on GitHub (asking the person first),
/// then update the files.
fn merge(r: &mut Repo, args: &[String]) -> R<()> {
    let mut words = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "-m" {
            it.next();
        } else if a == "--abort" {
            return Err("nothing to abort: a merge here either finishes on GitHub or changes nothing".into());
        } else if !a.starts_with('-') {
            words.push(a.trim_start_matches("origin/").to_string());
        }
    }
    let from = words.first().ok_or("usage: git merge BRANCH   (brings BRANCH's commits into the current branch; asks the person first)")?;
    let into = r.branch();
    if *from == into {
        return Err(format!("you are on {into}; to merge it into another branch, run git checkout OTHER first, then git merge {into}"));
    }
    ready_for(r, "git merge")?;
    let or = r.owner_repo();
    if branch_head(&or, &into)?.is_none() {
        return Err(format!("branch {into} is not on GitHub yet. First run: git push"));
    }
    if branch_head(&or, from)?.is_none() {
        return Err(no_branch(&or, from));
    }
    let cmp = api("GET", &format!("/repos/{or}/compare/{}...{}", url_path(&into), url_path(from)), None)?;
    let ahead = cmp["ahead_by"].as_u64().unwrap_or(0) as usize;
    if ahead == 0 {
        println!("Already up to date: {into} has every commit of {from}.");
        return Ok(());
    }
    let files = cmp["files"].as_array().map(|f| f.len()).unwrap_or(0);
    let default = if into == r.default_branch() { " (the default branch)" } else { "" };
    allowed(&format!("Merge {ahead} commit{} from {from} into {into}{default} on {or}, {files} file{} changed", plural(ahead), plural(files)), true)?;
    let made = match api("POST", &format!("/repos/{or}/merges"), Some(json!({ "base": into, "head": from, "commit_message": format!("Merge branch '{from}' into {into}") }))) {
        Err(e) if e.starts_with("GitHub 409") => return Err(format!("merge conflict: {from} and {into} change the same lines, so nothing was merged. Tell the person; they can resolve it on GitHub")),
        other => other?,
    };
    let Some(sha) = made["sha"].as_str().map(String::from) else {
        println!("Already up to date.");
        return Ok(());
    };
    let (a, m, d) = sync_to(r, &into, &sha)?;
    println!("Merged {from} into {into} ({}): {a} new, {m} changed, {d} removed files.", short(&sha));
    Ok(())
}

const GIT_HELP: &str = "git here works on GitHub repositories through GitHub's API. Every command runs inside a clone.

  Start:       gh repo clone OWNER/REPO, then cd REPO
  Look:        git status   git diff   git log --oneline   git branch
  New branch:  git checkout -b NAME          (do this before changing files)
  Save:        git commit -m \"MESSAGE\"       (includes every change; git add is not needed)
  Upload:      git push                      (may ask the person)
  Update:      git pull                      (the newest version of this branch)
  Switch:      git checkout BRANCH           (a branch on GitHub)
  Merge:       git merge BRANCH              (BRANCH into this branch; always asks the person)
  Undo:        git restore FILE              (or git restore . for every changed file)

Pull requests and CI: gh help";

fn git(args: &[String]) -> R<()> {
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rest = if args.is_empty() { &[][..] } else { &args[1..] };
    match cmd {
        "clone" => return clone(rest),
        "help" | "--help" | "-h" => {
            println!("{GIT_HELP}");
            return Ok(());
        }
        "init" => return Err("git init is not available: git here works on GitHub repositories. Clone one with: gh repo clone OWNER/REPO".into()),
        "config" => {
            eprintln!("git config: not needed here (commits are made with the person's GitHub account)");
            return Ok(());
        }
        _ => {}
    }
    let mut r = Repo::find()?;
    match cmd {
        "status" => status(&r),
        "diff" => diff(&r, &rest.iter().filter(|a| !a.starts_with('-')).cloned().collect::<Vec<_>>()),
        "add" => Ok(()), // Every change is part of the next commit.
        "commit" => commit(&r, rest),
        "log" => log_cmd(&r, rest),
        "push" => {
            let cur = r.branch();
            if let Some(b) = rest.iter().filter(|a| !a.starts_with('-') && *a != "origin" && *a != "HEAD").map(|a| a.trim_start_matches("HEAD:")).find(|b| *b != cur) {
                return Err(format!("you are on branch {cur}, and git push uploads the current branch. To push {b}, first run: git checkout {b}"));
            }
            push(&mut r)
        }
        "fetch" => {
            println!("Nothing to fetch: git pull, git checkout and git merge read GitHub directly.");
            Ok(())
        }
        "merge" => merge(&mut r, rest),
        "branch" => {
            let cur = r.branch();
            if rest.iter().any(|a| a == "--show-current") {
                println!("{cur}");
                return Ok(());
            }
            if let Some(a) = rest.iter().find(|a| !matches!(a.as_str(), "-a" | "-r" | "-v" | "-vv" | "--all" | "--list" | "--remotes")) {
                return Err(format!("git branch {a}: only listing branches works here. To start a new branch, run: git checkout -b NAME"));
            }
            let names = branch_names(&r.owner_repo())?;
            if !names.contains(&cur) {
                println!("* {cur} (not on GitHub yet; git push creates it)");
            }
            for n in names {
                println!("{} {n}", if n == cur { "*" } else { " " });
            }
            Ok(())
        }
        "checkout" | "switch" if rest.first().is_some_and(|a| a == "-b" || a == "-c" || a == "-B") => {
            let name = rest.get(1).ok_or("usage: git checkout -b NAME")?;
            if let Ok(Some(_)) = branch_head(&r.owner_repo(), name) {
                return Err(format!("branch {name} already exists on GitHub. Switch to it with: git checkout {name} (or pick a new name)"));
            }
            r.config["branch"] = Value::String(name.clone());
            r.save_config()?;
            println!("Switched to a new branch '{name}' (created on GitHub at the next push)");
            Ok(())
        }
        "checkout" | "switch" => {
            let names: Vec<String> = rest.iter().filter(|a| *a != "--" && !a.starts_with('-')).cloned().collect();
            // `git checkout FILE` (or `-- FILE`) restores files, as in Git.
            if cmd == "checkout" && (rest.iter().any(|a| a == "--") || names.iter().any(|p| p == "." || r.head().contains_key(p))) {
                return restore(&r, &names);
            }
            let name = names.first().ok_or("usage: git checkout BRANCH (or git checkout -b NEW-BRANCH)")?;
            switch(&mut r, name.trim_start_matches("origin/"))
        }
        "restore" => restore(&r, &rest.iter().filter(|a| *a != "--" && !a.starts_with('-')).cloned().collect::<Vec<_>>()),
        "stash" => Err("git stash is not available. Commit your changes (git commit -m \"MESSAGE\") or undo them (git restore FILE)".into()),
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
            let branch = r.branch();
            if let Some(b) = rest.iter().filter(|a| !a.starts_with('-') && *a != "origin").find(|b| **b != branch) {
                return Err(format!("git pull updates this branch ({branch}) only. To bring {b}'s commits into it, run: git merge {b}"));
            }
            ready_for(&r, "git pull")?;
            let commit = branch_head(&r.owner_repo(), &branch)?.ok_or(format!("branch {branch} is not on GitHub yet; git push creates it"))?;
            if Some(commit.as_str()) == r.config["upstream"]["commit"].as_str() {
                println!("Already up to date.");
                return Ok(());
            }
            let (a, m, d) = sync_to(&mut r, &branch, &commit)?;
            println!("Updated {branch} to {} ({a} new, {m} changed, {d} removed)", short(&commit));
            Ok(())
        }
        other => Err(format!("git {other}: not available here. Run git help for the commands that work")),
    }
}

const GH_HELP: &str = "gh here works on GitHub through its API.

  gh repo clone OWNER/REPO [DIR] [--branch B]    gh repo list [OWNER]    gh auth status
  gh pr create --title \"TITLE\" --body \"TEXT\" [--base BRANCH]   (push first; may ask the person)
  gh pr list            gh pr view [NUMBER]       gh pr checks [NUMBER]   (add -R OWNER/REPO outside a clone)
  gh pr merge [NUMBER] [--squash|--rebase]        (always asks the person; no NUMBER = this branch's pull request)
  gh run list [--branch B]    gh run view RUN_ID  gh run watch [RUN_ID]  (CI results: failed steps and errors; watch waits for the run)
  gh api PATH

Local work (git status, diff, commit, push, pull, checkout, merge): git help";

/// The repository for gh pr and gh run: `-R OWNER/REPO` (or `--repo`), as in GitHub's gh, works
/// anywhere; otherwise the clone around the working directory. Returns it and the other args.
fn repo_or_flag(args: &[String]) -> R<(Repo, Vec<String>)> {
    let (spec, rest): (String, Vec<String>) = match args.iter().position(|a| a == "-R" || a == "--repo") {
        Some(i) => (
            args.get(i + 1).ok_or("usage: -R OWNER/REPO")?.clone(),
            args.iter().enumerate().filter(|(j, _)| *j != i && *j != i + 1).map(|(_, a)| a.clone()).collect(),
        ),
        // Outside a clone, GH_REPO names the repository, as in GitHub's gh.
        None => match Repo::find() {
            Ok(r) => return Ok((r, args.to_vec())),
            Err(e) => (env::var("GH_REPO").map_err(|_| e)?, args.to_vec()),
        },
    };
    let spec = spec.trim_start_matches("https://github.com/").trim_end_matches(".git");
    let (owner, repo) = spec.split_once('/').ok_or("expected OWNER/REPO")?;
    let info = api("GET", &format!("/repos/{owner}/{repo}"), None)?;
    let default = info["default_branch"].as_str().unwrap_or("main").to_string();
    // Not a clone: no local commits, and the branch is the default one unless --head says.
    let r = Repo { root: PathBuf::from("/nonexistent-clone"), config: json!({ "owner": owner, "repo": repo, "default_branch": default, "branch": default }) };
    Ok((r, rest))
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
        (Some("pr"), _) => {
            let (r, rest) = repo_or_flag(&args[1..])?;
            pr(&r, &rest)
        }
        (Some("run"), _) => {
            let (r, rest) = repo_or_flag(&args[1..])?;
            run_cmd(&r, &rest)
        }
        (Some("api"), Some(path)) => {
            let method = args.iter().position(|a| a == "-X").and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("GET");
            if method != "GET" {
                allowed(&format!("GitHub API {method} {path}"), true)?;
            }
            println!("{}", serde_json::to_string_pretty(&api(method, &format!("/{}", path.trim_start_matches('/')), None)?).unwrap());
            Ok(())
        }
        (None, _) | (Some("help" | "--help" | "-h"), _) => {
            println!("{GH_HELP}");
            Ok(())
        }
        // A command that is not here fails, so a chain stops and the model sees it.
        (Some(a), b) => Err(format!("gh {a}{}: not available here. gh help lists what works", b.map(|b| format!(" {b}")).unwrap_or_default())),
    }
}

fn main() {
    // A reader that stopped early (`gh pr list | head -1`) ends the program quietly, as it would
    // a native one, instead of a panic about a broken pipe.
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        if msg.contains("Broken pipe") {
            std::process::exit(0);
        }
        eprintln!("{msg}");
    }));
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
