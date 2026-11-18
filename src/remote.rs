//! Models read over HTTP from where they are published (Hugging Face), for models larger than
//! the device's free storage. Each forward pass fetches the byte ranges it needs, many at once,
//! and a size-capped cache on disk keeps what it can: weights every token reads come first,
//! because a cache smaller than one token's reads gets no hits from plain least-recently-used
//! eviction (the reads cycle through the whole model in the same order every token).

use crate::source::{Ready, Span, WeightSource};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Requests are split into pieces of at most this size so large tensors download in parallel.
const PIECE: usize = 16 << 20;

/// What the network and the cache did since the counters were last taken.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct NetStats {
    pub fetched_bytes: u64,
    pub requests: u64,
    pub retries: u64,
    /// Wall time spent waiting for the network.
    pub fetch_seconds: f64,
    pub cache_hit_bytes: u64,
    pub cache_bytes: u64,
}

struct RemoteFile {
    url: String,
    /// The CDN address the file redirects to, and when its signature expires.
    direct: Mutex<Option<(String, Instant)>>,
}

type Key = (usize, u64, usize);

pub struct HttpSource {
    files: Vec<RemoteFile>,
    agent: ureq::Agent,
    /// Fetched spans the cache did not keep, waiting for the read that follows the prefetch.
    staged: Mutex<HashMap<Key, Vec<u8>>>,
    cache: Option<Mutex<DiskCache>>,
    /// Requests in flight at once.
    pub parallel: usize,
    stats: Mutex<NetStats>,
}

impl HttpSource {
    /// `urls` are the parts of the model in order; `cache` is a directory and its byte cap.
    pub fn open(urls: Vec<String>, cache: Option<(PathBuf, u64)>) -> Result<HttpSource, String> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_body(Some(Duration::from_secs(600)))
            .user_agent("evolve-llm")
            .build()
            .into();
        let cache = match cache {
            Some((dir, cap)) if cap > 0 => Some(Mutex::new(DiskCache::open(&dir, cap)?)),
            _ => None,
        };
        Ok(HttpSource {
            files: urls
                .into_iter()
                .map(|url| RemoteFile {
                    url,
                    direct: Mutex::new(None),
                })
                .collect(),
            agent,
            staged: Mutex::new(HashMap::new()),
            cache,
            parallel: 8,
            stats: Mutex::new(NetStats::default()),
        })
    }

    pub fn take_stats(&self) -> NetStats {
        let mut s = std::mem::take(&mut *self.stats.lock().unwrap());
        s.cache_bytes = self.cache.as_ref().map_or(0, |c| c.lock().unwrap().used);
        s
    }

    /// The address to read from: the redirect target of a Hugging Face `resolve` URL, reused
    /// until its signature is about to expire, so the rate-limited API sees one request per
    /// file and hour rather than one per read.
    fn direct_url(&self, file: usize) -> Result<String, String> {
        let f = self.files.get(file).ok_or("no such file")?;
        let mut direct = f.direct.lock().unwrap();
        if let Some((url, until)) = direct.as_ref() {
            if *until > Instant::now() + Duration::from_secs(120) {
                return Ok(url.clone());
            }
        }
        let resp = self
            .agent
            .get(&f.url)
            .header("Range", "bytes=0-0")
            .call()
            .map_err(|e| format!("{}: {e}", f.url))?;
        let status = resp.status().as_u16();
        let (url, until) = match status {
            301 | 302 | 303 | 307 | 308 => {
                let loc = resp
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or("a redirect without a location")?;
                let loc = if loc.starts_with('/') {
                    let origin: String = f.url.splitn(4, '/').take(3).collect::<Vec<_>>().join("/");
                    format!("{origin}{loc}")
                } else {
                    loc.to_string()
                };
                (loc.clone(), Instant::now() + expires_in(&loc))
            }
            200 | 206 => (
                f.url.clone(),
                Instant::now() + Duration::from_secs(365 * 86400),
            ),
            401 | 403 | 404 => {
                return Err(format!(
                    "{} answered HTTP {status}: the file is not public",
                    f.url
                ))
            }
            _ => return Err(format!("{} answered HTTP {status}", f.url)),
        };
        *direct = Some((url.clone(), until));
        Ok(url)
    }

    fn forget_direct(&self, file: usize) {
        *self.files[file].direct.lock().unwrap() = None;
    }

    /// One range request, retried with backoff on network errors and server trouble.
    fn fetch(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut last = String::new();
        for attempt in 0..8u32 {
            if attempt > 0 {
                self.stats.lock().unwrap().retries += 1;
                std::thread::sleep(Duration::from_millis(500 * (1 << attempt.min(5))));
            }
            let url = match self.direct_url(file) {
                Ok(u) => u,
                Err(e) => {
                    last = e;
                    continue;
                }
            };
            let resp = self
                .agent
                .get(&url)
                .header(
                    "Range",
                    &format!("bytes={}-{}", offset, offset + len as u64 - 1),
                )
                .call();
            let mut resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            match resp.status().as_u16() {
                206 => {}
                200 => {
                    return Err(
                        "the server ignored the byte range; it cannot stream this model".into(),
                    )
                }
                401 | 403 | 410 => {
                    // An expired or revoked signature: resolve the file again.
                    self.forget_direct(file);
                    last = format!("HTTP {}", resp.status());
                    continue;
                }
                s => {
                    last = format!("HTTP {s}");
                    continue;
                }
            }
            match resp
                .body_mut()
                .with_config()
                .limit(len as u64 + 1)
                .read_to_vec()
            {
                Ok(b) if b.len() == len => {
                    let mut st = self.stats.lock().unwrap();
                    st.fetched_bytes += len as u64;
                    st.requests += 1;
                    return Ok(b);
                }
                Ok(b) => last = format!("got {} of {len} bytes", b.len()),
                Err(e) => last = e.to_string(),
            }
        }
        Err(format!(
            "reading {} at {offset}: {last}",
            self.files[file].url
        ))
    }

    /// Fetch spans in parallel pieces and reassemble them.
    fn fetch_all(&self, spans: &[Span]) -> Result<Vec<Vec<u8>>, String> {
        let mut pieces = Vec::new();
        for (i, s) in spans.iter().enumerate() {
            let mut at = 0;
            while at < s.len {
                let n = PIECE.min(s.len - at);
                pieces.push((i, at, n));
                at += n;
            }
        }
        let mut out: Vec<Vec<u8>> = spans.iter().map(|s| vec![0u8; s.len]).collect();
        let next = Mutex::new(0usize);
        let done: Mutex<Vec<(usize, usize, Vec<u8>)>> = Mutex::new(Vec::new());
        let failed: Mutex<Option<String>> = Mutex::new(None);
        let t0 = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..self.parallel.max(1).min(pieces.len()) {
                scope.spawn(|| loop {
                    if failed.lock().unwrap().is_some() {
                        return;
                    }
                    let k = {
                        let mut n = next.lock().unwrap();
                        let k = *n;
                        *n += 1;
                        k
                    };
                    let Some(&(i, at, n)) = pieces.get(k) else {
                        return;
                    };
                    let s = spans[i];
                    match self.fetch(s.file, s.offset + at as u64, n) {
                        Ok(b) => done.lock().unwrap().push((i, at, b)),
                        Err(e) => {
                            *failed.lock().unwrap() = Some(e);
                            return;
                        }
                    }
                });
            }
        });
        // Wall time, which is what a token waits; requests overlap.
        self.stats.lock().unwrap().fetch_seconds += t0.elapsed().as_secs_f64();
        if let Some(e) = failed.into_inner().unwrap() {
            return Err(e);
        }
        for (i, at, b) in done.into_inner().unwrap() {
            out[i][at..at + b.len()].copy_from_slice(&b);
        }
        Ok(out)
    }
}

/// Seconds until a signed URL's `Expires=` timestamp, or half an hour when it has none.
fn expires_in(url: &str) -> Duration {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    url.split(['?', '&'])
        .find_map(|p| {
            p.strip_prefix("Expires=")
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|t| Duration::from_secs(t.saturating_sub(now)))
        .unwrap_or(Duration::from_secs(1800))
}

impl WeightSource for HttpSource {
    fn read(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        if let Some(b) = self.staged.lock().unwrap().remove(&(file, offset, len)) {
            return Ok(b);
        }
        if let Some(c) = &self.cache {
            if let Some(b) = c.lock().unwrap().get((file, offset, len)) {
                self.stats.lock().unwrap().cache_hit_bytes += len as u64;
                return Ok(b);
            }
        }
        let b = self
            .fetch_all(&[Span {
                file,
                offset,
                len,
                every_token: false,
            }])?
            .remove(0);
        if let Some(c) = &self.cache {
            c.lock().unwrap().put((file, offset, len), &b, false);
        }
        Ok(b)
    }

    fn prefetch<'a>(&'a self, spans: &'a [Span]) -> Ready<'a> {
        Box::pin(async move {
            // Anything still staged was never read (an earlier pass failed); drop it.
            self.staged.lock().unwrap().clear();
            let mut want: Vec<Span> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for s in spans {
                let key = (s.file, s.offset, s.len);
                if !seen.insert(key) {
                    continue;
                }
                let cached = self
                    .cache
                    .as_ref()
                    .is_some_and(|c| c.lock().unwrap().contains(key));
                if !cached {
                    want.push(*s);
                }
            }
            if want.is_empty() {
                return Ok(());
            }
            let data = self.fetch_all(&want)?;
            let mut kept = Vec::new();
            for (s, b) in want.iter().zip(data) {
                let key = (s.file, s.offset, s.len);
                let in_cache = self
                    .cache
                    .as_ref()
                    .is_some_and(|c| c.lock().unwrap().put(key, &b, s.every_token));
                if !in_cache {
                    kept.push((key, b));
                }
            }
            self.staged.lock().unwrap().extend(kept);
            Ok(())
        })
    }
}

/// Byte ranges kept as files in a directory, up to a byte cap. Ranges read for every token
/// are kept for good once admitted; experts share what is left, least recently used first out.
pub struct DiskCache {
    dir: PathBuf,
    cap: u64,
    used: u64,
    clock: u64,
    entries: HashMap<Key, (bool, u64)>,
}

impl DiskCache {
    pub fn open(dir: &Path, cap: u64) -> Result<DiskCache, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut found: Vec<(SystemTime, Key, bool, u64)> = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().to_string();
            let Some((key, every)) = parse_name(&name) else {
                // Partial writes from an interrupted run.
                if name.ends_with(".tmp") {
                    let _ = std::fs::remove_file(entry.path());
                }
                continue;
            };
            let meta = entry.metadata().map_err(|e| e.to_string())?;
            if meta.len() != key.2 as u64 {
                let _ = std::fs::remove_file(entry.path());
                continue;
            }
            found.push((
                meta.modified().unwrap_or(UNIX_EPOCH),
                key,
                every,
                meta.len(),
            ));
        }
        found.sort_by_key(|f| f.0);
        let mut c = DiskCache {
            dir: dir.to_path_buf(),
            cap,
            used: 0,
            clock: 0,
            entries: HashMap::new(),
        };
        for (_, key, every, len) in found {
            c.clock += 1;
            c.used += len;
            c.entries.insert(key, (every, c.clock));
        }
        // A smaller cap than last time: shed experts, then the newest every-token ranges.
        while c.used > c.cap {
            let victim = c
                .entries
                .iter()
                .min_by_key(|(_, (every, used))| (*every, *used))
                .map(|(k, _)| *k)
                .unwrap();
            c.remove(victim);
        }
        Ok(c)
    }

    fn path(&self, key: Key, every: bool) -> PathBuf {
        self.dir.join(format!(
            "{}-{}-{}.{}",
            key.0,
            key.1,
            key.2,
            if every { "t" } else { "e" }
        ))
    }

    pub fn contains(&self, key: Key) -> bool {
        self.entries.contains_key(&key)
    }

    pub fn get(&mut self, key: Key) -> Option<Vec<u8>> {
        let (every, _) = *self.entries.get(&key)?;
        match std::fs::read(self.path(key, every)) {
            Ok(b) if b.len() == key.2 => {
                self.clock += 1;
                self.entries.insert(key, (every, self.clock));
                Some(b)
            }
            _ => {
                self.remove(key);
                None
            }
        }
    }

    fn remove(&mut self, key: Key) {
        if let Some((every, _)) = self.entries.remove(&key) {
            let _ = std::fs::remove_file(self.path(key, every));
            self.used -= key.2 as u64;
        }
    }

    /// Keep a range if there is room, evicting least recently used experts to make it. An
    /// expert never evicts an every-token range. Returns whether the range is now cached.
    pub fn put(&mut self, key: Key, bytes: &[u8], every_token: bool) -> bool {
        if self.entries.contains_key(&key) {
            return true;
        }
        let len = bytes.len() as u64;
        if len > self.cap {
            return false;
        }
        let experts: u64 = self
            .entries
            .iter()
            .filter(|(_, v)| !v.0)
            .map(|(k, _)| k.2 as u64)
            .sum();
        if self.used - experts + len > self.cap {
            // Not even evicting every expert would make room.
            return false;
        }
        while self.used + len > self.cap {
            let victim = self
                .entries
                .iter()
                .filter(|(_, v)| !v.0)
                .min_by_key(|(_, v)| v.1)
                .map(|(k, _)| *k)
                .unwrap();
            self.remove(victim);
        }
        let path = self.path(key, every_token);
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, bytes)
            .and_then(|_| std::fs::rename(&tmp, &path))
            .is_err()
        {
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        self.clock += 1;
        self.used += len;
        self.entries.insert(key, (every_token, self.clock));
        true
    }
}

fn parse_name(name: &str) -> Option<(Key, bool)> {
    let (stem, ext) = name.rsplit_once('.')?;
    let every = match ext {
        "t" => true,
        "e" => false,
        _ => return None,
    };
    let mut it = stem.split('-');
    let key = (
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
    );
    it.next().is_none().then_some((key, every))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "evolve-llm-cache-test-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn every_token_ranges_are_never_evicted_by_experts() {
        let d = dir("priority");
        let mut c = DiskCache::open(&d, 100).unwrap();
        assert!(c.put((0, 0, 60), &[1; 60], true));
        assert!(c.put((0, 100, 30), &[2; 30], false));
        // A second expert evicts the first; the every-token range stays.
        assert!(c.put((0, 200, 30), &[3; 30], false));
        assert!(c.contains((0, 0, 60)) && !c.contains((0, 100, 30)));
        // An every-token range that does not fit is simply not kept.
        assert!(!c.put((0, 300, 50), &[4; 50], true));
        assert_eq!(c.get((0, 0, 60)).unwrap(), vec![1; 60]);
        // Entries survive a reopen, and a smaller cap sheds experts first.
        drop(c);
        let c = DiskCache::open(&d, 70).unwrap();
        assert!(c.contains((0, 0, 60)) && !c.contains((0, 200, 30)));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn signed_url_expiry() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let url = format!("https://cdn.example/x?a=1&Expires={}&Policy=y", now + 3600);
        let d = expires_in(&url).as_secs();
        assert!((3590..=3600).contains(&d), "{d}");
        assert_eq!(expires_in("https://cdn.example/x").as_secs(), 1800);
    }
}
