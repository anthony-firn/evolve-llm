//! The HTTP client: TLS (rustls) over Tor or Wisp streams, connection pooling, redirects, cookies
//! with SameSite rules, CORS for script requests, and content decoding.

use super::http::{self, Framing, ResponseHead};
use super::{Credentials, FetchRequest, FetchResponse, LogEntry, Mode, Redirect};
use super::policy::{self, Context};
use super::tor::{TorNet, TorStream};
use super::wisp::{Mux, WispStream};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::{poll_fn, Future};
use std::io::{Read, Write};
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Poll, Waker};
use url::Url;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

const PER_HOST: usize = 6;
const TIMEOUT_MS: i32 = 45_000;

/// The route to websites: `tor:` (the Tor network over Snowflake) or a Wisp relay's ws(s):// URL.
pub const TOR_ROUTE: &str = "tor:";

enum Route {
    Tor(TorNet),
    Wisp(Mux),
}

impl Route {
    fn new(url: &str) -> Self {
        if url.trim().eq_ignore_ascii_case(TOR_ROUTE) || url.trim().eq_ignore_ascii_case("tor") {
            Route::Tor(TorNet::default())
        } else {
            Route::Wisp(Mux::new(url))
        }
    }
    fn url(&self) -> String {
        match self {
            Route::Tor(_) => TOR_ROUTE.into(),
            Route::Wisp(m) => m.url(),
        }
    }
}

/// One TCP connection to a website, over whichever route is in use.
enum Stream {
    Tor(TorStream),
    Wisp(WispStream),
}

impl Stream {
    async fn read(&mut self) -> Result<Option<Vec<u8>>, String> {
        match self {
            Stream::Tor(s) => s.read().await,
            Stream::Wisp(s) => s.read().await,
        }
    }
    async fn write(&mut self, data: &[u8]) -> Result<(), String> {
        match self {
            Stream::Tor(s) => s.write(data).await,
            Stream::Wisp(s) => s.write(data).await,
        }
    }
}

struct TlsConn {
    tls: rustls::ClientConnection,
    io: Stream,
    pending: Vec<u8>,
    eof: bool,
}

impl TlsConn {
    async fn connect(io: Stream, host: &str, config: Arc<rustls::ClientConfig>) -> Result<Self, String> {
        let name = rustls::pki_types::ServerName::try_from(host.to_string()).map_err(|_| format!("Invalid TLS server name {host}"))?;
        let tls = rustls::ClientConnection::new(config, name).map_err(|e| format!("TLS setup failed: {e}"))?;
        let mut conn = TlsConn { tls, io, pending: vec![], eof: false };
        while conn.tls.is_handshaking() {
            conn.flush().await?;
            if conn.tls.is_handshaking() && conn.tls.wants_read() && !conn.fill().await? {
                return Err(format!("{host} closed the connection during the TLS handshake"));
            }
        }
        conn.flush().await?;
        Ok(conn)
    }

    async fn flush(&mut self) -> Result<(), String> {
        while self.tls.wants_write() {
            let mut buf = Vec::new();
            self.tls.write_tls(&mut buf).map_err(|e| e.to_string())?;
            self.io.write(&buf).await?;
        }
        Ok(())
    }

    /// Feed more ciphertext into rustls. Returns false at end of stream.
    async fn fill(&mut self) -> Result<bool, String> {
        if self.pending.is_empty() {
            match self.io.read().await? {
                Some(data) => self.pending = data,
                None => {
                    self.eof = true;
                    return Ok(false);
                }
            }
        }
        let mut reader = &self.pending[..];
        let mut consumed = 0;
        while !reader.is_empty() {
            match self.tls.read_tls(&mut reader) {
                Ok(0) => break,
                Ok(n) => consumed += n,
                Err(e) if e.kind() == std::io::ErrorKind::Other => break, // plaintext buffer full
                Err(e) => return Err(format!("TLS read failed: {e}")),
            }
            self.tls.process_new_packets().map_err(|e| format!("TLS error: {e}"))?;
        }
        self.pending.drain(..consumed);
        Ok(true)
    }

    async fn write_all(&mut self, data: &[u8]) -> Result<(), String> {
        self.tls.writer().write_all(data).map_err(|e| e.to_string())?;
        self.flush().await
    }

    async fn read(&mut self) -> Result<Option<Vec<u8>>, String> {
        loop {
            let mut buf = vec![0u8; 32 * 1024];
            match self.tls.reader().read(&mut buf) {
                Ok(0) => return Ok(None),
                Ok(n) => {
                    buf.truncate(n);
                    return Ok(Some(buf));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if self.eof || !self.fill().await? {
                        return Ok(None);
                    }
                    self.flush().await?;
                }
                // Many servers close without close_notify; treat it as the end of the body.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(format!("TLS read failed: {e}")),
            }
        }
    }
}

enum Io {
    Plain(Stream),
    Tls(Box<TlsConn>),
}

struct Conn {
    io: Io,
    buf: Vec<u8>,
}

impl Conn {
    async fn read(&mut self) -> Result<Option<Vec<u8>>, String> {
        if !self.buf.is_empty() {
            return Ok(Some(std::mem::take(&mut self.buf)));
        }
        match &mut self.io {
            Io::Plain(s) => s.read().await,
            Io::Tls(t) => t.read().await,
        }
    }
    async fn write(&mut self, data: &[u8]) -> Result<(), String> {
        match &mut self.io {
            Io::Plain(s) => s.write(data).await,
            Io::Tls(t) => t.write_all(data).await,
        }
    }
}

enum ExchangeError {
    /// Failed before any response byte arrived: safe to retry on a fresh connection
    Stale(String),
    Fatal(String),
}

pub struct Client {
    route: RefCell<Route>,
    tls: std::cell::RefCell<Arc<rustls::ClientConfig>>,
    pub jar: RefCell<cookie_store::CookieStore>,
    pool: RefCell<HashMap<String, Vec<Conn>>>,
    active: RefCell<HashMap<String, usize>>,
    /// Per host: (priority, arrival, waker); lowest priority value is served first, FIFO within.
    waiters: RefCell<HashMap<String, Vec<(u8, u64, Waker)>>>,
    seq: Cell<u64>,
    pub user_agent: RefCell<String>,
    pub accept_language: RefCell<String>,
    pub cookies_changed: Cell<bool>,
    pub log: RefCell<Vec<LogEntry>>,
}


impl Client {
    pub fn set_tls_profile(&self, profile: &str) -> Result<(), &'static str> {
        *self.tls.borrow_mut() = super::tls::config(profile)?;
        self.pool.borrow_mut().clear();
        Ok(())
    }

    pub fn new(relay_url: &str, cookies_json: Option<&str>) -> Self {
        let psl = Some(policy::suffix_list().clone());
        let jar = cookies_json
            .and_then(|json| cookie_store::serde::json::load_all(json.as_bytes()).ok())
            .map(|jar| jar.with_suffix_list(psl.clone().unwrap()))
            .unwrap_or_else(|| cookie_store::CookieStore::new_with_public_suffix(psl));
        Client {
            route: RefCell::new(Route::new(relay_url)),
            tls: RefCell::new(super::tls::config("default").expect("default TLS profile")),
            jar: RefCell::new(jar),
            pool: RefCell::default(),
            active: RefCell::default(),
            waiters: RefCell::default(),
            seq: Cell::new(0),
            user_agent: RefCell::new(crate::identity::Identity::default().user_agent().into()),
            accept_language: RefCell::new("en-US,en;q=0.9".into()),
            cookies_changed: Cell::new(false),
            log: RefCell::default(),
        }
    }

    /// Switch the route: `tor:` or a relay's ws(s):// URL. Open connections are dropped.
    pub fn set_relay(&self, url: &str) {
        if self.route.borrow().url() != Route::new(url).url() {
            *self.route.borrow_mut() = Route::new(url);
            self.pool.borrow_mut().clear();
        }
    }

    /// Start connecting the route before the first request needs it (the Tor bridge and first
    /// circuit take several seconds).
    pub fn prepare(&self) {
        if let Route::Tor(t) = &*self.route.borrow() {
            let t = t.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let _ = t.warm().await;
            });
        }
    }

    pub fn relay_url(&self) -> String {
        self.route.borrow().url()
    }

    pub fn relay_open(&self) -> bool {
        match &*self.route.borrow() {
            Route::Tor(t) => t.is_ready(),
            Route::Wisp(m) => m.is_open(),
        }
    }

    pub fn cookies_json(&self) -> String {
        let mut out = Vec::new();
        let _ = cookie_store::serde::json::save_incl_expired_and_nonpersistent(&self.jar.borrow(), &mut out);
        String::from_utf8(out).unwrap_or_default()
    }

    pub fn clear_cookies(&self) {
        self.jar.borrow_mut().clear();
        self.cookies_changed.set(true);
    }

    /// `document.cookie` for a page: every matching cookie except HttpOnly ones.
    pub fn script_cookies(&self, url: &Url) -> String {
        self.jar
            .borrow()
            .matches(url)
            .into_iter()
            .filter(|c| c.http_only() != Some(true))
            .map(|c| format!("{}={}", c.name(), c.value()))
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub fn set_script_cookie(&self, url: &Url, value: &str) {
        let Ok(raw) = cookie::Cookie::parse(value.to_string()) else { return };
        if raw.http_only() == Some(true) {
            return;
        }
        if self.jar.borrow_mut().insert_raw(&raw, url).is_ok() {
            self.cookies_changed.set(true);
        }
    }

    fn cookie_header(&self, url: &Url, initiator: Option<&Url>, ctx: Context) -> Option<String> {
        let jar = self.jar.borrow();
        let pairs: Vec<String> = jar
            .matches(url)
            .into_iter()
            .filter(|c| policy::cookie_allowed(c.same_site(), initiator, url, ctx))
            .map(|c| format!("{}={}", c.name(), c.value()))
            .collect();
        (!pairs.is_empty()).then(|| pairs.join("; "))
    }

    fn store_cookies(&self, url: &Url, head: &ResponseHead) {
        let mut jar = self.jar.borrow_mut();
        for value in head.all("set-cookie") {
            if jar.parse(value, url).is_ok() {
                self.cookies_changed.set(true);
            }
        }
    }

    pub async fn fetch(&self, mut req: FetchRequest) -> Result<FetchResponse, String> {
        let started = js_sys::Date::now();
        let first_url = req.url.to_string();
        let result = self.fetch_inner(&mut req).await;
        let mut log = self.log.borrow_mut();
        if log.len() > 400 {
            log.drain(..200);
        }
        log.push(LogEntry {
            method: req.method.clone(),
            url: first_url,
            status: result.as_ref().map(|r| r.status).unwrap_or(0),
            bytes: result.as_ref().map(|r| r.body.len()).unwrap_or(0),
            ms: js_sys::Date::now() - started,
            dest: req.dest,
            error: result.as_ref().err().cloned(),
        });
        result
    }

    async fn fetch_inner(&self, req: &mut FetchRequest) -> Result<FetchResponse, String> {
        let mut redirected = false;
        // The serialized origin of the initiator; becomes "null" after a cross-origin redirect.
        let mut origin = req.initiator.as_ref().map(|u| u.origin().ascii_serialization());
        for _ in 0..20 {
            match req.url.scheme() {
                "http" | "https" => {}
                "data" => {
                    let (mime, body) = http::parse_data_url(req.url.as_str())?;
                    return Ok(FetchResponse { url: req.url.clone(), status: 200, status_text: "OK".into(), headers: vec![("Content-Type".into(), mime)], body, redirected });
                }
                "about" => {
                    return Ok(FetchResponse { url: req.url.clone(), status: 200, status_text: "OK".into(), headers: vec![("Content-Type".into(), "text/html".into())], body: vec![], redirected });
                }
                other => return Err(format!("The browser cannot load {other}: URLs")),
            }
            let cross_origin = req.mode == Mode::Cors && req.initiator.as_ref().is_some_and(|i| !policy::same_origin(i, &req.url));
            let with_credentials = match req.credentials {
                Credentials::Include => true,
                Credentials::SameOrigin => !cross_origin,
                Credentials::Omit => false,
            };
            if cross_origin && policy::needs_preflight(&req.method, &req.headers) {
                self.preflight(req, origin.as_deref().unwrap_or("null"), with_credentials).await?;
            }
            let ctx = match req.mode {
                Mode::Navigate => Context::Navigation { safe_method: matches!(req.method.as_str(), "GET" | "HEAD") },
                _ => Context::Subresource,
            };
            let cookie = if with_credentials { self.cookie_header(&req.url, req.initiator.as_ref(), ctx) } else { None };
            let (head, body) = self.send(req, cookie, cross_origin.then_some(origin.as_deref().unwrap_or("null"))).await?;
            if with_credentials {
                self.store_cookies(&req.url, &head);
            }
            let location = head.header("location").and_then(|l| req.url.join(l).ok());
            if matches!(head.status, 301 | 302 | 303 | 307 | 308) && location.is_some() {
                match req.redirect {
                    Redirect::Follow => {}
                    Redirect::Error => return Err("Redirects are not allowed for this request".into()),
                    Redirect::Manual => {
                        return Ok(FetchResponse { url: req.url.clone(), status: 0, status_text: String::new(), headers: vec![], body: vec![], redirected: true });
                    }
                }
                if cross_origin {
                    policy::cors_allows(&head.headers, origin.as_deref().unwrap_or("null"), with_credentials)?;
                }
                let next = location.unwrap();
                if (head.status == 303 && req.method != "HEAD") || (matches!(head.status, 301 | 302) && req.method == "POST") {
                    req.method = "GET".into();
                    req.body.clear();
                    req.headers.retain(|(k, _)| !k.eq_ignore_ascii_case("content-type"));
                }
                if req.mode == Mode::Cors && req.initiator.as_ref().is_some_and(|i| !policy::same_origin(i, &next)) && !policy::same_origin(&req.url, &next) {
                    origin = Some("null".into());
                }
                // Keep the initiator so SameSite rules still see the original site.
                req.url = next;
                redirected = true;
                continue;
            }
            if cross_origin {
                policy::cors_allows(&head.headers, origin.as_deref().unwrap_or("null"), with_credentials)?;
            }
            let headers = if req.mode == Mode::Cors { policy::exposed_headers(&head.headers, cross_origin) } else { head.headers.clone() };
            return Ok(FetchResponse { url: req.url.clone(), status: head.status, status_text: head.reason, headers, body, redirected });
        }
        Err("Too many redirects".into())
    }

    async fn preflight(&self, req: &FetchRequest, origin: &str, with_credentials: bool) -> Result<(), String> {
        let names: Vec<String> = req.headers.iter().map(|(k, _)| k.to_ascii_lowercase()).collect();
        let pre = FetchRequest {
            url: req.url.clone(),
            method: "OPTIONS".into(),
            headers: vec![
                ("Access-Control-Request-Method".into(), req.method.clone()),
                ("Access-Control-Request-Headers".into(), names.join(",")),
                ("Accept".into(), "*/*".into()),
            ],
            body: vec![],
            initiator: req.initiator.clone(),
            mode: Mode::NoCors,
            credentials: Credentials::Omit,
            redirect: Redirect::Error,
            dest: "empty",
            user_activated: false,
        };
        let (head, _) = self.send(&pre, None, Some(origin)).await?;
        if !(200..300).contains(&head.status) {
            return Err(format!("CORS preflight failed with status {}", head.status));
        }
        policy::cors_allows(&head.headers, origin, with_credentials)?;
        policy::preflight_allows(&head.headers, &req.method, &req.headers, with_credentials)
    }

    fn request_headers(&self, req: &FetchRequest, cookie: Option<String>, cors_origin: Option<&str>) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = Vec::new();
        let has = |name: &str| req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
        headers.push(("User-Agent".into(), self.user_agent.borrow().clone()));
        if !has("accept") {
            let accept = match req.dest {
                "document" => "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "style" => "text/css,*/*;q=0.1",
                "image" => "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                _ => "*/*",
            };
            headers.push(("Accept".into(), accept.into()));
        }
        if !has("accept-language") {
            headers.push(("Accept-Language".into(), self.accept_language.borrow().clone()));
        }
        headers.push(("Accept-Encoding".into(), "gzip, deflate, br".into()));
        for (k, v) in &req.headers {
            if !policy::forbidden_request_header(k) || req.mode != Mode::Cors && k.starts_with("Access-Control-Request") {
                headers.push((k.clone(), v.clone()));
            }
        }
        if let Some(cookie) = cookie {
            headers.push(("Cookie".into(), cookie));
        }
        // Referer: origin only when crossing sites (strict-origin-when-cross-origin)
        if let Some(initiator) = &req.initiator {
            let downgrade = initiator.scheme() == "https" && req.url.scheme() == "http";
            if !downgrade && matches!(initiator.scheme(), "http" | "https") {
                let referer = if policy::same_origin(initiator, &req.url) {
                    let mut r = initiator.clone();
                    r.set_fragment(None);
                    let _ = r.set_username("");
                    let _ = r.set_password(None);
                    r.to_string()
                } else {
                    format!("{}/", initiator.origin().ascii_serialization())
                };
                headers.push(("Referer".into(), referer));
            }
        }
        let origin = cors_origin.map(str::to_string).or_else(|| {
            (!matches!(req.method.as_str(), "GET" | "HEAD")).then(|| req.initiator.as_ref().map(|u| u.origin().ascii_serialization()).unwrap_or_else(|| "null".into()))
        });
        if let Some(origin) = origin {
            headers.push(("Origin".into(), origin));
        }
        let site = match &req.initiator {
            None => "none",
            Some(i) if policy::same_origin(i, &req.url) => "same-origin",
            Some(i) if policy::same_site(i, &req.url) => "same-site",
            Some(_) => "cross-site",
        };
        headers.push(("Sec-Fetch-Site".into(), site.into()));
        headers.push(("Sec-Fetch-Mode".into(), match req.mode { Mode::Navigate => "navigate", Mode::NoCors => "no-cors", Mode::Cors => "cors" }.into()));
        headers.push(("Sec-Fetch-Dest".into(), req.dest.into()));
        if req.mode == Mode::Navigate {
            if req.user_activated {
                headers.push(("Sec-Fetch-User".into(), "?1".into()));
            }
            headers.push(("Upgrade-Insecure-Requests".into(), "1".into()));
        }
        headers
    }

    async fn send(&self, req: &FetchRequest, cookie: Option<String>, cors_origin: Option<&str>) -> Result<(ResponseHead, Vec<u8>), String> {
        let host = req.url.host_str().ok_or("URL has no host")?.to_string();
        let port = req.url.port_or_known_default().ok_or("URL has no port")?;
        let tls = req.url.scheme() == "https";
        // Over Tor, connections belong to the first-party site they were opened for, so that
        // site's circuits are never shared with another site (first-party isolation).
        let first_party = match req.mode {
            Mode::Navigate => policy::site(&req.url),
            _ => req.initiator.as_ref().filter(|u| matches!(u.scheme(), "http" | "https")).map(policy::site).unwrap_or_else(|| policy::site(&req.url)),
        };
        let tor = matches!(&*self.route.borrow(), Route::Tor(_));
        let key = if tor { format!("{}://{}:{}#{}", req.url.scheme(), host, port, first_party) } else { format!("{}://{}:{}", req.url.scheme(), host, port) };
        let host_header = match req.url.port() {
            Some(p) => format!("{host}:{p}"),
            None => host.clone(),
        };
        let mut target = req.url.path().to_string();
        if let Some(q) = req.url.query() {
            target.push('?');
            target.push_str(q);
        }
        let headers = self.request_headers(req, cookie, cors_origin);
        let bytes = http::build_request(&req.method, &target, &host_header, &headers, &req.body);

        let _slot = self.acquire(&key, req.dest).await;
        for attempt in 0..2 {
            let pooled = if attempt == 0 { self.pool.borrow_mut().get_mut(&key).and_then(Vec::pop) } else { None };
            let reused = pooled.is_some();
            let mut conn = match pooled {
                Some(c) => c,
                None => self.open(&host, port, tls, &first_party).await?,
            };
            let exchange = self.exchange(&mut conn, &bytes, &req.method);
            match with_timeout(exchange, TIMEOUT_MS).await {
                None => return Err(format!("{host} did not respond within {} seconds", TIMEOUT_MS / 1000)),
                Some(Ok((head, body, reusable))) => {
                    if reusable {
                        self.pool.borrow_mut().entry(key.clone()).or_default().push(conn);
                    }
                    return Ok((head, body));
                }
                Some(Err(ExchangeError::Stale(_))) if reused => continue,
                Some(Err(ExchangeError::Stale(e) | ExchangeError::Fatal(e))) => return Err(e),
            }
        }
        Err(format!("Could not complete the request to {host}"))
    }

    async fn open(&self, host: &str, port: u16, tls: bool, first_party: &str) -> Result<Conn, String> {
        let route = match &*self.route.borrow() {
            Route::Tor(t) => Route::Tor(t.clone()),
            Route::Wisp(m) => Route::Wisp(m.clone()),
        };
        let stream = match route {
            // The first Tor connection also waits for the bridge and the first circuit.
            Route::Tor(t) => match with_timeout(t.connect(host, port, first_party), 90_000).await {
                Some(stream) => Stream::Tor(stream?),
                None => return Err(format!("Tor could not reach {host}:{port} within 90 seconds")),
            },
            Route::Wisp(m) => Stream::Wisp(m.connect(host, port).await?),
        };
        let io = if tls {
            let config = self.tls.borrow().clone();
            match with_timeout(TlsConn::connect(stream, host, config), 20_000).await {
                Some(r) => Io::Tls(Box::new(r?)),
                None => return Err(format!("The TLS handshake with {host} timed out")),
            }
        } else {
            Io::Plain(stream)
        };
        Ok(Conn { io, buf: vec![] })
    }

    async fn exchange(&self, conn: &mut Conn, request: &[u8], method: &str) -> Result<(ResponseHead, Vec<u8>, bool), ExchangeError> {
        conn.write(request).await.map_err(ExchangeError::Stale)?;
        let mut buf: Vec<u8> = Vec::new();
        let mut got_any = false;
        let (head, mut rest) = loop {
            if let Some((head, len)) = http::parse_head(&buf).map_err(ExchangeError::Fatal)? {
                let rest = buf.split_off(len);
                if (100..200).contains(&head.status) && head.status != 101 {
                    buf = rest;
                    continue;
                }
                break (head, rest);
            }
            match conn.read().await {
                Ok(Some(chunk)) => {
                    got_any = true;
                    buf.extend_from_slice(&chunk);
                }
                Ok(None) if !got_any => return Err(ExchangeError::Stale("The server closed the connection".into())),
                Ok(None) => return Err(ExchangeError::Fatal("The server closed the connection mid-response".into())),
                Err(e) if !got_any => return Err(ExchangeError::Stale(e)),
                Err(e) => return Err(ExchangeError::Fatal(e)),
            }
        };
        let framing = http::framing(method, &head).map_err(ExchangeError::Fatal)?;
        let mut body = Vec::new();
        let mut leftover = Vec::new();
        match framing {
            Framing::None => leftover = rest,
            Framing::Length(len) => {
                while rest.len() < len {
                    match conn.read().await.map_err(ExchangeError::Fatal)? {
                        Some(chunk) => rest.extend_from_slice(&chunk),
                        None => return Err(ExchangeError::Fatal("The response ended early".into())),
                    }
                }
                leftover = rest.split_off(len);
                body = rest;
            }
            Framing::Chunked => {
                let mut dec = http::ChunkedDecoder::default();
                let mut input = rest;
                loop {
                    let used = dec.feed(&input, &mut body).map_err(ExchangeError::Fatal)?;
                    if dec.done {
                        leftover = input.split_off(used);
                        break;
                    }
                    match conn.read().await.map_err(ExchangeError::Fatal)? {
                        Some(chunk) => input = chunk,
                        None => return Err(ExchangeError::Fatal("The chunked response ended early".into())),
                    }
                }
            }
            Framing::Close => {
                body = rest;
                while let Some(chunk) = conn.read().await.map_err(ExchangeError::Fatal)? {
                    body.extend_from_slice(&chunk);
                    if body.len() > http::MAX_BODY {
                        return Err(ExchangeError::Fatal("Response body is too large".into()));
                    }
                }
            }
        }
        let reusable = http::keep_alive(&head, framing) && leftover.is_empty();
        conn.buf = leftover;
        let body = http::decode_content(head.header("content-encoding"), body).map_err(ExchangeError::Fatal)?;
        Ok((head, body, reusable))
    }

    async fn acquire(&self, key: &str, dest: &str) -> SlotGuard<'_> {
        // Render-blocking resources first: documents, stylesheets, fonts, then scripts and data,
        // then images.
        let priority = match dest {
            "document" => 0,
            "style" => 1,
            "font" => 2,
            "script" | "empty" => 3,
            _ => 4,
        };
        let ticket = self.seq.get();
        self.seq.set(ticket + 1);
        let _ticket = Ticket { client: self, key: key.to_string(), ticket };
        poll_fn(|cx| {
            let mut active = self.active.borrow_mut();
            let n = active.entry(key.to_string()).or_default();
            let mut waiters = self.waiters.borrow_mut();
            let queue = waiters.entry(key.to_string()).or_default();
            queue.retain(|(_, t, _)| *t != ticket);
            let ahead = queue.iter().any(|(p, t, _)| (*p, *t) < (priority, ticket));
            if *n < PER_HOST && !ahead {
                *n += 1;
                // Free slots left: let the next waiter in line take one too.
                if *n < PER_HOST {
                    if let Some((_, _, w)) = queue.iter().min_by_key(|(p, t, _)| (*p, *t)) {
                        w.wake_by_ref();
                    }
                }
                Poll::Ready(())
            } else {
                queue.push((priority, ticket, cx.waker().clone()));
                Poll::Pending
            }
        })
        .await;
        SlotGuard { client: self, key: key.to_string() }
    }
}

struct SlotGuard<'a> {
    client: &'a Client,
    key: String,
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if let Some(n) = self.client.active.borrow_mut().get_mut(&self.key) {
            *n = n.saturating_sub(1);
        }
        // Wake the first waiter in priority order; it removes its own entry when polled.
        if let Some(waiters) = self.client.waiters.borrow().get(&self.key) {
            if let Some((_, _, w)) = waiters.iter().min_by_key(|(p, t, _)| (*p, *t)) {
                w.wake_by_ref();
            }
        }
    }
}

/// Removes a waiter's queue entry if its request is dropped while waiting.
struct Ticket<'a> {
    client: &'a Client,
    key: String,
    ticket: u64,
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        let mut waiters = self.client.waiters.borrow_mut();
        if let Some(q) = waiters.get_mut(&self.key) {
            let had = q.len();
            q.retain(|(_, t, _)| *t != self.ticket);
            if q.len() != had {
                if let Some((_, _, w)) = q.iter().min_by_key(|(p, t, _)| (*p, *t)) {
                    w.wake_by_ref();
                }
            }
        }
    }
}

/// A future that resolves after `ms` milliseconds (uses the global `setTimeout`).
pub fn sleep(ms: i32) -> wasm_bindgen_futures::JsFuture {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        if let Ok(f) = js_sys::Reflect::get(&global, &"setTimeout".into()) {
            if let Some(f) = f.dyn_ref::<js_sys::Function>() {
                let _ = f.call2(&global, &resolve, &JsValue::from(ms));
            }
        }
    });
    wasm_bindgen_futures::JsFuture::from(promise)
}

/// Run a future with a deadline. `None` means it timed out.
pub async fn with_timeout<T>(fut: impl Future<Output = T>, ms: i32) -> Option<T> {
    let mut fut = pin!(fut);
    let mut timer = pin!(sleep(ms));
    poll_fn(|cx| {
        if let Poll::Ready(v) = fut.as_mut().poll(cx) {
            return Poll::Ready(Some(v));
        }
        if timer.as_mut().poll(cx).is_ready() {
            return Poll::Ready(None);
        }
        Poll::Pending
    })
    .await
}

#[allow(dead_code)]
fn _assert_rc(_: Rc<Client>) {}
