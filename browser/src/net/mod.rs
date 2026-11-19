//! Networking: HTTP/2 or HTTP/1.1 over TLS over Tor or Wisp streams, with browser cookie and
//! CORS rules.

pub mod headers;
pub mod http;
pub mod http2;
pub mod policy;
pub mod tls;
pub mod wisp;
#[cfg(target_arch = "wasm32")]
pub mod client;
#[cfg(target_arch = "wasm32")]
pub mod tor;

use url::Url;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// Top-level document navigation
    Navigate,
    /// Subresources requested by the engine (images, stylesheets, scripts, fonts)
    NoCors,
    /// fetch() and XMLHttpRequest from page scripts
    Cors,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Credentials {
    Omit,
    SameOrigin,
    Include,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Redirect {
    Follow,
    Manual,
    Error,
}

#[derive(Debug, Clone)]
pub struct FetchRequest {
    pub url: Url,
    pub method: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// The document that started the request; `None` for the address bar.
    pub initiator: Option<Url>,
    pub mode: Mode,
    pub credentials: Credentials,
    pub redirect: Redirect,
    /// The Fetch "destination": document, script, style, image, font, empty
    pub dest: &'static str,
    /// Whether a person's gesture started this navigation
    pub user_activated: bool,
}

impl FetchRequest {
    pub fn get(url: Url, initiator: Option<Url>, mode: Mode, dest: &'static str) -> Self {
        FetchRequest {
            url,
            method: "GET".into(),
            headers: vec![],
            body: vec![],
            initiator,
            mode,
            credentials: if mode == Mode::Cors { Credentials::SameOrigin } else { Credentials::Include },
            redirect: Redirect::Follow,
            dest,
            user_activated: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FetchResponse {
    pub url: Url,
    pub status: u16,
    pub status_text: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub redirected: bool,
}

impl FetchResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LogEntry {
    pub method: String,
    pub url: String,
    pub status: u16,
    pub bytes: usize,
    pub ms: f64,
    pub dest: &'static str,
    pub error: Option<String>,
}

