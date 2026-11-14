//! A native harness: an in-memory website and a driver for the engine (used by tests).

use crate::engine::Engine;
use crate::host::{self, Loader};
use crate::net::{FetchRequest, FetchResponse};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use url::Url;

#[derive(Default)]
pub struct Site {
    pub pages: RefCell<HashMap<String, (u16, String, Vec<u8>, Vec<(String, String)>)>>,
    pub requests: RefCell<Vec<FetchRequest>>,
    pub cookies: RefCell<cookie_store::CookieStore>,
}

impl Site {
    pub fn add(&self, url: &str, content_type: &str, body: &str) {
        self.pages.borrow_mut().insert(url.to_string(), (200, content_type.to_string(), body.as_bytes().to_vec(), vec![]));
    }
    pub fn add_with(&self, url: &str, status: u16, content_type: &str, body: &str, headers: &[(&str, &str)]) {
        let headers = headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        self.pages.borrow_mut().insert(url.to_string(), (status, content_type.to_string(), body.as_bytes().to_vec(), headers));
    }
}

pub struct SiteLoader(pub Rc<Site>);

impl Loader for SiteLoader {
    fn fetch(&self, req: FetchRequest, done: Box<dyn FnOnce(Result<FetchResponse, String>)>) {
        self.0.requests.borrow_mut().push(req.clone());
        let mut key = req.url.clone();
        key.set_fragment(None);
        let found = self.0.pages.borrow().get(key.as_str()).cloned();
        let result = match found {
            Some((status, ct, body, extra)) => {
                let mut headers = vec![("Content-Type".to_string(), ct)];
                headers.extend(extra);
                for (k, v) in &headers {
                    if k.eq_ignore_ascii_case("set-cookie") {
                        let _ = self.0.cookies.borrow_mut().parse(v, &req.url);
                    }
                }
                Ok(FetchResponse { url: req.url.clone(), status, status_text: "OK".into(), headers, body, redirected: false })
            }
            None => Err(format!("No route for {}", req.url)),
        };
        done(result);
    }
    fn script_cookies(&self, url: &Url) -> String {
        self.0.cookies.borrow().matches(url).into_iter().filter(|c| c.http_only() != Some(true)).map(|c| format!("{}={}", c.name(), c.value())).collect::<Vec<_>>().join("; ")
    }
    fn set_script_cookie(&self, url: &Url, value: &str) {
        let _ = self.0.cookies.borrow_mut().parse(value, url);
    }
}

pub struct Harness {
    pub engine: Engine,
    pub site: Rc<Site>,
    pub events: Vec<host::OutEvent>,
}

impl Harness {
    pub fn new() -> Self {
        let site = Rc::new(Site::default());
        host::LOADER.with(|l| *l.borrow_mut() = Some(Rc::new(SiteLoader(site.clone()))));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fonts");
        let fonts = crate::fonts::font_context(&crate::fonts::load_dir(&dir));
        let mut engine = Engine::new(fonts, None);
        engine.resize(800, 600, 1.0);
        Harness { engine, site, events: vec![] }
    }

    /// Tick until idle (no tasks, no due timers), advancing a virtual clock.
    pub fn settle(&mut self) {
        let mut now = crate::clock::now();
        for _ in 0..200 {
            let next = self.engine.tick(now);
            self.events.extend(host::OUT.with(|o| o.borrow_mut().drain(..).collect::<Vec<_>>()));
            let busy = host::TASKS.with(|t| !t.borrow().is_empty()) || host::NAVS.with(|n| !n.borrow().is_empty());
            if !busy && (next < 0.0 || next > 1000.0) {
                break;
            }
            now += next.max(1.0).min(50.0);
        }
    }

    pub fn open(&mut self, url: &str) {
        self.engine.navigate_typed(url);
        self.settle();
    }

    pub fn eval(&mut self, code: &str) -> String {
        let r = self.engine.evaluate(code);
        self.settle();
        r
    }

    pub fn console(&self) -> Vec<String> {
        self.events.iter().filter_map(|e| match e { host::OutEvent::Console { level, text } => Some(format!("{level}: {text}")), _ => None }).collect()
    }
}

impl Default for Harness {
    fn default() -> Self {
        Self::new()
    }
}
