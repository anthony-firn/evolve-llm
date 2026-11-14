//! Glue between Blitz's provider traits and the engine's single-threaded event loop.
//!
//! Blitz requires `Send + Sync` providers, but everything here runs on one thread (a Web
//! Worker in the browser). Providers are zero-sized handles that push work into thread-local
//! queues; `Engine::tick` drains them.

use crate::net::{FetchRequest, FetchResponse, Mode};
use blitz_traits::navigation::{NavigationOptions, NavigationProvider};
use blitz_traits::net::{Body, NetHandler, NetProvider, Request};
use blitz_traits::shell::ShellProvider;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use url::Url;

/// Fetches resources for the engine. In the browser this is the rustls/Wisp client; natively
/// (tests) it is an in-memory site.
pub trait Loader {
    fn fetch(&self, req: FetchRequest, done: Box<dyn FnOnce(Result<FetchResponse, String>)>);
    fn script_cookies(&self, url: &Url) -> String;
    fn set_script_cookie(&self, url: &Url, value: &str);
}

pub enum Task {
    /// A Blitz subresource (stylesheet, image, font, iframe)
    Resource { page: u64, handler: Box<dyn NetHandler>, result: Result<FetchResponse, String> },
    Navigation { nav: u64, result: Result<FetchResponse, String> },
    Script { page: u64, index: usize, result: Result<FetchResponse, String> },
    JsFetch { page: u64, id: u32, result: Result<FetchResponse, String> },
    Module { page: u64, sender: futures::channel::oneshot::Sender<Result<FetchResponse, String>>, result: Result<FetchResponse, String> },
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum OutEvent {
    Title { title: String },
    Url { url: String, can_back: bool, can_forward: bool },
    Loading { loading: bool, url: String },
    Cursor { cursor: String },
    Console { level: String, text: String },
    Dialog { kind: String, text: String },
    Clipboard { text: String },
    Storage,
    Cookies,
    Error { text: String },
    Download { url: String, name: String },
    Ime { enabled: bool, x: f32, y: f32, w: f32, h: f32 },
    /// The person clicked a <select>: show a menu. `id` is passed back to chooseOption.
    Select { id: f64, x: f32, y: f32, w: f32, h: f32, multiple: bool, options: Vec<SelectOption> },
    /// Progress of the network route: route "tor" | "relay", state "connecting" | "ready" | "failed".
    Network { route: String, state: String, detail: String },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SelectOption {
    pub label: String,
    pub value: String,
    pub selected: bool,
    pub disabled: bool,
    pub group: Option<String>,
}

pub struct NavRequest {
    pub options: NavigationOptions,
    pub replace: bool,
    pub user_activated: bool,
}

thread_local! {
    pub static LOADER: RefCell<Option<Rc<dyn Loader>>> = const { RefCell::new(None) };
    pub static TASKS: RefCell<VecDeque<Task>> = RefCell::new(VecDeque::new());
    pub static NAVS: RefCell<Vec<NavRequest>> = const { RefCell::new(Vec::new()) };
    pub static OUT: RefCell<Vec<OutEvent>> = const { RefCell::new(Vec::new()) };
    pub static REDRAW: Cell<bool> = const { Cell::new(true) };
    pub static WAKER: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    /// Set while a user gesture is being handled, so navigations it causes are marked as such.
    pub static USER_GESTURE: Cell<bool> = const { Cell::new(false) };
    pub static PENDING_FETCHES: Cell<usize> = const { Cell::new(0) };
}

/// Ask the host to run `Engine::tick` soon.
pub fn wake() {
    WAKER.with(|w| {
        if let Some(w) = &*w.borrow() {
            w()
        }
    });
}

pub fn push_task(task: Task) {
    TASKS.with(|t| t.borrow_mut().push_back(task));
    wake();
}

pub fn emit(event: OutEvent) {
    OUT.with(|o| o.borrow_mut().push(event));
    wake();
}

pub fn request_redraw() {
    REDRAW.with(|r| r.set(true));
    wake();
}

pub fn loader() -> Rc<dyn Loader> {
    LOADER.with(|l| l.borrow().clone().expect("loader installed"))
}

/// Start a fetch whose result is delivered as a task.
pub fn start_fetch(req: FetchRequest, make_task: impl FnOnce(Result<FetchResponse, String>) -> Task + 'static) {
    PENDING_FETCHES.with(|p| p.set(p.get() + 1));
    loader().fetch(
        req,
        Box::new(move |result| {
            PENDING_FETCHES.with(|p| p.set(p.get().saturating_sub(1)));
            push_task(make_task(result));
        }),
    );
}

/// Blitz's subresource fetcher for one page.
pub struct PageNet {
    pub page: u64,
    pub initiator: Url,
}

impl NetProvider for PageNet {
    fn fetch(&self, _doc_id: usize, request: Request, handler: Box<dyn NetHandler>) {
        let dest = guess_dest(&request.url);
        let mut req = FetchRequest::get(request.url.clone(), Some(self.initiator.clone()), Mode::NoCors, dest);
        req.method = request.method.to_string();
        for (k, v) in request.headers.iter() {
            if let Ok(v) = v.to_str() {
                req.headers.push((k.to_string(), v.to_string()));
            }
        }
        if let Body::Bytes(b) = &request.body {
            req.body = b.to_vec();
        }
        let page = self.page;
        start_fetch(req, move |result| Task::Resource { page, handler, result });
    }
}

fn guess_dest(url: &Url) -> &'static str {
    let path = url.path().to_ascii_lowercase();
    let query = url.query().unwrap_or("").to_ascii_lowercase();
    if path.ends_with(".css") || path.contains(".css") || query.contains("only=styles") || query.contains("type=css") || path.ends_with("/css") {
        "style"
    } else if [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".avif", ".bmp"].iter().any(|e| path.ends_with(e)) {
        "image"
    } else if [".woff", ".woff2", ".ttf", ".otf"].iter().any(|e| path.ends_with(e)) {
        "font"
    } else {
        "empty"
    }
}

pub struct PageNavigation;

impl NavigationProvider for PageNavigation {
    fn navigate_to(&self, options: NavigationOptions) {
        let user_activated = USER_GESTURE.with(|g| g.get());
        NAVS.with(|n| n.borrow_mut().push(NavRequest { options, replace: false, user_activated }));
        wake();
    }
}

pub struct PageShell;

impl ShellProvider for PageShell {
    fn request_redraw(&self) {
        request_redraw();
    }
    fn set_cursor(&self, icon: Option<cursor_icon::CursorIcon>) {
        let cursor = icon.map(|c| c.name().to_string()).unwrap_or_else(|| "default".into());
        emit(OutEvent::Cursor { cursor });
    }
    fn set_window_title(&self, _title: String) {
        // The engine reads the title from the document itself (see Engine::tick): Blitz reports
        // any <title> that changes, including detached ones built by scripts.
    }
    fn set_ime_enabled(&self, enabled: bool) {
        emit(OutEvent::Ime { enabled, x: 0.0, y: 0.0, w: 0.0, h: 0.0 });
    }
    fn set_ime_cursor_area(&self, x: f32, y: f32, w: f32, h: f32) {
        emit(OutEvent::Ime { enabled: true, x, y, w, h });
    }
    fn set_clipboard_text(&self, text: String) -> Result<(), blitz_traits::shell::ClipboardError> {
        emit(OutEvent::Clipboard { text });
        Ok(())
    }
}
