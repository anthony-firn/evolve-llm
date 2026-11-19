//! The browser engine: navigation and history, page lifecycle, script execution, input events,
//! the event loop, rendering and the agent's actions.

use crate::agent;
use crate::host::{self, NavRequest, OutEvent, PageNavigation, PageNet, PageShell, Task};
use crate::js::{self, HistoryOp, JsRuntime, PageState};
use crate::net::{Credentials, FetchRequest, FetchResponse, Mode, Redirect};
use anyrender::{ImageRenderer, PaintScene};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::{local_name, BaseDocument, DocumentConfig, EventDriver, EventHandler, NodeData, NodeId, ScrollBehavior, ScrollLogicalPosition};
use blitz_traits::events::{
    BlitzImeEvent, BlitzKeyEvent, BlitzPointerEvent, BlitzPointerId, BlitzWheelDelta, BlitzWheelEvent, DomEvent, DomEventData, EventState, KeyState,
    MouseEventButton, MouseEventButtons, PointerCoords, PointerDetails, UiEvent,
};
use blitz_traits::navigation::NavigationOptions;
use blitz_traits::net::{Body, EntryValue};
use blitz_traits::shell::{ColorScheme, Viewport};
use boa_engine::JsValue;
use keyboard_types::{Code, Key, Location, Modifiers, NamedKey};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;
use url::Url;

const EXTRA_UA_CSS: &str = "noscript { display: none !important; } evolve-fragment, evolve-shadow-root { display: contents; } template { display: none !important; }
select { display: inline-block; box-sizing: border-box; border: 1px solid #767676; border-radius: 3px; padding: 1px 22px 1px 4px; background: #fff linear-gradient(45deg, transparent 50%, #555 50%) no-repeat right 9px center / 5px 5px; color: #000; font: 13.33px sans-serif; min-width: 3em; min-height: 1.5em; white-space: nowrap; overflow: hidden; vertical-align: middle; }
select > option, select > optgroup { display: none; }
select > option[selected], select > optgroup > option[selected] { display: block; }
select:not(:has(option[selected])) > option:first-of-type { display: block; }
select[multiple], select[size] { padding: 1px 4px; background-image: none; }
select[multiple] > option, select[size] > option, select[multiple] > optgroup, select[size] > optgroup { display: block; }";
const HIGH_FREQUENCY: &[&str] = &["pointermove", "mousemove", "pointerover", "pointerout", "mouseover", "mouseout", "pointerenter", "pointerleave", "mouseenter", "mouseleave", "touchmove", "wheel"];

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScriptKind {
    Classic,
    Module,
}

#[derive(Debug, Clone, PartialEq)]
enum ScriptStatus {
    Fetching,
    Ready(String),
    Failed(String),
    Done,
}

struct ScriptItem {
    node: NodeId,
    kind: ScriptKind,
    url: Option<Url>,
    status: ScriptStatus,
    ordered_dynamic: bool,
}

pub struct Page {
    pub id: u64,
    pub doc: Box<BaseDocument>,
    pub state: PageState,
    pub js: Option<JsRuntime>,
    /// Parser-inserted scripts that must run in order (sync, then deferred and modules)
    ordered: Vec<ScriptItem>,
    next_ordered: usize,
    /// async and dynamically inserted scripts, run as they arrive
    unordered: Vec<ScriptItem>,
    dom_loaded: bool,
    load_fired: bool,
    dom_loaded_at: f64,
    last_frame_time: f64,
    /// iframe nodes whose sub-document scripts have already been run in their realm.
    framed: HashSet<NodeId>,
}

struct PendingNav {
    id: u64,
    url: Url,
    replace: bool,
    /// Set for back/forward: the history index this navigation lands on
    history_index: Option<usize>,
    referrer: String,
}

#[derive(Clone)]
struct HistoryEntry {
    url: Url,
    page: Option<u64>,
    state: Option<String>,
}

pub struct Engine {
    identity: crate::identity::Identity,
    pub css_width: u32,
    pub css_height: u32,
    pub dpr: f32,
    fonts: parley::FontContext,
    pub page: Option<Page>,
    next_page: u64,
    nav_seq: u64,
    pending: Option<PendingNav>,
    history: Vec<HistoryEntry>,
    history_index: usize,
    renderer: Option<VelloCpuImageRenderer>,
    frame: Vec<u8>,
    frame_ready: bool,
    frame_size: (u32, u32),
    local: HashMap<String, Rc<RefCell<HashMap<String, String>>>>,
    session: HashMap<String, Rc<RefCell<HashMap<String, String>>>>,
    /// Incremented on every person input; agent actions on an older snapshot are refused.
    pub version: u64,
    snapshot_version: u64,
    snapshot_page: u64,
    snapshot_refs: Vec<NodeId>,
    loading: bool,
    suppress_text: Option<String>,
    last_url_event: String,
    last_title: String,
}

impl Engine {
    pub fn can_configure(&self) -> bool { self.page.is_none() && self.pending.is_none() }

    /// Select an identity before navigation so every realm uses the same profile.
    pub fn set_identity(&mut self, profile: &str) -> Result<(), &'static str> {
        let identity = crate::identity::Identity::parse(profile)?;
        if !self.can_configure() {
            return Err("Select the browser identity before navigating; restart the browser to change it");
        }
        self.identity = identity;
        Ok(())
    }

    pub fn new(fonts: parley::FontContext, storage_json: Option<&str>) -> Self {
        let mut local = HashMap::new();
        if let Some(json) = storage_json {
            if let Ok(map) = serde_json::from_str::<HashMap<String, HashMap<String, String>>>(json) {
                for (origin, items) in map {
                    local.insert(origin, Rc::new(RefCell::new(items)));
                }
            }
        }
        Engine {
            css_width: 1024,
            css_height: 768,
            dpr: 1.0,
            fonts,
            page: None,
            next_page: 1,
            nav_seq: 0,
            pending: None,
            history: vec![],
            history_index: 0,
            renderer: None,
            frame: vec![],
            frame_ready: false,
            frame_size: (0, 0),
            local,
            session: HashMap::new(),
            version: 1,
            snapshot_version: 0,
            snapshot_page: 0,
            snapshot_refs: vec![],
            loading: false,
            suppress_text: None,
            last_url_event: String::new(),
            last_title: String::new(),
            identity: crate::identity::Identity::default(),
        }
    }

    fn device_size(&self) -> (u32, u32) {
        (((self.css_width as f32) * self.dpr).round().max(1.0) as u32, ((self.css_height as f32) * self.dpr).round().max(1.0) as u32)
    }

    fn blitz_viewport(&self) -> Viewport {
        let (w, h) = self.device_size();
        Viewport::new(w, h, self.dpr, ColorScheme::Light)
    }

    pub fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        let dpr = if dpr.is_finite() { dpr.clamp(0.5, 3.0) } else { 1.0 };
        if (width, height, dpr) == (self.css_width, self.css_height, self.dpr) {
            return;
        }
        self.css_width = width.clamp(1, 8192);
        self.css_height = height.clamp(1, 8192);
        self.dpr = dpr;
        let viewport = self.blitz_viewport();
        if let Some(page) = &mut self.page {
            page.doc.set_viewport(viewport);
            page.state.viewport = (self.css_width as f32, self.css_height as f32, dpr);
            page.state.layout_dirty = true;
            if let Some(js) = &page.js {
                let _s = js::enter(&mut page.doc, &mut page.state);
                let _ = js.evolve("resized", &[]);
            }
        }
        host::request_redraw();
    }

    // ------------------------------------------------------------------ navigation

    /// Navigate from the address bar.
    pub fn navigate_typed(&mut self, input: &str) {
        let input = input.trim();
        if input.is_empty() {
            return;
        }
        let url = Url::parse(input)
            .ok()
            .filter(|u| matches!(u.scheme(), "http" | "https" | "data" | "about"))
            .or_else(|| {
                if input.contains(' ') || !input.contains('.') && !input.starts_with("localhost") {
                    Url::parse_with_params("https://duckduckgo.com/html/", &[("q", input)]).ok()
                } else {
                    Url::parse(&format!("https://{input}")).ok()
                }
            });
        match url {
            Some(url) => self.start_navigation(FetchRequest::get(url, None, Mode::Navigate, "document"), false, None, true),
            None => host::emit(OutEvent::Error { text: format!("{input} is not a valid address") }),
        }
    }

    fn start_navigation(&mut self, mut req: FetchRequest, replace: bool, history_index: Option<usize>, user: bool) {
        req.user_activated = user;
        let referrer = req.initiator.as_ref().map(|u| u.to_string()).unwrap_or_default();
        // Same-document fragment navigation
        if let Some(page) = &mut self.page {
            let cur = page.state.url.clone();
            if req.method == "GET" && history_index.is_none() && req.url.fragment().is_some() && strip_fragment(&cur) == strip_fragment(&req.url) && cur != req.url {
                page.state.url = req.url.clone();
                let frag = req.url.fragment().unwrap_or("").to_string();
                page.doc.resolve(page.state.now / 1000.0);
                page.doc.scroll_to_fragment(&percent_encoding::percent_decode_str(&frag).decode_utf8_lossy());
                let init = serde_json::json!({ "oldURL": cur.as_str(), "newURL": req.url.as_str() }).to_string();
                if let Some(js) = &page.js {
                    let _s = js::enter(&mut page.doc, &mut page.state);
                    let _ = js.evolve("winEvent", &[js::js_str("hashchange"), js::js_str(&init)]);
                }
                let page_id = page.id;
                self.push_history(HistoryEntry { url: req.url.clone(), page: Some(page_id), state: None }, replace);
                self.emit_url();
                host::request_redraw();
                return;
            }
        }
        if req.url.scheme() == "javascript" {
            return;
        }
        self.nav_seq += 1;
        let id = self.nav_seq;
        self.pending = Some(PendingNav { id, url: req.url.clone(), replace, history_index, referrer });
        self.set_loading(true, req.url.as_str());
        host::start_fetch(req, move |result| Task::Navigation { nav: id, result });
    }

    fn navigation_from_options(&mut self, nav: NavRequest) {
        let NavRequest { options, replace, user_activated } = nav;
        let NavigationOptions { url, content_type, method, document_resource, .. } = options;
        let initiator = self.page.as_ref().map(|p| p.state.url.clone());
        let mut req = FetchRequest::get(url, initiator, Mode::Navigate, "document");
        req.method = method.to_string();
        let ct = content_type.unwrap_or_else(|| "application/x-www-form-urlencoded".into());
        match document_resource {
            Body::Form(form) => {
                let (body, ct) = encode_form(&form, &ct);
                req.body = body;
                req.headers.push(("Content-Type".into(), ct));
            }
            Body::Bytes(b) => {
                req.body = b.to_vec();
                req.headers.push(("Content-Type".into(), ct));
            }
            Body::Empty => {}
        }
        req.credentials = Credentials::Include;
        req.redirect = Redirect::Follow;
        self.start_navigation(req, replace, None, user_activated);
    }

    pub fn back(&mut self) {
        self.go(-1);
    }
    pub fn forward(&mut self) {
        self.go(1);
    }
    pub fn reload(&mut self) {
        if let Some(entry) = self.history.get(self.history_index).cloned() {
            let initiator = None;
            self.start_navigation(FetchRequest::get(entry.url, initiator, Mode::Navigate, "document"), true, None, true);
        }
    }
    pub fn stop(&mut self) {
        self.pending = None;
        let url = self.page.as_ref().map(|p| p.state.url.to_string()).unwrap_or_default();
        self.set_loading(false, &url);
    }

    fn go(&mut self, delta: i32) {
        let target = self.history_index as i64 + delta as i64;
        if delta == 0 {
            return self.reload();
        }
        if target < 0 || target as usize >= self.history.len() {
            return;
        }
        let target = target as usize;
        let entry = self.history[target].clone();
        let same_doc = self.page.as_ref().is_some_and(|p| Some(p.id) == entry.page);
        if same_doc {
            self.history_index = target;
            let page = self.page.as_mut().unwrap();
            let old = page.state.url.clone();
            page.state.url = entry.url.clone();
            if let Some(js) = &page.js {
                let _s = js::enter(&mut page.doc, &mut page.state);
                let state: serde_json::Value = entry.state.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(serde_json::Value::Null);
                let _ = js.evolve("winEvent", &[js::js_str("popstate"), js::js_str(&serde_json::json!({ "state": state }).to_string())]);
                if strip_fragment(&old) == strip_fragment(&entry.url) && old.fragment() != entry.url.fragment() {
                    let init = serde_json::json!({ "oldURL": old.as_str(), "newURL": entry.url.as_str() }).to_string();
                    let _ = js.evolve("winEvent", &[js::js_str("hashchange"), js::js_str(&init)]);
                }
            }
            if let Some(frag) = entry.url.fragment() {
                page.doc.scroll_to_fragment(frag);
            }
            self.emit_url();
            host::request_redraw();
            return;
        }
        self.start_navigation(FetchRequest::get(entry.url, None, Mode::Navigate, "document"), false, Some(target), true);
    }

    fn push_history(&mut self, entry: HistoryEntry, replace: bool) {
        if self.history.is_empty() {
            self.history.push(entry);
            self.history_index = 0;
        } else if replace {
            self.history[self.history_index] = entry;
        } else {
            self.history.truncate(self.history_index + 1);
            self.history.push(entry);
            self.history_index = self.history.len() - 1;
            if self.history.len() > 100 {
                self.history.remove(0);
                self.history_index -= 1;
            }
        }
        let len = self.history.len();
        if let Some(page) = &mut self.page {
            if let Some(js) = &page.js {
                let _s = js::enter(&mut page.doc, &mut page.state);
                let _ = js.evolve("historyLength", &[JsValue::from(len as f64)]);
            }
        }
    }

    fn set_loading(&mut self, loading: bool, url: &str) {
        if self.loading != loading {
            self.loading = loading;
            host::emit(OutEvent::Loading { loading, url: url.to_string() });
        }
    }

    fn emit_url(&mut self) {
        let url = self.page.as_ref().map(|p| p.state.url.to_string()).unwrap_or_default();
        let can_back = self.history_index > 0;
        let can_forward = self.history_index + 1 < self.history.len();
        let key = format!("{url}{can_back}{can_forward}");
        if key != self.last_url_event {
            self.last_url_event = key;
            host::emit(OutEvent::Url { url, can_back, can_forward });
        }
    }

    fn commit_navigation(&mut self, nav: PendingNav, result: Result<FetchResponse, String>) {
        let (url, html) = match result {
            Ok(resp) => {
                let ct = resp.header("content-type").unwrap_or("").to_ascii_lowercase();
                let mime = ct.split(';').next().unwrap_or("").trim().to_string();
                let disposition = resp.header("content-disposition").unwrap_or("").to_ascii_lowercase();
                if disposition.starts_with("attachment") {
                    let name = resp.url.path_segments().and_then(|mut s| s.next_back().map(str::to_string)).unwrap_or_else(|| "download".into());
                    host::emit(OutEvent::Download { url: resp.url.to_string(), name });
                    self.set_loading(false, resp.url.as_str());
                    return;
                }
                let html = if mime.contains("html") || mime.contains("xhtml") || (mime.is_empty() && sniff_html(&resp.body)) {
                    decode_html(&resp.body, &ct)
                } else if mime.starts_with("image/") {
                    format!(
                        "<!doctype html><title>{0}</title><body style=\"margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0e0e0e\"><img src=\"{1}\" style=\"max-width:100%;max-height:100vh\"></body>",
                        html_escape::encode_text(resp.url.path()),
                        html_escape::encode_double_quoted_attribute(resp.url.as_str())
                    )
                } else if mime.starts_with("text/") || mime.contains("json") || mime.contains("javascript") || mime.contains("xml") || mime.is_empty() {
                    let text = String::from_utf8_lossy(&resp.body);
                    format!("<!doctype html><body style=\"margin:8px\"><pre style=\"white-space:pre-wrap;word-wrap:break-word;font:13px monospace\">{}</pre></body>", html_escape::encode_text(&text))
                } else {
                    let name = resp.url.path_segments().and_then(|mut s| s.next_back().map(str::to_string)).unwrap_or_else(|| "download".into());
                    host::emit(OutEvent::Download { url: resp.url.to_string(), name });
                    self.set_loading(false, resp.url.as_str());
                    return;
                };
                (resp.url, html)
            }
            Err(error) => (nav.url.clone(), error_page(&nav.url, &error)),
        };
        // Unload the old page
        if let Some(old) = &mut self.page {
            if let Some(js) = &old.js {
                let _s = js::enter(&mut old.doc, &mut old.state);
                let _ = js.evolve("winEvent", &[js::js_str("pagehide"), js::js_str("{}")]);
            }
        }
        self.page = None;
        let page = self.build_page(url.clone(), &html, nav.referrer.clone());
        let page_id = page.id;
        self.page = Some(page);
        match nav.history_index {
            Some(i) if i < self.history.len() => {
                self.history_index = i;
                self.history[i].page = Some(page_id);
                self.history[i].url = url.clone();
            }
            _ => self.push_history(HistoryEntry { url: url.clone(), page: Some(page_id), state: None }, nav.replace),
        }
        if let Some(page) = &mut self.page {
            let len = self.history.len();
            if let Some(js) = &page.js {
                let _s = js::enter(&mut page.doc, &mut page.state);
                let _ = js.evolve("historyLength", &[JsValue::from(len as f64)]);
            }
        }
        self.last_title = String::from("\u{0}");
        self.emit_url();
        host::request_redraw();
    }

    fn build_page(&mut self, url: Url, html: &str, referrer: String) -> Page {
        let id = self.next_page;
        self.next_page += 1;
        let config = DocumentConfig {
            viewport: Some(self.blitz_viewport()),
            base_url: Some(url.to_string()),
            ua_stylesheets: vec![blitz_dom::UserAgentStylesheet::default_sheet(), blitz_dom::UserAgentStylesheet::parse(EXTRA_UA_CSS)],
            net_provider: Some(Arc::new(PageNet { page: id, initiator: url.clone() })),
            navigation_provider: Some(Arc::new(PageNavigation)),
            shell_provider: Some(Arc::new(PageShell)),
            html_parser_provider: Some(Arc::new(crate::html::ScriptHtmlProvider)),
            font_ctx: Some(self.fonts.clone()),
            ..Default::default()
        };
        let mut doc = Box::new(crate::html::document(html, config));
        let origin = url.origin().ascii_serialization();
        let local = self.local.entry(origin.clone()).or_default().clone();
        let session = self.session.entry(origin.clone()).or_default().clone();
        let mut state = PageState::new(id, url.clone(), referrer, local, session);
        state.identity = self.identity;
        state.viewport = (self.css_width as f32, self.css_height as f32, self.dpr);
        state.now = crate::clock::now();
        let js = match JsRuntime::new(id, &url).and_then(|js| {
            let _s = js::enter(&mut doc, &mut state);
            js.install_prelude().map(|_| {
                js.record_top_window(&origin);
                js
            })
        }) {
            Ok(js) => Some(js),
            Err(e) => {
                host::emit(OutEvent::Console { level: "error".into(), text: format!("JavaScript is unavailable on this page: {e}") });
                None
            }
        };
        let mut page = Page {
            id,
            doc,
            state,
            js,
            ordered: vec![],
            next_ordered: 0,
            unordered: vec![],
            dom_loaded: false,
            load_fired: false,
            dom_loaded_at: 0.0,
            last_frame_time: 0.0,
            framed: HashSet::new(),
        };
        if page.js.is_some() {
            collect_parser_scripts(&mut page);
        }
        page
    }

    // ------------------------------------------------------------------ event loop

    /// Run everything that is due. Returns ms until the next timer, or -1.
    pub fn tick(&mut self, now: f64) -> f64 {
        for _ in 0..4 {
            let tasks: Vec<Task> = host::TASKS.with(|t| t.borrow_mut().drain(..).collect());
            let navs: Vec<NavRequest> = host::NAVS.with(|n| n.borrow_mut().drain(..).collect());
            if tasks.is_empty() && navs.is_empty() {
                break;
            }
            for task in tasks {
                self.handle_task(task);
            }
            for nav in navs {
                self.navigation_from_options(nav);
            }
        }
        let Some(page) = &mut self.page else { return -1.0 };
        page.state.now = now;
        if page.js.is_some() {
            run_scripts(page);
            run_lifecycle(page, now);
            fire_timers(page, now);
            if page.state.animation_frame && now - page.last_frame_time >= 15.0 {
                page.state.animation_frame = false;
                page.last_frame_time = now;
                let js = page.js.as_ref().unwrap();
                let _s = js::enter(&mut page.doc, &mut page.state);
                let _ = js.evolve("frames", &[JsValue::from(now)]);
                report_errors(js);
            }
            flush_pending_events(page);
            if let Some(js) = &page.js {
                let _s = js::enter(&mut page.doc, &mut page.state);
                for e in js.poll() {
                    host::emit(OutEvent::Console { level: "error".into(), text: e });
                }
            }
        }
        // Requests from script
        let page = self.page.as_mut().unwrap();
        let navs: Vec<(String, bool)> = page.state.navigations.drain(..).collect();
        let ops: Vec<HistoryOp> = page.state.history.drain(..).collect();
        let page_url = page.state.url.clone();
        let page_id = page.id;
        for op in ops {
            match op {
                HistoryOp::Push { url, replace, state } => {
                    if let Ok(url) = Url::parse(&url) {
                        self.push_history(HistoryEntry { url, page: Some(page_id), state: Some(state) }, replace);
                        self.emit_url();
                    }
                }
                HistoryOp::Go(n) => self.go(n),
            }
        }
        if let Some((url, replace)) = navs.into_iter().last() {
            if let Ok(url) = Url::parse(&url) {
                let gesture = host::USER_GESTURE.with(|g| g.get());
                self.start_navigation(FetchRequest::get(url, Some(page_url), Mode::Navigate, "document"), replace, None, gesture);
            }
        }
        let Some(page) = &mut self.page else { return -1.0 };
        let title = page.doc.find_title_node().map(|n| n.text_content().split_whitespace().collect::<Vec<_>>().join(" ")).unwrap_or_default();
        if title != self.last_title {
            self.last_title = title.clone();
            host::emit(OutEvent::Title { title });
        }
        if page.state.storage_changed {
            page.state.storage_changed = false;
            host::emit(OutEvent::Storage);
        }
        // Loading indicator
        if self.pending.is_none() {
            let done = page.load_fired || page.js.is_none() && !page.doc.has_pending_critical_resources();
            let url = page.state.url.to_string();
            if done || !self.loading {
                self.set_loading(false, &url);
            }
        }
        self.render(now);
        // After render (which attaches any newly-loaded iframe sub-documents), run their scripts
        // so the frame can message the parent on the next turn.
        if let Some(page) = &mut self.page {
            if page.js.is_some() {
                run_iframe_scripts(page);
            }
        }
        self.next_delay(now)
    }

    fn handle_task(&mut self, task: Task) {
        match task {
            Task::Resource { handler, result, .. } => match result {
                Ok(resp) if resp.status < 400 || resp.status == 0 => handler.bytes(resp.url.to_string(), resp.body.into()),
                Ok(resp) => handler.bytes(resp.url.to_string(), Default::default()),
                Err(_) => handler.bytes(String::new(), Default::default()),
            },
            Task::Navigation { nav, result } => {
                if self.pending.as_ref().is_some_and(|p| p.id == nav) {
                    let pending = self.pending.take().unwrap();
                    self.commit_navigation(pending, result);
                }
            }
            Task::Script { page, index, result } => {
                let Some(p) = self.page.as_mut().filter(|p| p.id == page) else { return };
                let status = match result {
                    Ok(resp) if (200..300).contains(&resp.status) => ScriptStatus::Ready(decode_script(&resp.body, resp.header("content-type"))),
                    Ok(resp) => ScriptStatus::Failed(format!("HTTP {}", resp.status)),
                    Err(e) => ScriptStatus::Failed(e),
                };
                let (list, i) = if index >= 1 << 20 { (&mut p.unordered, index - (1 << 20)) } else { (&mut p.ordered, index) };
                if let Some(item) = list.get_mut(i) {
                    item.status = status;
                }
            }
            Task::JsFetch { page, id, result } => {
                let Some(p) = self.page.as_mut().filter(|p| p.id == page) else { return };
                let Some(js) = &p.js else { return };
                let _s = js::enter(&mut p.doc, &mut p.state);
                let args = js.with_context(|ctx| fetch_result_args(id, result, ctx));
                let _ = js.evolve("fetchDone", &args);
                report_errors(js);
            }
            Task::Module { sender, result, .. } => {
                let _ = sender.send(result);
            }
        }
    }

    fn next_delay(&self, now: f64) -> f64 {
        let Some(page) = &self.page else { return -1.0 };
        let mut next = f64::INFINITY;
        for due in page.state.timers.values() {
            next = next.min((due - now).max(0.0));
        }
        if page.state.animation_frame {
            next = next.min((16.0 - (now - page.last_frame_time)).max(0.0));
        }
        if page.doc.is_animating() {
            next = next.min(16.0);
        }
        if !page.load_fired && page.js.is_some() {
            next = next.min(250.0);
        }
        if next.is_finite() {
            next
        } else {
            -1.0
        }
    }

    // ------------------------------------------------------------------ rendering

    fn render(&mut self, now: f64) {
        let redraw = host::REDRAW.with(|r| r.replace(false));
        let Some(page) = &mut self.page else { return };
        let animating = page.doc.is_animating();
        if !redraw && !page.state.layout_dirty && !animating {
            return;
        }
        page.doc.resolve(now / 1000.0);
        page.state.layout_dirty = false;
        if page.doc.has_pending_critical_resources() {
            host::REDRAW.with(|r| r.set(true));
            return;
        }
        let (w, h) = (((self.css_width as f32) * self.dpr).round().max(1.0) as u32, ((self.css_height as f32) * self.dpr).round().max(1.0) as u32);
        let renderer = match &mut self.renderer {
            Some(r) if self.frame_size == (w, h) => r,
            _ => {
                self.renderer = Some(VelloCpuImageRenderer::new(w, h));
                self.frame_size = (w, h);
                self.renderer.as_mut().unwrap()
            }
        };
        let scale = self.dpr as f64;
        let doc: &mut BaseDocument = &mut page.doc;
        renderer.reset();
        renderer.render_to_vec(
            |scene| {
                scene.fill(peniko::Fill::NonZero, kurbo::Affine::IDENTITY, peniko::Color::WHITE, None, &kurbo::Rect::new(0.0, 0.0, w as f64, h as f64));
                blitz_paint::paint_scene(scene, doc, scale, w, h, 0, 0);
            },
            &mut self.frame,
        );
        self.frame_ready = true;
    }

    pub fn take_frame(&mut self) -> Option<&[u8]> {
        if !self.frame_ready {
            return None;
        }
        self.frame_ready = false;
        Some(&self.frame)
    }

    pub fn frame_size(&self) -> (u32, u32) {
        self.frame_size
    }

    // ------------------------------------------------------------------ input

    fn ensure_layout(&mut self) {
        if let Some(page) = &mut self.page {
            if page.state.layout_dirty {
                page.doc.resolve(page.state.now / 1000.0);
                page.state.layout_dirty = false;
            }
        }
    }

    fn dispatch_ui(&mut self, event: UiEvent, gesture: bool) {
        self.ensure_layout();
        let Some(page) = &mut self.page else { return };
        host::USER_GESTURE.with(|g| g.set(gesture));
        let Page { doc, state, js, .. } = page;
        let handler = JsHandler { js: js.as_ref(), state, suppress: &mut self.suppress_text };
        let mut driver = EventDriver::new(&mut **doc, handler);
        driver.handle_ui_event(event);
        drop(driver);
        flush_pending_events(page);
        host::USER_GESTURE.with(|g| g.set(false));
        host::request_redraw();
    }

    fn pointer_event(&self, x: f32, y: f32, button: i32, buttons: u32, mods: u32) -> BlitzPointerEvent {
        let (sx, sy) = self.page.as_ref().map(|p| {
            let s = p.doc.viewport_scroll();
            (s.x as f32, s.y as f32)
        }).unwrap_or((0.0, 0.0));
        BlitzPointerEvent {
            id: BlitzPointerId::Mouse,
            is_primary: true,
            coords: PointerCoords { page_x: x + sx, page_y: y + sy, screen_x: x, screen_y: y, client_x: x, client_y: y },
            button: match button {
                1 => MouseEventButton::Auxiliary,
                2 => MouseEventButton::Secondary,
                3 => MouseEventButton::Fourth,
                4 => MouseEventButton::Fifth,
                _ => MouseEventButton::Main,
            },
            buttons: MouseEventButtons::from_bits_truncate(buttons as u8),
            mods: modifiers(mods),
            details: PointerDetails { pressure: if buttons != 0 { 0.5 } else { 0.0 }, tangential_pressure: 0.0, tilt_x: 0, tilt_y: 0, twist: 0, altitude: std::f64::consts::FRAC_PI_2, azimuth: 0.0 },
            element: Default::default(),
            active_pointers: Default::default(),
        }
    }

    pub fn pointer(&mut self, kind: &str, x: f32, y: f32, button: i32, buttons: u32, mods: u32) {
        let ev = self.pointer_event(x, y, button, buttons, mods);
        let ui = match kind {
            "down" => {
                self.version += 1;
                UiEvent::PointerDown(ev)
            }
            "up" => {
                self.version += 1;
                UiEvent::PointerUp(ev)
            }
            "leave" => UiEvent::PointerLeave(ev),
            _ => UiEvent::PointerMove(ev),
        };
        self.dispatch_ui(ui, matches!(kind, "down" | "up"));
    }

    pub fn wheel(&mut self, x: f32, y: f32, dx: f64, dy: f64, mode: u32, mods: u32) {
        self.version += 1;
        let ev = self.pointer_event(x, y, 0, 0, mods);
        let delta = if mode == 1 { BlitzWheelDelta::Lines(dx, dy) } else if mode == 2 { BlitzWheelDelta::Pixels(dx * self.css_width as f64, dy * self.css_height as f64) } else { BlitzWheelDelta::Pixels(dx, dy) };
        let wheel = BlitzWheelEvent { delta, coords: ev.coords, buttons: ev.buttons, mods: ev.mods, element: Default::default() };
        self.dispatch_ui(UiEvent::Wheel(wheel), false);
    }

    pub fn key(&mut self, kind: &str, key: &str, code: &str, mods: u32, repeat: bool) {
        self.version += 1;
        let ev = key_event(kind, key, code, mods, repeat);
        let ui = if kind == "up" { UiEvent::KeyUp(ev) } else { UiEvent::KeyDown(ev) };
        self.dispatch_ui(ui, kind != "up");
    }

    pub fn ime_commit(&mut self, text: &str) {
        self.version += 1;
        if let Some(s) = self.suppress_text.take() {
            if s == text {
                return;
            }
        }
        self.dispatch_ui(UiEvent::Ime(BlitzImeEvent::Commit(text.to_string())), true);
    }

    pub fn ime_preedit(&mut self, text: &str) {
        let cursor = Some((text.len(), text.len()));
        self.dispatch_ui(UiEvent::Ime(BlitzImeEvent::Preedit(text.to_string(), cursor)), false);
    }

    // ------------------------------------------------------------------ data

    pub fn storage_json(&self) -> String {
        let map: HashMap<&String, HashMap<String, String>> = self.local.iter().filter(|(_, v)| !v.borrow().is_empty()).map(|(k, v)| (k, v.borrow().clone())).collect();
        serde_json::to_string(&map).unwrap_or_else(|_| "{}".into())
    }

    pub fn clear_storage(&mut self) {
        for v in self.local.values() {
            v.borrow_mut().clear();
        }
        for v in self.session.values() {
            v.borrow_mut().clear();
        }
    }

    pub fn evaluate(&mut self, code: &str) -> String {
        let Some(page) = &mut self.page else { return "No page is loaded".into() };
        let Some(js) = &page.js else { return "JavaScript is unavailable on this page".into() };
        let _s = js::enter(&mut page.doc, &mut page.state);
        let out = match js.eval_script(code, "console") {
            Ok(v) => js.with_context(|ctx| display_value(&v, ctx)),
            Err(e) => format!("Uncaught {e}"),
        };
        drop(_s);
        flush_pending_events(page);
        host::request_redraw();
        out
    }

    // ------------------------------------------------------------------ agent

    pub fn snapshot(&mut self, max_chars: usize) -> serde_json::Value {
        self.ensure_layout();
        let Some(page) = &mut self.page else {
            return serde_json::json!({ "url": "", "title": "", "text": "(no page loaded)", "refs": 0, "version": self.version, "scrollY": 0, "pageHeight": 0, "viewportHeight": self.css_height });
        };
        let snap = agent::snapshot(&page.doc, max_chars.clamp(200, 200_000));
        self.snapshot_refs = snap.refs;
        self.snapshot_version = self.version;
        self.snapshot_page = page.id;
        let title = page.doc.find_title_node().map(|n| n.text_content()).unwrap_or_default();
        let height = page.doc.try_root_element().map(|r| r.final_layout().size.height).unwrap_or(0.0);
        serde_json::json!({
            "url": page.state.url.as_str(),
            "title": title.trim(),
            "scrollY": page.doc.viewport_scroll().y,
            "pageHeight": height,
            "viewportHeight": self.css_height,
            "text": snap.text,
            "refs": self.snapshot_refs.len(),
            "version": self.version,
            "loading": self.loading,
        })
    }

    fn agent_target(&mut self, r: usize) -> Result<NodeId, String> {
        if self.snapshot_version != self.version {
            return Err("stale".into());
        }
        let page = self.page.as_ref().ok_or("no page")?;
        if page.id != self.snapshot_page {
            return Err("stale".into());
        }
        let node = *self.snapshot_refs.get(r.wrapping_sub(1)).ok_or_else(|| format!("ref {r} is not in the current snapshot"))?;
        if page.doc.get_node(node).is_none() {
            return Err("stale".into());
        }
        Ok(node)
    }

    /// Scroll the node into view and return its box.
    fn reveal(&mut self, node: NodeId) -> Result<(f32, f32, f32, f32), String> {
        self.ensure_layout();
        let page = self.page.as_mut().ok_or("no page")?;
        page.doc.scroll_into_view(node, ScrollBehavior::Instant, ScrollLogicalPosition::Center, ScrollLogicalPosition::Nearest);
        page.state.layout_dirty = true;
        self.ensure_layout();
        let page = self.page.as_ref().unwrap();
        let b = agent::element_box(&page.doc, node).ok_or("The element has no visible box")?;
        if b.2 <= 0.0 && b.3 <= 0.0 {
            return Err("The element is not visible".into());
        }
        Ok(b)
    }

    fn agent_click_at(&mut self, x: f32, y: f32) {
        let v = self.version;
        self.pointer("move", x, y, 0, 0, 0);
        self.pointer("down", x, y, 0, 1, 0);
        self.pointer("up", x, y, 0, 0, 0);
        self.version = v;
    }

    pub fn agent_click(&mut self, r: usize) -> serde_json::Value {
        let result = (|| {
            let node = self.agent_target(r)?;
            let b = self.reveal(node)?;
            let (x, y) = (b.0 + b.2 / 2.0, b.1 + b.3 / 2.0);
            let before = self.nav_seq;
            self.agent_click_at(x, y);
            let queued = host::NAVS.with(|n| !n.borrow().is_empty()) || self.page.as_ref().is_some_and(|p| !p.state.navigations.is_empty());
            Ok::<_, String>((b, self.nav_seq != before || queued))
        })();
        self.snapshot_version = self.version;
        match result {
            Ok((b, navigated)) => serde_json::json!({ "ok": true, "x": b.0 + b.2 / 2.0, "y": b.1 + b.3 / 2.0, "w": b.2, "h": b.3, "bx": b.0, "by": b.1, "navigated": navigated }),
            Err(e) => serde_json::json!({ "ok": false, "error": e }),
        }
    }

    pub fn agent_type(&mut self, r: usize, text: &str, submit: bool) -> serde_json::Value {
        let result = (|| {
            let node = self.agent_target(r)?;
            if agent::is_password(&self.page.as_ref().unwrap().doc, node) {
                return Err("password".to_string());
            }
            let b = self.reveal(node)?;
            let (x, y) = (b.0 + b.2 / 2.0, b.1 + b.3 / 2.0);
            let v = self.version;
            self.agent_click_at(x, y);
            let page = self.page.as_mut().ok_or("no page")?;
            if page.doc.get_focussed_node_id() != Some(node) {
                page.doc.set_focus_to(node);
            }
            let has_editor = page.doc.get_node(node).and_then(|n| n.element_data()).is_some_and(|el| el.text_input_data().is_some());
            if has_editor {
                page.doc.with_text_input(node, |mut d| d.select_all());
            }
            self.dispatch_ui(UiEvent::Ime(BlitzImeEvent::Commit(text.to_string())), true);
            if submit {
                self.key("down", "Enter", "Enter", 0, false);
                self.key("up", "Enter", "Enter", 0, false);
            }
            self.version = v;
            let before = self.nav_seq;
            Ok::<_, String>((b, before))
        })();
        self.snapshot_version = self.version;
        match result {
            Ok((b, _)) => serde_json::json!({ "ok": true, "x": b.0 + b.2 / 2.0, "y": b.1 + b.3 / 2.0, "w": b.2, "h": b.3, "bx": b.0, "by": b.1 }),
            Err(e) => serde_json::json!({ "ok": false, "error": e }),
        }
    }

    pub fn agent_select(&mut self, r: usize, value: &str) -> serde_json::Value {
        let result = (|| {
            let node = self.agent_target(r)?;
            let b = self.reveal(node)?;
            let page = self.page.as_mut().ok_or("no page")?;
            let js = page.js.as_ref().ok_or("JavaScript is unavailable on this page")?;
            let _s = js::enter(&mut page.doc, &mut page.state);
            let code = format!(
                "(function(){{const el=__evolve.node({node});if(!el||el.localName!=='select')return 'not a select';const v={v};const o=[...el.options].find(o=>o.value===v||o.text===v);if(!o)return 'no option '+v;el.value=o.value;el.dispatchEvent(new Event('input',{{bubbles:true}}));el.dispatchEvent(new Event('change',{{bubbles:true}}));return '';}})()",
                v = serde_json::to_string(value).unwrap(),
                node = node.as_u64()
            );
            let out = js.eval_script(&code, "evolve:agent").map_err(|e| e.to_string())?;
            let msg = js.with_context(|ctx| out.to_string(ctx).map(|s| s.to_std_string_lossy()).unwrap_or_default());
            drop(_s);
            if !msg.is_empty() {
                return Err(msg);
            }
            Ok::<_, String>(b)
        })();
        host::request_redraw();
        match result {
            Ok(b) => serde_json::json!({ "ok": true, "x": b.0 + b.2 / 2.0, "y": b.1 + b.3 / 2.0, "w": b.2, "h": b.3, "bx": b.0, "by": b.1 }),
            Err(e) => serde_json::json!({ "ok": false, "error": e }),
        }
    }

    /// The person picked option `index` (-1 to cancel) in the menu for select `id`.
    pub fn choose_option(&mut self, id: f64, index: i32) {
        self.version += 1;
        if index < 0 {
            return;
        }
        let Some(page) = &mut self.page else { return };
        let Some(js) = &page.js else { return };
        let code = format!(
            "(function(){{const el=__evolve.node({id});if(!el||el.localName!=='select')return;const o=el.options[{index}];if(!o||o.disabled)return;if(el.multiple)o.selected=!o.selected;else el.value=o.value;el.dispatchEvent(new Event('input',{{bubbles:true}}));el.dispatchEvent(new Event('change',{{bubbles:true}}));}})()",
            id = id as u64,
            index = index
        );
        let _s = js::enter(&mut page.doc, &mut page.state);
        if let Err(e) = js.eval_script(&code, "evolve:select") {
            host::emit(OutEvent::Console { level: "error".into(), text: e });
        }
        drop(_s);
        flush_pending_events(page);
        host::request_redraw();
    }

    pub fn agent_scroll(&mut self, dy: f64) -> serde_json::Value {
        let v = self.version;
        let (x, y) = (self.css_width as f32 / 2.0, self.css_height as f32 / 2.0);
        self.wheel(x, y, 0.0, dy, 0, 0);
        self.version = v;
        self.snapshot_version = self.version;
        let y = self.page.as_ref().map(|p| p.doc.viewport_scroll().y).unwrap_or(0.0);
        serde_json::json!({ "ok": true, "scrollY": y, "x": x, "y": self.css_height as f32 / 2.0 })
    }

    pub fn agent_key(&mut self, key: &str) -> serde_json::Value {
        let v = self.version;
        let code = match key {
            "Enter" | "Escape" | "Tab" | "Backspace" | "Delete" | "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight" | "Home" | "End" | "PageUp" | "PageDown" | "Space" => key,
            _ => "",
        };
        let key = if key == "Space" { " " } else { key };
        self.key("down", key, code, 0, false);
        self.key("up", key, code, 0, false);
        self.version = v;
        self.snapshot_version = self.version;
        serde_json::json!({ "ok": true })
    }
}

// ---------------------------------------------------------------------- scripts

/// Runs the inline classic scripts of any newly-attached iframe sub-document in its own realm,
/// so the frame has a real `window` that can message the parent (and be reached as
/// `contentWindow`). External and module frame scripts are not yet run here.
fn run_iframe_scripts(page: &mut Page) {
    if page.js.is_none() {
        return;
    }
    let frame_ids: Vec<NodeId> = page.doc.sub_document_node_ids();
    for node_id in frame_ids {
        if page.framed.contains(&node_id) {
            continue;
        }
        let Some((origin, url, scripts)) = collect_frame_scripts(&page.doc, node_id) else {
            continue;
        };
        page.framed.insert(node_id);
        if scripts.is_empty() {
            continue;
        }
        let js = page.js.as_ref().unwrap();
        let _s = js::enter(&mut page.doc, &mut page.state);
        if let Err(e) = js.run_frame(node_id.as_u64(), &origin, &url, &scripts) {
            host::emit(OutEvent::Console { level: "error".into(), text: format!("Frame script failed: {e}") });
        }
        report_errors(js);
    }
}

/// Collects an iframe sub-document's origin, URL and inline classic script sources.
fn collect_frame_scripts(doc: &BaseDocument, node_id: NodeId) -> Option<(String, String, Vec<String>)> {
    let guard = doc.get_node(node_id)?.element_data()?.sub_doc_data()?.inner();
    let sub: &BaseDocument = &guard;
    let url = sub.base_url().clone();
    let origin = url.origin().ascii_serialization();
    let mut scripts = Vec::new();
    let script_ids = sub.query_selector_all("script").map(|v| v.to_vec()).unwrap_or_default();
    for sid in script_ids {
        let Some(node) = sub.get_node(sid) else { continue };
        let ty = node.attr(local_name!("type")).unwrap_or("").trim().to_ascii_lowercase();
        let ty = ty.split(';').next().unwrap_or("").trim();
        let is_classic_js = matches!(ty, "" | "text/javascript" | "application/javascript" | "text/ecmascript" | "application/ecmascript" | "application/x-javascript" | "text/jscript");
        if !is_classic_js || node.attr(local_name!("src")).is_some() {
            continue;
        }
        let code = node.text_content();
        if !code.trim().is_empty() {
            scripts.push(code);
        }
    }
    Some((origin, url.to_string(), scripts))
}

fn collect_parser_scripts(page: &mut Page) {
    let ids: Vec<NodeId> = page.doc.query_selector_all("script").map(|v| v.to_vec()).unwrap_or_default();
    let base = page.doc.base_url().clone();
    let mut deferred = vec![];
    for id in ids {
        page.state.started_scripts.insert(id);
        let Some(item) = prepare_script(&mut page.doc, id, &base) else { continue };
        let node = page.doc.get_node(id).unwrap();
        let is_async = node.data.has_attr(local_name!("async")) && item.url.is_some();
        let defer = item.kind == ScriptKind::Module || (node.data.has_attr(local_name!("defer")) && item.url.is_some());
        if is_async {
            page.unordered.push(item);
        } else if defer {
            deferred.push(item);
        } else {
            page.ordered.push(item);
        }
    }
    page.ordered.extend(deferred);
    let id = page.id;
    for (i, item) in page.ordered.iter().enumerate() {
        if let Some(url) = &item.url {
            fetch_script(id, i, url.clone(), item.kind, page.state.url.clone());
        }
    }
    for (i, item) in page.unordered.iter().enumerate() {
        if let Some(url) = &item.url {
            fetch_script(id, i + (1 << 20), url.clone(), item.kind, page.state.url.clone());
        }
    }
}

/// Classify a `<script>`; import maps are applied immediately.
fn prepare_script(doc: &mut BaseDocument, id: NodeId, base: &Url) -> Option<ScriptItem> {
    let node = doc.get_node(id)?;
    let ty = node.attr(local_name!("type")).unwrap_or("").trim().to_ascii_lowercase();
    let kind = match ty.split(';').next().unwrap_or("").trim() {
        "" | "text/javascript" | "application/javascript" | "text/ecmascript" | "application/ecmascript" | "application/x-javascript" | "text/jscript" | "text/livescript" | "text/x-javascript" | "module" => {
            if ty == "module" {
                ScriptKind::Module
            } else {
                ScriptKind::Classic
            }
        }
        "importmap" => {
            let text = node.text_content();
            if let Ok(map) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(imports) = map.get("imports").and_then(|i| i.as_object()) {
                    let pairs: Vec<(String, String)> = imports.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect();
                    IMPORT_MAP.with(|m| m.borrow_mut().extend(pairs));
                }
            }
            return None;
        }
        _ => return None,
    };
    if kind == ScriptKind::Classic && node.data.has_attr(local_name!("nomodule")) {
        return None;
    }
    match node.attr(local_name!("src")) {
        Some(src) => {
            let url = base.join(src.trim()).ok()?;
            Some(ScriptItem { node: id, kind, url: Some(url), status: ScriptStatus::Fetching, ordered_dynamic: false })
        }
        None => {
            let code = node.text_content();
            Some(ScriptItem { node: id, kind, url: None, status: ScriptStatus::Ready(code), ordered_dynamic: false })
        }
    }
}

thread_local! {
    static IMPORT_MAP: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
}

/// `blob:` script URLs come from `URL.createObjectURL` in the page; read them from the page.
fn resolve_blob_scripts(page: &mut Page) {
    let Some(js) = &page.js else { return };
    let mut pending: Vec<(bool, usize, String)> = vec![];
    for (i, item) in page.ordered.iter().enumerate() {
        if let (Some(url), ScriptStatus::Fetching) = (&item.url, &item.status) {
            if url.scheme() == "blob" {
                pending.push((true, i, url.to_string()));
            }
        }
    }
    for (i, item) in page.unordered.iter().enumerate() {
        if let (Some(url), ScriptStatus::Fetching) = (&item.url, &item.status) {
            if url.scheme() == "blob" {
                pending.push((false, i, url.to_string()));
            }
        }
    }
    if pending.is_empty() {
        return;
    }
    let _s = js::enter(&mut page.doc, &mut page.state);
    let results: Vec<_> = pending
        .into_iter()
        .map(|(ordered, i, url)| {
            let text = js.evolve("blobText", &[js::js_str(&url)]).ok().and_then(|v| v.as_string().map(|s| s.to_std_string_lossy()));
            (ordered, i, text.map(ScriptStatus::Ready).unwrap_or_else(|| ScriptStatus::Failed(format!("{url} is not available"))))
        })
        .collect();
    drop(_s);
    for (ordered, i, status) in results {
        let list = if ordered { &mut page.ordered } else { &mut page.unordered };
        list[i].status = status;
    }
}

fn fetch_script(page: u64, index: usize, url: Url, kind: ScriptKind, initiator: Url) {
    if url.scheme() == "blob" {
        return; // resolved from the page by resolve_blob_scripts
    }
    let mode = if kind == ScriptKind::Module { Mode::Cors } else { Mode::NoCors };
    let mut req = FetchRequest::get(url, Some(initiator), mode, "script");
    req.credentials = if kind == ScriptKind::Module { Credentials::SameOrigin } else { Credentials::Include };
    host::start_fetch(req, move |result| Task::Script { page, index, result });
}

fn execute(page: &mut Page, item_node: NodeId, kind: ScriptKind, url: Option<&Url>, code: &str) {
    let Some(js) = &page.js else { return };
    IMPORT_MAP.with(|m| {
        let pairs: Vec<_> = m.borrow_mut().drain(..).collect();
        js.modules.imports.borrow_mut().extend(pairs);
    });
    let name = url.map(|u| u.to_string()).unwrap_or_else(|| format!("{}#inline-{}", page.state.url, item_node.as_u64()));
    let _s = js::enter(&mut page.doc, &mut page.state);
    let _ = js.evolve("currentScript", &[JsValue::from(if kind == ScriptKind::Classic { item_node.as_u64() as f64 } else { -1.0 })]);
    let result = match kind {
        ScriptKind::Classic => js.eval_script(code, &name).map(|_| ()),
        ScriptKind::Module => js.run_module(code, &name),
    };
    let _ = js.evolve("currentScript", &[JsValue::from(-1.0)]);
    if let Err(e) = result {
        host::emit(OutEvent::Console { level: "error".into(), text: format!("Uncaught {e}\n    at {name}") });
    }
    report_errors(js);
    if url.is_some() {
        let _ = js.evolve("scriptEvent", &[JsValue::from(item_node.as_u64() as f64), JsValue::from(true)]);
    }
    drop(_s);
    // document.write output goes right after the script.
    if let Some(html) = page.state.write_buffer.take() {
        insert_written_html(page, item_node, &html);
    }
    flush_pending_events(page);
}

fn insert_written_html(page: &mut Page, after: NodeId, html: &str) {
    let doc: &mut BaseDocument = &mut page.doc;
    let Some(parent) = doc.get_node(after).and_then(|n| n.parent) else { return };
    let mut m = doc.mutate();
    let tmp = m.create_element(blitz_dom::QualName::new(None, blitz_dom::ns!(html), blitz_dom::LocalName::from("div")), vec![]);
    m.set_inner_html(tmp, html);
    let kids = m.child_ids(tmp).to_vec();
    for k in &kids {
        m.remove_node(*k);
    }
    m.insert_nodes_after(after, &kids);
    drop(m);
    let _ = parent;
    // Scripts in the written markup run next, in order.
    let base = page.doc.base_url().clone();
    let mut new_items = vec![];
    for k in kids {
        let mut ids = vec![];
        if page.doc.get_node(k).is_some_and(|n| n.data.is_element_with_tag_name(&local_name!("script"))) {
            ids.push(k);
        }
        ids.extend(page.doc.query_selector_all_in(k, "script").map(|v| v.to_vec()).unwrap_or_default());
        for id in ids {
            if page.state.started_scripts.insert(id) {
                if let Some(item) = prepare_script(&mut page.doc, id, &base) {
                    new_items.push(item);
                }
            }
        }
    }
    let at = page.next_ordered;
    let count = new_items.len();
    for (j, item) in new_items.into_iter().enumerate() {
        page.ordered.insert(at + j, item);
    }
    if count > 0 {
        // Indices shifted: refetch any external scripts by their new positions.
        let id = page.id;
        for i in at..page.ordered.len() {
            if let (Some(url), ScriptStatus::Fetching) = (&page.ordered[i].url, &page.ordered[i].status) {
                fetch_script(id, i, url.clone(), page.ordered[i].kind, page.state.url.clone());
            }
        }
    }
    page.state.layout_dirty = true;
}

fn run_scripts(page: &mut Page) {
    // Dynamically inserted scripts
    let dynamic: Vec<NodeId> = page.state.dynamic_scripts.drain(..).collect();
    let base = page.doc.base_url().clone();
    for id in dynamic {
        if !page.state.started_scripts.insert(id) {
            continue;
        }
        if let Some(mut item) = prepare_script(&mut page.doc, id, &base) {
            item.ordered_dynamic = item.url.is_some()
                && !page.state.force_async_scripts.contains(&id)
                && !page.doc.get_node(id).is_some_and(|n| n.data.has_attr(local_name!("async")));
            let index = page.unordered.len();
            if let Some(url) = &item.url {
                fetch_script(page.id, index + (1 << 20), url.clone(), item.kind, page.state.url.clone());
            }
            page.unordered.push(item);
        }
    }
    resolve_blob_scripts(page);
    // Ordered scripts
    for _ in 0..64 {
        let Some(item) = page.ordered.get(page.next_ordered) else { break };
        match item.status.clone() {
            ScriptStatus::Fetching => break,
            ScriptStatus::Done => page.next_ordered += 1,
            ScriptStatus::Failed(e) => {
                let (node, url) = (item.node, item.url.clone());
                page.ordered[page.next_ordered].status = ScriptStatus::Done;
                page.next_ordered += 1;
                script_failed(page, node, url.as_ref(), &e);
            }
            ScriptStatus::Ready(code) => {
                let (node, kind, url) = (item.node, item.kind, item.url.clone());
                page.ordered[page.next_ordered].status = ScriptStatus::Done;
                page.next_ordered += 1;
                execute(page, node, kind, url.as_ref(), &code);
            }
        }
    }
    // Unordered scripts
    // Explicit async=false scripts download concurrently but execute in insertion order.
    // A waiting ordered script must not block the ordinary async scripts in the same list.
    let mut ordered_blocked = false;
    for i in 0..page.unordered.len() {
        if page.unordered[i].ordered_dynamic {
            if ordered_blocked { continue; }
            if page.unordered[i].status == ScriptStatus::Fetching {
                ordered_blocked = true;
                continue;
            }
        }
        match page.unordered[i].status.clone() {
            ScriptStatus::Ready(code) => {
                let (node, kind, url) = (page.unordered[i].node, page.unordered[i].kind, page.unordered[i].url.clone());
                page.unordered[i].status = ScriptStatus::Done;
                execute(page, node, kind, url.as_ref(), &code);
            }
            ScriptStatus::Failed(e) => {
                let (node, url) = (page.unordered[i].node, page.unordered[i].url.clone());
                page.unordered[i].status = ScriptStatus::Done;
                script_failed(page, node, url.as_ref(), &e);
            }
            _ => {}
        }
    }
}

fn script_failed(page: &mut Page, node: NodeId, url: Option<&Url>, error: &str) {
    host::emit(OutEvent::Console { level: "error".into(), text: format!("Failed to load script {}: {error}", url.map(|u| u.as_str()).unwrap_or("")) });
    if let Some(js) = &page.js {
        let _s = js::enter(&mut page.doc, &mut page.state);
        let _ = js.evolve("scriptEvent", &[JsValue::from(node.as_u64() as f64), JsValue::from(false)]);
    }
}

fn run_lifecycle(page: &mut Page, now: f64) {
    let Some(js) = &page.js else { return };
    if !page.dom_loaded && page.next_ordered >= page.ordered.len() {
        page.dom_loaded = true;
        page.dom_loaded_at = now;
        page.state.ready_state = "interactive";
        let _s = js::enter(&mut page.doc, &mut page.state);
        let _ = js.evolve("upgrade", &[]);
        let _ = js.evolve("docEvent", &[js::js_str("readystatechange")]);
        let _ = js.evolve("docEvent", &[js::js_str("DOMContentLoaded")]);
        report_errors(js);
    }
    let resources_done = host::PENDING_FETCHES.with(|p| p.get()) == 0 && !page.doc.has_pending_critical_resources();
    let scripts_done = page.unordered.iter().all(|s| s.status == ScriptStatus::Done) && !js.modules_pending();
    if page.dom_loaded && !page.load_fired && ((resources_done && scripts_done) || now - page.dom_loaded_at > 8000.0) {
        page.load_fired = true;
        page.state.ready_state = "complete";
        let _s = js::enter(&mut page.doc, &mut page.state);
        let _ = js.evolve("docEvent", &[js::js_str("readystatechange")]);
        let _ = js.evolve("winEvent", &[js::js_str("load"), js::js_str("{}")]);
        let _ = js.evolve("winEvent", &[js::js_str("pageshow"), js::js_str("{\"persisted\":false}")]);
        report_errors(js);
    }
}

fn fire_timers(page: &mut Page, now: f64) {
    for _ in 0..200 {
        let due = page.state.timers.iter().filter(|(_, d)| **d <= now).min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(id, _)| *id);
        let Some(id) = due else { break };
        page.state.timers.remove(&id);
        let js = page.js.as_ref().unwrap();
        let _s = js::enter(&mut page.doc, &mut page.state);
        let _ = js.evolve("fireTimer", &[JsValue::from(id as f64)]);
        report_errors(js);
    }
}

/// Dispatch events generated by default actions run from script (e.g. input events after
/// `el.click()` on a checkbox).
fn flush_pending_events(page: &mut Page) {
    for _ in 0..8 {
        let events: Vec<DomEvent> = page.state.pending_events.drain(..).collect();
        if events.is_empty() {
            break;
        }
        let Page { doc, state, js, .. } = page;
        let mut suppress = None;
        let handler = JsHandler { js: js.as_ref(), state, suppress: &mut suppress };
        let mut driver = EventDriver::new(&mut **doc, handler);
        for e in events {
            driver.handle_dom_event(e);
        }
    }
}

fn report_errors(js: &JsRuntime) {
    for e in js.executor.errors.borrow_mut().drain(..) {
        host::emit(OutEvent::Console { level: "error".into(), text: format!("Uncaught (in promise) {e}") });
    }
}

// ---------------------------------------------------------------------- events into script

struct JsHandler<'a> {
    js: Option<&'a JsRuntime>,
    state: &'a mut PageState,
    suppress: &'a mut Option<String>,
}

impl EventHandler for JsHandler<'_> {
    fn handle_event(&mut self, _chain: &[NodeId], event: &mut DomEvent, doc: &mut dyn blitz_dom::Document, event_state: &mut EventState) {
        // Typed text arrives through imeCommit (from the panel's hidden textarea), so printable
        // keydowns reach scripts but must not also insert characters.
        let text_key = matches!(&event.data, DomEventData::KeyDown(k) if matches!(&k.key, Key::Character(_)) && !k.modifiers.intersects(Modifiers::CONTROL | Modifiers::META));
        self.handle(event, doc, event_state);
        if text_key {
            event_state.prevent_default();
        }
    }
}

impl JsHandler<'_> {
    fn handle(&mut self, event: &mut DomEvent, doc: &mut dyn blitz_dom::Document, event_state: &mut EventState) {
        let Some(js) = self.js else { return };
        let name = event.data.name();
        if HIGH_FREQUENCY.contains(&name) && !self.state.listened.contains(name) {
            return;
        }
        let mut guard = doc.inner_mut();
        let base: &mut BaseDocument = &mut guard;
        let target = event.target;
        let init = event_init(&event.data, base, target);
        let _scope = js::enter(base, self.state);
        let prevented = js.evolve("dispatch", &[js::js_str(name), JsValue::from(target.as_u64() as f64), js::js_str(&init)]).ok().and_then(|v| v.as_number()).unwrap_or(0.0) == 1.0;
        report_errors(js);
        if prevented && event.cancelable {
            event_state.prevent_default();
            if let DomEventData::KeyDown(k) = &event.data {
                if let Key::Character(c) = &k.key {
                    *self.suppress = Some(c.clone());
                }
            }
        }
        drop(_scope);
        if event_state.is_cancelled() {
            return;
        }
        // Give scripts the submit event before Blitz submits the form.
        let submit = match &event.data {
            DomEventData::Click(_) => submit_target(base, target),
            DomEventData::KeyDown(k) if k.key == Key::Named(NamedKey::Enter) && is_single_line_input(base, target) => form_owner(base, target).map(|f| (f, f)),
            _ => None,
        };
        if let Some((form, submitter)) = submit {
            let _scope = js::enter(base, self.state);
            let sub = if submitter == form { -1.0 } else { submitter.as_u64() as f64 };
            let cancelled = js.evolve("submit", &[JsValue::from(form.as_u64() as f64), JsValue::from(sub)]).ok().and_then(|v| v.as_number()).unwrap_or(0.0) == 1.0;
            report_errors(js);
            if cancelled {
                event_state.prevent_default();
            } else {
                js::prepare_form_submission(base, form);
            }
        }
        // change events: on blur for text fields, immediately for checkable inputs
        let change = match &event.data {
            DomEventData::Blur(_) => true,
            DomEventData::Input(_) => base.get_node(target).and_then(|n| n.attr(local_name!("type"))).is_some_and(|t| matches!(t, "checkbox" | "radio")),
            _ => false,
        };
        if change {
            let _scope = js::enter(base, self.state);
            // Checkboxes and radios fire change right after input; text fields on blur if edited.
            let force = matches!(event.data, DomEventData::Input(_));
            let _ = js.evolve("change", &[JsValue::from(target.as_u64() as f64), JsValue::from(force)]);
        }
        if matches!(event.data, DomEventData::Click(_)) && host::USER_GESTURE.with(|g| g.get()) {
            if let Some(menu) = select_menu(base, target) {
                host::emit(menu);
            }
        }
    }
}

fn event_init(data: &DomEventData, doc: &BaseDocument, _target: NodeId) -> String {
    let mods_json = |m: &Modifiers| serde_json::json!({ "ctrlKey": m.contains(Modifiers::CONTROL), "shiftKey": m.contains(Modifiers::SHIFT), "altKey": m.contains(Modifiers::ALT), "metaKey": m.contains(Modifiers::META) });
    let v = match data {
        DomEventData::PointerMove(p) | DomEventData::PointerDown(p) | DomEventData::PointerUp(p) | DomEventData::PointerCancel(p) | DomEventData::PointerEnter(p) | DomEventData::PointerLeave(p)
        | DomEventData::PointerOver(p) | DomEventData::PointerOut(p) | DomEventData::MouseMove(p) | DomEventData::MouseDown(p) | DomEventData::MouseUp(p) | DomEventData::MouseEnter(p)
        | DomEventData::MouseLeave(p) | DomEventData::MouseOver(p) | DomEventData::MouseOut(p) | DomEventData::Click(p) | DomEventData::ContextMenu(p) | DomEventData::DoubleClick(p)
        | DomEventData::TouchStart(p) | DomEventData::TouchMove(p) | DomEventData::TouchEnd(p) | DomEventData::TouchCancel(p) => {
            let mut o = mods_json(&p.mods);
            let m = o.as_object_mut().unwrap();
            m.insert("clientX".into(), p.coords.client_x.into());
            m.insert("clientY".into(), p.coords.client_y.into());
            m.insert("screenX".into(), p.coords.screen_x.into());
            m.insert("screenY".into(), p.coords.screen_y.into());
            m.insert("pageX".into(), p.coords.page_x.into());
            m.insert("pageY".into(), p.coords.page_y.into());
            m.insert("offsetX".into(), p.element.x.into());
            m.insert("offsetY".into(), p.element.y.into());
            m.insert("button".into(), (p.button as u8 as i32 - if matches!(data, DomEventData::PointerMove(_) | DomEventData::MouseMove(_)) { 0 } else { 0 }).into());
            m.insert("buttons".into(), (p.buttons.bits() as i32).into());
            m.insert("detail".into(), (if matches!(data, DomEventData::DoubleClick(_)) { 2 } else { 1 }).into());
            m.insert("pointerType".into(), "mouse".into());
            m.insert("pressure".into(), p.details.pressure.into());
            if matches!(data, DomEventData::PointerMove(_) | DomEventData::MouseMove(_)) {
                m.insert("button".into(), 0.into());
            }
            if matches!(p.button, MouseEventButton::Secondary) {
                m.insert("button".into(), 2.into());
            } else if matches!(p.button, MouseEventButton::Auxiliary) {
                m.insert("button".into(), 1.into());
            }
            o
        }
        DomEventData::Wheel(w) => {
            let (dx, dy, mode) = match w.delta {
                BlitzWheelDelta::Pixels(x, y) => (x, y, 0),
                BlitzWheelDelta::Lines(x, y) => (x, y, 1),
            };
            let mut o = mods_json(&w.mods);
            let m = o.as_object_mut().unwrap();
            m.insert("deltaX".into(), dx.into());
            m.insert("deltaY".into(), dy.into());
            m.insert("deltaMode".into(), mode.into());
            m.insert("clientX".into(), w.coords.client_x.into());
            m.insert("clientY".into(), w.coords.client_y.into());
            o
        }
        DomEventData::KeyDown(k) | DomEventData::KeyUp(k) | DomEventData::KeyPress(k) => {
            let mut o = mods_json(&k.modifiers);
            let m = o.as_object_mut().unwrap();
            m.insert("key".into(), k.key.to_string().into());
            m.insert("code".into(), k.code.to_string().into());
            m.insert("repeat".into(), k.is_auto_repeating.into());
            m.insert("isComposing".into(), k.is_composing.into());
            m.insert("location".into(), (match k.location { Location::Left => 1, Location::Right => 2, Location::Numpad => 3, _ => 0 }).into());
            o
        }
        DomEventData::Input(i) => serde_json::json!({ "data": serde_json::Value::Null, "inputType": "insertText", "value": i.value }),
        DomEventData::Ime(BlitzImeEvent::Commit(t)) => serde_json::json!({ "data": t }),
        _ => serde_json::json!({}),
    };
    let _ = doc;
    v.to_string()
}

/// The menu for a <select> the person clicked (the engine draws no popup itself).
fn select_menu(doc: &BaseDocument, node: NodeId) -> Option<OutEvent> {
    let mut cur = Some(node);
    let mut select = None;
    while let Some(id) = cur {
        let n = doc.get_node(id)?;
        if n.data.is_element_with_tag_name(&local_name!("select")) {
            select = Some(id);
            break;
        }
        cur = n.parent;
    }
    let select = select?;
    let el = doc.get_node(select)?;
    if el.attr(local_name!("disabled")).is_some() {
        return None;
    }
    let multiple = el.attr(local_name!("multiple")).is_some();
    let options: Vec<NodeId> = doc.query_selector_all_in(select, "option").map(|v| v.to_vec()).unwrap_or_default();
    let any_selected = options.iter().any(|o| doc.get_node(*o).is_some_and(|n| n.attr(local_name!("selected")).is_some()));
    let list = options
        .iter()
        .enumerate()
        .filter_map(|(i, o)| {
            let n = doc.get_node(*o)?;
            let text = n.text_content().split_whitespace().collect::<Vec<_>>().join(" ");
            let label = n.attr(local_name!("label")).map(str::to_string).unwrap_or_else(|| text.clone());
            let group = n.parent.and_then(|p| doc.get_node(p)).filter(|p| p.data.is_element_with_tag_name(&local_name!("optgroup"))).and_then(|p| p.attr(local_name!("label")).map(str::to_string));
            Some(crate::host::SelectOption {
                value: n.attr(local_name!("value")).map(str::to_string).unwrap_or(text),
                label,
                selected: n.attr(local_name!("selected")).is_some() || (!any_selected && !multiple && i == 0),
                disabled: n.attr(local_name!("disabled")).is_some(),
                group,
            })
        })
        .collect();
    let r = doc.get_client_bounding_rect(select)?;
    Some(OutEvent::Select { id: select.as_u64() as f64, x: r.x as f32, y: r.y as f32, w: r.width as f32, h: r.height as f32, multiple, options: list })
}

fn form_owner(doc: &BaseDocument, node: NodeId) -> Option<NodeId> {
    let n = doc.get_node(node)?;
    if let Some(id) = n.attr(local_name!("form")) {
        return doc.get_element_by_id(id);
    }
    let mut cur = n.parent;
    while let Some(p) = cur {
        let pn = doc.get_node(p)?;
        if pn.data.is_element_with_tag_name(&local_name!("form")) {
            return Some(p);
        }
        cur = pn.parent;
    }
    None
}

/// (form, submitter) when clicking `node` would submit a form.
fn submit_target(doc: &BaseDocument, node: NodeId) -> Option<(NodeId, NodeId)> {
    let mut cur = Some(node);
    while let Some(id) = cur {
        let n = doc.get_node(id)?;
        if let Some(el) = n.element_data() {
            let tag: &str = &el.name.local;
            let ty = el.attr(local_name!("type")).unwrap_or("").to_ascii_lowercase();
            if (tag == "button" && (ty.is_empty() || ty == "submit")) || (tag == "input" && (ty == "submit" || ty == "image")) {
                if el.attr(local_name!("disabled")).is_some() {
                    return None;
                }
                return form_owner(doc, id).map(|f| (f, id));
            }
            if matches!(tag, "a" | "form" | "body") {
                return None;
            }
        }
        cur = n.parent;
    }
    None
}

fn is_single_line_input(doc: &BaseDocument, node: NodeId) -> bool {
    doc.get_node(node).and_then(|n| n.element_data()).is_some_and(|el| {
        &*el.name.local == "input" && !matches!(el.attr(local_name!("type")).unwrap_or("text").to_ascii_lowercase().as_str(), "checkbox" | "radio" | "submit" | "button" | "reset" | "image" | "file" | "hidden")
    })
}

fn modifiers(bits: u32) -> Modifiers {
    let mut m = Modifiers::empty();
    if bits & 1 != 0 {
        m |= Modifiers::SHIFT;
    }
    if bits & 2 != 0 {
        m |= Modifiers::CONTROL;
    }
    if bits & 4 != 0 {
        m |= Modifiers::ALT;
    }
    if bits & 8 != 0 {
        m |= Modifiers::META;
    }
    m
}

fn key_event(kind: &str, key: &str, code: &str, mods: u32, repeat: bool) -> BlitzKeyEvent {
    let k = if key.chars().count() == 1 { Key::Character(key.to_string()) } else { Key::from_str(key).unwrap_or(Key::Named(NamedKey::Unidentified)) };
    let c = Code::from_str(code).unwrap_or(Code::Unidentified);
    let location = if code.ends_with("Left") && matches!(key, "Shift" | "Control" | "Alt" | "Meta") {
        Location::Left
    } else if code.ends_with("Right") && matches!(key, "Shift" | "Control" | "Alt" | "Meta") {
        Location::Right
    } else if code.starts_with("Numpad") {
        Location::Numpad
    } else {
        Location::Standard
    };
    BlitzKeyEvent {
        key: k,
        code: c,
        modifiers: modifiers(mods),
        location,
        is_auto_repeating: repeat,
        is_composing: false,
        state: if kind == "up" { KeyState::Released } else { KeyState::Pressed },
        // Text arrives separately through imeCommit, so key events never insert characters.
        text: None,
    }
}

// ---------------------------------------------------------------------- helpers

fn fetch_result_args(id: u32, result: Result<FetchResponse, String>, ctx: &mut boa_engine::Context) -> Vec<JsValue> {
    use boa_engine::object::builtins::{JsArray, JsUint8Array};
    match result {
        Err(e) => vec![JsValue::from(id as f64), js::js_str(&e), JsValue::from(0), js::js_str(""), js::js_str(""), JsArray::new(ctx).map(JsValue::from).unwrap_or(JsValue::null()), JsValue::null(), JsValue::from(false)],
        Ok(resp) => {
            let pairs: Vec<JsValue> = resp.headers.iter().map(|(k, v)| JsValue::from(JsArray::from_iter([js::js_str(&k.to_ascii_lowercase()), js::js_str(v)], ctx))).collect();
            let headers = JsArray::from_iter(pairs, ctx);
            let body = JsUint8Array::from_iter(resp.body, ctx).map(JsValue::from).unwrap_or(JsValue::null());
            vec![JsValue::from(id as f64), JsValue::null(), JsValue::from(resp.status as f64), js::js_str(&resp.status_text), js::js_str(resp.url.as_str()), headers.into(), body, JsValue::from(resp.redirected)]
        }
    }
}

fn display_value(v: &JsValue, ctx: &mut boa_engine::Context) -> String {
    if v.is_undefined() {
        return "undefined".into();
    }
    if let Some(obj) = v.as_object() {
        if obj.is_callable() {
            return "ƒ".into();
        }
        let json = ctx.global_object().get(boa_engine::JsString::from("JSON"), ctx).ok().and_then(|j| j.as_object());
        if let Some(json) = json {
            if let Ok(f) = json.get(boa_engine::JsString::from("stringify"), ctx) {
                if let Some(f) = f.as_callable() {
                    if let Ok(s) = f.call(&JsValue::undefined(), &[v.clone()], ctx) {
                        if let Ok(s) = s.to_string(ctx) {
                            let s = s.to_std_string_lossy();
                            return if s.len() > 4000 { format!("{}…", &s[..s.floor_char_boundary(4000)]) } else { s };
                        }
                    }
                }
            }
        }
    }
    v.display().to_string()
}

fn strip_fragment(u: &Url) -> String {
    let mut u = u.clone();
    u.set_fragment(None);
    u.to_string()
}

fn sniff_html(body: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&body[..body.len().min(512)]).to_ascii_lowercase();
    let t = head.trim_start_matches('\u{feff}').trim_start();
    t.starts_with("<!doctype html") || t.starts_with("<html") || t.starts_with("<head") || t.starts_with("<body") || t.starts_with("<!--") || t.starts_with("<div") || t.starts_with("<p")
}

fn charset_of(content_type: &str) -> Option<String> {
    content_type.split(';').filter_map(|p| p.trim().strip_prefix("charset=")).next().map(|c| c.trim_matches('"').to_string())
}

pub fn decode_html(body: &[u8], content_type: &str) -> String {
    let label = charset_of(content_type).or_else(|| {
        let head = String::from_utf8_lossy(&body[..body.len().min(2048)]).to_ascii_lowercase();
        let i = head.find("charset=")?;
        let rest = &head[i + 8..];
        let rest = rest.trim_start_matches(['"', '\'']);
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_')).unwrap_or(rest.len());
        Some(rest[..end].to_string())
    });
    let enc = label.and_then(|l| encoding_rs::Encoding::for_label(l.as_bytes())).unwrap_or(encoding_rs::UTF_8);
    let (text, _, _) = enc.decode(body);
    text.into_owned()
}

fn decode_script(body: &[u8], content_type: Option<&str>) -> String {
    match content_type.and_then(charset_of).and_then(|l| encoding_rs::Encoding::for_label(l.as_bytes())) {
        Some(enc) => enc.decode(body).0.into_owned(),
        None => encoding_rs::UTF_8.decode(body).0.into_owned(),
    }
}

fn encode_form(form: &blitz_traits::net::FormData, enctype: &str) -> (Vec<u8>, String) {
    let pairs: Vec<(String, String)> = form.iter().map(|e| (e.name.clone(), match &e.value { EntryValue::String(s) => s.clone(), _ => String::new() })).collect();
    let enctype = enctype.to_ascii_lowercase();
    if enctype.starts_with("multipart/form-data") {
        let boundary = format!("----EvolveFormBoundary{:016x}", crate::clock::now().to_bits() ^ 0x5eed);
        let mut body = Vec::new();
        for (k, v) in &pairs {
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n{v}\r\n", k.replace('"', "%22")).as_bytes());
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (body, format!("multipart/form-data; boundary={boundary}"))
    } else if enctype.starts_with("text/plain") {
        let body: String = pairs.iter().map(|(k, v)| format!("{k}={v}\r\n")).collect();
        (body.into_bytes(), "text/plain".into())
    } else {
        let mut s = String::new();
        url::form_urlencoded::Serializer::new(&mut s).extend_pairs(&pairs);
        (s.into_bytes(), "application/x-www-form-urlencoded".into())
    }
}

fn error_page(url: &Url, error: &str) -> String {
    let help = if error.contains("Tor") {
        "<p>The browser reaches websites through the Tor network. Some sites refuse Tor exits or stop answering; try again for a new circuit, or pick another route with the network button.</p>"
    } else if error.contains("relay") {
        "<p>The browser reaches websites through a small relay that forwards encrypted bytes. Start it on this computer with <code>evolve-relay</code>, or pick another route with the network button.</p>"
    } else {
        "<p>Check the address, or try again in a moment.</p>"
    };
    format!(
        "<!doctype html><html><head><title>Can’t open this page</title><style>body{{font:16px/1.5 system-ui,sans-serif;color:#1d2433;max-width:640px;margin:12vh auto;padding:0 24px}}h1{{font-size:24px;margin:0 0 12px}}code{{background:#eef0f5;padding:2px 6px;border-radius:4px}}.u{{color:#5b6475;word-break:break-all}}</style></head><body><h1>Can’t open this page</h1><p class=\"u\">{}</p><p><strong>{}</strong></p>{help}</body></html>",
        html_escape::encode_text(url.as_str()),
        html_escape::encode_text(error)
    )
}

#[allow(dead_code)]
fn is_element(doc: &BaseDocument, id: NodeId) -> bool {
    doc.get_node(id).is_some_and(|n| matches!(n.data, NodeData::Element(_)))
}
