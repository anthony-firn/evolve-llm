//! JavaScript for pages: Boa, with the DOM and web APIs written in JavaScript (`prelude.js`)
//! over a compact host API (`host_api.rs`) that reads and mutates the Blitz document.

mod host_api;
pub use host_api::{form_owner_of, prepare_form_submission};

use crate::host;
use crate::net::{Credentials, FetchRequest, Mode};
use blitz_dom::{BaseDocument, NodeId};
use blitz_traits::events::DomEvent;
use boa_engine::job::{GenericJob, Job, JobExecutor, NativeAsyncJob, PromiseJob};
use boa_engine::module::{ModuleLoader, Referrer};
use boa_engine::object::builtins::JsPromise;
use boa_engine::property::Attribute;
use boa_engine::realm::Realm;
use boa_engine::{Context, JsObject, JsResult, JsString, JsValue, Module, Source};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Wake, Waker};
use url::Url;

pub const PRELUDE: &str = include_str!("prelude.js");
const WINDOW_BRIDGE: &str = include_str!("window_bridge.js");
pub const FRAME_PRELUDE: &str = include_str!("frame.js");

/// Private per-page registry of iframe realms and queued posted messages. Raw globals stay
/// in the host; page scripts receive realm-local, origin-checked facades. Keyed by the iframe
/// node id; the top window is id 0 (no real iframe is node 0).
#[derive(Default)]
struct FrameTable {
    next_epoch: Cell<u64>,
    messages: RefCell<VecDeque<FrameMessage>>,
    top: RefCell<Option<FrameEntry>>,
    frames: RefCell<HashMap<u64, FrameEntry>>,
}

#[derive(Clone)]
struct FrameEntry {
    #[allow(dead_code)] // held so the realm's intrinsics outlive the windows that use them
    realm: Option<Realm>,
    window: JsObject,
    origin: String,
    url: String,
    epoch: u64,
    deliver: Option<JsObject>,
}

struct FrameMessage {
    target: u64,
    target_epoch: u64,
    source: u64,
    source_epoch: u64,
    source_origin: String,
    wire: String,
    target_origin: String,
}

fn frame_entry(id: u64) -> Option<FrameEntry> {
    FRAMES.with(|f| if id == 0 { f.top.borrow().clone() } else { f.frames.borrow().get(&id).cloned() })
}

pub(crate) fn checked_frame_window(target: u64, source: u64, epoch: u64) -> Option<JsObject> {
    let sender = frame_entry(source)?;
    let receiver = frame_entry(target)?;
    (sender.epoch == epoch && (target == source || sender.origin != "null" && sender.origin == receiver.origin)).then_some(receiver.window)
}

pub(crate) fn queue_frame_message(target: u64, source: u64, epoch: u64, wire: String, target_origin: String) {
    let Some(sender) = frame_entry(source).filter(|e| e.epoch == epoch) else { return };
    let Some(receiver) = frame_entry(target) else { return };
    let allowed = target_origin == "*" || target_origin == receiver.origin;
    if !allowed { return; }
    FRAMES.with(|f| f.messages.borrow_mut().push_back(FrameMessage {
        target, target_epoch: receiver.epoch, source, source_epoch: epoch,
        source_origin: sender.origin, wire, target_origin,
    }));
    host::wake();
}

thread_local! {
    static FRAMES: FrameTable = FrameTable::default();
}

/// Clears every frame realm. Must run before the owning `Context` is dropped, since the stored
/// window objects belong to it.
pub(crate) fn frames_clear() {
    FRAMES.with(|f| {
        f.next_epoch.set(0);
        f.messages.borrow_mut().clear();
        *f.top.borrow_mut() = None;
        f.frames.borrow_mut().clear();
    });
}

/// The registered origin of a window, used for postMessage targetOrigin checks.
pub(crate) fn frame_origin(id: u64) -> Option<String> {
    FRAMES.with(|f| {
        if id == 0 { f.top.borrow().as_ref().map(|e| e.origin.clone()) } else { f.frames.borrow().get(&id).map(|e| e.origin.clone()) }
    })
}

/// The window object for an iframe node, for `iframe.contentWindow`.
pub(crate) fn frame_window(id: u64) -> Option<JsObject> {
    FRAMES.with(|f| f.frames.borrow().get(&id).map(|e| e.window.clone()))
}

/// Per-page state the host API reads and writes while script runs.
pub struct PageState {
    pub identity: crate::identity::Identity,
    pub id: u64,
    pub url: Url,
    pub referrer: String,
    pub ready_state: &'static str,
    /// Timer id → due time (ms, performance clock)
    pub timers: HashMap<u32, f64>,
    pub animation_frame: bool,
    pub listened: HashSet<String>,
    /// `<script>` elements inserted by script, waiting to run
    pub dynamic_scripts: Vec<NodeId>,
    /// Script elements already prepared, including synchronously inserted inline scripts.
    pub started_scripts: HashSet<NodeId>,
    /// Script-created elements start with force-async set, until the async setter clears it.
    pub force_async_scripts: HashSet<NodeId>,
    /// Navigations requested by script: (url, replace)
    pub navigations: Vec<(String, bool)>,
    /// history.pushState/replaceState: (url, replace, state json)
    pub history: Vec<HistoryOp>,
    pub layout_dirty: bool,
    pub local: Rc<RefCell<HashMap<String, String>>>,
    pub session: Rc<RefCell<HashMap<String, String>>>,
    pub storage_changed: bool,
    /// Events produced by default actions run from script (e.g. `el.click()`)
    pub pending_events: Vec<DomEvent>,
    pub viewport: (f32, f32, f32),
    pub now: f64,
    pub console_budget: u32,
    pub write_buffer: Option<String>,
}

pub enum HistoryOp {
    Push { url: String, replace: bool, state: String },
    Go(i32),
}

impl PageState {
    pub fn new(id: u64, url: Url, referrer: String, local: Rc<RefCell<HashMap<String, String>>>, session: Rc<RefCell<HashMap<String, String>>>) -> Self {
        PageState {
            identity: crate::identity::Identity::default(),
            id,
            url,
            referrer,
            ready_state: "loading",
            timers: HashMap::new(),
            animation_frame: false,
            listened: HashSet::new(),
            dynamic_scripts: vec![],
            started_scripts: HashSet::new(),
            force_async_scripts: HashSet::new(),
            navigations: vec![],
            history: vec![],
            layout_dirty: true,
            local,
            session,
            storage_changed: false,
            pending_events: vec![],
            viewport: (1024.0, 768.0, 1.0),
            now: 0.0,
            console_budget: 400,
            write_buffer: None,
        }
    }
}

thread_local! {
    static HOST: Cell<(*mut BaseDocument, *mut PageState)> = const { Cell::new((std::ptr::null_mut(), std::ptr::null_mut())) };
}

/// Makes `doc` and `state` available to host functions while script runs.
pub struct HostScope {
    prev: (*mut BaseDocument, *mut PageState),
}

pub fn enter(doc: &mut BaseDocument, state: &mut PageState) -> HostScope {
    let prev = HOST.with(|h| h.replace((doc as *mut _, state as *mut _)));
    HostScope { prev }
}

impl Drop for HostScope {
    fn drop(&mut self) {
        HOST.with(|h| h.set(self.prev));
    }
}

pub(crate) fn with_host<R>(f: impl FnOnce(&mut BaseDocument, &mut PageState) -> R) -> Option<R> {
    let (doc, state) = HOST.with(|h| h.get());
    if doc.is_null() || state.is_null() {
        return None;
    }
    // SAFETY: the pointers are set by `enter` for the duration of a script call, during which
    // the engine does not otherwise touch the document or state. A host function must end
    // its with_host closure before invoking script or converting values via page callbacks,
    // so these borrows do not overlap.
    Some(unsafe { f(&mut *doc, &mut *state) })
}

struct EngineWaker;
impl Wake for EngineWaker {
    fn wake(self: Arc<Self>) {
        host::wake();
    }
}

/// Runs promise jobs synchronously at microtask checkpoints and polls async jobs (module
/// loading) without blocking, from the engine's event loop.
#[derive(Default)]
pub struct Executor {
    promise: RefCell<VecDeque<PromiseJob>>,
    generic: RefCell<VecDeque<GenericJob>>,
    new_async: RefCell<Vec<NativeAsyncJob>>,
    running: RefCell<Vec<Pin<Box<dyn Future<Output = JsResult<JsValue>>>>>>,
    pub errors: RefCell<Vec<String>>,
}

impl JobExecutor for Executor {
    fn enqueue_job(self: Rc<Self>, job: Job, _context: &mut Context) {
        match job {
            Job::PromiseJob(p) => self.promise.borrow_mut().push_back(p),
            Job::GenericJob(g) => self.generic.borrow_mut().push_back(g),
            Job::AsyncJob(a) => {
                self.new_async.borrow_mut().push(a);
                host::wake();
            }
            _ => {}
        }
    }

    fn run_jobs(self: Rc<Self>, context: &mut Context) -> JsResult<()> {
        for _ in 0..200_000 {
            let job = self.promise.borrow_mut().pop_front();
            if let Some(job) = job {
                if let Err(e) = job.call(context) {
                    self.errors.borrow_mut().push(e.to_string());
                }
                continue;
            }
            let job = self.generic.borrow_mut().pop_front();
            match job {
                Some(job) => {
                    if let Err(e) = job.call(context) {
                        self.errors.borrow_mut().push(e.to_string());
                    }
                }
                None => break,
            }
        }
        context.clear_kept_objects();
        Ok(())
    }
}

impl Executor {
    fn has_async(&self) -> bool {
        !self.new_async.borrow().is_empty() || !self.running.borrow().is_empty()
    }
}

/// Fetches and caches ES modules for a page, honoring `<script type="importmap">`.
pub struct PageModules {
    page: u64,
    base: RefCell<Url>,
    cache: RefCell<HashMap<String, Module>>,
    pub imports: RefCell<Vec<(String, String)>>,
}

impl PageModules {
    pub fn resolve(&self, specifier: &str, base: &Url) -> Result<Url, String> {
        let imports = self.imports.borrow();
        if let Some((_, target)) = imports.iter().find(|(k, _)| k == specifier) {
            return self.base.borrow().join(target).map_err(|e| e.to_string());
        }
        if let Some((k, target)) = imports.iter().filter(|(k, _)| k.ends_with('/') && specifier.starts_with(k.as_str())).max_by_key(|(k, _)| k.len()) {
            return self.base.borrow().join(&format!("{target}{}", &specifier[k.len()..])).map_err(|e| e.to_string());
        }
        if specifier.starts_with('/') || specifier.starts_with("./") || specifier.starts_with("../") {
            return base.join(specifier).map_err(|e| e.to_string());
        }
        Url::parse(specifier).map_err(|_| format!("Cannot resolve bare module specifier \"{specifier}\" (no import map entry)"))
    }
}

impl ModuleLoader for PageModules {
    async fn load_imported_module(self: Rc<Self>, referrer: Referrer, request: boa_engine::module::ModuleRequest, context: &RefCell<&mut Context>) -> JsResult<Module> {
        let spec = request.specifier().to_std_string_lossy();
        let base = referrer
            .path()
            .and_then(|p| p.to_str())
            .and_then(|p| Url::parse(p).ok())
            .unwrap_or_else(|| self.base.borrow().clone());
        let url = self.resolve(&spec, &base).map_err(|e| boa_engine::JsNativeError::typ().with_message(e))?;
        if let Some(m) = self.cache.borrow().get(url.as_str()) {
            return Ok(m.clone());
        }
        let (tx, rx) = futures::channel::oneshot::channel();
        let mut req = FetchRequest::get(url.clone(), Some(self.base.borrow().clone()), Mode::Cors, "script");
        req.credentials = Credentials::SameOrigin;
        let _ = self.page;
        host::loader().fetch(
            req,
            Box::new(move |result| {
                let _ = tx.send(result);
                host::wake();
            }),
        );
        let resp = rx
            .await
            .map_err(|_| boa_engine::JsNativeError::error().with_message("Module load cancelled"))?
            .map_err(|e| boa_engine::JsNativeError::typ().with_message(format!("Failed to load module {url}: {e}")))?;
        if !(200..300).contains(&resp.status) {
            return Err(boa_engine::JsNativeError::typ().with_message(format!("Failed to load module {url}: HTTP {}", resp.status)).into());
        }
        let src = String::from_utf8_lossy(&resp.body).into_owned();
        let module = Module::parse(Source::from_bytes(src.as_bytes()).with_path(Path::new(resp.url.as_str())), None, &mut context.borrow_mut())?;
        self.cache.borrow_mut().insert(url.to_string(), module.clone());
        self.cache.borrow_mut().insert(resp.url.to_string(), module.clone());
        Ok(module)
    }

    /// `import.meta.url` (webpack and Vite derive their public path from it) and `resolve()`.
    fn init_import_meta(self: Rc<Self>, import_meta: &boa_engine::JsObject, module: &Module, context: &mut Context) {
        let url = module.path().and_then(|p| p.to_str()).map(str::to_string).unwrap_or_else(|| self.base.borrow().to_string());
        let url = url.split("#inline-").next().unwrap_or(&url).to_string();
        let _ = import_meta.set(JsString::from("url"), JsString::from(url.as_str()), false, context);
        let base = Url::parse(&url).unwrap_or_else(|_| self.base.borrow().clone());
        let loader = self.clone();
        // SAFETY: the closure captures only Rust values (no garbage-collected objects).
        let resolve = unsafe {
            boa_engine::NativeFunction::from_closure(move |_, args, ctx| {
                let spec = args.first().cloned().unwrap_or_default().to_string(ctx)?.to_std_string_lossy();
                let url = loader.resolve(&spec, &base).map_err(|e| boa_engine::JsNativeError::typ().with_message(e))?;
                Ok(JsString::from(url.as_str()).into())
            })
        };
        let f = boa_engine::object::FunctionObjectBuilder::new(context.realm(), resolve).name(JsString::from("resolve")).length(1).build();
        let _ = import_meta.set(JsString::from("resolve"), f, false, context);
    }
}

pub struct JsRuntime {
    cell: &'static RefCell<&'static mut Context>,
    raw_cell: *mut RefCell<&'static mut Context>,
    raw_ctx: *mut Context,
    pub executor: Rc<Executor>,
    pub modules: Rc<PageModules>,
    pending_modules: RefCell<Vec<(String, JsPromise)>>,
    waker: Waker,
}

impl JsRuntime {
    pub fn new(page: u64, url: &Url) -> Result<Self, String> {
        let executor = Rc::new(Executor::default());
        let modules = Rc::new(PageModules { page, base: RefCell::new(url.clone()), cache: RefCell::default(), imports: RefCell::default() });
        let mut ctx = Context::builder()
            .job_executor(executor.clone())
            .module_loader(modules.clone())
            .build()
            .map_err(|e| e.to_string())?;
        ctx.runtime_limits_mut().set_loop_iteration_limit(200_000_000);
        ctx.runtime_limits_mut().set_recursion_limit(2_000);
        let raw_ctx = Box::into_raw(Box::new(ctx));
        // SAFETY: freed in Drop after every future that borrows the cell is dropped.
        let raw_cell = Box::into_raw(Box::new(RefCell::new(unsafe { &mut *raw_ctx })));
        let cell: &'static RefCell<&'static mut Context> = unsafe { &*raw_cell };
        host_api::install(&mut cell.borrow_mut()).map_err(|e| e.to_string())?;
        let rt = JsRuntime {
            cell,
            raw_cell,
            raw_ctx,
            executor,
            modules,
            pending_modules: RefCell::default(),
            waker: Waker::from(Arc::new(EngineWaker)),
        };
        Ok(rt)
    }

    /// Install the DOM and web APIs. Must run inside `enter(doc, state)`.
    pub fn install_prelude(&self) -> Result<(), String> {
        self.eval_script(&PRELUDE.replace("/*__WINDOW_BRIDGE__*/", WINDOW_BRIDGE), "evolve:prelude").map(|_| ()).map_err(|e| format!("Prelude failed: {e}"))
    }

    /// Run a classic script, then a microtask checkpoint.
    pub fn eval_script(&self, code: &str, url: &str) -> Result<JsValue, String> {
        let result = {
            let mut ctx = self.cell.borrow_mut();
            let result = ctx.eval(Source::from_bytes(code.as_bytes()).with_path(Path::new(url)));
            let _ = ctx.run_jobs();
            result
        };
        result.map_err(|e| self.describe(e))
    }

    pub fn run_module(&self, code: &str, url: &str) -> Result<(), String> {
        let mut ctx = self.cell.borrow_mut();
        let module = Module::parse(Source::from_bytes(code.as_bytes()).with_path(Path::new(url)), None, &mut ctx).map_err(|e| e.to_string())?;
        let promise = module.load_link_evaluate(&mut ctx);
        let _ = ctx.run_jobs();
        drop(ctx);
        self.pending_modules.borrow_mut().push((url.to_string(), promise));
        Ok(())
    }

    /// Call a global function defined by the prelude.
    pub fn call(&self, name: &str, args: &[JsValue]) -> Result<JsValue, String> {
        let result = {
            let mut ctx = self.cell.borrow_mut();
            let global = ctx.global_object();
            let f = global.get(JsString::from(name), &mut ctx).map_err(|e| e.to_string())?;
            let result = match f.as_callable() {
                Some(f) => f.call(&JsValue::undefined(), args, &mut ctx),
                None => return Err(format!("{name} is not defined")),
            };
            let _ = ctx.run_jobs();
            result
        };
        result.map_err(|e| self.describe(e))
    }

    /// Call a method of the prelude's `__evolve` object.
    pub fn evolve(&self, method: &str, args: &[JsValue]) -> Result<JsValue, String> {
        let result = {
            let mut ctx = self.cell.borrow_mut();
            let global = ctx.global_object();
            let obj = global.get(JsString::from("__evolve"), &mut ctx).map_err(|e| e.to_string())?;
            let obj = obj.as_object().ok_or("__evolve is missing")?;
            let f = obj.get(JsString::from(method), &mut ctx).map_err(|e| e.to_string())?;
            let result = match f.as_callable() {
                Some(f) => f.call(&JsValue::from(obj.clone()), args, &mut ctx),
                None => return Err(format!("__evolve.{method} is not defined")),
            };
            let _ = ctx.run_jobs();
            result
        };
        result.map_err(|e| self.describe(e))
    }

    pub fn with_context<R>(&self, f: impl FnOnce(&mut Context) -> R) -> R {
        f(&mut self.cell.borrow_mut())
    }

    fn describe(&self, e: boa_engine::JsError) -> String {
        let mut ctx = self.cell.borrow_mut();
        match e.try_native(&mut ctx) {
            Ok(native) => format!("{}: {}", native.kind(), native.message()),
            Err(_) => {
                let fallback = e.to_string();
                e.into_opaque(&mut ctx)
                    .ok()
                    .and_then(|v| v.to_string(&mut ctx).ok())
                    .map(|s| s.to_std_string_lossy())
                    .unwrap_or(fallback)
            }
        }
    }

    /// Poll module loading. Returns errors from modules that failed.
    pub fn poll(&self) -> Vec<String> {
        let mut errors: Vec<String> = self.executor.errors.borrow_mut().drain(..).collect();
        // Posted messages are tasks, never synchronous calls across a realm boundary.
        let messages: Vec<_> = FRAMES.with(|f| f.messages.borrow_mut().drain(..).collect());
        for message in messages {
            let Some(target) = frame_entry(message.target).filter(|e| e.epoch == message.target_epoch) else { continue };
            if frame_entry(message.source).is_none_or(|e| e.epoch != message.source_epoch) { continue; }
            if message.target_origin != "*" && message.target_origin != target.origin { continue; }
            let Some(deliver) = target.deliver.filter(|o| o.is_callable()) else { continue };
            let mut ctx = self.cell.borrow_mut();
            let previous = target.realm.map(|realm| ctx.enter_realm(realm));
            let result = deliver.call(&JsValue::from(target.window), &[js_str(&message.wire), js_str(&message.source_origin), JsValue::from(message.source as f64)], &mut ctx);
            if let Err(error) = result { errors.push(error.to_string()); }
            let _ = ctx.run_jobs();
            if let Some(previous) = previous { ctx.enter_realm(previous); }
        }
        if self.executor.has_async() {
            let fresh: Vec<_> = self.executor.new_async.borrow_mut().drain(..).collect();
            for job in fresh {
                let fut = job.call(self.cell);
                self.executor.running.borrow_mut().push(Box::pin(fut));
            }
            let mut cx = std::task::Context::from_waker(&self.waker);
            let running: Vec<_> = self.executor.running.borrow_mut().drain(..).collect();
            let mut still = Vec::new();
            for mut fut in running {
                match fut.as_mut().poll(&mut cx) {
                    std::task::Poll::Ready(Err(e)) => errors.push(e.to_string()),
                    std::task::Poll::Ready(Ok(_)) => {}
                    std::task::Poll::Pending => still.push(fut),
                }
            }
            self.executor.running.borrow_mut().extend(still);
            let _ = self.cell.borrow_mut().run_jobs();
        }
        let mut pending = self.pending_modules.borrow_mut();
        pending.retain(|(url, promise)| match promise.state() {
            boa_engine::builtins::promise::PromiseState::Pending => true,
            boa_engine::builtins::promise::PromiseState::Fulfilled(_) => false,
            boa_engine::builtins::promise::PromiseState::Rejected(err) => {
                let mut ctx = self.cell.borrow_mut();
                let text = err.to_string(&mut ctx).map(|s| s.to_std_string_lossy()).unwrap_or_default();
                errors.push(format!("{text} (module {url})"));
                false
            }
        });
        errors.extend(self.executor.errors.borrow_mut().drain(..));
        errors
    }

    pub fn frame_messages_pending(&self) -> bool {
        FRAMES.with(|f| !f.messages.borrow().is_empty())
    }

    pub fn modules_pending(&self) -> bool {
        !self.pending_modules.borrow().is_empty() || self.executor.has_async()
    }

    /// Records the top document's window as frame 0, so cross-frame messaging can reach it.
    /// Call once after `install_prelude`, with the document's origin.
    pub fn record_top_window(&self, origin: &str) {
        frames_clear();
        let mut ctx = self.cell.borrow_mut();
        let window = ctx.global_object();
        let deliver = window.get(JsString::from("__deliverMessage"), &mut ctx).ok().and_then(|v| v.as_object());
        let realm = Some(ctx.realm().clone());
        FRAMES.with(|f| *f.top.borrow_mut() = Some(FrameEntry { realm, window, origin: origin.to_string(), url: String::new(), epoch: 0, deliver }));
    }

    /// Runs an iframe's classic scripts in its own realm (fresh global, shared heap) so the
    /// frame has a distinct global behind its contentWindow facade. Queued postMessage
    /// delivery reconstructs values and window references in the receiving realm. `node_id` is the
    /// iframe's node id in the parent document (used as the frame id). The realm may already exist
    /// (created eagerly on a synchronous `contentWindow` access); its scripts then run in it.
    /// Returns the first error.
    pub fn run_frame(&self, node_id: u64, origin: &str, url: &str, scripts: &[String]) -> Result<(), String> {
        let mut guard = self.cell.borrow_mut();
        let ctx: &mut Context = &mut guard;
        // A newly attached document gets a fresh realm even on same-URL reloads. WindowProxy
        // identity remains stable; old source capabilities and queued messages lose their epoch.
        FRAMES.with(|f| f.frames.borrow_mut().remove(&node_id));
        create_frame_realm(ctx, node_id, origin, url)?;
        let Some(realm) = FRAMES.with(|f| f.frames.borrow().get(&node_id).and_then(|e| e.realm.clone())) else {
            return Ok(());
        };
        let prev = ctx.enter_realm(realm);
        let mut first_err = None;
        for code in scripts {
            if let Err(e) = ctx.eval(Source::from_bytes(code.as_bytes()).with_path(Path::new("evolve:frame-script"))) {
                if first_err.is_none() { first_err = Some(e.to_string()); }
            }
            let _ = ctx.run_jobs();
        }
        ctx.enter_realm(prev);
        match first_err { Some(e) => Err(e), None => Ok(()) }
    }
}

/// Creates (or returns the existing) realm and window for an iframe node: a fresh global running
/// the frame prelude, registered in the frame table so `postMessage` and `contentWindow` resolve
/// it. No scripts run here. Used both eagerly (synchronous same-origin `contentWindow` access,
/// matching a real browser's immediate about:blank window) and when the sub-document loads.
pub(crate) fn create_frame_realm(ctx: &mut Context, node_id: u64, origin: &str, url: &str) -> Result<JsObject, String> {
    if let Some(entry) = frame_entry(node_id) {
        if entry.origin == origin && entry.url == url { return Ok(entry.window); }
    }
    let epoch = FRAMES.with(|f| { let epoch = f.next_epoch.get() + 1; f.next_epoch.set(epoch); epoch });
    let realm = ctx.create_realm().map_err(|e| e.to_string())?;
    let prev = ctx.enter_realm(realm.clone());
    let result = (|| -> Result<JsObject, String> {
        host_api::install_frame(ctx).map_err(|e| e.to_string())?;
        let attr = Attribute::all();
        ctx.register_global_property(JsString::from("__frameId_value"), JsValue::from(node_id as f64), attr).map_err(|e| e.to_string())?;
        ctx.register_global_property(JsString::from("__frameEpoch_value"), JsValue::from(epoch as f64), attr).map_err(|e| e.to_string())?;
        ctx.register_global_property(JsString::from("__frameOrigin"), js_str(origin), attr).map_err(|e| e.to_string())?;
        ctx.register_global_property(JsString::from("__frameUrl"), js_str(url), attr).map_err(|e| e.to_string())?;
        ctx.eval(Source::from_bytes(FRAME_PRELUDE.replace("/*__WINDOW_BRIDGE__*/", WINDOW_BRIDGE).as_bytes()).with_path(Path::new("evolve:frame"))).map_err(|e| e.to_string())?;
        Ok(ctx.global_object())
    })();
    let window = match result {
        Ok(w) => w,
        Err(e) => { ctx.enter_realm(prev); return Err(e); }
    };
    let deliver = window.get(JsString::from("__deliverMessage"), ctx).ok().and_then(|v| v.as_object());
    ctx.enter_realm(prev);
    FRAMES.with(|f| f.frames.borrow_mut().insert(node_id, FrameEntry { realm: Some(realm), window: window.clone(), origin: origin.to_string(), url: url.to_string(), epoch, deliver }));
    Ok(window)
}

impl Drop for JsRuntime {
    fn drop(&mut self) {
        // Frame realms hold window objects from this context; release them before it is dropped.
        frames_clear();
        // Futures borrow the context cell; drop them first.
        self.executor.running.borrow_mut().clear();
        self.executor.new_async.borrow_mut().clear();
        self.pending_modules.borrow_mut().clear();
        self.modules.cache.borrow_mut().clear();
        self.executor.promise.borrow_mut().clear();
        self.executor.generic.borrow_mut().clear();
        unsafe {
            drop(Box::from_raw(self.raw_cell));
            drop(Box::from_raw(self.raw_ctx));
        }
    }
}

pub fn js_str(s: &str) -> JsValue {
    JsString::from(s).into()
}
