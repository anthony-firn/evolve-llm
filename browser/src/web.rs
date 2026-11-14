//! WebAssembly bindings: the `Browser` class used by `web/browser-host.mjs` (see API.md §1).

use crate::engine::Engine;
use crate::host::{self, OutEvent};
use crate::net::client::Client;
use crate::net::{FetchRequest, FetchResponse};
use std::rc::Rc;
use std::sync::Arc;
use url::Url;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

struct WasmLoader {
    client: Rc<Client>,
}

impl host::Loader for WasmLoader {
    fn fetch(&self, req: FetchRequest, done: Box<dyn FnOnce(Result<FetchResponse, String>)>) {
        let client = self.client.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = client.fetch(req).await;
            if client.cookies_changed.replace(false) {
                host::emit(OutEvent::Cookies);
            }
            done(result);
        });
    }
    fn script_cookies(&self, url: &Url) -> String {
        self.client.script_cookies(url)
    }
    fn set_script_cookie(&self, url: &Url, value: &str) {
        self.client.set_script_cookie(url, value);
    }
}

#[wasm_bindgen]
pub struct Browser {
    engine: Engine,
    client: Rc<Client>,
    _wake: js_sys::Function,
}

#[wasm_bindgen]
impl Browser {
    #[wasm_bindgen(constructor)]
    pub fn new(fonts: js_sys::Array, relay_url: String, cookies_json: Option<String>, storage_json: Option<String>, wake: js_sys::Function) -> Result<Browser, JsValue> {
        console_error_panic_hook::set_once();
        let mut files = Vec::new();
        for entry in fonts.iter() {
            let pair: js_sys::Array = entry.dyn_into().map_err(|_| JsValue::from_str("fonts must be [name, Uint8Array] pairs"))?;
            let name = pair.get(0).as_string().unwrap_or_default();
            let bytes = js_sys::Uint8Array::new(&pair.get(1)).to_vec();
            files.push((name, Arc::new(bytes)));
        }
        let client = Rc::new(Client::new(&relay_url, cookies_json.as_deref()));
        host::LOADER.with(|l| *l.borrow_mut() = Some(Rc::new(WasmLoader { client: client.clone() })));
        let w = wake.clone();
        host::WAKER.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                let _ = w.call0(&JsValue::NULL);
            }))
        });
        let engine = Engine::new(crate::fonts::font_context(&files), storage_json.as_deref());
        Ok(Browser { engine, client, _wake: wake })
    }

    pub fn resize(&mut self, css_width: u32, css_height: u32, dpr: f32) {
        self.engine.resize(css_width, css_height, dpr);
    }

    #[wasm_bindgen(js_name = setRelay)]
    pub fn set_relay(&self, url: &str) {
        self.client.set_relay(url);
    }

    pub fn navigate(&mut self, url: &str) {
        self.engine.navigate_typed(url);
    }
    pub fn back(&mut self) {
        self.engine.back();
    }
    pub fn forward(&mut self) {
        self.engine.forward();
    }
    pub fn reload(&mut self) {
        self.engine.reload();
    }
    pub fn stop(&mut self) {
        self.engine.stop();
    }

    pub fn tick(&mut self, now: f64) -> f64 {
        let next = self.engine.tick(now);
        if self.client.cookies_changed.replace(false) {
            host::emit(OutEvent::Cookies);
        }
        next
    }

    #[wasm_bindgen(js_name = takeFrame)]
    pub fn take_frame(&mut self) -> Option<js_sys::Uint8Array> {
        self.engine.take_frame().map(js_sys::Uint8Array::from)
    }
    #[wasm_bindgen(js_name = frameWidth)]
    pub fn frame_width(&self) -> u32 {
        self.engine.frame_size().0
    }
    #[wasm_bindgen(js_name = frameHeight)]
    pub fn frame_height(&self) -> u32 {
        self.engine.frame_size().1
    }

    pub fn pointer(&mut self, kind: &str, x: f32, y: f32, button: i32, buttons: u32, mods: u32) {
        self.engine.pointer(kind, x, y, button, buttons, mods);
    }
    pub fn wheel(&mut self, x: f32, y: f32, dx: f64, dy: f64, delta_mode: u32, mods: u32) {
        self.engine.wheel(x, y, dx, dy, delta_mode, mods);
    }
    pub fn key(&mut self, kind: &str, key: &str, code: &str, mods: u32, repeat: bool) {
        self.engine.key(kind, key, code, mods, repeat);
    }
    #[wasm_bindgen(js_name = imeCommit)]
    pub fn ime_commit(&mut self, text: &str) {
        self.engine.ime_commit(text);
    }
    #[wasm_bindgen(js_name = imePreedit)]
    pub fn ime_preedit(&mut self, text: &str) {
        self.engine.ime_preedit(text);
    }

    /// The person picked an option from a `select` event's menu (-1 = cancelled).
    #[wasm_bindgen(js_name = chooseOption)]
    pub fn choose_option(&mut self, id: f64, index: i32) {
        self.engine.choose_option(id, index);
    }

    #[wasm_bindgen(js_name = takeEvents)]
    pub fn take_events(&mut self) -> String {
        let events: Vec<OutEvent> = host::OUT.with(|o| o.borrow_mut().drain(..).collect());
        serde_json::to_string(&events).unwrap_or_else(|_| "[]".into())
    }
    #[wasm_bindgen(js_name = cookiesJson)]
    pub fn cookies_json(&self) -> String {
        self.client.cookies_json()
    }
    #[wasm_bindgen(js_name = storageJson)]
    pub fn storage_json(&self) -> String {
        self.engine.storage_json()
    }
    #[wasm_bindgen(js_name = clearData)]
    pub fn clear_data(&mut self) {
        self.client.clear_cookies();
        self.engine.clear_storage();
        host::emit(OutEvent::Cookies);
        host::emit(OutEvent::Storage);
    }
    #[wasm_bindgen(js_name = networkLog)]
    pub fn network_log(&self) -> String {
        serde_json::to_string(&*self.client.log.borrow()).unwrap_or_else(|_| "[]".into())
    }
    pub fn evaluate(&mut self, code: &str) -> String {
        self.engine.evaluate(code)
    }

    pub fn snapshot(&mut self, max_chars: u32) -> String {
        self.engine.snapshot(max_chars as usize).to_string()
    }
    #[wasm_bindgen(js_name = agentClick)]
    pub fn agent_click(&mut self, r: u32) -> String {
        self.engine.agent_click(r as usize).to_string()
    }
    #[wasm_bindgen(js_name = agentType)]
    pub fn agent_type(&mut self, r: u32, text: &str, submit: bool) -> String {
        self.engine.agent_type(r as usize, text, submit).to_string()
    }
    #[wasm_bindgen(js_name = agentSelect)]
    pub fn agent_select(&mut self, r: u32, value: &str) -> String {
        self.engine.agent_select(r as usize, value).to_string()
    }
    #[wasm_bindgen(js_name = agentScroll)]
    pub fn agent_scroll(&mut self, dy: f64) -> String {
        self.engine.agent_scroll(dy).to_string()
    }
    #[wasm_bindgen(js_name = agentKey)]
    pub fn agent_key(&mut self, key: &str) -> String {
        self.engine.agent_key(key).to_string()
    }
}
