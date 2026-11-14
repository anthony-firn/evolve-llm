//! The page's or worker's global scope. Evolve Browser runs webtor inside a dedicated Web Worker,
//! where `window` does not exist, so these call `fetch`, `setTimeout` and `performance` on
//! `globalThis` instead.

use js_sys::{Function, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};

fn method(name: &str) -> Option<(JsValue, Function)> {
    let global: JsValue = js_sys::global().into();
    let f = Reflect::get(&global, &JsValue::from_str(name)).ok()?;
    f.is_function().then(|| (global, f.unchecked_into()))
}

pub fn fetch(request: &web_sys::Request) -> Promise {
    match method("fetch") {
        Some((global, f)) => match f.call1(&global, request) {
            Ok(p) => p.unchecked_into(),
            Err(e) => Promise::reject(&e),
        },
        None => Promise::reject(&JsValue::from_str("fetch is not available")),
    }
}

pub fn set_timeout(callback: &Function, ms: i32) {
    if let Some((global, f)) = method("setTimeout") {
        let _ = f.call2(&global, callback, &JsValue::from(ms));
    }
}

/// Milliseconds from a monotonic clock (performance.now, or Date.now if it is missing).
pub fn now_ms() -> f64 {
    let global: JsValue = js_sys::global().into();
    Reflect::get(&global, &JsValue::from_str("performance"))
        .ok()
        .filter(|p| p.is_object())
        .and_then(|p| {
            let now: Function = Reflect::get(&p, &JsValue::from_str("now")).ok()?.dyn_into().ok()?;
            now.call0(&p).ok()?.as_f64()
        })
        .unwrap_or_else(js_sys::Date::now)
}
