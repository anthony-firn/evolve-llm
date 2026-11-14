//! Monotonic milliseconds, matching `performance.now()` in the browser.

#[cfg(target_arch = "wasm32")]
pub fn now() -> f64 {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    js_sys::Reflect::get(&global, &"performance".into())
        .ok()
        .and_then(|p| {
            let now = js_sys::Reflect::get(&p, &"now".into()).ok()?;
            now.dyn_ref::<js_sys::Function>()?.call0(&p).ok()?.as_f64()
        })
        .unwrap_or_else(js_sys::Date::now)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn now() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}
