//! Evolve Browser: a web browser engine that runs inside a web page.
//!
//! - HTML parsing, the DOM, CSS (Stylo), layout (Taffy), text (Parley) and painting come from
//!   Blitz; frames are rasterized on the CPU by vello_cpu.
//! - JavaScript runs in Boa, with the DOM and web APIs in `js/prelude.js`.
//! - Networking is HTTP/1.1 over rustls over Wisp streams, so TLS ends inside this engine and
//!   the relay only forwards ciphertext. Cookies, SameSite and CORS follow browser rules.
//!
//! See `API.md` for the interfaces between the engine, the worker, the page UI and the relay.

pub mod agent;
pub mod clock;
pub mod engine;
pub mod fonts;
pub mod host;
pub mod js;
pub mod net;
#[cfg(not(target_arch = "wasm32"))]
pub mod testing;
#[cfg(target_arch = "wasm32")]
pub mod web;

pub use engine::Engine;
