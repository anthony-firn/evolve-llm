//! Evolve LLM: GGUF language models on the GPU, natively or from WebAssembly through WebGPU.

pub mod chat;
pub mod cpu;
pub mod gguf;
pub mod gpu;
pub mod iq_tables;
pub mod model;
pub mod quant;
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod remote;
pub mod sampler;
pub mod session;
pub mod vision;
pub mod source;
pub mod tokenizer;

#[cfg(target_arch = "wasm32")]
pub mod web;
