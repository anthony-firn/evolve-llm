//! Evolve LLM: GGUF language models on the GPU, natively or from WebAssembly through WebGPU.

pub mod chat;
pub mod cpu;
pub mod gguf;
pub mod gpu;
pub mod iq_tables;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod session;
pub mod source;
pub mod tokenizer;

#[cfg(target_arch = "wasm32")]
pub mod web;
