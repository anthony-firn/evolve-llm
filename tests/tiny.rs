//! Mixture-of-experts and latent-attention architectures against transformers: tiny random
//! models from tools/make_tiny_models.py (Qwen3-MoE, Mixtral, DeepSeek V3 and V2 Lite layouts).
//! The GPU must reproduce PyTorch's float32 logits for a whole prompt at once and token by token,
//! in both weight layouts. Needs the GGUF files in $EVOLVE_LLM_MODELS and a GPU; skips otherwise.

use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::{Engine, Options}};
use evolve_llm::quant::Layout;
use serde_json::Value;
use std::path::PathBuf;

fn b64f(s: &str) -> Vec<f32> {
    let table = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|&c| c != b'=') {
        acc = (acc << 6) | table.iter().position(|&t| t == c).unwrap() as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn models_dir() -> PathBuf {
    std::env::var("EVOLVE_LLM_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/evolve-llm/models"))
}

fn worst(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

fn check(name: &str) {
    run(name, "", 5e-3);
    // The same model with 8-bit weights: quantization moves the logits a little, kernels must not.
    run(name, "-q8_0", 4e-2);
}

fn run(name: &str, suffix: &str, tol: f32) {
    let fx: Value = serde_json::from_str(
        &std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/tiny-{name}.json")))
            .unwrap(),
    )
    .unwrap();
    let path = models_dir().join(fx["file"].as_str().unwrap().replace(".gguf", &format!("{suffix}.gguf")));
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping {name}: {} not found (run tools/make_tiny_models.py)", path.display());
        return;
    };
    let g = gguf::parse(&bytes).unwrap();
    let tokens: Vec<u32> = fx["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let positions: Vec<usize> = fx["positions"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let want: Vec<Vec<f32>> = fx["logits"].as_array().unwrap().iter().map(|v| b64f(v.as_str().unwrap())).collect();
    let range = fx["range"].as_f64().unwrap() as f32;
    for layout in [Layout::Fast, Layout::Blocks] {
        let Ok(gpu) = pollster::block_on(gpu::init()) else {
            eprintln!("skipping {name}: no GPU adapter");
            return;
        };
        let opts = Options { binding_limit: None, layout };
        let mut e = Engine::with_options(gpu, &g, opts).unwrap_or_else(|err| panic!("{name}: {err}"));
        for p in e.plan().to_vec() {
            e.upload(&p.name, 0, &bytes[p.offset as usize..(p.offset + p.bytes) as usize]).unwrap();
        }
        e.finish(64).unwrap();
        // The whole prompt in one batch (matrix-matrix kernels, causal attention within the batch).
        let batched = pollster::block_on(e.forward(&tokens, 0)).unwrap();
        let b_err = worst(&batched, want.last().unwrap()) / range;
        // Token by token (matrix-vector kernels, attention over the cache).
        let mut step_err = 0f32;
        for (pos, &t) in tokens.iter().enumerate() {
            let got = pollster::block_on(e.forward(&[t], pos)).unwrap();
            if let Some(i) = positions.iter().position(|&p| p == pos) {
                step_err = step_err.max(worst(&got, &want[i]) / range);
            }
        }
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        eprintln!(
            "{name:14}{suffix:6} {layout:?}: {} weight MB, batched error {b_err:.2e}, step error {step_err:.2e} of a {range:.2} logit range",
            e.weight_bytes / 1_000_000
        );
        if suffix.is_empty() {
            assert_eq!(argmax(&batched), argmax(want.last().unwrap()), "{name}: different top token");
        }
        assert!(b_err < tol && step_err < tol, "{name}{suffix} {layout:?}: GPU differs from transformers");
    }
}

#[test]
fn qwen3_moe_matches_transformers() {
    check("qwen3moe");
}

#[test]
fn mixtral_matches_transformers() {
    check("mixtral");
}

#[test]
fn deepseek_v3_matches_transformers() {
    check("deepseek3");
}

#[test]
fn deepseek_v2_lite_matches_transformers() {
    check("deepseek2lite");
}
