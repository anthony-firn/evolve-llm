//! Mixture-of-experts and latent-attention architectures against transformers: tiny random
//! models from tools/make_tiny_models.py (Qwen3-MoE, Mixtral, DeepSeek V3 and V2 Lite layouts).
//! The GPU must reproduce PyTorch's float32 logits for a whole prompt at once and token by token,
//! in both weight layouts. Needs the GGUF files in $EVOLVE_LLM_MODELS and a GPU; skips otherwise.

use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::{Engine, Options}};
use evolve_llm::quant::Layout;
use evolve_llm::source::MemorySource;
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
    // Resident in both layouts, then streamed: a third of the weights on the GPU, and almost
    // nothing (every layer streamed, an expert cache of a few slots, several passes per prompt).
    let runs = [
        (Layout::Fast, None),
        (Layout::Blocks, None),
        (Layout::Fast, Some(bytes.len() as u64 / 3)),
        (Layout::Fast, Some(1)),
    ];
    for (layout, budget) in runs {
        let Ok(gpu) = pollster::block_on(gpu::init()) else {
            eprintln!("skipping {name}: no GPU adapter");
            return;
        };
        let opts = Options { binding_limit: None, layout, budget };
        let mut e = Engine::with_options(gpu, &g, opts).unwrap_or_else(|err| panic!("{name}: {err}"));
        for p in e.plan().to_vec().iter().filter(|p| p.resident) {
            e.upload(&p.name, 0, &bytes[p.offset as usize..(p.offset + p.bytes) as usize]).unwrap();
        }
        e.set_source(Box::new(MemorySource(vec![bytes.clone()])));
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
        let st = e.take_stream_stats();
        let mode = match budget {
            None => format!("{layout:?}"),
            Some(_) => format!(
                "streamed ({} of {} layers resident, {} expert slots, {} passes)",
                e.placement.resident_layers, e.cfg.n_layer, e.placement.expert_slots, st.passes
            ),
        };
        eprintln!(
            "{name:14}{suffix:6} {mode}: {:.1} weight MB, batched error {b_err:.2e}, step error {step_err:.2e} of a {range:.2} logit range",
            e.weight_bytes as f64 / 1e6
        );
        if suffix.is_empty() {
            assert_eq!(argmax(&batched), argmax(want.last().unwrap()), "{name}: different top token");
        }
        assert!(b_err < tol && step_err < tol, "{name}{suffix} {mode}: GPU differs from transformers");
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
fn olmoe_matches_transformers() {
    check("olmoe");
}

#[test]
fn deepseek_v3_matches_transformers() {
    check("deepseek3");
}

#[test]
fn deepseek_v2_lite_matches_transformers() {
    check("deepseek2lite");
}

/// Kimi K3.1 (linear attention, latent attention without rotation, block residuals, latent
/// experts, situ) against tools/ref_kimi_k31.py, a NumPy implementation of llama.cpp's graph,
/// on the tiny random model akopytko/tiny-kimi-k3.1-pro-GGUF.
#[test]
fn kimi_k31_matches_reference() {
    run("kimik31", "", 5e-3);
}

/// The same with the indexer's top_k patched to 8 (tools/ref_kimi_k31.py --top-k 8), so the
/// 20-token prompt goes through DSA's selection of cached positions.
#[test]
fn kimi_k31_sparse_attention_matches_reference() {
    run("kimik31-topk8", "", 5e-3);
}

/// A prompt longer than one chunk read by a model that streams goes layer by layer across its
/// chunks (each streamed weight read once for all of them). Its logits must match the same model
/// with every weight on the GPU, which reads the prompt chunk by chunk.
fn long_prompt_matches_resident(file: &str, budget: u64, want: u64) {
    let path = models_dir().join(file);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping {file}: not found");
        return;
    };
    let g = gguf::parse(&bytes).unwrap();
    let cfg = evolve_llm::model::config(&g).unwrap();
    let vocab = cfg.n_vocab as u64;
    // Several chunks (two whole ones and part of a third for 300), within the model's context.
    let n = want.min(cfg.ctx_train as u64 - 1);
    let tokens: Vec<u32> = (0..n).map(|i| ((i * 2654435761 + 12345) % vocab.min(32000)) as u32).collect();
    let mut logits = Vec::new();
    for budget in [None, Some(budget)] {
        let Ok(gpu) = pollster::block_on(gpu::init()) else {
            eprintln!("skipping {file}: no GPU adapter");
            return;
        };
        let mut e = Engine::with_options(gpu, &g, Options { binding_limit: None, layout: Layout::Fast, budget }).unwrap();
        for p in e.plan().to_vec().iter().filter(|p| p.resident) {
            e.upload(&p.name, 0, &bytes[p.offset as usize..(p.offset + p.bytes) as usize]).unwrap();
        }
        e.set_source(Box::new(MemorySource(vec![bytes.clone()])));
        e.finish(n as usize + 64).unwrap();
        assert_eq!(e.placement.streaming, budget.is_some());
        logits.push(pollster::block_on(e.forward(&tokens, 0)).unwrap());
        let st = e.take_stream_stats();
        if budget.is_some() {
            eprintln!("{file}: streamed prompt read {:.1} MB in {} expert passes", st.bytes_read as f64 / 1e6, st.passes);
        }
    }
    let range = logits[0].iter().cloned().fold(f32::MIN, f32::max) - logits[0].iter().cloned().fold(f32::MAX, f32::min);
    let err = worst(&logits[1], &logits[0]) / range;
    eprintln!("{file}: layer-by-layer streamed prompt vs resident: {err:.2e} of a {range:.2} logit range");
    assert!(err < 5e-3, "{file}: streamed long prompt differs from resident");
}

#[test]
fn long_streamed_prompts_match_resident() {
    long_prompt_matches_resident("tiny-k31-bf16.gguf", 1, 300);
    long_prompt_matches_resident("tiny-k31-bf16-topk8.gguf", 1, 300);
    long_prompt_matches_resident("tiny-deepseek3.gguf", 1, 300);
    long_prompt_matches_resident("tiny-qwen3moe.gguf", 1, 300);
    long_prompt_matches_resident("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 300 << 20, 300);
    // Longer than one layer-by-layer pass (SUPER_BATCH tokens): two passes.
    long_prompt_matches_resident("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 300 << 20, 1100);
}

/// Speculative decoding (drafts from the context, verified in one pass) must give exactly the
/// tokens one-at-a-time greedy decoding gives, with the model streaming its weights.
fn speculation_matches(file: &str, budget: u64) {
    use evolve_llm::sampler::SamplerConfig;
    use evolve_llm::session::Session;
    let path = models_dir().join(file);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping {file}: not found");
        return;
    };
    let g = gguf::parse(&bytes).unwrap();
    let Ok(gpu) = pollster::block_on(gpu::init()) else {
        eprintln!("skipping {file}: no GPU adapter");
        return;
    };
    let mut e = Engine::with_options(gpu, &g, Options { binding_limit: None, layout: Layout::Fast, budget: Some(budget) }).unwrap();
    for p in e.plan().to_vec().iter().filter(|p| p.resident) {
        e.upload(&p.name, 0, &bytes[p.offset as usize..(p.offset + p.bytes) as usize]).unwrap();
    }
    e.set_source(Box::new(MemorySource(vec![bytes.clone()])));
    e.finish(256).unwrap();
    let mut s = Session::new(e, &g).unwrap();
    s.set_sampler(SamplerConfig { temperature: 0.0, repeat_penalty: 1.0, ..Default::default() });
    // A repetitive prompt, so earlier text predicts what comes next.
    let prompt: Vec<u32> = (0..48u32).map(|i| [11, 220, 1000, 4096, 77, 31][(i % 6) as usize]).collect();
    let n = 40;
    let mut runs = Vec::new();
    for speculate in [false, true] {
        s.reset();
        s.set_speculation(speculate);
        pollster::block_on(s.prefill(&prompt)).unwrap();
        let mut out: Vec<u32> = Vec::new();
        let mut steps = 0;
        while out.len() < n {
            let ids = pollster::block_on(s.next_many()).unwrap();
            steps += 1;
            let end = ids.iter().any(|&id| s.tok.is_end_of_generation(id));
            out.extend(ids.into_iter().filter(|&id| !s.tok.is_end_of_generation(id)));
            if end {
                break;
            }
        }
        out.truncate(n);
        eprintln!("{file}: speculation {speculate}: {} tokens in {steps} steps", out.len());
        runs.push(out);
    }
    assert_eq!(runs[0], runs[1], "{file}: speculative decoding changed the greedy text");
}

#[test]
fn speculative_decoding_matches_greedy() {
    speculation_matches("tiny-k31-bf16.gguf", 1);
    speculation_matches("tiny-deepseek3.gguf", 1);
    speculation_matches("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 300 << 20);
}
