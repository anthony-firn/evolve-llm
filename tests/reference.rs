//! Checks against independent implementations (fixtures from tools/make_fixtures.py):
//! gguf-py dequantization, Hugging Face tokenizers and chat templates, and
//! transformers' float32 logits for the same GGUF weights.
//!
//! Model tests need the GGUF files in $EVOLVE_LLM_MODELS (default ~/.cache/evolve-llm/models)
//! and a GPU; they are skipped with a note when either is missing.

use evolve_llm::chat::{Message, Options};
use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::Engine};
use evolve_llm::quant;
use evolve_llm::session::Session;
use evolve_llm::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::PathBuf;

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn b64(s: &str) -> Vec<u8> {
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
    out
}

#[test]
fn dequantization_matches_gguf_py() {
    for case in fixture("quant.json").as_array().unwrap() {
        let t = case["type"].as_u64().unwrap() as u32;
        let bytes = b64(case["bytes"].as_str().unwrap());
        let want: Vec<f32> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let mut got = vec![0f32; want.len()];
        quant::dequantize(t, &bytes, &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1e-3),
                "{} value {i}: {g} vs {w}",
                case["name"]
            );
        }
        // The GPU layout must hold exactly the same values (F32 sources are rounded to f16 by design).
        let fmt = quant::GpuFormat::for_type(t).unwrap();
        let k = want.len();
        let packed = quant::repack(t, &bytes, k, 1).unwrap();
        let unpacked = quant::unpack(fmt, &packed, k);
        for (i, (u, g)) in unpacked.iter().zip(&got).enumerate() {
            let tol = if t == quant::BF16 {
                1e-3 * g.abs()
            } else {
                0.0
            };
            assert!(
                (u - g).abs() <= tol,
                "{} packed value {i}: {u} vs {g}",
                case["name"]
            );
        }
    }
}

fn model_path(file: &str) -> Option<PathBuf> {
    let dir = std::env::var("EVOLVE_LLM_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/evolve-llm/models")
        });
    let p = dir.join(file);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping {file}: not found in {}", dir.display());
        None
    }
}

const MODELS: &[&str] = &["stories15m", "smollm2", "qwen25", "qwen3", "llama32"];

#[test]
fn tokenizers_match_hugging_face() {
    for key in MODELS {
        let fx = fixture(&format!("{key}.json"));
        let Some(path) = model_path(fx["file"].as_str().unwrap()) else {
            continue;
        };
        let bytes = std::fs::read(path).unwrap();
        let g = gguf::parse(&bytes).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let mut failures = Vec::new();
        for case in fx["tokenizer"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let got = tok.encode(text, false, true);
            if got != want {
                failures.push(format!("{key} {text:?}\n  got  {got:?}\n  want {want:?}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

#[test]
fn chat_templates_match_transformers() {
    for key in MODELS {
        let fx = fixture(&format!("{key}.json"));
        let Some(chats) = fx.get("chat").and_then(Value::as_array) else {
            continue;
        };
        let Some(path) = model_path(fx["file"].as_str().unwrap()) else {
            continue;
        };
        let bytes = std::fs::read(path).unwrap();
        let g = gguf::parse(&bytes).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let tmpl = evolve_llm::chat::ChatTemplate::from_gguf(&g, &tok);
        for chat in chats {
            let messages: Vec<Message> = serde_json::from_value(chat["messages"].clone()).unwrap();
            let opts: Options = serde_json::from_value(chat["options"].clone()).unwrap();
            let got = tmpl.render(&messages, true, &opts).unwrap();
            assert_eq!(
                got,
                chat["text"].as_str().unwrap(),
                "{key} {:?}",
                chat["options"]
            );
        }
    }
}

#[test]
fn gpu_logits_match_transformers() {
    let Ok(gpu) = pollster::block_on(gpu::init()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut gpu = Some(gpu);
    for key in MODELS {
        let fx = fixture(&format!("{key}.json"));
        let Some(path) = model_path(fx["file"].as_str().unwrap()) else {
            continue;
        };
        let bytes = std::fs::read(path).unwrap();
        let g = gguf::parse(&bytes).unwrap();
        let mut engine = Engine::new(gpu.take().unwrap(), &g).unwrap();
        for p in engine.plan().to_vec() {
            engine
                .upload(
                    &p.name,
                    0,
                    &bytes[p.offset as usize..(p.offset + p.bytes) as usize],
                )
                .unwrap();
        }
        engine.finish(1024).unwrap();
        let ids: Vec<u32> = fx["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let logits = pollster::block_on(engine.forward(&ids, 0)).unwrap();
        let top = fx["logits"]["top"].as_array().unwrap();
        let range = fx["logits"]["max"].as_f64().unwrap() - fx["logits"]["min"].as_f64().unwrap();
        let best = evolve_llm::sampler::argmax(&logits);
        let mut worst = 0f64;
        for pair in top {
            let id = pair[0].as_u64().unwrap() as usize;
            let want = pair[1].as_f64().unwrap();
            worst = worst.max((logits[id] as f64 - want).abs());
        }
        eprintln!("{key}: {} prompt tokens, top-20 max |gpu - transformers| = {worst:.4} (range {range:.1}), argmax {best}", ids.len());
        assert_eq!(
            best as u64,
            top[0][0].as_u64().unwrap(),
            "{key}: argmax differs from transformers"
        );
        assert!(worst < 0.01 * range, "{key}: logits differ by {worst}");
        // Also exercise the chat session path, which must reproduce the same logits.
        let mut s = Session::new(engine, &g).unwrap();
        pollster::block_on(s.prefill(&ids)).unwrap();
        assert_eq!(evolve_llm::sampler::argmax(s.last_logits()), best);
        gpu = Some(s.engine.gpu);
    }
}
