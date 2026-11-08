//! Native command line for the same engine the browser runs, used for testing and benchmarks.

use evolve_llm::chat::{Message, Options};
use evolve_llm::cpu::CpuModel;
use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::Engine};
use evolve_llm::sampler::{argmax, SamplerConfig};
use evolve_llm::session::Session;
use evolve_llm::tokenizer::{StreamDecoder, Tokenizer};
use std::io::Write;
use std::time::Instant;

struct Args {
    cmd: String,
    model: String,
    prompt: String,
    chat: bool,
    n: usize,
    temp: f32,
    ctx: usize,
    seed: u64,
}

fn parse() -> Args {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_default();
    let model = it.next().unwrap_or_default();
    let mut a = Args {
        cmd,
        model,
        prompt: "Once upon a time".into(),
        chat: false,
        n: 64,
        temp: 0.0,
        ctx: 2048,
        seed: 1,
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().expect("missing value");
        match flag.as_str() {
            "--prompt" | "-p" => a.prompt = val(),
            "--chat" => a.chat = true,
            "-n" => a.n = val().parse().unwrap(),
            "--temp" => a.temp = val().parse().unwrap(),
            "--ctx" => a.ctx = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn header(bytes: &[u8]) -> gguf::Gguf {
    gguf::parse(bytes).unwrap_or_else(|e| panic!("{e}"))
}

fn load(bytes: &[u8], g: &gguf::Gguf, ctx: usize) -> Session {
    let t = Instant::now();
    let gpu = pollster::block_on(gpu::init()).unwrap();
    eprintln!(
        "GPU: {} ({}, {})",
        gpu.info.name, gpu.info.backend, gpu.info.driver
    );
    let mut engine = Engine::new(gpu, g).unwrap();
    for p in engine.plan().to_vec() {
        let start = p.offset as usize;
        engine
            .upload(&p.name, 0, &bytes[start..start + p.bytes as usize])
            .unwrap();
    }
    engine.finish(ctx).unwrap();
    eprintln!(
        "loaded {} in {:.2}s: {:.0} MB weights, {:.0} MB cache and activations, context {}",
        engine.cfg.name,
        t.elapsed().as_secs_f64(),
        engine.weight_bytes as f64 / 1e6,
        engine.state_memory() as f64 / 1e6,
        engine.context_length()
    );
    Session::new(engine, g).unwrap()
}

fn main() {
    let a = parse();
    if a.model.is_empty() {
        eprintln!("usage: evolve-llm <info|tokenize|run|check|bench> model.gguf [--prompt text] [--chat] [-n N] [--temp T] [--ctx N]");
        std::process::exit(2);
    }
    let bytes = std::fs::read(&a.model).expect("read model");
    let g = header(&bytes);
    match a.cmd.as_str() {
        "info" => {
            let cfg = evolve_llm::model::config(&g).unwrap();
            println!("{}", serde_json::to_string_pretty(&cfg).unwrap());
        }
        "tokenize" => {
            let tok = Tokenizer::from_gguf(&g).unwrap();
            let ids = tok.encode(&a.prompt, false, true);
            println!("{}", serde_json::to_string(&ids).unwrap());
        }
        "run" => {
            let mut s = load(&bytes, &g, a.ctx);
            s.set_sampler(SamplerConfig {
                temperature: a.temp,
                seed: a.seed,
                ..Default::default()
            });
            let tokens = if a.chat {
                let m = vec![Message {
                    role: "user".into(),
                    content: a.prompt.clone(),
                }];
                s.prompt_tokens(
                    &m,
                    &Options {
                        date: Some((2026, 11, 8)),
                        enable_thinking: Some(false),
                    },
                )
                .unwrap()
            } else {
                s.encode_prompt(&a.prompt)
            };
            let t0 = Instant::now();
            let pre = pollster::block_on(s.prefill(&tokens)).unwrap();
            let t_pre = t0.elapsed().as_secs_f64();
            if !a.chat {
                print!("{}", a.prompt);
            }
            let mut dec = StreamDecoder::default();
            let t1 = Instant::now();
            let mut n = 0;
            let mut out = Vec::new();
            for _ in 0..a.n {
                let id = pollster::block_on(s.next()).unwrap();
                n += 1;
                if s.tok.is_end_of_generation(id) {
                    break;
                }
                out.push(id);
                print!("{}", dec.push(&s.tok, id));
                std::io::stdout().flush().unwrap();
            }
            println!("{}", dec.flush());
            let t_gen = t1.elapsed().as_secs_f64();
            eprintln!(
                "prompt: {} tokens in {:.3}s ({:.0} tok/s); generated {} tokens in {:.3}s ({:.1} tok/s)",
                pre.computed, t_pre, pre.computed as f64 / t_pre, n, t_gen, n as f64 / t_gen
            );
            eprintln!("ids: {}", serde_json::to_string(&out).unwrap());
        }
        "check" => {
            // Compare GPU logits with the CPU reference after every prompt position.
            let tok = Tokenizer::from_gguf(&g).unwrap();
            let ids = tok.encode(&a.prompt, tok.add_bos, true);
            let mut cpu = CpuModel::new(&g, &bytes).unwrap();
            let s = load(&bytes, &g, a.ctx);
            let gpu_batch = pollster::block_on(s.engine.forward(&ids, 0)).unwrap();
            let mut worst: f32 = 0.0;
            for (i, &id) in ids.iter().enumerate() {
                let c = cpu.forward(&[id], i);
                let gl = pollster::block_on(s.engine.forward(&[id], i)).unwrap();
                let (d, scale) = diff(&c, &gl);
                worst = worst.max(d / scale);
                println!("pos {i:3} token {id:6}: max |gpu - cpu| = {d:.5} (logit range {scale:.2}), argmax cpu {} gpu {}", argmax(&c), argmax(&gl));
                if i + 1 == ids.len() {
                    let (d, scale) = diff(&c, &gpu_batch);
                    println!(
                        "batched prompt of {} tokens: max |gpu - cpu| = {d:.5}, argmax {} vs {}",
                        ids.len(),
                        argmax(&c),
                        argmax(&gpu_batch)
                    );
                    worst = worst.max(d / scale);
                }
            }
            println!("worst relative difference {worst:.2e}");
            if worst > 2e-2 {
                std::process::exit(1);
            }
        }
        "bench" => {
            let mut s = load(&bytes, &g, a.ctx);
            s.set_sampler(SamplerConfig {
                temperature: 0.0,
                ..Default::default()
            });
            let mut prompt: Vec<u32> = s.tok.encode(&a.prompt.repeat(64), s.tok.add_bos, false);
            prompt.truncate(256);
            for round in 0..2 {
                s.reset();
                let t0 = Instant::now();
                pollster::block_on(s.prefill(&prompt)).unwrap();
                let tp = t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                let mut n = 0;
                while n < a.n {
                    let id = pollster::block_on(s.next()).unwrap();
                    n += 1;
                    if s.tok.is_end_of_generation(id) {
                        s.prefill(&[id]).now_or_never();
                        break;
                    }
                }
                let tg = t1.elapsed().as_secs_f64();
                println!(
                    "round {round}: prefill {} tokens {:.0} tok/s, decode {n} tokens {:.1} tok/s",
                    prompt.len(),
                    prompt.len() as f64 / tp,
                    n as f64 / tg
                );
            }
        }
        "profile" => {
            let mut s = load(&bytes, &g, a.ctx);
            let prompt: Vec<u32> = s
                .tok
                .encode(&a.prompt.repeat(64), s.tok.add_bos, false)
                .into_iter()
                .take(256)
                .collect();
            pollster::block_on(s.prefill(&prompt)).unwrap();
            for t in [1usize, 128] {
                let rows = s.engine.profile(t, if t == 1 { 256 } else { 0 }, 20);
                let total: f64 = rows.iter().map(|r| r.2).sum();
                println!(
                    "{t} token(s) per step: {:.2} ms of kernel time",
                    total * 1e3
                );
                for (k, n, secs) in rows {
                    println!(
                        "  {k:<16} x{n:<4} {:8.3} ms  {:5.1}%",
                        secs * 1e3,
                        secs / total * 100.0
                    );
                }
            }
        }
        other => panic!("unknown command {other}"),
    }
}

fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let d = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let max = a.iter().cloned().fold(f32::MIN, f32::max);
    let min = a.iter().cloned().fold(f32::MAX, f32::min);
    (d, (max - min).max(1e-6))
}

trait NowOrNever {
    fn now_or_never(self);
}
impl<F: std::future::Future> NowOrNever for F {
    fn now_or_never(self) {}
}
