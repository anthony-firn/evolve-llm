//! Native command line for the same engine the browser runs, used for testing and benchmarks.

use evolve_llm::chat::{Message, Options};
use evolve_llm::cpu::CpuModel;
use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::{Engine, Options as EngineOptions}};
use evolve_llm::sampler::{argmax, SamplerConfig};
use evolve_llm::session::Session;
use evolve_llm::source::{FileSource, WeightSource};
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
    /// GPU memory for weights in MB (0 = no limit); larger models stream the rest from disk.
    gpu_mem: u64,
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
        gpu_mem: std::env::var("EVOLVE_LLM_GPU_MEM").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
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
            "--gpu-mem" => a.gpu_mem = val().parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn header(bytes: &[u8]) -> gguf::Gguf {
    gguf::parse(bytes).unwrap_or_else(|e| panic!("{e}"))
}

/// The model's files (every part of a split model) and their merged header, read without
/// loading the weights.
fn open_model(path: &str) -> (Vec<std::path::PathBuf>, gguf::Gguf) {
    let p = std::path::PathBuf::from(path);
    let name = p.file_name().unwrap().to_string_lossy().to_string();
    let paths: Vec<_> = match gguf::split_names(&name) {
        Some(names) => names.iter().map(|n| p.with_file_name(n)).collect(),
        None => vec![p.clone()],
    };
    let mut parts = Vec::new();
    for path in &paths {
        let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut want = 1 << 20;
        let mut buf = Vec::new();
        loop {
            use std::io::Read;
            buf.resize(want, 0);
            let mut got = 0;
            f.rewind_to_start();
            while got < want {
                let n = f.read(&mut buf[got..]).unwrap();
                if n == 0 {
                    break;
                }
                got += n;
            }
            buf.truncate(got);
            match gguf::parse(&buf) {
                Ok(g) => break parts.push(g),
                Err(gguf::ParseError::NeedMore(n)) if got == want => want = (n as usize).max(want * 2),
                Err(e) => panic!("{}: {e}", path.display()),
            }
        }
    }
    (paths, gguf::Gguf::merge(parts).unwrap_or_else(|e| panic!("{e}")))
}

trait Rewind {
    fn rewind_to_start(&mut self);
}

impl Rewind for std::fs::File {
    fn rewind_to_start(&mut self) {
        use std::io::Seek;
        self.seek(std::io::SeekFrom::Start(0)).unwrap();
    }
}

fn load_from(paths: &[std::path::PathBuf], g: &gguf::Gguf, ctx: usize, gpu_mem: u64) -> Session {
    let t = Instant::now();
    let gpu = pollster::block_on(gpu::init()).unwrap();
    eprintln!(
        "GPU: {} ({}, {})",
        gpu.info.name, gpu.info.backend, gpu.info.driver
    );
    let opts = EngineOptions {
        binding_limit: None,
        layout: match std::env::var("EVOLVE_LLM_LAYOUT").as_deref() {
            Ok("compact") => evolve_llm::quant::Layout::Compact,
            Ok("blocks") => evolve_llm::quant::Layout::Blocks,
            _ => evolve_llm::quant::Layout::Fast,
        },
        budget: (gpu_mem > 0).then_some(gpu_mem << 20),
    };
    let mut engine = Engine::with_options(gpu, g, opts).unwrap();
    let source = FileSource::open(paths).unwrap();
    for p in engine.plan().to_vec().iter().filter(|p| p.resident) {
        let bytes = source.read(p.file, p.offset, p.bytes as usize).unwrap();
        engine.upload(&p.name, 0, &bytes).unwrap();
    }
    engine.set_source(Box::new(source));
    engine.finish(ctx).unwrap();
    eprintln!(
        "loaded {} in {:.2}s: {:.0} MB weights, {:.0} MB cache and activations, context {}",
        engine.cfg.name,
        t.elapsed().as_secs_f64(),
        engine.weight_bytes as f64 / 1e6,
        engine.state_memory() as f64 / 1e6,
        engine.context_length()
    );
    if engine.placement.streaming {
        eprintln!("streaming: {}", serde_json::to_string(&engine.placement).unwrap());
    }
    Session::new(engine, g).unwrap()
}

fn stats(s: &Session) -> String {
    let st = s.engine.take_stream_stats();
    if st.passes == 0 && st.bytes_read == 0 {
        return String::new();
    }
    format!(
        "; read {:.1} MB from disk in {:.2}s, uploads {:.2}s, experts: {} requested, {:.0}% cached, {} loads",
        st.bytes_read as f64 / 1e6,
        st.read_seconds,
        st.upload_seconds,
        st.expert_requests,
        100.0 * st.expert_hits as f64 / st.expert_requests.max(1) as f64,
        st.expert_loads
    )
}

fn main() {
    let a = parse();
    if a.model.is_empty() {
        eprintln!("usage: evolve-llm <info|tokenize|run|check|bench> model.gguf [--prompt text] [--chat] [-n N] [--temp T] [--ctx N]");
        std::process::exit(2);
    }
    let (paths, g) = open_model(&a.model);
    let full = || std::fs::read(&a.model).expect("read model");
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
            let mut s = load_from(&paths, &g, a.ctx, a.gpu_mem);
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
                "prompt: {} tokens in {:.3}s ({:.0} tok/s); generated {} tokens in {:.3}s ({:.1} tok/s){}",
                pre.computed, t_pre, pre.computed as f64 / t_pre, n, t_gen, n as f64 / t_gen, stats(&s)
            );
            eprintln!("ids: {}", serde_json::to_string(&out).unwrap());
        }
        "check" => {
            // Compare GPU logits with the CPU reference after every prompt position.
            let tok = Tokenizer::from_gguf(&g).unwrap();
            let ids = tok.encode(&a.prompt, tok.add_bos, true);
            let bytes = full();
            let g = header(&bytes);
            let mut cpu = CpuModel::new(&g, &bytes).unwrap();
            let s = load_from(&paths, &g, a.ctx, a.gpu_mem);
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
            let mut s = load_from(&paths, &g, a.ctx, a.gpu_mem);
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
                    "round {round}: prefill {} tokens {:.0} tok/s, decode {n} tokens {:.1} tok/s{}",
                    prompt.len(),
                    prompt.len() as f64 / tp,
                    n as f64 / tg,
                    stats(&s)
                );
            }
        }
        "profile" => {
            let mut s = load_from(&paths, &g, a.ctx, a.gpu_mem);
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
