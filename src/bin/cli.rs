//! Native command line for the same engine the browser runs, used for testing and benchmarks.

use evolve_llm::chat::{Message, Options};
use evolve_llm::cpu::CpuModel;
use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::{Engine, Options as EngineOptions}};
use evolve_llm::sampler::{argmax, SamplerConfig};
use evolve_llm::remote::HttpSource;
use evolve_llm::session::Session;
use evolve_llm::source::{FileSource, Span, WeightSource};
use evolve_llm::tokenizer::{StreamDecoder, Tokenizer};
use std::io::Write;
use std::sync::Arc;
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
    /// For models read over the network: where to cache, how much, and requests in flight.
    cache_dir: Option<String>,
    cache_gb: f64,
    parallel: usize,
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
        cache_dir: None,
        cache_gb: 8.0,
        parallel: 8,
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
            "--cache-dir" => a.cache_dir = Some(val()),
            "--cache-gb" => a.cache_gb = val().parse().unwrap(),
            "--parallel" => a.parallel = val().parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn header(bytes: &[u8]) -> gguf::Gguf {
    gguf::parse(bytes).unwrap_or_else(|e| panic!("{e}"))
}

/// An opened model: where its weights are read from and its merged header.
struct Model {
    src: Arc<dyn WeightSource>,
    /// Set when the weights are read over the network.
    net: Option<Arc<HttpSource>>,
    paths: Vec<std::path::PathBuf>,
    g: gguf::Gguf,
}

/// `hf:owner/repo/dir/file.gguf` or a huggingface.co resolve (or blob) URL, as the download
/// address of each part of the model; None for a local path.
fn remote_urls(spec: &str) -> Option<Vec<String>> {
    let url = if let Some(rest) = spec.strip_prefix("hf:") {
        let mut parts = rest.splitn(3, '/');
        let (owner, repo, file) = (parts.next()?, parts.next()?, parts.next()?);
        format!("https://huggingface.co/{owner}/{repo}/resolve/main/{file}")
    } else if spec.starts_with("https://") || spec.starts_with("http://") {
        spec.replacen("/blob/", "/resolve/", 1)
    } else {
        return None;
    };
    let (dir, name) = url.rsplit_once('/')?;
    Some(match gguf::split_names(name) {
        Some(names) => names.iter().map(|n| format!("{dir}/{n}")).collect(),
        None => vec![url.clone()],
    })
}

fn open_model(a: &Args) -> Model {
    let Some(urls) = remote_urls(&a.model) else {
        let (paths, g) = open_files(&a.model);
        let src: Arc<dyn WeightSource> = Arc::new(FileSource::open(&paths).unwrap());
        return Model { src, net: None, paths, g };
    };
    let key: String = urls[0]
        .trim_start_matches("https://")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    let dir = a.cache_dir.clone().map(std::path::PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        std::path::PathBuf::from(home).join(".cache/evolve-llm/stream").join(key)
    });
    let cap = (a.cache_gb * 1e9) as u64;
    let mut net = HttpSource::open(urls.clone(), Some((dir.clone(), cap))).unwrap_or_else(|e| panic!("{e}"));
    net.parallel = a.parallel;
    let net = Arc::new(net);
    eprintln!(
        "reading {} part(s) from {} over the network; cache {} ({:.1} GB cap)",
        urls.len(),
        urls[0],
        dir.display(),
        a.cache_gb
    );
    let mut parts = Vec::new();
    for (file, url) in urls.iter().enumerate() {
        let mut want = 4usize << 20;
        loop {
            let span = Span { file, offset: 0, len: want, every_token: true };
            pollster::block_on(net.prefetch(std::slice::from_ref(&span))).unwrap_or_else(|e| panic!("{e}"));
            let buf = net.read(file, 0, want).unwrap_or_else(|e| panic!("{e}"));
            match gguf::parse(&buf) {
                Ok(g) => break parts.push(g),
                Err(gguf::ParseError::NeedMore(n)) => want = (n as usize).max(want * 2),
                Err(e) => panic!("{url}: {e}"),
            }
        }
    }
    let g = gguf::Gguf::merge(parts).unwrap_or_else(|e| panic!("{e}"));
    let src: Arc<dyn WeightSource> = net.clone();
    Model { src, net: Some(net), paths: Vec::new(), g }
}

/// The model's files (every part of a split model) and their merged header, read without
/// loading the weights.
fn open_files(path: &str) -> (Vec<std::path::PathBuf>, gguf::Gguf) {
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

fn load_from(m: &Model, ctx: usize, gpu_mem: u64) -> Session {
    let g = &m.g;
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
        stage: None,
    };
    let mut engine = Engine::with_options(gpu, g, opts).unwrap();
    // Resident weights in slices of up to 64 MB, fetched about 512 MB at a time.
    let mut slices: Vec<(String, usize, Span)> = Vec::new();
    for p in engine.plan().iter().filter(|p| p.resident) {
        let step = ((64u64 << 20) / p.row_bytes).max(1);
        let mut r = 0;
        while r < p.rows {
            let n = step.min(p.rows - r);
            let span = Span {
                file: p.file,
                offset: p.offset + r * p.row_bytes,
                len: (n * p.row_bytes) as usize,
                every_token: false,
            };
            slices.push((p.name.clone(), r as usize, span));
            r += n;
        }
    }
    let mut i = 0;
    while i < slices.len() {
        let mut j = i;
        let mut bytes = 0;
        while j < slices.len() && (j == i || bytes + slices[j].2.len <= 512 << 20) {
            bytes += slices[j].2.len;
            j += 1;
        }
        let spans: Vec<Span> = slices[i..j].iter().map(|s| s.2).collect();
        pollster::block_on(m.src.prefetch(&spans)).unwrap_or_else(|e| panic!("{e}"));
        for (name, row0, s) in &slices[i..j] {
            let b = m.src.read(s.file, s.offset, s.len).unwrap_or_else(|e| panic!("{e}"));
            engine.upload(name, *row0, &b).unwrap();
        }
        if m.net.is_some() {
            eprintln!("  loaded {:.0} MB of resident weights", slices[..j].iter().map(|s| s.2.len as f64).sum::<f64>() / 1e6);
        }
        i = j;
    }
    engine.set_source(Box::new(m.src.clone()));
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

fn stats(s: &Session, net: Option<&HttpSource>) -> String {
    let st = s.engine.take_stream_stats();
    if st.passes == 0 && st.bytes_read == 0 {
        return String::new();
    }
    let mut out = format!(
        "; read {:.1} MB of streamed weights, waiting {:.2}s, uploads {:.2}s, experts: {} requested, {:.0}% on the GPU, {} loads",
        st.bytes_read as f64 / 1e6,
        st.read_seconds,
        st.upload_seconds,
        st.expert_requests,
        100.0 * st.expert_hits as f64 / st.expert_requests.max(1) as f64,
        st.expert_loads
    );
    if let Some(n) = net {
        let ns = n.take_stats();
        out += &format!(
            "; network {:.1} MB in {} requests ({} retries) over {:.1}s, {:.1} MB/s; disk cache hits {:.1} MB, cache holds {:.2} GB",
            ns.fetched_bytes as f64 / 1e6,
            ns.requests,
            ns.retries,
            ns.fetch_seconds,
            ns.fetched_bytes as f64 / 1e6 / ns.fetch_seconds.max(1e-9),
            ns.cache_hit_bytes as f64 / 1e6,
            ns.cache_bytes as f64 / 1e9
        );
    }
    out
}

fn main() {
    let a = parse();
    if a.model.is_empty() {
        eprintln!("usage: evolve-llm <info|tokenize|run|check|bench> model.gguf [--prompt text] [--chat] [-n N] [--temp T] [--ctx N]");
        std::process::exit(2);
    }
    let m = open_model(&a);
    let g = &m.g;
    let net = m.net.as_deref();
    match a.cmd.as_str() {
        "info" => {
            let cfg = evolve_llm::model::config(g).unwrap();
            println!("{}", serde_json::to_string_pretty(&cfg).unwrap());
        }
        "plan" => {
            // Where the weights would live and what each token reads, without loading them.
            let gpu = pollster::block_on(gpu::init()).unwrap();
            eprintln!("GPU: {} ({})", gpu.info.name, gpu.info.backend);
            let opts = EngineOptions {
                binding_limit: None,
                layout: evolve_llm::quant::Layout::Fast,
                budget: (a.gpu_mem > 0).then_some(a.gpu_mem << 20),
                stage: None,
            };
            let mut e = Engine::with_options(gpu, g, opts).unwrap_or_else(|err| panic!("{err}"));
            e.finish(a.ctx).unwrap_or_else(|err| panic!("{err}"));
            let cfg = &e.cfg;
            let (total, active) = cfg.parameters();
            let plan = e.plan();
            let expert = |n: &str| n.contains("_exps");
            let per_expert: f64 = plan.iter().filter(|p| expert(&p.name)).map(|p| p.bytes as f64).sum::<f64>()
                / cfg.moe.as_ref().map_or(1, |m| m.n_expert) as f64;
            let streamed_dense: u64 = plan.iter().filter(|p| !p.resident && !expert(&p.name) && p.name != "token_embd.weight").map(|p| p.bytes).sum();
            let used = cfg.moe.as_ref().map_or(0, |m| m.n_used) as f64;
            println!("{} parameters, {} active per token", total, active);
            println!("GPU: {:.2} GB of weights, {:.2} GB of context and state", e.weight_bytes as f64 / 1e9, e.state_memory() as f64 / 1e9);
            println!("placement: {}", serde_json::to_string(&e.placement).unwrap());
            println!(
                "each token reads {:.2} GB of streamed attention and shared weights and up to {:.2} GB of experts (less what the expert cache holds)",
                streamed_dense as f64 / 1e9,
                per_expert * used / 1e9
            );
        }
        "render" => {
            // The chat template applied to one user message, as the model will read it.
            let tok = Tokenizer::from_gguf(g).unwrap();
            let t = evolve_llm::chat::ChatTemplate::from_gguf(g, &tok);
            // `EVOLVE_LLM_MESSAGES` (a JSON file of messages) and `EVOLVE_LLM_TOOLS` (a JSON file
            // of tool schemas) render a whole conversation, for checking tool-call syntax.
            let m: Vec<Message> = match std::env::var("EVOLVE_LLM_MESSAGES") {
                Ok(path) => serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap(),
                Err(_) => vec![Message::new("user", &a.prompt)],
            };
            let tools = std::env::var("EVOLVE_LLM_TOOLS")
                .ok()
                .map(|path| serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap());
            let opts = Options { date: Some((2026, 11, 18)), enable_thinking: Some(a.chat), thinking_effort: None, tools };
            println!("{}", t.render(&m, true, &opts).unwrap_or_else(|e| panic!("{e}")));
        }
        "tokenize" => {
            let tok = Tokenizer::from_gguf(g).unwrap();
            let ids = tok.encode(&a.prompt, false, true);
            println!("{}", serde_json::to_string(&ids).unwrap());
        }
        "run" => {
            let mut s = load_from(&m, a.ctx, a.gpu_mem);
            s.set_sampler(SamplerConfig {
                temperature: a.temp,
                seed: a.seed,
                ..Default::default()
            });
            let tokens = if a.chat {
                let m = vec![Message::new("user", &a.prompt)];
                s.prompt_tokens(
                    &m,
                    &Options {
                        date: Some((2026, 11, 8)),
                        enable_thinking: Some(false),
                        thinking_effort: None,
                        tools: None,
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
            // Several tokens per pass when speculation is on (the model streams).
            'gen: while n < a.n {
                let tt = Instant::now();
                let ids = pollster::block_on(s.next_many()).unwrap();
                if net.is_some() {
                    eprintln!("\n[{} token(s) in {:.1}s{}]", ids.len(), tt.elapsed().as_secs_f64(), stats(&s, net));
                }
                for id in ids {
                    n += 1;
                    if s.tok.is_end_of_generation(id) || n > a.n {
                        break 'gen;
                    }
                    out.push(id);
                    print!("{}", dec.push(&s.tok, id));
                }
                std::io::stdout().flush().unwrap();
            }
            println!("{}", dec.flush());
            let t_gen = t1.elapsed().as_secs_f64();
            eprintln!(
                "prompt: {} tokens in {:.3}s ({:.0} tok/s); generated {} tokens in {:.3}s ({:.1} tok/s){}",
                pre.computed, t_pre, pre.computed as f64 / t_pre, n, t_gen, n as f64 / t_gen, stats(&s, net)
            );
            eprintln!("ids: {}", serde_json::to_string(&out).unwrap());
        }
        "check" => {
            // Compare GPU logits with the CPU reference after every prompt position.
            let tok = Tokenizer::from_gguf(g).unwrap();
            let ids = tok.encode(&a.prompt, tok.add_bos, true);
            assert!(m.net.is_none(), "check needs a local model");
            let bytes = std::fs::read(&m.paths[0]).expect("read model");
            let g = header(&bytes);
            let mut cpu = CpuModel::new(&g, &bytes).unwrap();
            let s = load_from(&m, a.ctx, a.gpu_mem);
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
            let mut s = load_from(&m, a.ctx, a.gpu_mem);
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
                    stats(&s, net)
                );
            }
        }
        "profile" => {
            let mut s = load_from(&m, a.ctx, a.gpu_mem);
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
