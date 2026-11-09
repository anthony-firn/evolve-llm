//! Every weight format through the GPU kernels: a tiny random llama-layout model is written as
//! GGUF in each format, and the GPU's logits must match the CPU reference for single tokens
//! and for a batched prompt. Skipped when no GPU adapter is available.

use evolve_llm::cpu::CpuModel;
use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::{Engine, Options}};
use evolve_llm::quant::{self, block_info};

const D: u64 = 256;
const FF: u64 = 512;
const VOCAB: u64 = 384;
const LAYERS: u64 = 2;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Offsets of the f16 scale fields inside one block, which must hold sane values.
fn half_fields(t: u32) -> &'static [usize] {
    match t {
        quant::Q4_0 | quant::Q5_0 | quant::Q8_0 | quant::IQ4_NL | quant::IQ4_XS => &[0],
        quant::Q4_1 | quant::Q5_1 | quant::Q4_K | quant::Q5_K => &[0, 2],
        quant::Q2_K => &[80, 82],
        quant::Q3_K => &[108],
        quant::Q6_K => &[208],
        quant::IQ2_XXS | quant::IQ2_XS | quant::IQ2_S | quant::IQ3_XXS | quant::IQ3_S | quant::IQ1_S => &[0],
        quant::TQ1_0 => &[52],
        quant::TQ2_0 => &[64],
        _ => &[],
    }
}

/// Block scale that keeps weights around a few hundredths, given each format's largest code.
fn unit_scale(t: u32) -> f32 {
    match t {
        quant::Q8_0 | quant::Q6_K => 0.0012,
        quant::IQ4_NL | quant::IQ4_XS => 0.0006,
        quant::IQ2_XXS | quant::IQ2_XS | quant::IQ2_S => 0.0003,
        quant::IQ3_XXS | quant::IQ3_S => 0.0001,
        quant::IQ1_S | quant::IQ1_M => 0.003,
        quant::TQ1_0 | quant::TQ2_0 => 0.04,
        _ => 0.006,
    }
}

fn tensor_bytes(t: u32, k: u64, rows: u64, rng: &mut Rng) -> Vec<u8> {
    let n = (k * rows) as usize;
    match t {
        quant::F32 => (0..n)
            .flat_map(|_| (rng.unit() * 0.08).to_le_bytes())
            .collect(),
        quant::F16 => (0..n)
            .flat_map(|_| quant::f32_to_f16(rng.unit() * 0.08).to_le_bytes())
            .collect(),
        quant::BF16 => (0..n)
            .flat_map(|_| (((rng.unit() * 0.08).to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => {
            let (vals, size) = block_info(t).unwrap();
            let blocks = n / vals;
            let mut b: Vec<u8> = (0..blocks * size).map(|_| rng.next() as u8).collect();
            // Scales small enough that activations stay in range through the layers.
            let k_quant_super = vals == 256 && matches!(t, quant::Q2_K | quant::Q3_K | quant::Q4_K | quant::Q5_K | quant::Q6_K);
            for bi in 0..blocks {
                let blk = &mut b[bi * size..(bi + 1) * size];
                for (j, &at) in half_fields(t).iter().enumerate() {
                    // K-quant super-block scales multiply 6-bit sub-scales, so they are smaller.
                    let s = if k_quant_super {
                        unit_scale(t) / 24.0
                    } else {
                        unit_scale(t)
                    } * (0.6 + 0.4 * rng.unit().abs());
                    let v = if j == 1 { s * 0.5 } else { s };
                    blk[at..at + 2].copy_from_slice(&quant::f32_to_f16(v).to_le_bytes());
                }
                match t {
                    // MXFP4: an E8M0 exponent near 2^-8.
                    quant::MXFP4 => blk[0] = 117 + (rng.next() % 5) as u8,
                    // IQ1_M spreads its f16 scale over the top nibbles of four 16-bit scale words.
                    quant::IQ1_M => {
                        let d = quant::f32_to_f16(unit_scale(t) * (0.6 + 0.4 * rng.unit().abs()));
                        for i in 0..4 {
                            let w = u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]);
                            let w = (w & 0x0fff) | (((d >> (4 * i)) & 15) << 12);
                            blk[48 + 2 * i..50 + 2 * i].copy_from_slice(&w.to_le_bytes());
                        }
                    }
                    _ => {}
                }
            }
            b
        }
    }
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn kv_u32(out: &mut Vec<u8>, k: &str, v: u32) {
    push_str(out, k);
    out.extend(4u32.to_le_bytes());
    out.extend(v.to_le_bytes());
}

fn kv_f32(out: &mut Vec<u8>, k: &str, v: f32) {
    push_str(out, k);
    out.extend(6u32.to_le_bytes());
    out.extend(v.to_le_bytes());
}

fn model(t: u32, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let mut mat = |name: String, rows: u64, k: u64, rng: &mut Rng| {
        let bytes = tensor_bytes(t, k, rows, rng);
        tensors.push((name, vec![k, rows], t, bytes));
    };
    mat("token_embd.weight".into(), VOCAB, D, &mut rng);
    for l in 0..LAYERS {
        mat(format!("blk.{l}.attn_q.weight"), D, D, &mut rng);
        mat(format!("blk.{l}.attn_k.weight"), D / 2, D, &mut rng);
        mat(format!("blk.{l}.attn_v.weight"), D / 2, D, &mut rng);
        mat(format!("blk.{l}.attn_output.weight"), D, D, &mut rng);
        mat(format!("blk.{l}.ffn_gate.weight"), FF, D, &mut rng);
        mat(format!("blk.{l}.ffn_up.weight"), FF, D, &mut rng);
        mat(format!("blk.{l}.ffn_down.weight"), D, FF, &mut rng);
    }
    let mut norm = |name: String, rng: &mut Rng| {
        let bytes: Vec<u8> = (0..D)
            .flat_map(|_| (1.0 + 0.2 * rng.unit()).to_le_bytes())
            .collect();
        tensors.push((name, vec![D], quant::F32, bytes));
    };
    norm("output_norm.weight".into(), &mut rng);
    for l in 0..LAYERS {
        norm(format!("blk.{l}.attn_norm.weight"), &mut rng);
        norm(format!("blk.{l}.ffn_norm.weight"), &mut rng);
    }
    let mut out = Vec::new();
    out.extend(0x4655_4747u32.to_le_bytes());
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend(9u64.to_le_bytes());
    push_str(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    push_str(&mut out, "llama");
    kv_u32(&mut out, "llama.embedding_length", D as u32);
    kv_u32(&mut out, "llama.block_count", LAYERS as u32);
    kv_u32(&mut out, "llama.feed_forward_length", FF as u32);
    kv_u32(&mut out, "llama.attention.head_count", 4);
    kv_u32(&mut out, "llama.attention.head_count_kv", 2);
    kv_u32(&mut out, "llama.context_length", 256);
    kv_f32(&mut out, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_f32(&mut out, "llama.rope.freq_base", 10000.0);
    let mut offset = 0u64;
    let mut offsets = Vec::new();
    for (name, dims, ty, bytes) in &tensors {
        push_str(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(ty.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offsets.push(offset);
        offset += (bytes.len() as u64).div_ceil(32) * 32;
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    let base = out.len() as u64;
    for ((_, _, _, bytes), off) in tensors.iter().zip(offsets) {
        out.resize((base + off) as usize, 0);
        out.extend(bytes);
    }
    out
}

#[test]
fn every_format_matches_the_cpu_reference_on_the_gpu() {
    let only = std::env::var("EVOLVE_LLM_FORMAT").ok();
    let formats = [
        quant::Q4_0,
        quant::Q4_1,
        quant::Q5_0,
        quant::Q5_1,
        quant::Q8_0,
        quant::Q2_K,
        quant::Q3_K,
        quant::Q4_K,
        quant::Q5_K,
        quant::Q6_K,
        quant::IQ4_NL,
        quant::IQ4_XS,
        quant::IQ2_XXS,
        quant::IQ2_XS,
        quant::IQ2_S,
        quant::IQ3_XXS,
        quant::IQ3_S,
        quant::IQ1_S,
        quant::IQ1_M,
        quant::TQ1_0,
        quant::TQ2_0,
        quant::MXFP4,
        quant::F16,
        quant::BF16,
        quant::F32,
    ];
    let Ok(first) = pollster::block_on(gpu::init()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut device = Some(first);
    let tokens = [3u32, 141, 7, 300, 77, 12];
    // Every format runs in its fast and its compact layout, whole and again with matrices split
    // into 64-row shards, as on devices with small storage-binding limits.
    // Finally every layer streamed from storage (ring buffers, embeddings on the CPU).
    let runs = [false, true]
        .into_iter()
        .flat_map(|compact| {
            formats
                .iter()
                .map(move |&t| (t, None, compact, None))
                .chain(formats.iter().map(move |&t| (t, Some(16 * 1024u64), compact, None)))
        })
        .chain(formats.iter().map(|&t| (t, None, true, Some(1u64))));
    for (i, (t, limit, compact, budget)) in runs.enumerate() {
        if only.as_deref().is_some_and(|f| !quant::type_name(t).eq_ignore_ascii_case(f)) {
            continue;
        }
        let bytes = model(t, 0x9e37 + (i % formats.len()) as u64);
        let g = gguf::parse(&bytes).unwrap();
        let opts = Options {
            binding_limit: limit,
            layout: if compact { quant::Layout::Blocks } else { quant::Layout::Fast },
            budget,
        };
        let mut engine = Engine::with_options(device.take().unwrap(), &g, opts).unwrap();
        engine.set_source(Box::new(evolve_llm::source::MemorySource(vec![bytes.clone()])));
        for p in engine.plan().to_vec().iter().filter(|p| p.resident) {
            engine
                .upload(
                    &p.name,
                    0,
                    &bytes[p.offset as usize..(p.offset + p.bytes) as usize],
                )
                .unwrap();
        }
        engine.finish(64).unwrap();
        let mut cpu = CpuModel::new(&g, &bytes).unwrap();
        let mut worst = 0f32;
        let mut scale = 0f32;
        for (pos, &tok) in tokens.iter().enumerate() {
            let want = cpu.forward(&[tok], pos);
            let got = pollster::block_on(engine.forward(&[tok], pos)).unwrap();
            let range = want.iter().cloned().fold(f32::MIN, f32::max)
                - want.iter().cloned().fold(f32::MAX, f32::min);
            scale = scale.max(range);
            worst = worst.max(
                want.iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max)
                    / range,
            );
            assert!(
                want.iter().all(|v| v.is_finite()),
                "{}: CPU logits overflowed",
                quant::type_name(t)
            );
        }
        let want = {
            let mut cpu = CpuModel::new(&g, &bytes).unwrap();
            cpu.forward(&tokens, 0)
        };
        let got = pollster::block_on(engine.forward(&tokens, 0)).unwrap();
        let range = want.iter().cloned().fold(f32::MIN, f32::max)
            - want.iter().cloned().fold(f32::MAX, f32::min);
        let batched = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
            / range;
        eprintln!(
            "{:7} {:8} in {:3} buffers: logit range {scale:7.3}, one-token max error {:.2e}, batched {:.2e}",
            quant::type_name(t),
            if budget.is_some() { "streamed" } else if compact { "compact" } else { "fast" },
            engine.shard_count(),
            worst,
            batched
        );
        assert!(
            scale > 0.05,
            "{}: degenerate test model",
            quant::type_name(t)
        );
        assert!(
            worst < 2e-3 && batched < 2e-3,
            "{}: GPU differs from the CPU reference",
            quant::type_name(t)
        );
        device = Some(engine.gpu);
    }
}
