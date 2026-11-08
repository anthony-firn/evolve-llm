//! Model hyperparameters and tensor names for the supported decoder architectures.

use crate::gguf::Gguf;
use crate::quant;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Arch {
    /// Llama 2/3, Mistral, SmolLM2, TinyLlama and other llama-layout models.
    Llama,
    /// Qwen2 and Qwen2.5: biased Q/K/V projections.
    Qwen2,
    /// Qwen3 dense models: per-head RMSNorm on queries and keys.
    Qwen3,
}

#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub arch: Arch,
    pub name: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub ctx_train: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    /// NeoX-style rotation pairs (i, i + d/2) instead of adjacent pairs.
    pub rope_neox: bool,
    /// Linear position scaling factor (1 = none).
    pub rope_linear_scale: f32,
    pub has_rope_freqs: bool,
    pub tied_embeddings: bool,
    pub qkv_bias: bool,
    pub qk_norm: bool,
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
}

pub fn arch_of(g: &Gguf) -> Result<Arch, String> {
    let arch = g
        .str("general.architecture")
        .ok_or("missing general.architecture")?;
    Ok(match arch {
        "llama" | "mistral" => Arch::Llama,
        "qwen2" => Arch::Qwen2,
        "qwen3" => Arch::Qwen3,
        other => {
            return Err(format!(
                "the {other} architecture is not supported yet (supported: llama, mistral, qwen2, qwen3)"
            ))
        }
    })
}

pub fn config(g: &Gguf) -> Result<Config, String> {
    let arch = arch_of(g)?;
    let prefix = g.str("general.architecture").unwrap().to_string();
    let key = |k: &str| format!("{prefix}.{k}");
    let need = |k: &str| {
        g.u64(&key(k))
            .map(|v| v as usize)
            .ok_or_else(|| format!("missing {}", key(k)))
    };
    let n_embd = need("embedding_length")?;
    let n_head = need("attention.head_count")?;
    let n_head_kv = g
        .u64(&key("attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(n_head);
    let head_dim = g
        .u64(&key("attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or(n_embd / n_head.max(1));
    let embd = g
        .tensor("token_embd.weight")
        .ok_or("missing token_embd.weight")?;
    let n_vocab = g
        .u64(&key("vocab_size"))
        .map(|v| v as usize)
        .unwrap_or(embd.dims[1] as usize);
    let scaling = g.str(&key("rope.scaling.type")).unwrap_or("none");
    let rope_linear_scale = match scaling {
        "linear" => g.f64(&key("rope.scaling.factor")).unwrap_or(1.0) as f32,
        _ => 1.0,
    };
    let cfg = Config {
        arch,
        name: g.str("general.name").unwrap_or("model").to_string(),
        n_vocab,
        n_embd,
        n_layer: need("block_count")?,
        n_head,
        n_head_kv,
        head_dim,
        n_ff: need("feed_forward_length")?,
        ctx_train: g.u64(&key("context_length")).unwrap_or(2048) as usize,
        rms_eps: g
            .f64(&key("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5) as f32,
        rope_base: g.f64(&key("rope.freq_base")).unwrap_or(10000.0) as f32,
        rope_dim: g
            .u64(&key("rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim),
        rope_neox: arch != Arch::Llama,
        rope_linear_scale,
        has_rope_freqs: g.tensor("rope_freqs.weight").is_some(),
        tied_embeddings: g.tensor("output.weight").is_none(),
        qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
        qk_norm: g.tensor("blk.0.attn_q_norm.weight").is_some(),
    };
    validate(g, &cfg)?;
    Ok(cfg)
}

fn validate(g: &Gguf, c: &Config) -> Result<(), String> {
    if c.n_head == 0 || c.n_head_kv == 0 || c.n_head % c.n_head_kv != 0 {
        return Err(format!(
            "unsupported head layout {} / {}",
            c.n_head, c.n_head_kv
        ));
    }
    if c.head_dim % 4 != 0 || c.head_dim > 256 || c.rope_dim > c.head_dim || c.rope_dim % 2 != 0 {
        return Err(format!("unsupported head size {}", c.head_dim));
    }
    if scaling_is_unsupported(g) {
        return Err("this model needs YaRN or LongRoPE scaling, which is not supported yet".into());
    }
    for (name, rows, cols) in matrices(c) {
        let t = g
            .tensor(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if t.dims.len() != 2 || t.dims[0] as usize != cols || t.dims[1] as usize != rows {
            return Err(format!(
                "tensor {name} has shape {:?}, expected [{cols}, {rows}]",
                t.dims
            ));
        }
        quant::GpuFormat::for_type(t.ggml_type).ok_or_else(|| {
            format!(
                "{name} uses {}, which is not supported yet",
                quant::type_name(t.ggml_type)
            )
        })?;
        quant::row_bytes(t.ggml_type, cols).map_err(|e| format!("{name}: {e}"))?;
    }
    for (name, len) in vectors(c) {
        let t = g
            .tensor(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if t.elements() as usize != len {
            return Err(format!(
                "tensor {name} has {} values, expected {len}",
                t.elements()
            ));
        }
        quant::row_bytes(t.ggml_type, len).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

fn scaling_is_unsupported(g: &Gguf) -> bool {
    let arch = g.str("general.architecture").unwrap_or("");
    matches!(
        g.str(&format!("{arch}.rope.scaling.type")),
        Some("yarn") | Some("longrope")
    )
}

/// Every weight matrix as (name, rows = outputs, cols = inputs).
pub fn matrices(c: &Config) -> Vec<(String, usize, usize)> {
    let mut m = vec![("token_embd.weight".to_string(), c.n_vocab, c.n_embd)];
    if !c.tied_embeddings {
        m.push(("output.weight".into(), c.n_vocab, c.n_embd));
    }
    for i in 0..c.n_layer {
        let b = |s: &str| format!("blk.{i}.{s}.weight");
        m.push((b("attn_q"), c.q_dim(), c.n_embd));
        m.push((b("attn_k"), c.kv_dim(), c.n_embd));
        m.push((b("attn_v"), c.kv_dim(), c.n_embd));
        m.push((b("attn_output"), c.n_embd, c.q_dim()));
        m.push((b("ffn_gate"), c.n_ff, c.n_embd));
        m.push((b("ffn_up"), c.n_ff, c.n_embd));
        m.push((b("ffn_down"), c.n_embd, c.n_ff));
    }
    m
}

/// Every vector (norm weights, biases, rope factors) as (name, length).
pub fn vectors(c: &Config) -> Vec<(String, usize)> {
    let mut v = vec![("output_norm.weight".to_string(), c.n_embd)];
    if c.has_rope_freqs {
        v.push(("rope_freqs.weight".into(), c.rope_dim / 2));
    }
    for i in 0..c.n_layer {
        v.push((format!("blk.{i}.attn_norm.weight"), c.n_embd));
        v.push((format!("blk.{i}.ffn_norm.weight"), c.n_embd));
        if c.qkv_bias {
            v.push((format!("blk.{i}.attn_q.bias"), c.q_dim()));
            v.push((format!("blk.{i}.attn_k.bias"), c.kv_dim()));
            v.push((format!("blk.{i}.attn_v.bias"), c.kv_dim()));
        }
        if c.qk_norm {
            v.push((format!("blk.{i}.attn_q_norm.weight"), c.head_dim));
            v.push((format!("blk.{i}.attn_k_norm.weight"), c.head_dim));
        }
    }
    v
}

/// Rotation frequency for pair `i` (of `rope_dim / 2`), before position scaling.
pub fn rope_inv_freq(c: &Config, i: usize, factors: Option<&[f32]>) -> f32 {
    let base = c.rope_base.powf(-(2.0 * i as f32) / c.rope_dim as f32);
    let f = factors.map(|f| f[i]).unwrap_or(1.0);
    base / f / c.rope_linear_scale
}
