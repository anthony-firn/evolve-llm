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
    /// Qwen3 mixture-of-experts models (30B-A3B, 235B-A22B, Coder 480B-A35B).
    Qwen3Moe,
    /// DeepSeek V2, V3 and R1, Kimi K2 and Moonlight: multi-head latent attention and experts.
    Deepseek2,
}

/// How router scores become expert probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Gating {
    Softmax,
    Sigmoid,
}

/// Mixture-of-experts feed-forward layers.
#[derive(Clone, Debug, Serialize)]
pub struct Moe {
    pub n_expert: usize,
    /// Experts each token uses.
    pub n_used: usize,
    /// Hidden width of one expert.
    pub n_ff: usize,
    /// Hidden width of the always-on shared experts (0 when there are none).
    pub n_ff_shared: usize,
    pub gating: Gating,
    /// Renormalise the chosen experts' weights to sum to one.
    pub norm: bool,
    /// Factor on the chosen weights (1 = none).
    pub scale: f32,
    /// A learned per-expert bias that only affects which experts are chosen (DeepSeek V3).
    pub probs_bias: bool,
    /// Experts are chosen within the best `n_group_used` of `n_group` groups.
    pub n_group: usize,
    pub n_group_used: usize,
    /// Leading layers that use a dense feed-forward block instead.
    pub dense_lead: usize,
}

/// Multi-head latent attention: keys and values are cached as one compressed vector per token.
#[derive(Clone, Debug, Serialize)]
pub struct Mla {
    /// Rank of the query compression (0 = queries are projected directly).
    pub q_lora: usize,
    /// Size of the cached latent vector.
    pub kv_lora: usize,
    /// Per-head query/key dimensions without rotation.
    pub nope: usize,
    /// Rotated dimensions, shared by all heads on the key side.
    pub rope: usize,
    pub v_dim: usize,
}

/// YaRN context extension: rotation frequencies blend between interpolated and original.
#[derive(Clone, Debug, Serialize)]
pub struct Yarn {
    pub factor: f32,
    pub orig_ctx: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub ext_factor: f32,
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
    pub moe: Option<Moe>,
    pub mla: Option<Mla>,
    pub yarn: Option<Yarn>,
    /// Multiplier on rotary cos/sin (YaRN magnitude scaling; 1 = none).
    pub rope_mscale: f32,
    /// Scale of query-key dot products.
    pub kq_scale: f32,
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    /// True when layer `l` uses experts.
    pub fn is_moe(&self, l: usize) -> bool {
        self.moe.as_ref().is_some_and(|m| l >= m.dense_lead)
    }
    /// Values cached per token and layer: f16 keys and values, or the MLA latent and rotated key.
    pub fn cache_dim(&self) -> usize {
        match &self.mla {
            Some(m) => m.kv_lora + m.rope,
            None => 2 * self.kv_dim(),
        }
    }
    /// Total and per-token active parameters of the weight matrices.
    pub fn parameters(&self) -> (u64, u64) {
        let (mut total, mut active) = (0u64, 0u64);
        for (name, rows, k, experts) in matrices(self) {
            let n = (rows * k * experts) as u64;
            total += n;
            active += if experts > 1 && name.contains("_exps") {
                n / experts as u64 * self.moe.as_ref().map_or(1, |m| m.n_used) as u64
            } else {
                n
            };
        }
        (total, active)
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
        "qwen3moe" => Arch::Qwen3Moe,
        "deepseek2" => Arch::Deepseek2,
        other => {
            return Err(format!(
                "the {other} architecture is not supported yet (supported: llama, mistral, qwen2, qwen3, qwen3moe, deepseek2)"
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
    let factor = g.f64(&key("rope.scaling.factor")).unwrap_or(1.0) as f32;
    let rope_linear_scale = match scaling {
        "linear" => factor,
        _ => 1.0,
    };
    let opt = |k: &str| g.u64(&key(k)).map(|v| v as usize);
    let moe = match opt("expert_count") {
        Some(n_expert) if n_expert > 1 => {
            let n_ff = opt("expert_feed_forward_length").unwrap_or(need("feed_forward_length")?);
            let shared = g
                .tensor("blk.0.ffn_gate_shexp.weight")
                .or_else(|| {
                    let lead = opt("leading_dense_block_count").unwrap_or(0);
                    g.tensor(&format!("blk.{lead}.ffn_gate_shexp.weight"))
                })
                .map(|t| t.dims[1] as usize)
                .unwrap_or(0);
            let gating = match (arch, opt("expert_gating_func")) {
                (_, Some(2)) => Gating::Sigmoid,
                _ => Gating::Softmax,
            };
            let lead = opt("leading_dense_block_count").unwrap_or(0);
            Some(Moe {
                n_expert,
                n_used: need("expert_used_count")?,
                n_ff,
                n_ff_shared: shared,
                gating,
                // Llama (Mixtral) and Qwen3 always renormalise; DeepSeek says so in its metadata.
                norm: match arch {
                    Arch::Deepseek2 => g.bool(&key("expert_weights_norm")).unwrap_or(false),
                    _ => true,
                },
                scale: g.f64(&key("expert_weights_scale")).filter(|&v| v != 0.0).unwrap_or(1.0) as f32,
                probs_bias: g.tensor(&format!("blk.{lead}.exp_probs_b.bias")).is_some(),
                n_group: opt("expert_group_count").unwrap_or(1).max(1),
                n_group_used: opt("expert_group_used_count").unwrap_or(1).max(1),
                dense_lead: lead,
            })
        }
        _ => None,
    };
    let mla = if arch == Arch::Deepseek2 {
        let kv_lora = need("attention.kv_lora_rank")?;
        let k_mla = opt("attention.key_length_mla").ok_or(
            "this DeepSeek file predates split attn_k_b/attn_v_b tensors; use a GGUF converted by llama.cpp from mid-2025 or later",
        )?;
        let rope = need("rope.dimension_count")?;
        Some(Mla {
            q_lora: opt("attention.q_lora_rank").unwrap_or(0),
            kv_lora,
            nope: k_mla - rope,
            rope,
            v_dim: need("attention.value_length_mla")?,
        })
    } else {
        None
    };
    let yarn = (scaling == "yarn").then(|| Yarn {
        factor,
        orig_ctx: opt("rope.scaling.original_context_length").unwrap_or(4096),
        beta_fast: g.f64(&key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0) as f32,
        beta_slow: g.f64(&key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0) as f32,
        ext_factor: g.f64(&key("rope.scaling.yarn_ext_factor")).filter(|&v| v >= 0.0).unwrap_or(1.0) as f32,
    });
    // MLA heads are nope + rope wide; the file describes them as one latent head.
    let head_dim = match &mla {
        Some(m) => m.nope + m.rope,
        None => head_dim,
    };
    let n_head_kv = if mla.is_some() { 1 } else { n_head_kv };
    // YaRN: DeepSeek folds its magnitude correction into the attention scale (as llama.cpp does);
    // other models scale cos and sin by 0.1 ln(factor) + 1.
    let (rope_mscale, kq_scale) = match (&yarn, arch) {
        (Some(y), Arch::Deepseek2) => {
            let log_mul = g.f64(&key("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0) as f32;
            let m = 1.0 + log_mul * y.factor.ln();
            (1.0, m * m / (head_dim as f32).sqrt())
        }
        (Some(y), _) => (0.1 * y.factor.ln() + 1.0, 1.0 / (head_dim as f32).sqrt()),
        (None, _) => (1.0, 1.0 / (head_dim as f32).sqrt()),
    };
    let first_attn = if arch == Arch::Deepseek2 { "blk.0.attn_kv_a_mqa.weight" } else { "blk.0.attn_q.weight" };
    if g.tensor(first_attn).is_none() && g.tensor("blk.0.attn_qkv.weight").is_some() {
        return Err("fused attn_qkv tensors are not supported yet".into());
    }
    if g.tensor(&format!("blk.{}.ffn_gate_up_exps.weight", moe.as_ref().map_or(0, |m| m.dense_lead))).is_some() {
        return Err("fused ffn_gate_up_exps tensors are not supported yet".into());
    }
    let cfg = Config {
        arch,
        name: g.str("general.name").unwrap_or("model").to_string(),
        n_vocab,
        n_embd,
        n_layer: need("block_count")?,
        n_head,
        n_head_kv,
        head_dim,
        n_ff: opt("feed_forward_length").unwrap_or(0),
        ctx_train: g.u64(&key("context_length")).unwrap_or(2048) as usize,
        rms_eps: g
            .f64(&key("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5) as f32,
        rope_base: g.f64(&key("rope.freq_base")).unwrap_or(10000.0) as f32,
        rope_dim: g
            .u64(&key("rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim),
        rope_neox: !matches!(arch, Arch::Llama | Arch::Deepseek2),
        rope_linear_scale,
        has_rope_freqs: g.tensor("rope_freqs.weight").is_some(),
        tied_embeddings: g.tensor("output.weight").is_none(),
        qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
        qk_norm: g.tensor("blk.0.attn_q_norm.weight").is_some(),
        moe,
        mla,
        yarn,
        rope_mscale,
        kq_scale,
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
    if c.head_dim % 4 != 0 || (c.mla.is_none() && c.head_dim > 256) || c.rope_dim > c.head_dim || c.rope_dim % 2 != 0 {
        return Err(format!("unsupported head size {}", c.head_dim));
    }
    if let Some(m) = &c.mla {
        if m.kv_lora + m.rope > 1024 || m.kv_lora % 64 != 0 || m.v_dim % 64 != 0 || m.nope % 8 != 0 {
            return Err(format!("unsupported latent attention sizes {m:?}"));
        }
    }
    if let Some(m) = &c.moe {
        if m.n_expert > 1024 || m.n_used > 32 || m.n_used > m.n_expert || m.n_expert % m.n_group != 0 {
            return Err(format!("unsupported expert layout {} / {}", m.n_used, m.n_expert));
        }
    }
    if scaling_is_unsupported(g) {
        return Err("this model needs LongRoPE scaling, which is not supported yet".into());
    }
    for (name, rows, cols, experts) in matrices(c) {
        let t = g
            .tensor(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let want: Vec<u64> = if experts > 1 {
            vec![cols as u64, rows as u64, experts as u64]
        } else {
            vec![cols as u64, rows as u64]
        };
        if t.dims != want {
            return Err(format!(
                "tensor {name} has shape {:?}, expected {want:?}",
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
        Some("longrope")
    )
}

/// Every weight matrix as (name, rows = outputs, cols = inputs, experts). Expert tensors and
/// MLA's per-head matrices stack `experts` matrices of `rows` x `cols`.
pub fn matrices(c: &Config) -> Vec<(String, usize, usize, usize)> {
    let mut m = vec![("token_embd.weight".to_string(), c.n_vocab, c.n_embd, 1)];
    if !c.tied_embeddings {
        m.push(("output.weight".into(), c.n_vocab, c.n_embd, 1));
    }
    for i in 0..c.n_layer {
        let b = |s: &str| format!("blk.{i}.{s}.weight");
        match &c.mla {
            Some(a) => {
                if a.q_lora > 0 {
                    m.push((b("attn_q_a"), a.q_lora, c.n_embd, 1));
                    m.push((b("attn_q_b"), c.q_dim(), a.q_lora, 1));
                } else {
                    m.push((b("attn_q"), c.q_dim(), c.n_embd, 1));
                }
                m.push((b("attn_kv_a_mqa"), a.kv_lora + a.rope, c.n_embd, 1));
                m.push((b("attn_k_b"), a.kv_lora, a.nope, c.n_head));
                m.push((b("attn_v_b"), a.v_dim, a.kv_lora, c.n_head));
                m.push((b("attn_output"), c.n_embd, c.n_head * a.v_dim, 1));
            }
            None => {
                m.push((b("attn_q"), c.q_dim(), c.n_embd, 1));
                m.push((b("attn_k"), c.kv_dim(), c.n_embd, 1));
                m.push((b("attn_v"), c.kv_dim(), c.n_embd, 1));
                m.push((b("attn_output"), c.n_embd, c.q_dim(), 1));
            }
        }
        match c.moe.as_ref().filter(|_| c.is_moe(i)) {
            Some(e) => {
                m.push((b("ffn_gate_inp"), e.n_expert, c.n_embd, 1));
                m.push((b("ffn_gate_exps"), e.n_ff, c.n_embd, e.n_expert));
                m.push((b("ffn_up_exps"), e.n_ff, c.n_embd, e.n_expert));
                m.push((b("ffn_down_exps"), c.n_embd, e.n_ff, e.n_expert));
                if e.n_ff_shared > 0 {
                    m.push((b("ffn_gate_shexp"), e.n_ff_shared, c.n_embd, 1));
                    m.push((b("ffn_up_shexp"), e.n_ff_shared, c.n_embd, 1));
                    m.push((b("ffn_down_shexp"), c.n_embd, e.n_ff_shared, 1));
                }
            }
            None => {
                m.push((b("ffn_gate"), c.n_ff, c.n_embd, 1));
                m.push((b("ffn_up"), c.n_ff, c.n_embd, 1));
                m.push((b("ffn_down"), c.n_embd, c.n_ff, 1));
            }
        }
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
        if let Some(a) = &c.mla {
            if a.q_lora > 0 {
                v.push((format!("blk.{i}.attn_q_a_norm.weight"), a.q_lora));
            }
            v.push((format!("blk.{i}.attn_kv_a_norm.weight"), a.kv_lora));
        }
        if let Some(e) = c.moe.as_ref().filter(|e| e.probs_bias && c.is_moe(i)) {
            v.push((format!("blk.{i}.exp_probs_b.bias"), e.n_expert));
        }
    }
    v
}

/// YaRN's blend between original (1) and interpolated (0) frequency for pair `i`.
fn yarn_ramp(c: &Config, y: &Yarn, i: usize) -> f32 {
    let dims = c.rope_dim as f32;
    let corr = |rot: f32| dims * (y.orig_ctx as f32 / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * c.rope_base.ln());
    let low = corr(y.beta_fast).floor().max(0.0);
    let high = corr(y.beta_slow).ceil().min(dims - 1.0);
    let t = ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
    (1.0 - t) * y.ext_factor
}

/// Rotation frequency for pair `i` (of `rope_dim / 2`), before position scaling.
pub fn rope_inv_freq(c: &Config, i: usize, factors: Option<&[f32]>) -> f32 {
    let base = c.rope_base.powf(-(2.0 * i as f32) / c.rope_dim as f32);
    let f = factors.map(|f| f[i]).unwrap_or(1.0);
    if let Some(y) = &c.yarn {
        let mix = yarn_ramp(c, y, i);
        return (base / y.factor * (1.0 - mix) + base * mix) / f;
    }
    base / f / c.rope_linear_scale
}
