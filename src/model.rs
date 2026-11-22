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
    /// OLMoE: experts, with RMSNorm over the whole query and key projections.
    Olmoe,
    /// Kimi K3.1: Kimi delta attention (a linear-attention recurrence) in most layers, latent
    /// attention without rotation in the rest, residuals that attend across earlier blocks,
    /// and experts that work in a smaller latent space.
    KimiK31,
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
    /// The `rope` part is rotated (false: no position encoding, Kimi K3).
    pub rotate: bool,
}

/// Kimi delta attention: per head, a head_dim x head_dim state that decays per key channel and
/// is corrected toward each new value (the gated delta rule), after a short causal convolution.
#[derive(Clone, Debug, Serialize)]
pub struct Kda {
    pub n_head: usize,
    pub head_dim: usize,
    /// Convolution width, including the current token.
    pub conv: usize,
    /// Decay gates are `lower_bound * sigmoid(...)`, so each step keeps at least exp(lower_bound).
    pub gate_lower_bound: f32,
    /// Which layers use it; the others use latent attention.
    pub layers: Vec<bool>,
}

impl Kda {
    pub fn width(&self) -> usize {
        self.n_head * self.head_dim
    }
}

/// The lightning indexer of DeepSeek sparse attention: it scores every cached position and
/// latent attention reads only the best `top_k`.
#[derive(Clone, Debug, Serialize)]
pub struct Indexer {
    pub n_head: usize,
    pub head_dim: usize,
    pub top_k: usize,
    /// Layers that score their own selection; other latent-attention layers reuse the last one.
    pub full: Vec<bool>,
    pub eps: f32,
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
    /// Multimodal rotary positions (qwen2vl): rotary pairs per section (time, height, width),
    /// each section rotated by its own position. None for text-only models.
    pub rope_sections: Option<[usize; 3]>,
    /// Linear position scaling factor (1 = none).
    pub rope_linear_scale: f32,
    pub has_rope_freqs: bool,
    pub tied_embeddings: bool,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    /// The query and key norms cover the whole projection rather than each head (OLMoE).
    pub qk_norm_full: bool,
    pub moe: Option<Moe>,
    pub mla: Option<Mla>,
    pub yarn: Option<Yarn>,
    /// Multiplier on rotary cos/sin (YaRN magnitude scaling; 1 = none).
    pub rope_mscale: f32,
    /// Scale of query-key dot products.
    pub kq_scale: f32,
    pub kda: Option<Kda>,
    /// Layers per block of residuals that attend across blocks (0: a plain residual stream).
    pub attn_res_block: usize,
    /// Kimi K3's situ gate (beta, linear beta) in place of SiLU.
    pub situ: Option<(f32, f32)>,
    /// Width of the space routed experts work in (0: they read the hidden state).
    pub expert_latent: usize,
    /// A sigmoid gate on the attention output before its projection.
    pub attn_gate: bool,
    pub indexer: Option<Indexer>,
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
    /// True when layer `l` is a linear-attention (KDA) layer.
    pub fn is_kda(&self, l: usize) -> bool {
        self.kda.as_ref().is_some_and(|k| k.layers[l])
    }
    /// Width of what routed experts read and write.
    pub fn expert_in(&self) -> usize {
        if self.expert_latent > 0 { self.expert_latent } else { self.n_embd }
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
        // Qwen2-VL and Qwen2.5-VL (and their fine-tunes such as UI-TARS and Holo): Qwen2 with
        // multimodal rotary positions, which equal plain ones for text.
        "qwen2" | "qwen2vl" => Arch::Qwen2,
        "qwen3" => Arch::Qwen3,
        "qwen3moe" => Arch::Qwen3Moe,
        "deepseek2" => Arch::Deepseek2,
        "olmoe" => Arch::Olmoe,
        // Kimi K3 is the same model without the sparse-attention indexer.
        "kimi-k31" | "kimi-k3" => Arch::KimiK31,
        other => {
            return Err(format!(
                "the {other} architecture is not supported yet (supported: llama, mistral, qwen2, qwen3, qwen3moe, deepseek2, olmoe, kimi-k3, kimi-k31)"
            ))
        }
    })
}

/// Rotary pairs per multimodal section (time, height, width), from `ARCH.rope.dimension_sections`.
pub fn rope_sections(g: &Gguf, prefix: &str) -> Option<[usize; 3]> {
    let a = g.get(&format!("{prefix}.rope.dimension_sections"))?.as_array()?;
    let n: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
    (n.len() >= 3 && n[..3].iter().any(|&x| x > 0)).then(|| [n[0], n[1], n[2]])
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
    // Kimi K3.1 lists it per layer: 0 marks a linear-attention layer.
    let kv_per_layer: Option<Vec<u64>> = g
        .get(&key("attention.head_count_kv"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect());
    let n_head_kv = g
        .u64(&key("attention.head_count_kv"))
        .map(|v| v as usize)
        .or_else(|| kv_per_layer.as_ref().and_then(|a| a.iter().copied().find(|&v| v > 0)).map(|v| v as usize))
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
                // Llama (Mixtral) and Qwen3 always renormalise, OLMoE never; DeepSeek says in its metadata.
                norm: match arch {
                    Arch::Deepseek2 | Arch::KimiK31 => g.bool(&key("expert_weights_norm")).unwrap_or(false),
                    Arch::Olmoe => false,
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
    let mla = if matches!(arch, Arch::Deepseek2 | Arch::KimiK31) {
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
            // Kimi K3 latent attention has no position encoding (the recurrent layers carry
            // order); its files still describe a 64-wide unrotated key part.
            rotate: arch != Arch::KimiK31,
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
    if arch != Arch::KimiK31 && g.tensor(first_attn).is_none() && g.tensor("blk.0.attn_qkv.weight").is_some() {
        return Err("fused attn_qkv tensors are not supported yet".into());
    }
    if g.tensor(&format!("blk.{}.ffn_gate_up_exps.weight", moe.as_ref().map_or(0, |m| m.dense_lead))).is_some() {
        return Err("fused ffn_gate_up_exps tensors are not supported yet".into());
    }
    let n_layer = need("block_count")?;
    let kda = match (arch, &kv_per_layer) {
        (Arch::KimiK31, Some(per)) => {
            if per.len() != n_layer {
                return Err("attention.head_count_kv does not list every layer".into());
            }
            Some(Kda {
                n_head,
                head_dim: need("kda.head_dim")?,
                conv: need("ssm.conv_kernel")?,
                gate_lower_bound: g.f64(&key("kda.gate_lower_bound")).unwrap_or(f64::NEG_INFINITY) as f32,
                layers: per.iter().map(|&v| v == 0).collect(),
            })
        }
        (Arch::KimiK31, None) => return Err("Kimi K3 files list attention.head_count_kv per layer".into()),
        _ => None,
    };
    if kda.as_ref().is_some_and(|k| !k.gate_lower_bound.is_finite()) {
        return Err("Kimi Linear's softplus decay gate is not supported yet (only Kimi K3's bounded gate)".into());
    }
    let first_mla = (0..n_layer).find(|&l| !kda.as_ref().is_some_and(|k| k.layers[l])).unwrap_or(0);
    let indexer = match opt("attention.indexer.top_k") {
        Some(top_k) if top_k > 0 => {
            let full: Vec<bool> = match g.get(&key("attention.indexer.types")).and_then(|v| v.as_array()) {
                Some(a) => a.iter().map(|v| v.as_bool().unwrap_or(true)).collect(),
                None => vec![true; n_layer],
            };
            Some(Indexer {
                n_head: need("attention.indexer.head_count")?,
                head_dim: need("attention.indexer.key_length")?,
                top_k,
                full,
                eps: g.f64(&key("attention.layer_norm_epsilon")).unwrap_or(1e-5) as f32,
            })
        }
        _ => None,
    };
    let situ = g
        .f64(&key("activation.situ_beta"))
        .map(|b| (b as f32, g.f64(&key("activation.situ_linear_beta")).unwrap_or(0.0) as f32));
    let cfg = Config {
        arch,
        name: g.str("general.name").unwrap_or("model").to_string(),
        n_vocab,
        n_embd,
        n_layer,
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
        rope_sections: rope_sections(g, &prefix),
        rope_linear_scale,
        has_rope_freqs: g.tensor("rope_freqs.weight").is_some(),
        tied_embeddings: g.tensor("output.weight").is_none(),
        qkv_bias: g.tensor("blk.0.attn_q.bias").is_some(),
        qk_norm: g.tensor("blk.0.attn_q_norm.weight").is_some(),
        qk_norm_full: arch == Arch::Olmoe,
        moe,
        mla,
        yarn,
        rope_mscale,
        kq_scale,
        kda,
        attn_res_block: opt("attn_res.block_size").unwrap_or(0),
        situ,
        expert_latent: opt("expert_latent_length").unwrap_or(0),
        attn_gate: arch == Arch::KimiK31 && g.tensor(&format!("blk.{first_mla}.attn_gate.weight")).is_some(),
        indexer,
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
    if let Some(k) = &c.kda {
        if k.head_dim != 128 || k.conv != 4 {
            return Err(format!("unsupported linear-attention sizes: head {} and convolution {}", k.head_dim, k.conv));
        }
    }
    if c.attn_res_block > 0 && c.n_layer.div_ceil(c.attn_res_block) > 16 {
        return Err("more than 16 residual blocks are not supported".into());
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
            _ if c.is_kda(i) => {
                let k = c.kda.as_ref().unwrap();
                let d = k.width();
                m.push((b("attn_q"), d, c.n_embd, 1));
                m.push((b("attn_k"), d, c.n_embd, 1));
                m.push((b("attn_v"), d, c.n_embd, 1));
                m.push((b("ssm_f_a"), k.head_dim, c.n_embd, 1));
                m.push((b("ssm_f_b"), d, k.head_dim, 1));
                m.push((b("ssm_beta"), k.n_head, c.n_embd, 1));
                m.push((b("ssm_g"), d, c.n_embd, 1));
                m.push((b("attn_output"), c.n_embd, d, 1));
            }
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
                if c.attn_gate {
                    m.push((b("attn_gate"), c.n_head * a.v_dim, c.n_embd, 1));
                }
                if let Some(ix) = c.indexer.as_ref().filter(|ix| ix.full[i]) {
                    m.push((b("indexer.attn_q_b"), ix.n_head * ix.head_dim, a.q_lora, 1));
                    m.push((b("indexer.attn_k"), ix.head_dim, c.n_embd, 1));
                    m.push((b("indexer.proj"), ix.n_head, c.n_embd, 1));
                }
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
                let w = c.expert_in();
                m.push((b("ffn_gate_exps"), e.n_ff, w, e.n_expert));
                m.push((b("ffn_up_exps"), e.n_ff, w, e.n_expert));
                m.push((b("ffn_down_exps"), w, e.n_ff, e.n_expert));
                if c.expert_latent > 0 {
                    m.push((b("ffn_routed_down"), w, c.n_embd, 1));
                    m.push((b("ffn_routed_up"), c.n_embd, w, 1));
                }
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
    if c.attn_res_block > 0 {
        v.push(("output_res_score.weight".into(), c.n_embd));
    }
    for i in 0..c.n_layer {
        v.push((format!("blk.{i}.attn_norm.weight"), c.n_embd));
        v.push((format!("blk.{i}.ffn_norm.weight"), c.n_embd));
        if c.attn_res_block > 0 {
            v.push((format!("blk.{i}.attn_res_score.weight"), c.n_embd));
            v.push((format!("blk.{i}.ffn_res_score.weight"), c.n_embd));
        }
        if let Some(k) = c.kda.as_ref().filter(|_| c.is_kda(i)) {
            for x in ["q", "k", "v"] {
                v.push((format!("blk.{i}.ssm_conv1d_{x}.weight"), k.width() * k.conv));
            }
            v.push((format!("blk.{i}.ssm_dt.bias"), k.width()));
            v.push((format!("blk.{i}.ssm_a"), k.n_head));
            v.push((format!("blk.{i}.ssm_norm.weight"), k.head_dim));
            if let Some(e) = c.moe.as_ref().filter(|e| e.probs_bias && c.is_moe(i)) {
                v.push((format!("blk.{i}.exp_probs_b.bias"), e.n_expert));
            }
            if c.expert_latent > 0 && c.is_moe(i) {
                v.push((format!("blk.{i}.ffn_routed_norm.weight"), c.expert_latent));
            }
            continue;
        }
        if c.qkv_bias {
            v.push((format!("blk.{i}.attn_q.bias"), c.q_dim()));
            v.push((format!("blk.{i}.attn_k.bias"), c.kv_dim()));
            v.push((format!("blk.{i}.attn_v.bias"), c.kv_dim()));
        }
        if c.qk_norm {
            let (q, k) = if c.qk_norm_full { (c.q_dim(), c.kv_dim()) } else { (c.head_dim, c.head_dim) };
            v.push((format!("blk.{i}.attn_q_norm.weight"), q));
            v.push((format!("blk.{i}.attn_k_norm.weight"), k));
        }
        if let Some(a) = &c.mla {
            if a.q_lora > 0 {
                v.push((format!("blk.{i}.attn_q_a_norm.weight"), a.q_lora));
            }
            v.push((format!("blk.{i}.attn_kv_a_norm.weight"), a.kv_lora));
        }
        if let Some(ix) = c.indexer.as_ref().filter(|ix| ix.full[i]) {
            v.push((format!("blk.{i}.indexer.k_norm.weight"), ix.head_dim));
            v.push((format!("blk.{i}.indexer.k_norm.bias"), ix.head_dim));
        }
        if let Some(e) = c.moe.as_ref().filter(|e| e.probs_bias && c.is_moe(i)) {
            v.push((format!("blk.{i}.exp_probs_b.bias"), e.n_expert));
        }
        if c.expert_latent > 0 && c.is_moe(i) {
            v.push((format!("blk.{i}.ffn_routed_norm.weight"), c.expert_latent));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF header with string, u32 and u32-array keys and F32 tensor infos (no data needed).
    fn header(keys: &[(&str, &dyn Fn(&mut Vec<u8>))], tensors: &[(&str, &[u64])]) -> Vec<u8> {
        let s = |b: &mut Vec<u8>, t: &str| {
            b.extend((t.len() as u64).to_le_bytes());
            b.extend(t.as_bytes());
        };
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((keys.len() as u64).to_le_bytes());
        for (k, v) in keys {
            s(&mut b, k);
            v(&mut b);
        }
        for (name, dims) in tensors {
            s(&mut b, name);
            b.extend((dims.len() as u32).to_le_bytes());
            for d in *dims {
                b.extend(d.to_le_bytes());
            }
            b.extend(0u32.to_le_bytes());
            b.extend(0u64.to_le_bytes());
        }
        b
    }

    #[test]
    fn qwen2vl_loads_as_qwen2_with_rotary_sections() {
        let text = |v: &'static str| move |b: &mut Vec<u8>| {
            b.extend(8u32.to_le_bytes());
            b.extend((v.len() as u64).to_le_bytes());
            b.extend(v.as_bytes());
        };
        let int = |v: u32| move |b: &mut Vec<u8>| {
            b.extend(4u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        };
        let sections = |b: &mut Vec<u8>| {
            b.extend(9u32.to_le_bytes());
            b.extend(5u32.to_le_bytes());
            b.extend(4u64.to_le_bytes());
            for x in [16i32, 24, 24, 0] {
                b.extend(x.to_le_bytes());
            }
        };
        let (arch, n_embd, blocks, heads, kv, ff) = (text("qwen2vl"), int(256), int(2), int(4), int(2), int(512));
        let bytes = header(
            &[("general.architecture", &arch), ("qwen2vl.embedding_length", &n_embd), ("qwen2vl.block_count", &blocks), ("qwen2vl.attention.head_count", &heads),
              ("qwen2vl.attention.head_count_kv", &kv), ("qwen2vl.feed_forward_length", &ff), ("qwen2vl.rope.dimension_sections", &sections)],
            &[("token_embd.weight", &[256, 1000]), ("blk.0.attn_q.weight", &[256, 256]), ("blk.0.attn_k.weight", &[256, 128]), ("blk.0.attn_q.bias", &[256])],
        );
        let g = crate::gguf::parse(&bytes).unwrap();
        assert_eq!(arch_of(&g).unwrap(), Arch::Qwen2);
        assert_eq!(rope_sections(&g, "qwen2vl"), Some([16, 24, 24]));
        assert_eq!(rope_sections(&g, "qwen2"), None);
    }
}
