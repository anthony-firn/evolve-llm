//! The GPU forward pass: weight storage, activation buffers and the dispatch plan.

use super::{read_back, shaders, Gpu};
use crate::gguf::Gguf;
use crate::model::{self, Config, Gating};
use crate::quant::{self, GpuFormat};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Tokens processed per submission while reading a prompt.
pub const MAX_BATCH: usize = 128;

/// Marks an expert that is not resident in the expert-to-slot map.
pub const ABSENT: u32 = u32::MAX;

struct Shard {
    row0: usize,
    rows: usize,
    codes: wgpu::Buffer,
    /// Per-group f32 scales of an expanded layout.
    scales: Option<wgpu::Buffer>,
}

struct Matrix {
    fmt: GpuFormat,
    ggml_type: u32,
    k: usize,
    /// Rows of one expert (or one head group); the matrix stacks `experts` of them.
    rows: usize,
    experts: usize,
    shards: Vec<Shard>,
}

impl Matrix {
    fn total_rows(&self) -> usize {
        self.rows * self.experts
    }
}

/// How weights are laid out on the GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Lower cap on buffer size (tests use it to force row shards).
    pub binding_limit: Option<u64>,
    /// Expanded layouts for speed, or file-sized blocks when memory is tight.
    pub layout: quant::Layout,
}

/// Where a tensor lives in the file and how to upload it in row chunks.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TensorPlan {
    pub name: String,
    pub offset: u64,
    pub bytes: u64,
    pub row_bytes: u64,
    pub rows: u64,
    pub ggml_type: u32,
}

#[derive(Clone, Copy)]
enum Grid {
    /// One workgroup per row: n_tokens * rows_per_token.
    Rows(u32),
    /// Exactly one workgroup.
    One,
    /// Matrix product over `n` output rows; `gemv` forces the single-token kernel.
    Matmul { n: u32, gemv: bool },
    /// Embedding lookup of width `k`.
    Embed { k: u32 },
    /// (heads, n_tokens).
    Heads(u32),
    /// Expert projections: (row blocks of `rows`, pairs = n_tokens * `used`).
    Pairs { rows: u32, used: u32 },
    /// (row blocks of `rows`, n_tokens).
    TokenRows { rows: u32 },
    /// One workgroup per token.
    Tokens,
}

#[derive(Clone)]
enum Kernel {
    Matmul(GpuFormat),
    MatmulId(GpuFormat),
    MatmulIdDown(GpuFormat),
    Embed(GpuFormat),
    Norm,
    NormInPlace,
    Rope,
    Attention,
    TopK,
    MlaStore,
    MlaAttention,
}

struct Op {
    kernel: Kernel,
    grid: Grid,
    bind: wgpu::BindGroup,
}

struct Layouts {
    matmul: wgpu::BindGroupLayout,
    matmul_id: wgpu::BindGroupLayout,
    norm: wgpu::BindGroupLayout,
    norm_inplace: wgpu::BindGroupLayout,
    rope: wgpu::BindGroupLayout,
    attention: wgpu::BindGroupLayout,
    topk: wgpu::BindGroupLayout,
    mla: wgpu::BindGroupLayout,
}

struct State {
    ctx: usize,
    step: wgpu::Buffer,
    tokens: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    ops: Vec<Op>,
    /// Index of the first op that only matters for the last token (final norm + head).
    head_start: usize,
    bytes: u64,
}

pub struct Engine {
    pub gpu: Gpu,
    pub cfg: Config,
    mats: HashMap<String, Matrix>,
    vecs: HashMap<String, wgpu::Buffer>,
    rope_factors: Option<Vec<f32>>,
    plan: Vec<TensorPlan>,
    layouts: Layouts,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    dummy: wgpu::Buffer,
    /// Code books of the IQ and MXFP4 block formats.
    tables: wgpu::Buffer,
    state: Option<State>,
    pub weight_bytes: u64,
}

fn storage(ro: bool, binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Layout from a spec: `u` uniform, `r` read-only storage, `w` read-write storage.
fn layout(device: &wgpu::Device, spec: &str) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = spec
        .chars()
        .enumerate()
        .map(|(i, c)| match c {
            'u' => uniform(i as u32),
            'r' => storage(true, i as u32),
            _ => storage(false, i as u32),
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(spec),
        entries: &entries,
    })
}

/// Lanes cooperating on one output row in the single-token kernel.
fn gemv_lanes() -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(v) = std::env::var("EVOLVE_LLM_GEMV_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return v;
    }
    4
}

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// How a matrix reads its input: row stride per token, and per-group input slices.
#[derive(Clone, Copy)]
struct Input {
    stride: usize,
    group_rows: usize,
    group_stride: usize,
}

impl Input {
    fn rows(stride: usize) -> Input {
        Input {
            stride,
            group_rows: 0,
            group_stride: 0,
        }
    }
}

/// Query heads per MLA attention workgroup: up to four, within the shared query buffer.
fn mla_heads(c: &Config) -> usize {
    let m = c.mla.as_ref().unwrap();
    let d = m.kv_lora + m.rope;
    [4, 2, 1]
        .into_iter()
        .find(|&h| c.n_head % h == 0 && h * d <= 2304)
        .unwrap_or(1)
}

impl Engine {
    /// Allocate weight buffers for a model. Weights are uploaded afterwards with `upload`.
    pub fn new(gpu: Gpu, g: &Gguf) -> Result<Engine, String> {
        Self::with_options(gpu, g, Options::default())
    }

    /// Like `new`, with an optional lower cap on buffer size (tests use it to force row shards).
    pub fn with_binding_limit(gpu: Gpu, g: &Gguf, limit: Option<u64>) -> Result<Engine, String> {
        Self::with_options(
            gpu,
            g,
            Options {
                binding_limit: limit,
                layout: quant::Layout::Fast,
            },
        )
    }

    pub fn with_options(gpu: Gpu, g: &Gguf, opts: Options) -> Result<Engine, String> {
        let cfg = model::config(g)?;
        if let Some(m) = &cfg.mla {
            if mla_heads(&cfg) * (m.kv_lora + m.rope) > 2304 {
                return Err("the latent attention vector is too large".into());
            }
        }
        let device = &gpu.device;
        let max_binding = gpu
            .info
            .max_storage_binding
            .min(gpu.info.max_buffer_size)
            .min(opts.binding_limit.unwrap_or(u64::MAX));
        let mut mats = HashMap::new();
        let mut plan = Vec::new();
        let mut weight_bytes = 0u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        for (name, rows, k, experts) in model::matrices(&cfg) {
            let t = g.tensor(&name).unwrap();
            let fmt = GpuFormat::choose(t.ggml_type, opts.layout).unwrap();
            let total = rows * experts;
            let rb = quant::row_bytes(t.ggml_type, k)? as u64;
            plan.push(TensorPlan {
                name: name.clone(),
                offset: g.data_offset + t.offset,
                bytes: rb * total as u64,
                row_bytes: rb,
                rows: total as u64,
                ggml_type: t.ggml_type,
            });
            let (codes_row, scales_row) = fmt.sizes(k, 1);
            let fit = ((max_binding / codes_row.max(scales_row).max(1)) as usize)
                .min(total)
                .max(1);
            // Shards hold whole experts (or head groups); otherwise multiples of 64 rows, so
            // matrix tiles never straddle them.
            let per_shard = if fit >= total {
                total
            } else if experts > 1 {
                if fit < rows {
                    return Err(format!("{name}: one expert does not fit in a GPU buffer"));
                }
                fit / rows * rows
            } else if fit >= 64 {
                fit / 64 * 64
            } else {
                fit
            };
            let mut shards = Vec::new();
            let mut row0 = 0;
            while row0 < total {
                let n = per_shard.min(total - row0);
                let (cb, sb) = fmt.sizes(k, n);
                let codes = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&name),
                    size: cb,
                    usage,
                    mapped_at_creation: false,
                });
                let scales = fmt.has_scales().then(|| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&name),
                        size: sb.max(16),
                        usage,
                        mapped_at_creation: false,
                    })
                });
                weight_bytes += cb + sb;
                shards.push(Shard {
                    row0,
                    rows: n,
                    codes,
                    scales,
                });
                row0 += n;
            }
            mats.insert(
                name,
                Matrix {
                    fmt,
                    ggml_type: t.ggml_type,
                    k,
                    rows,
                    experts,
                    shards,
                },
            );
        }
        let mut vecs = HashMap::new();
        for (name, len) in model::vectors(&cfg) {
            let t = g.tensor(&name).unwrap();
            let rb = quant::row_bytes(t.ggml_type, len)? as u64;
            plan.push(TensorPlan {
                name: name.clone(),
                offset: g.data_offset + t.offset,
                bytes: rb,
                row_bytes: rb,
                rows: 1,
                ggml_type: t.ggml_type,
            });
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&name),
                size: (len * 4) as u64,
                usage,
                mapped_at_creation: false,
            });
            weight_bytes += (len * 4) as u64;
            vecs.insert(name, buf);
        }
        plan.sort_by_key(|p| p.offset);
        let layouts = Layouts {
            matmul: layout(device, "uurrrwrr"),
            matmul_id: layout(device, "uurrrwrrr"),
            norm: layout(device, "uurrw"),
            norm_inplace: layout(device, "uuwr"),
            rope: layout(device, "uuwrrwwr"),
            attention: layout(device, "uurrrw"),
            topk: layout(device, "uurrw"),
            mla: layout(device, "uurrrw"),
        };
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage,
            mapped_at_creation: false,
        });
        let tables = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("code books"),
            contents: &words(&quant::Tables::words()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut e = Engine {
            gpu,
            cfg,
            mats,
            vecs,
            rope_factors: None,
            plan,
            layouts,
            pipelines: HashMap::new(),
            dummy,
            tables,
            state: None,
            weight_bytes,
        };
        e.compile()?;
        Ok(e)
    }

    /// Number of GPU buffer pairs holding weight matrices.
    pub fn shard_count(&self) -> usize {
        self.mats.values().map(|m| m.shards.len()).sum()
    }

    pub fn plan(&self) -> &[TensorPlan] {
        &self.plan
    }

    fn pipeline(&mut self, key: String, source: String, layout: &wgpu::BindGroupLayout) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let device = &self.gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&key),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&key),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&key),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipelines.insert(key, p);
    }

    fn compile(&mut self) -> Result<(), String> {
        let mut dense: Vec<GpuFormat> = Vec::new();
        let mut expert: Vec<GpuFormat> = Vec::new();
        for (name, m) in &self.mats {
            if name.contains("_exps") {
                expert.push(m.fmt);
            } else {
                dense.push(m.fmt);
            }
        }
        for v in [&mut dense, &mut expert] {
            v.sort_by_key(|f| f.key());
            v.dedup();
        }
        let (mm, mi) = (self.layouts.matmul.clone(), self.layouts.matmul_id.clone());
        let lanes = gemv_lanes();
        for f in dense {
            self.pipeline(format!("gemv-{}", f.key()), shaders::gemv(f, lanes), &mm);
            self.pipeline(format!("gemm-{}", f.key()), shaders::gemm(f), &mm);
        }
        for f in expert {
            self.pipeline(format!("gemv-id-{}", f.key()), shaders::gemv_id(f, lanes), &mi);
            self.pipeline(format!("gemv-id-down-{}", f.key()), shaders::gemv_id_down(f, lanes), &mi);
        }
        let ef = self.mats["token_embd.weight"].fmt;
        self.pipeline(format!("embed-{}", ef.key()), shaders::embed(ef), &mm);
        let l = &self.layouts;
        let (n, ni, r, a, tk, ml) = (
            l.norm.clone(),
            l.norm_inplace.clone(),
            l.rope.clone(),
            l.attention.clone(),
            l.topk.clone(),
            l.mla.clone(),
        );
        self.pipeline("norm".into(), shaders::rmsnorm(false), &n);
        self.pipeline("norm-inplace".into(), shaders::rmsnorm(true), &ni);
        self.pipeline("rope".into(), shaders::rope(), &r);
        self.pipeline("attention".into(), shaders::attention(), &a);
        if self.cfg.moe.is_some() {
            self.pipeline("topk".into(), shaders::topk(), &tk);
        }
        if self.cfg.mla.is_some() {
            self.pipeline("mla-store".into(), shaders::mla_store(), &ml);
            self.pipeline("mla-attention".into(), shaders::mla_attention(), &ml);
        }
        match self.gpu.take_error() {
            Some(e) => Err(format!("shader compilation failed: {e}")),
            None => Ok(()),
        }
    }

    /// Upload whole rows `row0..` of a tensor from its GGUF bytes.
    pub fn upload(&mut self, name: &str, row0: usize, bytes: &[u8]) -> Result<(), String> {
        let queue = &self.gpu.queue;
        if let Some(m) = self.mats.get(name) {
            let rb = quant::row_bytes(m.ggml_type, m.k)?;
            if bytes.len() % rb != 0 {
                return Err(format!("{name}: upload is not a whole number of rows"));
            }
            let rows = bytes.len() / rb;
            if row0 + rows > m.total_rows() {
                return Err(format!("{name}: rows out of range"));
            }
            let mut r = row0;
            while r < row0 + rows {
                let s = m
                    .shards
                    .iter()
                    .find(|s| r >= s.row0 && r < s.row0 + s.rows)
                    .unwrap();
                let n = (s.row0 + s.rows - r).min(row0 + rows - r);
                let src = &bytes[(r - row0) * rb..(r - row0 + n) * rb];
                if let Some(scales) = &s.scales {
                    let packed = quant::repack(m.ggml_type, src, m.k, n)?;
                    let at = m.fmt.planes(m.k, r - s.row0);
                    let whole = m.fmt.planes(m.k, s.rows);
                    queue.write_buffer(&s.codes, at.low, &words(&packed.codes));
                    if !packed.high.is_empty() {
                        queue.write_buffer(&s.codes, whole.low + at.high, &words(&packed.high));
                    }
                    queue.write_buffer(scales, at.scales, &floats(&packed.scales));
                } else {
                    let packed = quant::to_gpu(m.ggml_type, src, m.k, n)?;
                    queue.write_buffer(&s.codes, m.fmt.bytes(m.k, r - s.row0), &words(&packed));
                }
                r += n;
            }
            return Ok(());
        }
        if let Some(buf) = self.vecs.get(name) {
            let t = self.plan.iter().find(|p| p.name == name).unwrap();
            if bytes.len() as u64 != t.bytes {
                return Err(format!("{name}: expected {} bytes", t.bytes));
            }
            let len = (buf.size() / 4) as usize;
            let ty = t.ggml_type;
            let mut out = vec![0f32; len];
            quant::dequantize(ty, bytes, &mut out)?;
            queue.write_buffer(buf, 0, &floats(&out));
            if name == "rope_freqs.weight" {
                self.rope_factors = Some(out);
            }
            return Ok(());
        }
        Err(format!("unknown tensor {name}"))
    }

    /// GPU memory for activations and the KV cache at a given context length.
    pub fn state_bytes(cfg: &Config, ctx: usize) -> u64 {
        let t = MAX_BATCH as u64;
        let mut acts = 2 * cfg.n_embd + 2 * cfg.q_dim() + 2 * cfg.kv_dim() + cfg.n_ff;
        if let Some(e) = &cfg.moe {
            acts += e.n_expert + 2 * e.n_used + e.n_used * e.n_ff + e.n_ff_shared;
        }
        if let Some(m) = &cfg.mla {
            acts += m.q_lora + m.kv_lora + m.rope + 2 * cfg.n_head * m.kv_lora + cfg.n_head * m.v_dim;
        }
        let kv = cfg.n_layer as u64 * ctx as u64 * cfg.cache_dim() as u64 * 2;
        let rope = ctx as u64 * (cfg.rope_dim / 2) as u64 * 8;
        t * acts as u64 * 4 + kv + rope + cfg.n_vocab as u64 * 8
    }

    /// Allocate activations and the KV cache, and record the dispatch plan.
    pub fn finish(&mut self, ctx: usize) -> Result<(), String> {
        let c = self.cfg.clone();
        let ctx = ctx.clamp(16, c.ctx_train.max(16));
        let device = &self.gpu.device;
        let t = MAX_BATCH;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buf = |label: &str, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (floats.max(4) * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let moe = c.moe.clone();
        let mla = c.mla.clone();
        let v_width = mla.as_ref().map_or(c.q_dim(), |m| c.n_head * m.v_dim);
        let ff_width = c.n_ff.max(moe.as_ref().map_or(0, |e| e.n_ff_shared));
        let x = buf("x", t * c.n_embd);
        let xn = buf("xn", t * c.n_embd);
        let q = buf("q", t * c.q_dim());
        let k = buf("k", t * c.kv_dim());
        let v = buf("v", t * c.kv_dim());
        let att = buf("att", t * c.q_dim().max(v_width));
        let gate = buf("gate", t * ff_width);
        let xf = buf("xf", c.n_embd);
        let logits = buf("logits", c.n_vocab);
        let tokens = buf("tokens", t);
        let (kc, vc): (Vec<_>, Vec<_>) = (0..c.n_layer)
            .map(|_| match &mla {
                Some(m) => (buf("latent-cache", ctx * (m.kv_lora + m.rope) / 2), buf("unused", 4)),
                None => (
                    buf("k-cache", ctx * c.kv_dim() / 2),
                    buf("v-cache", ctx * c.kv_dim() / 2),
                ),
            })
            .unzip();
        // Experts: router scores, chosen experts and weights, per-pair hidden states, and the
        // expert-to-slot map (identity while every expert is resident).
        let (rlog, route, hexp, emap) = match &moe {
            Some(e) => {
                let ident: Vec<u32> = (0..c.n_layer)
                    .flat_map(|_| 0..e.n_expert as u32)
                    .collect();
                let emap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("expert map"),
                    contents: &words(&ident),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                (
                    buf("router", t * e.n_expert),
                    buf("route", 2 * t * e.n_used),
                    buf("expert hidden", t * e.n_used * e.n_ff),
                    emap,
                )
            }
            None => (buf("unused", 4), buf("unused", 4), buf("unused", 4), buf("unused", 4)),
        };
        let (qa, kva, qlat, olat) = match &mla {
            Some(m) => (
                buf("q-a", t * m.q_lora.max(1)),
                buf("kv-a", t * (m.kv_lora + m.rope)),
                buf("q-latent", t * c.n_head * m.kv_lora),
                buf("o-latent", t * c.n_head * m.kv_lora),
            ),
            None => (buf("unused", 4), buf("unused", 4), buf("unused", 4), buf("unused", 4)),
        };
        let half = c.rope_dim / 2;
        let mut table = Vec::with_capacity(ctx * half * 2);
        for pos in 0..ctx {
            for i in 0..half {
                let angle = pos as f32 * model::rope_inv_freq(&c, i, self.rope_factors.as_deref());
                let (s, co) = (angle as f64).sin_cos();
                table.push(co as f32 * c.rope_mscale);
                table.push(s as f32 * c.rope_mscale);
            }
        }
        let rope_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope"),
            contents: &floats(&table),
            usage,
        });
        let step = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (c.n_vocab * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = |data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        fn entry(i: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding: i,
                resource: b.as_entire_binding(),
            }
        }
        let bind = |layout: &wgpu::BindGroupLayout, label: &str, bufs: &[&wgpu::Buffer]| {
            let entries: Vec<_> = bufs
                .iter()
                .enumerate()
                .map(|(i, b)| entry(i as u32, b))
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        };
        let mut ops: Vec<Op> = Vec::new();
        let dummy = &self.dummy;
        let tables = &self.tables;
        // Writable placeholders: a buffer cannot be bound read-only and writable in one dispatch.
        let (spare_a, spare_b) = (buf("unused", 4), buf("unused", 4));
        let route_w = (t * moe.as_ref().map_or(0, |e| e.n_used)) as u32;

        // y (+)= W x for every shard of a matrix.
        let matmul = |ops: &mut Vec<Op>,
                      name: &str,
                      x: &wgpu::Buffer,
                      y: &wgpu::Buffer,
                      mode: u32,
                      bias: Option<&wgpu::Buffer>,
                      gemv: bool,
                      input: Input| {
            let m = &self.mats[name];
            for s in &m.shards {
                let p = params(&words(&[
                    m.k as u32,
                    s.rows as u32,
                    m.total_rows() as u32,
                    s.row0 as u32,
                    mode,
                    bias.is_some() as u32,
                    0,
                    input.stride as u32,
                    input.group_rows as u32,
                    input.group_stride as u32,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]));
                let codes4 = if matches!(m.fmt, GpuFormat::Block(_)) { &s.codes } else { dummy };
                let b = bind(
                    &self.layouts.matmul,
                    name,
                    &[
                        &step,
                        &p,
                        &s.codes,
                        s.scales.as_ref().unwrap_or(tables),
                        x,
                        y,
                        bias.unwrap_or(dummy),
                        codes4,
                    ],
                );
                ops.push(Op {
                    kernel: Kernel::Matmul(m.fmt),
                    grid: Grid::Matmul {
                        n: s.rows as u32,
                        gemv,
                    },
                    bind: b,
                });
            }
        };
        // Expert projections of layer `l`: gate/up (`down` = false) or the weighted down sum.
        let matmul_id = |ops: &mut Vec<Op>,
                         l: usize,
                         name: &str,
                         x: &wgpu::Buffer,
                         y: &wgpu::Buffer,
                         mode: u32,
                         down: bool| {
            let e = moe.as_ref().unwrap();
            let m = &self.mats[name];
            for s in &m.shards {
                let p = params(&words(&[
                    m.k as u32,
                    s.rows as u32,
                    if down { c.n_embd } else { m.rows } as u32,
                    0,
                    mode,
                    0,
                    0,
                    if down { m.k } else { c.n_embd } as u32,
                    0,
                    0,
                    e.n_used as u32,
                    (s.row0 / m.rows) as u32,
                    m.rows as u32,
                    (l * e.n_expert) as u32,
                    route_w,
                    0,
                ]));
                let codes4 = if matches!(m.fmt, GpuFormat::Block(_)) { &s.codes } else { dummy };
                let b = bind(
                    &self.layouts.matmul_id,
                    name,
                    &[
                        &step,
                        &p,
                        &s.codes,
                        s.scales.as_ref().unwrap_or(tables),
                        x,
                        y,
                        &route,
                        codes4,
                        &emap,
                    ],
                );
                ops.push(Op {
                    kernel: if down {
                        Kernel::MatmulIdDown(m.fmt)
                    } else {
                        Kernel::MatmulId(m.fmt)
                    },
                    grid: if down {
                        Grid::TokenRows { rows: m.rows as u32 }
                    } else {
                        Grid::Pairs {
                            rows: m.rows as u32,
                            used: e.n_used as u32,
                        }
                    },
                    bind: b,
                });
            }
        };
        let norm = |ops: &mut Vec<Op>,
                    src: &wgpu::Buffer,
                    w: &wgpu::Buffer,
                    dst: &wgpu::Buffer,
                    len: usize,
                    last_only: bool| {
            let mut data = words(&[len as u32, 1, last_only as u32]);
            data.extend(c.rms_eps.to_le_bytes());
            let p = params(&data);
            ops.push(Op {
                kernel: Kernel::Norm,
                grid: if last_only { Grid::One } else { Grid::Rows(1) },
                bind: bind(&self.layouts.norm, "norm", &[&step, &p, src, w, dst]),
            });
        };
        // RMSNorm in place over `rows` consecutive slices of `len` per token.
        let norm_inplace = |ops: &mut Vec<Op>, b: &wgpu::Buffer, w: &wgpu::Buffer, len: usize, rows: usize| {
            let mut data = words(&[len as u32, rows as u32, 0]);
            data.extend(c.rms_eps.to_le_bytes());
            let p = params(&data);
            ops.push(Op {
                kernel: Kernel::NormInPlace,
                grid: Grid::Rows(rows as u32),
                bind: bind(&self.layouts.norm_inplace, "norm-inplace", &[&step, &p, b, w]),
            });
        };

        let emb = &self.mats["token_embd.weight"];
        for s in &emb.shards {
            let p = params(&words(&[
                c.n_embd as u32,
                s.rows as u32,
                0,
                s.row0 as u32,
                0,
                0,
                0,
                c.n_embd as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]));
            let b = bind(
                &self.layouts.matmul,
                "embed",
                &[
                    &step,
                    &p,
                    &s.codes,
                    s.scales.as_ref().unwrap_or(tables),
                    dummy,
                    &x,
                    dummy,
                    &tokens,
                ],
            );
            ops.push(Op {
                kernel: Kernel::Embed(emb.fmt),
                grid: Grid::Embed { k: c.n_embd as u32 },
                bind: b,
            });
        }
        let rope_p = params(&words(&[
            c.n_head as u32,
            c.n_head_kv as u32,
            c.head_dim as u32,
            half as u32,
            c.rope_neox as u32,
            c.kv_dim() as u32,
            c.q_dim() as u32,
            mla.as_ref().map_or(0, |m| m.nope) as u32,
        ]));
        let mut attn_data = words(&[
            c.n_head as u32,
            c.n_head_kv as u32,
            c.head_dim as u32,
            c.kv_dim() as u32,
            c.q_dim() as u32,
        ]);
        attn_data.extend(c.kq_scale.to_le_bytes());
        attn_data.extend(words(&[0, 0]));
        let attn_p = params(&attn_data);
        let (mla_store_p, mla_attn_p) = match &mla {
            Some(m) => {
                let mut sd = words(&[m.kv_lora as u32, m.rope as u32, (m.kv_lora + m.rope) as u32, c.rope_neox as u32]);
                sd.extend(c.rms_eps.to_le_bytes());
                sd.extend(words(&[0, 0, 0]));
                let mut ad = words(&[
                    c.n_head as u32,
                    m.kv_lora as u32,
                    m.rope as u32,
                    c.head_dim as u32,
                    m.nope as u32,
                    (m.kv_lora + m.rope) as u32,
                ]);
                ad.extend(c.kq_scale.to_le_bytes());
                ad.extend(words(&[mla_heads(&c) as u32]));
                (params(&sd), params(&ad))
            }
            None => (params(&[0u8; 32]), params(&[0u8; 32])),
        };
        let topk_p = match &moe {
            Some(e) => {
                let mut d = words(&[
                    e.n_expert as u32,
                    e.n_used as u32,
                    (e.gating == Gating::Sigmoid) as u32,
                    e.norm as u32,
                ]);
                d.extend(e.scale.to_le_bytes());
                d.extend(words(&[
                    e.probs_bias as u32,
                    e.n_group as u32,
                    e.n_group_used as u32,
                    route_w,
                    0,
                    0,
                    0,
                ]));
                params(&d)
            }
            None => params(&[0u8; 48]),
        };
        for l in 0..c.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            norm(&mut ops, &x, &self.vecs[&n("attn_norm.weight")], &xn, c.n_embd, false);
            if let Some(m) = &mla {
                if m.q_lora > 0 {
                    matmul(&mut ops, &n("attn_q_a.weight"), &xn, &qa, 0, None, false, Input::rows(c.n_embd));
                    norm_inplace(&mut ops, &qa, &self.vecs[&n("attn_q_a_norm.weight")], m.q_lora, 1);
                    matmul(&mut ops, &n("attn_q_b.weight"), &qa, &q, 0, None, false, Input::rows(m.q_lora));
                } else {
                    matmul(&mut ops, &n("attn_q.weight"), &xn, &q, 0, None, false, Input::rows(c.n_embd));
                }
                matmul(&mut ops, &n("attn_kv_a_mqa.weight"), &xn, &kva, 0, None, false, Input::rows(c.n_embd));
                // Rotate the tail of every query head (the key's rotated part goes to the cache).
                ops.push(Op {
                    kernel: Kernel::Rope,
                    grid: Grid::Heads(c.n_head as u32),
                    bind: bind(&self.layouts.rope, "rope-q", &[&step, &rope_p, &q, dummy, dummy, &spare_a, &spare_b, &rope_table]),
                });
                ops.push(Op {
                    kernel: Kernel::MlaStore,
                    grid: Grid::Tokens,
                    bind: bind(&self.layouts.mla, "mla-store", &[&step, &mla_store_p, &kva, &self.vecs[&n("attn_kv_a_norm.weight")], &rope_table, &kc[l]]),
                });
                // Absorb the key projection into the queries: one kv_lora-row block per head.
                matmul(
                    &mut ops,
                    &n("attn_k_b.weight"),
                    &q,
                    &qlat,
                    0,
                    None,
                    false,
                    Input {
                        stride: c.q_dim(),
                        group_rows: m.kv_lora,
                        group_stride: c.head_dim,
                    },
                );
                ops.push(Op {
                    kernel: Kernel::MlaAttention,
                    grid: Grid::Heads((c.n_head / mla_heads(&c)) as u32),
                    bind: bind(&self.layouts.mla, "mla-attention", &[&step, &mla_attn_p, &qlat, &q, &kc[l], &olat]),
                });
                matmul(
                    &mut ops,
                    &n("attn_v_b.weight"),
                    &olat,
                    &att,
                    0,
                    None,
                    false,
                    Input {
                        stride: c.n_head * m.kv_lora,
                        group_rows: m.v_dim,
                        group_stride: m.kv_lora,
                    },
                );
                matmul(&mut ops, &n("attn_output.weight"), &att, &x, 1, None, false, Input::rows(v_width));
            } else {
                let bias = |s: &str| {
                    if c.qkv_bias {
                        Some(&self.vecs[&n(s)])
                    } else {
                        None
                    }
                };
                let input = Input::rows(c.n_embd);
                matmul(&mut ops, &n("attn_q.weight"), &xn, &q, 0, bias("attn_q.bias"), false, input);
                matmul(&mut ops, &n("attn_k.weight"), &xn, &k, 0, bias("attn_k.bias"), false, input);
                matmul(&mut ops, &n("attn_v.weight"), &xn, &v, 0, bias("attn_v.bias"), false, input);
                if c.qk_norm {
                    norm_inplace(&mut ops, &q, &self.vecs[&n("attn_q_norm.weight")], c.head_dim, c.n_head);
                    norm_inplace(&mut ops, &k, &self.vecs[&n("attn_k_norm.weight")], c.head_dim, c.n_head_kv);
                }
                ops.push(Op {
                    kernel: Kernel::Rope,
                    grid: Grid::Heads((c.n_head + c.n_head_kv) as u32),
                    bind: bind(&self.layouts.rope, "rope", &[&step, &rope_p, &q, &k, &v, &kc[l], &vc[l], &rope_table]),
                });
                ops.push(Op {
                    kernel: Kernel::Attention,
                    grid: Grid::Heads(c.n_head as u32),
                    bind: bind(&self.layouts.attention, "attention", &[&step, &attn_p, &q, &kc[l], &vc[l], &att]),
                });
                matmul(&mut ops, &n("attn_output.weight"), &att, &x, 1, None, false, Input::rows(c.q_dim()));
            }
            norm(&mut ops, &x, &self.vecs[&n("ffn_norm.weight")], &xn, c.n_embd, false);
            let input = Input::rows(c.n_embd);
            match moe.as_ref().filter(|_| c.is_moe(l)) {
                Some(e) => {
                    matmul(&mut ops, &n("ffn_gate_inp.weight"), &xn, &rlog, 0, None, false, input);
                    let bias = if e.probs_bias { &self.vecs[&n("exp_probs_b.bias")] } else { dummy };
                    ops.push(Op {
                        kernel: Kernel::TopK,
                        grid: Grid::Tokens,
                        bind: bind(&self.layouts.topk, "topk", &[&step, &topk_p, &rlog, bias, &route]),
                    });
                    matmul_id(&mut ops, l, &n("ffn_gate_exps.weight"), &xn, &hexp, 0, false);
                    matmul_id(&mut ops, l, &n("ffn_up_exps.weight"), &xn, &hexp, 2, false);
                    matmul_id(&mut ops, l, &n("ffn_down_exps.weight"), &hexp, &x, 1, true);
                    if e.n_ff_shared > 0 {
                        matmul(&mut ops, &n("ffn_gate_shexp.weight"), &xn, &gate, 0, None, false, input);
                        matmul(&mut ops, &n("ffn_up_shexp.weight"), &xn, &gate, 2, None, false, input);
                        matmul(&mut ops, &n("ffn_down_shexp.weight"), &gate, &x, 1, None, false, Input::rows(e.n_ff_shared));
                    }
                }
                None => {
                    matmul(&mut ops, &n("ffn_gate.weight"), &xn, &gate, 0, None, false, input);
                    matmul(&mut ops, &n("ffn_up.weight"), &xn, &gate, 2, None, false, input);
                    matmul(&mut ops, &n("ffn_down.weight"), &gate, &x, 1, None, false, Input::rows(c.n_ff));
                }
            }
        }
        let head_start = ops.len();
        norm(&mut ops, &x, &self.vecs["output_norm.weight"], &xf, c.n_embd, true);
        let head = if c.tied_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        matmul(&mut ops, head, &xf, &logits, 0, None, true, Input::rows(c.n_embd));

        let bytes = Self::state_bytes(&c, ctx);
        self.state = Some(State {
            ctx,
            step,
            tokens,
            logits,
            staging,
            ops,
            head_start,
            bytes,
        });
        match self.gpu.take_error() {
            Some(e) => Err(format!("could not prepare the model on the GPU: {e}")),
            None => Ok(()),
        }
    }

    pub fn context_length(&self) -> usize {
        self.state.as_ref().map(|s| s.ctx).unwrap_or(0)
    }

    pub fn state_memory(&self) -> u64 {
        self.state.as_ref().map(|s| s.bytes).unwrap_or(0)
    }

    fn dispatch_of(op: &Op, t: u32) -> (String, (u32, u32)) {
        let rows_per_wg = 256 / gemv_lanes();
        match (&op.kernel, op.grid) {
            (Kernel::Matmul(f), Grid::Matmul { n, gemv }) => {
                if gemv || t == 1 {
                    let groups = n.div_ceil(rows_per_wg);
                    let gx = groups.min(65535);
                    (format!("gemv-{}", f.key()), (gx, groups.div_ceil(gx)))
                } else {
                    (
                        format!("gemm-{}", f.key()),
                        (n.div_ceil(64), t.div_ceil(32)),
                    )
                }
            }
            (Kernel::MatmulId(f), Grid::Pairs { rows, used }) => {
                (format!("gemv-id-{}", f.key()), (rows.div_ceil(rows_per_wg), t * used))
            }
            (Kernel::MatmulIdDown(f), Grid::TokenRows { rows }) => {
                (format!("gemv-id-down-{}", f.key()), (rows.div_ceil(rows_per_wg), t))
            }
            (Kernel::Embed(f), Grid::Embed { k }) => {
                (format!("embed-{}", f.key()), ((k / 8).div_ceil(64), t))
            }
            (Kernel::Norm, Grid::One) => ("norm".into(), (1, 1)),
            (Kernel::Norm, Grid::Rows(r)) => ("norm".into(), (t * r, 1)),
            (Kernel::NormInPlace, Grid::Rows(r)) => ("norm-inplace".into(), (t * r, 1)),
            (Kernel::Rope, Grid::Heads(h)) => ("rope".into(), (h, t)),
            (Kernel::Attention, Grid::Heads(h)) => ("attention".into(), (h, t)),
            (Kernel::MlaAttention, Grid::Heads(h)) => ("mla-attention".into(), (h, t)),
            (Kernel::TopK, Grid::Tokens) => ("topk".into(), (t, 1)),
            (Kernel::MlaStore, Grid::Tokens) => ("mla-store".into(), (t, 1)),
            _ => unreachable!(),
        }
    }

    fn encode_chunk(&self, n_tokens: usize, with_head: bool) -> wgpu::CommandBuffer {
        let st = self.state.as_ref().unwrap();
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward"),
                timestamp_writes: None,
            });
            let t = n_tokens as u32;
            let end = if with_head {
                st.ops.len()
            } else {
                st.head_start
            };
            for op in &st.ops[..end] {
                let (key, (gx, gy)) = Self::dispatch_of(op, t);
                pass.set_pipeline(&self.pipelines[&key]);
                pass.set_bind_group(0, &op.bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        if with_head {
            enc.copy_buffer_to_buffer(&st.logits, 0, &st.staging, 0, st.staging.size());
        }
        enc.finish()
    }

    /// Time each kernel on its own (native only): (kernel, calls, seconds).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn profile(&self, n_tokens: usize, pos0: usize, reps: usize) -> Vec<(String, usize, f64)> {
        let st = self.state.as_ref().unwrap();
        let t = n_tokens as u32;
        self.gpu.queue.write_buffer(
            &st.step,
            0,
            &words(&[t, pos0 as u32, pos0 as u32 + t, t - 1]),
        );
        let mut totals: Vec<(String, usize, f64)> = Vec::new();
        for op in &st.ops {
            let (key, (gx, gy)) = Self::dispatch_of(op, t);
            let run = |count: usize| {
                let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipelines[&key]);
                    pass.set_bind_group(0, &op.bind, &[]);
                    for _ in 0..count {
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
                self.gpu.queue.submit([enc.finish()]);
                self.gpu
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            };
            run(reps);
            let start = std::time::Instant::now();
            run(reps);
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            match totals.iter_mut().find(|e| e.0 == key) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += secs;
                }
                None => totals.push((key, 1, secs)),
            }
        }
        totals
    }

    /// Run `tokens` at positions `pos0..` and return the logits after the last one.
    pub async fn forward(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, String> {
        let st = self.state.as_ref().ok_or("model is not loaded")?;
        if tokens.is_empty() {
            return Err("nothing to process".into());
        }
        if pos0 + tokens.len() > st.ctx {
            return Err(format!(
                "the conversation is longer than the {}-token context",
                st.ctx
            ));
        }
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_BATCH).collect();
        let mut pos = pos0;
        for (i, chunk) in chunks.iter().enumerate() {
            let n = chunk.len() as u32;
            let last = i + 1 == chunks.len();
            self.gpu.queue.write_buffer(
                &st.step,
                0,
                &words(&[n, pos as u32, pos as u32 + n, n - 1]),
            );
            self.gpu.queue.write_buffer(&st.tokens, 0, &words(chunk));
            let cmd = self.encode_chunk(chunk.len(), last);
            self.gpu.queue.submit([cmd]);
            pos += chunk.len();
        }
        let bytes = read_back(&self.gpu, &st.staging, st.staging.size()).await?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
