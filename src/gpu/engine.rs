//! The GPU forward pass: weight storage, activation buffers and the dispatch plan.

use super::{read_back, shaders, Gpu};
use crate::gguf::Gguf;
use crate::model::{self, Config};
use crate::quant::{self, GpuFormat};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Tokens processed per submission while reading a prompt.
pub const MAX_BATCH: usize = 128;

struct Shard {
    row0: usize,
    rows: usize,
    codes: wgpu::Buffer,
    scales: wgpu::Buffer,
}

struct Matrix {
    fmt: GpuFormat,
    ggml_type: u32,
    k: usize,
    rows: usize,
    shards: Vec<Shard>,
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
}

#[derive(Clone)]
enum Kernel {
    Matmul(GpuFormat),
    Embed(GpuFormat),
    Norm,
    NormInPlace,
    Rope,
    Attention,
}

struct Op {
    kernel: Kernel,
    grid: Grid,
    bind: wgpu::BindGroup,
}

struct Layouts {
    matmul: wgpu::BindGroupLayout,
    norm: wgpu::BindGroupLayout,
    norm_inplace: wgpu::BindGroupLayout,
    rope: wgpu::BindGroupLayout,
    attention: wgpu::BindGroupLayout,
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

impl Engine {
    /// Allocate weight buffers for a model. Weights are uploaded afterwards with `upload`.
    pub fn new(gpu: Gpu, g: &Gguf) -> Result<Engine, String> {
        let cfg = model::config(g)?;
        let device = &gpu.device;
        let max_binding = gpu.info.max_storage_binding.min(gpu.info.max_buffer_size);
        let mut mats = HashMap::new();
        let mut plan = Vec::new();
        let mut weight_bytes = 0u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        for (name, rows, k) in model::matrices(&cfg) {
            let t = g.tensor(&name).unwrap();
            let fmt = GpuFormat::for_type(t.ggml_type).unwrap();
            let rb = quant::row_bytes(t.ggml_type, k)? as u64;
            plan.push(TensorPlan {
                name: name.clone(),
                offset: g.data_offset + t.offset,
                bytes: rb * rows as u64,
                row_bytes: rb,
                rows: rows as u64,
                ggml_type: t.ggml_type,
            });
            let (codes_row, scales_row) = fmt.sizes(k, 1);
            let per_shard = ((max_binding / codes_row.max(scales_row).max(1)) as usize)
                .min(rows)
                .max(1);
            // Keep shard boundaries on multiples of 64 rows so matrix tiles never straddle them.
            let per_shard = if per_shard < rows {
                (per_shard / 64 * 64).max(1)
            } else {
                rows
            };
            let mut shards = Vec::new();
            let mut row0 = 0;
            while row0 < rows {
                let n = per_shard.min(rows - row0);
                let (cb, sb) = fmt.sizes(k, n);
                let codes = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&name),
                    size: cb,
                    usage,
                    mapped_at_creation: false,
                });
                let scales = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&name),
                    size: sb.max(16),
                    usage,
                    mapped_at_creation: false,
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
            norm: layout(device, "uurrw"),
            norm_inplace: layout(device, "uuwr"),
            rope: layout(device, "uuwrrwwr"),
            attention: layout(device, "uurrrw"),
        };
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage,
            mapped_at_creation: false,
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
            state: None,
            weight_bytes,
        };
        e.compile()?;
        Ok(e)
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
        let fmts: Vec<GpuFormat> = {
            let mut f: Vec<GpuFormat> = self.mats.values().map(|m| m.fmt).collect();
            f.sort_by_key(|f| f.key());
            f.dedup();
            f
        };
        let mm = self.layouts.matmul.clone();
        for f in fmts {
            self.pipeline(
                format!("gemv-{}", f.key()),
                shaders::gemv(f, gemv_lanes()),
                &mm,
            );
            self.pipeline(format!("gemm-{}", f.key()), shaders::gemm(f), &mm);
        }
        let ef = self.mats["token_embd.weight"].fmt;
        self.pipeline(format!("embed-{}", ef.key()), shaders::embed(ef), &mm);
        let (n, ni, r, a) = (
            self.layouts.norm.clone(),
            self.layouts.norm_inplace.clone(),
            self.layouts.rope.clone(),
            self.layouts.attention.clone(),
        );
        self.pipeline("norm".into(), shaders::rmsnorm(false), &n);
        self.pipeline("norm-inplace".into(), shaders::rmsnorm(true), &ni);
        self.pipeline("rope".into(), shaders::rope(), &r);
        self.pipeline("attention".into(), shaders::attention(), &a);
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
            if row0 + rows > m.rows {
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
                let packed = quant::repack(m.ggml_type, src, m.k, n)?;
                let (cb, sb) = m.fmt.sizes(m.k, r - s.row0);
                queue.write_buffer(&s.codes, cb, &words(&packed.codes));
                if !packed.scales.is_empty() {
                    queue.write_buffer(&s.scales, sb, &floats(&packed.scales));
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
        let acts = t * (2 * cfg.n_embd + 2 * cfg.q_dim() + 2 * cfg.kv_dim() + cfg.n_ff) as u64 * 4;
        let kv = cfg.n_layer as u64 * ctx as u64 * cfg.kv_dim() as u64 * 2 * 2;
        let rope = ctx as u64 * (cfg.rope_dim / 2) as u64 * 8;
        acts + kv + rope + cfg.n_vocab as u64 * 8
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
        let x = buf("x", t * c.n_embd);
        let xn = buf("xn", t * c.n_embd);
        let q = buf("q", t * c.q_dim());
        let k = buf("k", t * c.kv_dim());
        let v = buf("v", t * c.kv_dim());
        let att = buf("att", t * c.q_dim());
        let gate = buf("gate", t * c.n_ff);
        let xf = buf("xf", c.n_embd);
        let logits = buf("logits", c.n_vocab);
        let tokens = buf("tokens", t);
        let kc: Vec<_> = (0..c.n_layer)
            .map(|_| buf("k-cache", ctx * c.kv_dim() / 2))
            .collect();
        let vc: Vec<_> = (0..c.n_layer)
            .map(|_| buf("v-cache", ctx * c.kv_dim() / 2))
            .collect();
        let half = c.rope_dim / 2;
        let mut table = Vec::with_capacity(ctx * half * 2);
        for pos in 0..ctx {
            for i in 0..half {
                let angle = pos as f32 * model::rope_inv_freq(&c, i, self.rope_factors.as_deref());
                let (s, co) = (angle as f64).sin_cos();
                table.push(co as f32);
                table.push(s as f32);
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
        let mut ops: Vec<Op> = Vec::new();
        let dummy = &self.dummy;

        // y (+)= W x for every shard of a matrix.
        let matmul = |ops: &mut Vec<Op>,
                      name: &str,
                      x: &wgpu::Buffer,
                      y: &wgpu::Buffer,
                      mode: u32,
                      bias: Option<&wgpu::Buffer>,
                      gemv: bool| {
            let m = &self.mats[name];
            for s in &m.shards {
                let p = params(&words(&[
                    m.k as u32,
                    s.rows as u32,
                    m.rows as u32,
                    s.row0 as u32,
                    mode,
                    bias.is_some() as u32,
                    0,
                    0,
                ]));
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(name),
                    layout: &self.layouts.matmul,
                    entries: &[
                        entry(0, &step),
                        entry(1, &p),
                        entry(2, &s.codes),
                        entry(3, &s.scales),
                        entry(4, x),
                        entry(5, y),
                        entry(6, bias.unwrap_or(dummy)),
                        entry(7, dummy),
                    ],
                });
                ops.push(Op {
                    kernel: Kernel::Matmul(m.fmt),
                    grid: Grid::Matmul {
                        n: s.rows as u32,
                        gemv,
                    },
                    bind,
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
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("norm"),
                layout: &self.layouts.norm,
                entries: &[
                    entry(0, &step),
                    entry(1, &p),
                    entry(2, src),
                    entry(3, w),
                    entry(4, dst),
                ],
            });
            ops.push(Op {
                kernel: Kernel::Norm,
                grid: if last_only { Grid::One } else { Grid::Rows(1) },
                bind,
            });
        };
        let head_norm = |ops: &mut Vec<Op>, b: &wgpu::Buffer, w: &wgpu::Buffer, heads: usize| {
            let mut data = words(&[c.head_dim as u32, heads as u32, 0]);
            data.extend(c.rms_eps.to_le_bytes());
            let p = params(&data);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("head-norm"),
                layout: &self.layouts.norm_inplace,
                entries: &[entry(0, &step), entry(1, &p), entry(2, b), entry(3, w)],
            });
            ops.push(Op {
                kernel: Kernel::NormInPlace,
                grid: Grid::Rows(heads as u32),
                bind,
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
                0,
            ]));
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embed"),
                layout: &self.layouts.matmul,
                entries: &[
                    entry(0, &step),
                    entry(1, &p),
                    entry(2, &s.codes),
                    entry(3, &s.scales),
                    entry(4, dummy),
                    entry(5, &x),
                    entry(6, dummy),
                    entry(7, &tokens),
                ],
            });
            ops.push(Op {
                kernel: Kernel::Embed(emb.fmt),
                grid: Grid::Embed { k: c.n_embd as u32 },
                bind,
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
            0,
        ]));
        let mut attn_data = words(&[
            c.n_head as u32,
            c.n_head_kv as u32,
            c.head_dim as u32,
            c.kv_dim() as u32,
            c.q_dim() as u32,
        ]);
        attn_data.extend((1.0 / (c.head_dim as f32).sqrt()).to_le_bytes());
        attn_data.extend(words(&[0, 0]));
        let attn_p = params(&attn_data);
        for l in 0..c.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            norm(
                &mut ops,
                &x,
                &self.vecs[&n("attn_norm.weight")],
                &xn,
                c.n_embd,
                false,
            );
            let bias = |s: &str| {
                if c.qkv_bias {
                    Some(&self.vecs[&n(s)])
                } else {
                    None
                }
            };
            matmul(
                &mut ops,
                &n("attn_q.weight"),
                &xn,
                &q,
                0,
                bias("attn_q.bias"),
                false,
            );
            matmul(
                &mut ops,
                &n("attn_k.weight"),
                &xn,
                &k,
                0,
                bias("attn_k.bias"),
                false,
            );
            matmul(
                &mut ops,
                &n("attn_v.weight"),
                &xn,
                &v,
                0,
                bias("attn_v.bias"),
                false,
            );
            if c.qk_norm {
                head_norm(&mut ops, &q, &self.vecs[&n("attn_q_norm.weight")], c.n_head);
                head_norm(
                    &mut ops,
                    &k,
                    &self.vecs[&n("attn_k_norm.weight")],
                    c.n_head_kv,
                );
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rope"),
                layout: &self.layouts.rope,
                entries: &[
                    entry(0, &step),
                    entry(1, &rope_p),
                    entry(2, &q),
                    entry(3, &k),
                    entry(4, &v),
                    entry(5, &kc[l]),
                    entry(6, &vc[l]),
                    entry(7, &rope_table),
                ],
            });
            ops.push(Op {
                kernel: Kernel::Rope,
                grid: Grid::Heads((c.n_head + c.n_head_kv) as u32),
                bind,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("attention"),
                layout: &self.layouts.attention,
                entries: &[
                    entry(0, &step),
                    entry(1, &attn_p),
                    entry(2, &q),
                    entry(3, &kc[l]),
                    entry(4, &vc[l]),
                    entry(5, &att),
                ],
            });
            ops.push(Op {
                kernel: Kernel::Attention,
                grid: Grid::Heads(c.n_head as u32),
                bind,
            });
            matmul(&mut ops, &n("attn_output.weight"), &att, &x, 1, None, false);
            norm(
                &mut ops,
                &x,
                &self.vecs[&n("ffn_norm.weight")],
                &xn,
                c.n_embd,
                false,
            );
            matmul(&mut ops, &n("ffn_gate.weight"), &xn, &gate, 0, None, false);
            matmul(&mut ops, &n("ffn_up.weight"), &xn, &gate, 2, None, false);
            matmul(&mut ops, &n("ffn_down.weight"), &gate, &x, 1, None, false);
        }
        let head_start = ops.len();
        norm(
            &mut ops,
            &x,
            &self.vecs["output_norm.weight"],
            &xf,
            c.n_embd,
            true,
        );
        let head = if c.tied_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        matmul(&mut ops, head, &xf, &logits, 0, None, true);

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
        match (&op.kernel, op.grid) {
            (Kernel::Matmul(f), Grid::Matmul { n, gemv }) => {
                if gemv || t == 1 {
                    let groups = n.div_ceil(256 / gemv_lanes());
                    let gx = groups.min(65535);
                    (format!("gemv-{}", f.key()), (gx, groups.div_ceil(gx)))
                } else {
                    (
                        format!("gemm-{}", f.key()),
                        (n.div_ceil(64), t.div_ceil(32)),
                    )
                }
            }
            (Kernel::Embed(f), Grid::Embed { k }) => {
                (format!("embed-{}", f.key()), ((k / 8).div_ceil(64), t))
            }
            (Kernel::Norm, Grid::One) => ("norm".into(), (1, 1)),
            (Kernel::Norm, Grid::Rows(r)) => ("norm".into(), (t * r, 1)),
            (Kernel::NormInPlace, Grid::Rows(r)) => ("norm-inplace".into(), (t * r, 1)),
            (Kernel::Rope, Grid::Heads(h)) => ("rope".into(), (h, t)),
            (Kernel::Attention, Grid::Heads(h)) => ("attention".into(), (h, t)),
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
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines[&key]);
                pass.set_bind_group(0, &op.bind, &[]);
                for _ in 0..reps {
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            self.gpu.queue.submit([enc.finish()]);
            self.gpu
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let start = std::time::Instant::now();
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines[&key]);
                pass.set_bind_group(0, &op.bind, &[]);
                for _ in 0..reps {
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            self.gpu.queue.submit([enc.finish()]);
            self.gpu
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
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
