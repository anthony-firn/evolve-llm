//! The GPU forward pass: weight storage, activation buffers and the dispatch plan.

use super::{read_back, shaders, Gpu};
use crate::gguf::Gguf;
use crate::model::{self, Config, Gating};
use crate::quant::{self, GpuFormat, Layout};
use crate::source::WeightSource;
use std::cell::RefCell;
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

/// Where a weight matrix lives while the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// Uploaded once at load.
    Resident,
    /// Read from storage before each use into a buffer it shares with the same matrix of other
    /// layers (a ring of two, so one layer's weights can upload while the previous one runs).
    Ring,
    /// Experts: the chosen ones are read into slots of an expert cache shared by all layers.
    Pool(usize),
    /// Token embeddings looked up on the CPU, one row per token.
    Cpu,
}

struct Matrix {
    fmt: GpuFormat,
    ggml_type: u32,
    k: usize,
    /// Rows of one expert (or one head group); the matrix stacks `experts` of them.
    rows: usize,
    experts: usize,
    shards: Vec<Shard>,
    place: Place,
}

/// One buffer per expert matrix (gate, up, down) holding `n` cache slots from `slot0`.
struct PoolPage {
    slot0: usize,
    n: usize,
    bufs: [wgpu::Buffer; 3],
}

/// An expert cache for the layers whose experts share three weight formats.
struct Pool {
    types: [u32; 3],
    /// (rows, k) of one expert's gate, up and down matrices.
    shape: [(usize, usize); 3],
    slots: usize,
    pages: Vec<PoolPage>,
}

/// Least-recently-used slot ownership of one expert cache.
struct Lru {
    owner: Vec<Option<(usize, u32)>>,
    used: Vec<u64>,
    slot_of: HashMap<(usize, u32), usize>,
}

/// What streaming did since the counters were last read.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct StreamStats {
    pub bytes_read: u64,
    pub read_seconds: f64,
    /// Rewriting blocks into their GPU layout and handing them to the GPU queue.
    pub upload_seconds: f64,
    pub expert_requests: u64,
    pub expert_hits: u64,
    pub expert_loads: u64,
    pub passes: u64,
}

/// How the weights were placed, for the interface.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Placement {
    pub streaming: bool,
    pub resident_bytes: u64,
    pub streamed_bytes: u64,
    pub resident_layers: usize,
    pub streamed_layers: usize,
    pub expert_slots: usize,
    pub experts_total: usize,
    pub cpu_embeddings: bool,
}

struct Stream {
    lru: Vec<Lru>,
    emap: Vec<u32>,
    clock: u64,
    stats: StreamStats,
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
    /// GPU memory for weights. Models larger than this keep only part of their weights on the
    /// GPU and read the rest from storage while they run (None = no limit).
    pub budget: Option<u64>,
}

/// Where a tensor lives in the file and how to upload it in row chunks.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TensorPlan {
    pub name: String,
    /// File of a split model.
    pub file: usize,
    pub offset: u64,
    pub bytes: u64,
    pub row_bytes: u64,
    pub rows: u64,
    pub ggml_type: u32,
    /// Uploaded at load; other tensors are read from storage while the model runs.
    pub resident: bool,
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
    /// Expert matrix-matrix products: (64-row blocks of `rows`, 32-token tiles, experts).
    ExpertTiles { rows: u32, experts: u32 },
    /// A single workgroup whatever the token count.
    Single,
    /// (256-value blocks of `width`, n_tokens).
    Width(u32),
}

/// Which token counts an op runs for: some steps have a one-token and a prompt variant.
#[derive(Clone, Copy, PartialEq, Eq)]
enum When {
    Always,
    OneToken,
    Prompt,
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
    MatmulIdGemm(GpuFormat),
    RouteSort,
    MoeSum,
}

struct Op {
    kernel: Kernel,
    grid: Grid,
    bind: wgpu::BindGroup,
    when: When,
}

struct Layouts {
    matmul: wgpu::BindGroupLayout,
    matmul_id: wgpu::BindGroupLayout,
    route_sort: wgpu::BindGroupLayout,
    moe_sum: wgpu::BindGroupLayout,
    norm: wgpu::BindGroupLayout,
    norm_inplace: wgpu::BindGroupLayout,
    rope: wgpu::BindGroupLayout,
    attention: wgpu::BindGroupLayout,
    topk: wgpu::BindGroupLayout,
    mla: wgpu::BindGroupLayout,
}

/// Op ranges of one layer: attention up to the expert choice, the expert projections, the rest.
#[derive(Clone, Copy)]
struct LayerOps {
    start: usize,
    route_end: usize,
    experts_end: usize,
    end: usize,
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
    embed_end: usize,
    layers: Vec<LayerOps>,
    x: wgpu::Buffer,
    route: wgpu::Buffer,
    route_staging: wgpu::Buffer,
    emap: wgpu::Buffer,
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
    pools: Vec<Pool>,
    /// Expert cache of each layer (None for dense layers and resident experts).
    layer_pool: Vec<Option<usize>>,
    /// Ring matrices to read before each layer, and before the output head.
    ring_layers: Vec<Vec<String>>,
    ring_head: Vec<String>,
    plan_index: HashMap<String, usize>,
    stream: RefCell<Stream>,
    source: RefCell<Option<Box<dyn WeightSource>>>,
    pub placement: Placement,
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
                budget: None,
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
        let mats_list = model::matrices(&cfg);
        let vec_bytes: u64 = model::vectors(&cfg).iter().map(|v| v.1 as u64 * 4).sum();
        let ty = |name: &str| g.tensor(name).unwrap().ggml_type;
        let size_in = |layout: Layout, name: &str, rows: usize, k: usize| {
            GpuFormat::choose(ty(name), layout).unwrap().bytes(k, rows)
        };
        let total = |layout: Layout| {
            vec_bytes
                + mats_list
                    .iter()
                    .map(|(n, r, k, e)| size_in(layout, n, r * e, *k))
                    .sum::<u64>()
        };
        let budget = opts.budget.unwrap_or(u64::MAX);
        let (wlayout, streaming) = if total(opts.layout) <= budget {
            (opts.layout, false)
        } else if total(Layout::Compact) <= budget {
            (Layout::Compact, false)
        } else {
            (Layout::Blocks, true)
        };
        let blk_of = |name: &str| -> Option<usize> {
            name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
        };
        let head_name = if cfg.tied_embeddings { "token_embd.weight" } else { "output.weight" };
        let mut place: HashMap<String, Place> =
            mats_list.iter().map(|m| (m.0.clone(), Place::Resident)).collect();
        // Expert caches: one per combination of gate, up and down formats.
        let mut pools: Vec<Pool> = Vec::new();
        let mut layer_pool: Vec<Option<usize>> = vec![None; cfg.n_layer];
        let mut placement = Placement {
            streaming,
            ..Default::default()
        };
        if streaming {
            let mut used = vec_bytes;
            if !cfg.tied_embeddings {
                place.insert("token_embd.weight".into(), Place::Cpu);
                placement.cpu_embeddings = true;
            }
            let compact = |name: &str, rows: usize, k: usize| size_in(Layout::Blocks, name, rows, k);
            let mut pool_layers: Vec<usize> = Vec::new();
            if let Some(e) = &cfg.moe {
                for l in (0..cfg.n_layer).filter(|&l| cfg.is_moe(l)) {
                    let names = ["gate", "up", "down"].map(|k| format!("blk.{l}.ffn_{k}_exps.weight"));
                    let types = [ty(&names[0]), ty(&names[1]), ty(&names[2])];
                    let shape = [(e.n_ff, cfg.n_embd), (e.n_ff, cfg.n_embd), (cfg.n_embd, e.n_ff)];
                    let p = match pools.iter().position(|p| p.types == types) {
                        Some(p) => p,
                        None => {
                            pools.push(Pool { types, shape, slots: 0, pages: Vec::new() });
                            pool_layers.push(0);
                            pools.len() - 1
                        }
                    };
                    pool_layers[p] += 1;
                    layer_pool[l] = Some(p);
                    for n in &names {
                        place.insert(n.clone(), Place::Pool(p));
                    }
                }
                placement.experts_total = pool_layers.iter().sum::<usize>() * e.n_expert;
            }
            let unit = |p: &Pool| -> u64 {
                (0..3)
                    .map(|i| GpuFormat::compact(p.types[i]).unwrap().bytes(p.shape[i].1, p.shape[i].0))
                    .sum()
            };
            let n_used = cfg.moe.as_ref().map_or(0, |e| e.n_used);
            let min_slots = |p: usize| (2 * n_used).min(pool_layers[p] * cfg.moe.as_ref().unwrap().n_expert);
            let pool_min: u64 = (0..pools.len()).map(|p| unit(&pools[p]) * min_slots(p) as u64).sum();
            // Dense weights of each layer, then the head; experts get what is left.
            let dense: Vec<(String, usize, usize, u64)> = mats_list
                .iter()
                .filter(|m| place[&m.0] == Place::Resident && m.0.starts_with("blk."))
                .map(|m| (m.0.clone(), m.1 * m.3, m.2, compact(&m.0, m.1 * m.3, m.2)))
                .collect();
            let head = mats_list.iter().find(|m| m.0 == head_name).unwrap();
            let head_bytes = compact(head_name, head.1 * head.3, head.2);
            let dense_total: u64 = dense.iter().map(|d| d.3).sum();
            if used + head_bytes + dense_total + pool_min <= budget {
                used += head_bytes + dense_total;
                placement.resident_layers = cfg.n_layer;
            } else {
                // Keep the head if it is small next to the budget; stream it otherwise.
                if head_bytes * 4 <= budget.saturating_sub(used) {
                    used += head_bytes;
                } else {
                    place.insert(head_name.into(), Place::Ring);
                    used += head_bytes;
                }
                // Two ring slots per kind of layer matrix, then whole layers while they fit.
                let kind = |n: &str| n.splitn(3, '.').nth(2).unwrap_or(n).to_string();
                let mut ring_reserve: HashMap<(String, u32), u64> = HashMap::new();
                for d in &dense {
                    let e = ring_reserve.entry((kind(&d.0), ty(&d.0))).or_default();
                    *e = (*e).max(d.3 * 2);
                }
                let reserve: u64 = ring_reserve.values().sum();
                let mut l = 0;
                while l < cfg.n_layer {
                    let layer: u64 = dense.iter().filter(|d| blk_of(&d.0) == Some(l)).map(|d| d.3).sum();
                    if used + layer + reserve + pool_min > budget {
                        break;
                    }
                    used += layer;
                    l += 1;
                }
                placement.resident_layers = l;
                used += reserve;
                for d in &dense {
                    if blk_of(&d.0).unwrap_or(0) >= l {
                        place.insert(d.0.clone(), Place::Ring);
                    }
                }
            }
            placement.streamed_layers = cfg.n_layer - placement.resident_layers;
            // Share what is left among the expert caches by layer count.
            let left = budget.saturating_sub(used);
            let layers_total: usize = pool_layers.iter().sum();
            for p in 0..pools.len() {
                let share = left / layers_total.max(1) as u64 * pool_layers[p] as u64;
                let cap = pool_layers[p] * cfg.moe.as_ref().unwrap().n_expert;
                pools[p].slots = ((share / unit(&pools[p]).max(1)) as usize).clamp(min_slots(p).max(1), cap);
                placement.expert_slots += pools[p].slots;
            }
        }
        let mut mats = HashMap::new();
        let mut plan = Vec::new();
        let mut weight_bytes = 0u64;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let new_buf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        // Expert cache pages: as many slots per buffer as the binding limit allows.
        for pool in &mut pools {
            let per_slot: Vec<u64> = (0..3)
                .map(|i| GpuFormat::compact(pool.types[i]).unwrap().bytes(pool.shape[i].1, pool.shape[i].0))
                .collect();
            let per_page = ((max_binding / per_slot.iter().max().unwrap().max(&1)) as usize).max(1);
            let mut slot0 = 0;
            while slot0 < pool.slots {
                let n = per_page.min(pool.slots - slot0);
                let bufs = [0, 1, 2].map(|i| new_buf("expert cache", per_slot[i] * n as u64));
                weight_bytes += per_slot.iter().sum::<u64>() * n as u64;
                pool.pages.push(PoolPage { slot0, n, bufs });
                slot0 += n;
            }
        }
        // Ring slots, shared by the same matrix kind of the streamed layers.
        let mut rings: HashMap<(String, u32, usize, usize), (Vec<wgpu::Buffer>, usize)> = HashMap::new();
        let mut ring_layers: Vec<Vec<String>> = vec![Vec::new(); cfg.n_layer];
        let mut ring_head = Vec::new();
        for (name, rows, k, experts) in &mats_list {
            let name = name.clone();
            let (rows, k, experts) = (*rows, *k, *experts);
            let t = g.tensor(&name).unwrap();
            let pl = place[&name];
            let fmt = match pl {
                Place::Resident => GpuFormat::choose(t.ggml_type, wlayout).unwrap(),
                _ => GpuFormat::compact(t.ggml_type).unwrap(),
            };
            let total = rows * experts;
            let rb = quant::row_bytes(t.ggml_type, k)? as u64;
            plan.push(TensorPlan {
                name: name.clone(),
                file: t.file,
                offset: t.start(),
                bytes: rb * total as u64,
                row_bytes: rb,
                rows: total as u64,
                ggml_type: t.ggml_type,
                resident: pl == Place::Resident,
            });
            let mut shards = Vec::new();
            match pl {
                Place::Resident => {
                    let (codes_row, scales_row) = fmt.sizes(k, 1);
                    let fit = ((max_binding / codes_row.max(scales_row).max(1)) as usize)
                        .min(total)
                        .max(1);
                    // Shards hold whole experts (or head groups); otherwise multiples of 64 rows,
                    // so matrix tiles never straddle them.
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
                    let mut row0 = 0;
                    while row0 < total {
                        let n = per_shard.min(total - row0);
                        let (cb, sb) = fmt.sizes(k, n);
                        let codes = new_buf(&name, cb);
                        let scales = fmt.has_scales().then(|| new_buf(&name, sb));
                        weight_bytes += cb + sb;
                        placement.resident_bytes += cb + sb;
                        shards.push(Shard {
                            row0,
                            rows: n,
                            codes,
                            scales,
                        });
                        row0 += n;
                    }
                }
                Place::Ring => {
                    let bytes = fmt.bytes(k, total);
                    if bytes > max_binding {
                        return Err(format!("{name} is too large to stream through one GPU buffer"));
                    }
                    let kind = name.splitn(3, '.').nth(2).unwrap_or(&name).to_string();
                    let key = (kind, t.ggml_type, k, total);
                    let slots = if name == head_name { 1 } else { 2 };
                    let entry = rings.entry(key).or_insert_with(|| {
                        weight_bytes += bytes * slots as u64;
                        ((0..slots).map(|_| new_buf("ring", bytes)).collect(), 0)
                    });
                    let codes = entry.0[entry.1 % entry.0.len()].clone();
                    entry.1 += 1;
                    placement.streamed_bytes += rb * total as u64;
                    shards.push(Shard {
                        row0: 0,
                        rows: total,
                        codes,
                        scales: None,
                    });
                    match blk_of(&name) {
                        Some(l) => ring_layers[l].push(name.clone()),
                        None => ring_head.push(name.clone()),
                    }
                }
                Place::Pool(p) => {
                    let i = ["gate", "up", "down"].iter().position(|w| name.contains(&format!("ffn_{w}_exps"))).unwrap();
                    placement.streamed_bytes += rb * total as u64;
                    for page in &pools[p].pages {
                        shards.push(Shard {
                            row0: page.slot0 * rows,
                            rows: page.n * rows,
                            codes: page.bufs[i].clone(),
                            scales: None,
                        });
                    }
                }
                Place::Cpu => {
                    placement.streamed_bytes += rb * total as u64;
                }
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
                    place: pl,
                },
            );
        }
        let mut vecs = HashMap::new();
        for (name, len) in model::vectors(&cfg) {
            let t = g.tensor(&name).unwrap();
            let rb = quant::row_bytes(t.ggml_type, len)? as u64;
            plan.push(TensorPlan {
                name: name.clone(),
                file: t.file,
                offset: t.start(),
                bytes: rb,
                row_bytes: rb,
                rows: 1,
                ggml_type: t.ggml_type,
                resident: true,
            });
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&name),
                size: (len * 4) as u64,
                usage,
                mapped_at_creation: false,
            });
            weight_bytes += (len * 4) as u64;
            placement.resident_bytes += (len * 4) as u64;
            vecs.insert(name, buf);
        }
        plan.sort_by_key(|p| (p.file, p.offset));
        let plan_index = plan.iter().enumerate().map(|(i, p)| (p.name.clone(), i)).collect();
        let lru = pools
            .iter()
            .map(|p| Lru {
                owner: vec![None; p.slots],
                used: vec![0; p.slots],
                slot_of: HashMap::new(),
            })
            .collect();
        let layouts = Layouts {
            matmul: layout(device, "uurrrwrr"),
            matmul_id: layout(device, "uurrrwrrrr"),
            route_sort: layout(device, "uurw"),
            moe_sum: layout(device, "uurrrw"),
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
            pools,
            layer_pool,
            ring_layers,
            ring_head,
            plan_index,
            stream: RefCell::new(Stream {
                lru,
                emap: Vec::new(),
                clock: 0,
                stats: StreamStats::default(),
            }),
            source: RefCell::new(None),
            placement,
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
            self.pipeline(format!("gemm-id-{}", f.key()), shaders::gemm_id(f), &mi);
        }
        if self.cfg.moe.is_some() {
            let (rs, ms) = (self.layouts.route_sort.clone(), self.layouts.moe_sum.clone());
            self.pipeline("route-sort".into(), shaders::route_sort(), &rs);
            self.pipeline("moe-sum".into(), shaders::moe_sum(), &ms);
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
            if m.place != Place::Resident {
                return Ok(());
            }
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
            acts += e.n_expert + 3 * e.n_used + e.n_used * e.n_ff + e.n_ff_shared + e.n_used * cfg.n_embd;
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
                    .flat_map(|l| {
                        let cached = self.layer_pool[l].is_some();
                        (0..e.n_expert as u32).map(move |i| if cached { ABSENT } else { i })
                    })
                    .collect();
                self.stream.borrow_mut().emap = ident.clone();
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
        let (rsort, dout) = match &moe {
            Some(e) => (
                buf("route sort", 2 * e.n_expert + t * e.n_used),
                buf("expert out", t * e.n_used * c.n_embd),
            ),
            None => (buf("unused", 4), buf("unused", 4)),
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
                    when: When::Always,
                    kernel: Kernel::Matmul(m.fmt),
                    grid: Grid::Matmul {
                        n: s.rows as u32,
                        gemv,
                    },
                    bind: b,
                });
            }
        };
        // Expert projections of layer `l`: gate/up (`down` = false) or down. One token runs the
        // matrix-vector kernels (down adds the weighted sum to x); a prompt runs matrix-matrix
        // products per expert (down writes each pair's row for `moe_sum`).
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
                let fields = |mode: u32| {
                    params(&words(&[
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
                        e.n_expert as u32,
                    ]))
                };
                let codes4 = if matches!(m.fmt, GpuFormat::Block(_)) { &s.codes } else { dummy };
                let scales = s.scales.as_ref().unwrap_or(tables);
                let one = fields(mode);
                let b = bind(&self.layouts.matmul_id, name, &[&step, &one, &s.codes, scales, x, y, &route, codes4, &emap, &rsort]);
                ops.push(Op {
                    when: When::OneToken,
                    kernel: if down { Kernel::MatmulIdDown(m.fmt) } else { Kernel::MatmulId(m.fmt) },
                    grid: if down {
                        Grid::TokenRows { rows: m.rows as u32 }
                    } else {
                        Grid::Pairs { rows: m.rows as u32, used: e.n_used as u32 }
                    },
                    bind: b,
                });
                let many = fields(if down { 3 } else { mode });
                let target = if down { &dout } else { y };
                let b = bind(&self.layouts.matmul_id, name, &[&step, &many, &s.codes, scales, x, target, &route, codes4, &emap, &rsort]);
                ops.push(Op {
                    when: When::Prompt,
                    kernel: Kernel::MatmulIdGemm(m.fmt),
                    grid: Grid::ExpertTiles { rows: m.rows as u32, experts: e.n_expert as u32 },
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
                when: When::Always,
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
                when: When::Always,
                kernel: Kernel::NormInPlace,
                grid: Grid::Rows(rows as u32),
                bind: bind(&self.layouts.norm_inplace, "norm-inplace", &[&step, &p, b, w]),
            });
        };

        let emb = &self.mats["token_embd.weight"];
        let gpu_embed = emb.place == Place::Resident;
        for s in emb.shards.iter().filter(|_| gpu_embed) {
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
                when: When::Always,
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
        let sort_p = params(&words(&[
            moe.as_ref().map_or(0, |e| e.n_expert) as u32,
            moe.as_ref().map_or(0, |e| e.n_used) as u32,
            0,
            0,
        ]));
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
        let embed_end = ops.len();
        let mut layers = Vec::with_capacity(c.n_layer);
        for l in 0..c.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}");
            let start = ops.len();
            let (mut route_end, mut experts_end) = (0, 0);
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
                    when: When::Always,
                    kernel: Kernel::Rope,
                    grid: Grid::Heads(c.n_head as u32),
                    bind: bind(&self.layouts.rope, "rope-q", &[&step, &rope_p, &q, dummy, dummy, &spare_a, &spare_b, &rope_table]),
                });
                ops.push(Op {
                    when: When::Always,
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
                    when: When::Always,
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
                if c.qk_norm_full {
                    norm_inplace(&mut ops, &q, &self.vecs[&n("attn_q_norm.weight")], c.q_dim(), 1);
                    norm_inplace(&mut ops, &k, &self.vecs[&n("attn_k_norm.weight")], c.kv_dim(), 1);
                } else if c.qk_norm {
                    norm_inplace(&mut ops, &q, &self.vecs[&n("attn_q_norm.weight")], c.head_dim, c.n_head);
                    norm_inplace(&mut ops, &k, &self.vecs[&n("attn_k_norm.weight")], c.head_dim, c.n_head_kv);
                }
                ops.push(Op {
                    when: When::Always,
                    kernel: Kernel::Rope,
                    grid: Grid::Heads((c.n_head + c.n_head_kv) as u32),
                    bind: bind(&self.layouts.rope, "rope", &[&step, &rope_p, &q, &k, &v, &kc[l], &vc[l], &rope_table]),
                });
                ops.push(Op {
                    when: When::Always,
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
                        when: When::Always,
                        kernel: Kernel::TopK,
                        grid: Grid::Tokens,
                        bind: bind(&self.layouts.topk, "topk", &[&step, &topk_p, &rlog, bias, &route]),
                    });
                    route_end = ops.len();
                    ops.push(Op {
                        when: When::Prompt,
                        kernel: Kernel::RouteSort,
                        grid: Grid::Single,
                        bind: bind(&self.layouts.route_sort, "route-sort", &[&step, &sort_p, &route, &rsort]),
                    });
                    matmul_id(&mut ops, l, &n("ffn_gate_exps.weight"), &xn, &hexp, 0, false);
                    matmul_id(&mut ops, l, &n("ffn_up_exps.weight"), &xn, &hexp, 2, false);
                    matmul_id(&mut ops, l, &n("ffn_down_exps.weight"), &hexp, &x, 1, true);
                    let sum_p = params(&words(&[c.n_embd as u32, e.n_used as u32, route_w, (l * e.n_expert) as u32]));
                    ops.push(Op {
                        when: When::Prompt,
                        kernel: Kernel::MoeSum,
                        grid: Grid::Width(c.n_embd as u32),
                        bind: bind(&self.layouts.moe_sum, "moe-sum", &[&step, &sum_p, &route, &emap, &dout, &x]),
                    });
                    experts_end = ops.len();
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
            let end = ops.len();
            if route_end == 0 {
                route_end = end;
                experts_end = end;
            }
            layers.push(LayerOps {
                start,
                route_end,
                experts_end,
                end,
            });
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
        let route_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("route staging"),
            size: route.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.state = Some(State {
            ctx,
            step,
            tokens,
            logits,
            staging,
            ops,
            head_start,
            embed_end,
            layers,
            x,
            route,
            route_staging,
            emap,
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

    fn dispatch_of(op: &Op, t: u32) -> (String, (u32, u32, u32)) {
        let (key, (gx, gy)) = match (&op.kernel, op.grid) {
            (Kernel::MatmulIdGemm(f), Grid::ExpertTiles { rows, experts }) => {
                return (format!("gemm-id-{}", f.key()), (rows.div_ceil(64), t.div_ceil(32), experts));
            }
            _ => Self::dispatch_2d(op, t),
        };
        (key, (gx, gy, 1))
    }

    fn runs(op: &Op, t: u32) -> bool {
        match op.when {
            When::Always => true,
            When::OneToken => t == 1,
            When::Prompt => t > 1,
        }
    }

    fn dispatch_2d(op: &Op, t: u32) -> (String, (u32, u32)) {
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
            (Kernel::RouteSort, Grid::Single) => ("route-sort".into(), (1, 1)),
            (Kernel::MoeSum, Grid::Width(w)) => ("moe-sum".into(), (w.div_ceil(256), t)),
            _ => unreachable!(),
        }
    }

    fn encode_chunk(&self, n_tokens: usize, with_head: bool) -> wgpu::CommandBuffer {
        let st = self.state.as_ref().unwrap();
        let end = if with_head { st.ops.len() } else { st.head_start };
        self.encode_range(0..end, n_tokens, with_head)
    }

    fn encode_range(&self, range: std::ops::Range<usize>, n_tokens: usize, copy_logits: bool) -> wgpu::CommandBuffer {
        let st = self.state.as_ref().unwrap();
        let with_head = copy_logits;
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
            for op in st.ops[range].iter().filter(|op| Self::runs(op, t)) {
                let (key, (gx, gy, gz)) = Self::dispatch_of(op, t);
                pass.set_pipeline(&self.pipelines[&key]);
                pass.set_bind_group(0, &op.bind, &[]);
                pass.dispatch_workgroups(gx, gy, gz);
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
        for op in st.ops.iter().filter(|op| Self::runs(op, t)) {
            let (key, (gx, gy, gz)) = Self::dispatch_of(op, t);
            let run = |count: usize| {
                let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipelines[&key]);
                    pass.set_bind_group(0, &op.bind, &[]);
                    for _ in 0..count {
                        pass.dispatch_workgroups(gx, gy, gz);
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
        if self.placement.streaming {
            return self.forward_streamed(tokens, pos0).await;
        }
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

    /// Free the weights and hand back the GPU (to retry with a smaller budget).
    pub fn into_gpu(self) -> Gpu {
        self.gpu
    }

    /// Where weights that are not resident are read from (required when `placement.streaming`).
    pub fn set_source(&self, source: Box<dyn WeightSource>) {
        *self.source.borrow_mut() = Some(source);
    }

    /// Streaming counters since the last call.
    pub fn take_stream_stats(&self) -> StreamStats {
        std::mem::take(&mut self.stream.borrow_mut().stats)
    }

    fn read(&self, plan: &TensorPlan, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let src = self.source.borrow();
        let src = src.as_ref().ok_or("this model streams weights but has no weight source")?;
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let bytes = src.read(plan.file, plan.offset + offset, len)?;
        let mut st = self.stream.borrow_mut();
        st.stats.bytes_read += len as u64;
        #[cfg(not(target_arch = "wasm32"))]
        {
            st.stats.read_seconds += t0.elapsed().as_secs_f64();
        }
        Ok(bytes)
    }

    fn tensor(&self, name: &str) -> &TensorPlan {
        &self.plan[self.plan_index[name]]
    }

    /// Read whole ring matrices into their slots before the ops that use them.
    fn upload_rings(&self, names: &[String]) -> Result<u64, String> {
        let mut bytes = 0;
        for name in names {
            let m = &self.mats[name];
            let tp = self.tensor(name).clone();
            // In slices of about 32 MB, so a large head never needs one huge buffer.
            let step = ((32u64 << 20) / tp.row_bytes).max(1) as usize;
            let total = m.total_rows();
            let mut r = 0;
            while r < total {
                let n = step.min(total - r);
                let raw = self.read(&tp, r as u64 * tp.row_bytes, n * tp.row_bytes as usize)?;
                #[cfg(not(target_arch = "wasm32"))]
                let t0 = std::time::Instant::now();
                let w = quant::to_gpu_bytes(m.ggml_type, &raw, m.k, n)?;
                self.gpu.queue.write_buffer(&m.shards[0].codes, m.fmt.bytes(m.k, r), &w);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.stream.borrow_mut().stats.upload_seconds += t0.elapsed().as_secs_f64();
                }
                r += n;
            }
            bytes += tp.bytes;
        }
        Ok(bytes)
    }

    /// Embedding rows of `tokens`, dequantized on the CPU.
    fn embed_cpu(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let tp = self.tensor("token_embd.weight").clone();
        let k = self.cfg.n_embd;
        let mut out = vec![0f32; tokens.len() * k];
        for (i, &t) in tokens.iter().enumerate() {
            if t as u64 >= tp.rows {
                return Err(format!("token {t} is outside the vocabulary"));
            }
            let row = self.read(&tp, t as u64 * tp.row_bytes, tp.row_bytes as usize)?;
            quant::dequantize(tp.ggml_type, &row, &mut out[i * k..(i + 1) * k])?;
        }
        Ok(out)
    }

    /// Read expert `e` of layer `l` into cache slot `slot` of pool `p`.
    fn load_expert(&self, p: usize, slot: usize, l: usize, e: u32) -> Result<(), String> {
        let pool = &self.pools[p];
        let page = pool.pages.iter().find(|pg| slot >= pg.slot0 && slot < pg.slot0 + pg.n).unwrap();
        for (i, kind) in ["gate", "up", "down"].iter().enumerate() {
            let name = format!("blk.{l}.ffn_{kind}_exps.weight");
            let tp = self.tensor(&name).clone();
            let (rows, k) = pool.shape[i];
            let len = rows * tp.row_bytes as usize;
            let raw = self.read(&tp, e as u64 * len as u64, len)?;
            let w = quant::to_gpu_bytes(tp.ggml_type, &raw, k, rows)?;
            let at = GpuFormat::compact(tp.ggml_type).unwrap().bytes(k, rows) * (slot - page.slot0) as u64;
            self.gpu.queue.write_buffer(&page.bufs[i], at, &w);
        }
        Ok(())
    }

    /// The experts layer `l` needs, grouped into passes that each fit the cache. Each pass is its
    /// (expert, slot) map and the (slot, expert) loads to do before it runs: misses go to the least
    /// recently used slots, so a later pass may reuse a slot an earlier pass read from.
    #[allow(clippy::type_complexity)]
    fn plan_experts(&self, l: usize, ids: &[u32]) -> Result<Vec<(Vec<(u32, usize)>, Vec<(usize, u32)>)>, String> {
        let p = self.layer_pool[l].unwrap();
        let mut need: Vec<u32> = ids.to_vec();
        need.sort_unstable();
        need.dedup();
        let mut passes = Vec::new();
        {
            let mut st = self.stream.borrow_mut();
            st.stats.expert_requests += need.len() as u64;
            let slots = self.pools[p].slots;
            while !need.is_empty() {
                st.clock += 1;
                let clock = st.clock;
                let lru = &mut st.lru[p];
                let mut pass: Vec<(u32, usize)> = Vec::new();
                let mut locked = vec![false; slots];
                let mut rest = Vec::new();
                for &e in &need {
                    match lru.slot_of.get(&(l, e)) {
                        Some(&s) => {
                            pass.push((e, s));
                            locked[s] = true;
                            lru.used[s] = clock;
                        }
                        None => rest.push(e),
                    }
                }
                let hits = pass.len() as u64;
                let mut loads: Vec<(usize, u32)> = Vec::new();
                let mut left = Vec::new();
                for e in rest {
                    if pass.len() == slots {
                        left.push(e);
                        continue;
                    }
                    let s = (0..slots).filter(|&s| !locked[s]).min_by_key(|&s| lru.used[s]).unwrap();
                    if let Some(old) = lru.owner[s].take() {
                        lru.slot_of.remove(&old);
                    }
                    lru.owner[s] = Some((l, e));
                    lru.slot_of.insert((l, e), s);
                    lru.used[s] = clock;
                    locked[s] = true;
                    loads.push((s, e));
                    pass.push((e, s));
                }
                st.stats.expert_hits += hits;
                st.stats.expert_loads += loads.len() as u64;
                st.stats.passes += 1;
                passes.push((pass, loads));
                need = left;
            }
        }
        Ok(passes)
    }

    /// Forward pass for models that do not fit: layer by layer, reading streamed weights and
    /// the experts each token chose.
    async fn forward_streamed(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, String> {
        let st = self.state.as_ref().ok_or("model is not loaded")?;
        if tokens.is_empty() {
            return Err("nothing to process".into());
        }
        if pos0 + tokens.len() > st.ctx {
            return Err(format!("the conversation is longer than the {}-token context", st.ctx));
        }
        let q = &self.gpu.queue;
        let n_expert = self.cfg.moe.as_ref().map_or(0, |e| e.n_expert);
        let n_used = self.cfg.moe.as_ref().map_or(0, |e| e.n_used);
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_BATCH).collect();
        let mut pos = pos0;
        for (ci, chunk) in chunks.iter().enumerate() {
            let n = chunk.len();
            let last = ci + 1 == chunks.len();
            q.write_buffer(&st.step, 0, &words(&[n as u32, pos as u32, (pos + n) as u32, n as u32 - 1]));
            q.write_buffer(&st.tokens, 0, &words(chunk));
            if self.placement.cpu_embeddings || st.embed_end == 0 {
                q.write_buffer(&st.x, 0, &floats(&self.embed_cpu(chunk)?));
            } else {
                q.submit([self.encode_range(0..st.embed_end, n, false)]);
            }
            let mut queued = 0u64;
            for (l, lo) in st.layers.iter().enumerate() {
                queued += self.upload_rings(&self.ring_layers[l])?;
                match self.layer_pool[l] {
                    Some(_) => {
                        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                        {
                            let mut pass = enc.begin_compute_pass(&Default::default());
                            for op in st.ops[lo.start..lo.route_end].iter().filter(|op| Self::runs(op, n as u32)) {
                                let (key, (gx, gy, gz)) = Self::dispatch_of(op, n as u32);
                                pass.set_pipeline(&self.pipelines[&key]);
                                pass.set_bind_group(0, &op.bind, &[]);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                        }
                        let route_len = (n * n_used * 4) as u64;
                        enc.copy_buffer_to_buffer(&st.route, 0, &st.route_staging, 0, route_len);
                        q.submit([enc.finish()]);
                        let raw = read_back(&self.gpu, &st.route_staging, route_len).await?;
                        let ids: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
                        if ids.iter().any(|&e| e as usize >= n_expert) {
                            return Err("expert routing produced an invalid expert".into());
                        }
                        let pool = self.layer_pool[l].unwrap();
                        for (pass, loads) in self.plan_experts(l, &ids)? {
                            // Queued after the previous pass's work, so reused slots are safe.
                            for (slot, e) in loads {
                                self.load_expert(pool, slot, l, e)?;
                            }
                            let mut map = vec![ABSENT; n_expert];
                            for (e, s) in pass {
                                map[e as usize] = s as u32;
                            }
                            q.write_buffer(&st.emap, (l * n_expert * 4) as u64, &words(&map));
                            q.submit([self.encode_range(lo.route_end..lo.experts_end, n, false)]);
                        }
                        q.submit([self.encode_range(lo.experts_end..lo.end, n, false)]);
                        queued = 0;
                    }
                    None => {
                        q.submit([self.encode_range(lo.start..lo.end, n, false)]);
                        // Bound the weight uploads waiting in the queue.
                        if queued > 512 << 20 {
                            super::wait_idle(&self.gpu).await?;
                            queued = 0;
                        }
                    }
                }
            }
            if last {
                self.upload_rings(&self.ring_head)?;
                q.submit([self.encode_range(st.head_start..st.ops.len(), n, true)]);
            }
            pos += n;
        }
        let bytes = read_back(&self.gpu, &st.staging, st.staging.size()).await?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
