//! The GPU forward pass: weight storage, activation buffers and the dispatch plan.

use super::{read_back, shaders, Gpu};
use crate::gguf::Gguf;
use crate::model::{self, Config, Gating};
use std::cell::Cell;
use crate::quant::{self, GpuFormat, Layout};
use crate::source::{Span, WeightSource};
use std::cell::RefCell;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Tokens processed per submission while reading a prompt.
pub const MAX_BATCH: usize = 128;
/// The smallest prompt chunk, the first chunk of a pass (it grows while chunks finish fast), and
/// the time past which the next chunk is halved. Starting small matters: on a slow GPU a first
/// full chunk can already outlast the driver's limit, and cancelled work leaves wrong numbers,
/// not an error.
const MIN_BATCH: usize = 16;
const FIRST_BATCH: usize = 32;
const SLOW_CHUNK_MS: f64 = 3000.0;

/// Marks an expert that is not resident in the expert-to-slot map.
pub const ABSENT: u32 = u32::MAX;

/// Most tokens one pass can return logits for (a speculative draft and the token before it).
pub const SPEC_MAX: usize = 8;

/// Prompt tokens a model that streams reads layer by layer at once: each streamed weight is
/// read once per this many tokens instead of once per `MAX_BATCH`.
pub const SUPER_BATCH: usize = 1024;

/// Activations of every chunk of a prompt read layer by layer, parked between phases.
struct Super {
    tokens: usize,
    x: wgpu::Buffer,
    xn: wgpu::Buffer,
    xl: wgpu::Buffer,
    ml: wgpu::Buffer,
    route: wgpu::Buffer,
    route_staging: wgpu::Buffer,
}

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

/// The part of a model one device runs when several devices share it: layers `first..end`,
/// and whether it reads tokens (the embedding) and makes logits (the output head). The device
/// the chat runs on holds the embedding, the head and the first layers; others hold the rest
/// and pass the hidden state on (see [`Remote`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Stage {
    pub first: usize,
    pub end: usize,
    pub embed: bool,
    pub head: bool,
}

impl Stage {
    pub fn whole(n_layer: usize) -> Stage {
        Stage { first: 0, end: n_layer, embed: true, head: true }
    }

    pub fn has_layer(&self, l: usize) -> bool {
        l >= self.first && l < self.end
    }

    /// Whether this stage needs the tensor.
    pub fn holds(&self, name: &str, tied_embeddings: bool) -> bool {
        let layer = name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok());
        match layer {
            Some(l) => self.has_layer(l),
            None => match name {
                "token_embd.weight" => self.embed || (self.head && tied_embeddings),
                "rope_freqs.weight" => true,
                _ => self.head,
            },
        }
    }
}

/// The layers after this device's, run elsewhere (other devices of the same person). `forward`
/// takes the hidden state after this device's last layer for `n` tokens from `pos0` and returns
/// the hidden state after the model's last layer (see [`Engine::hidden_width`] for the layout).
/// `snapshot` and `restore` mirror this engine's, for models with a recurrent state.
pub trait Remote {
    fn forward<'a>(&'a self, hidden: Vec<f32>, n: usize, pos0: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>, String>> + 'a>>;
    fn snapshot(&self);
    fn restore(&self);
}

/// What a pass reads: tokens, or the hidden state another device's layers produced (`n` tokens,
/// token-major).
#[derive(Clone, Copy)]
enum Feed<'a> {
    Tokens(&'a [u32]),
    Hidden(&'a [f32], usize),
}

impl<'a> Feed<'a> {
    fn len(&self) -> usize {
        match self {
            Feed::Tokens(t) => t.len(),
            Feed::Hidden(_, n) => *n,
        }
    }

    fn slice(&self, s: usize, n: usize) -> Feed<'a> {
        match *self {
            Feed::Tokens(t) => Feed::Tokens(&t[s..s + n]),
            Feed::Hidden(h, total) => {
                let w = h.len() / total;
                Feed::Hidden(&h[s * w..(s + n) * w], n)
            }
        }
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
    /// Run only part of the model (None = all of it).
    pub stage: Option<Stage>,
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
    /// A fixed number of workgroups whatever the token count (kernels that walk the tokens).
    Groups(u32),
    /// (blocks of 256 cached positions, n_tokens).
    KeyBlocks(u32),
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
    ResMix,
    Fill,
    KdaConv,
    L2Norm,
    Kda,
    IdxStore,
    IdxScore,
    IdxTopK,
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
    res_mix: wgpu::BindGroupLayout,
    fill: wgpu::BindGroupLayout,
    kda_conv: wgpu::BindGroupLayout,
    l2norm: wgpu::BindGroupLayout,
    kda: wgpu::BindGroupLayout,
    mla_attn: wgpu::BindGroupLayout,
    idx: wgpu::BindGroupLayout,
    idx_topk: wgpu::BindGroupLayout,
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
    mpos: wgpu::Buffer,
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
    /// Per linear-attention layer: its recurrent state and convolution inputs, and a copy of
    /// both taken at the end of the last prompt (a recurrent state cannot be rewound).
    kda: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    kda_snap: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    /// Ops after `head_end` produce logits for every token (up to SPEC_MAX) into `logits_all`.
    head_end: usize,
    logits_all: wgpu::Buffer,
    staging_all: wgpu::Buffer,
    // Working buffers the layer-by-layer prompt copies chunks in and out of.
    xn: wgpu::Buffer,
    xl: wgpu::Buffer,
    ml: wgpu::Buffer,
    sup: Option<Super>,
    // Block checkpoints (models with residuals across blocks) and the token stride of their
    // buffer; a staging buffer for handing the hidden state to another device.
    ck: wgpu::Buffer,
    ck_tokens: usize,
    hidden_staging: Option<wgpu::Buffer>,
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
    source: RefCell<Option<std::rc::Rc<dyn WeightSource>>>,
    pub placement: Placement,
    /// Tokens the recurrent state has read, and the position of its snapshot.
    recurrent_pos: Cell<usize>,
    snapshot_pos: Cell<Option<usize>>,
    /// The next pass returns logits for every token (speculative verification).
    all_logits: Cell<bool>,
    /// Tokens per prompt chunk, sized so one chunk finishes in about a second or two: on a slow
    /// GPU with a long context, a full chunk can outlast the driver's limit on one piece of GPU
    /// work (Linux i915 cancels after about 10 s). Learned from each chunk, kept between passes.
    batch: Cell<usize>,
    rope_delta: Cell<i32>,
    rope_positions: std::cell::RefCell<Option<(usize, Vec<u32>)>>,
    /// A second copy of the recurrent state, taken before verifying a draft (made on first use).
    spec_state: RefCell<Vec<(wgpu::Buffer, wgpu::Buffer)>>,
    spec_pos: Cell<usize>,
    /// The part of the model this engine runs, and who runs the layers after it.
    pub stage: Stage,
    remote: RefCell<Option<std::rc::Rc<dyn Remote>>>,
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
                stage: None,
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
        let stage = opts.stage.unwrap_or(Stage::whole(cfg.n_layer));
        if stage.first > stage.end || stage.end > cfg.n_layer || (stage.embed && stage.first != 0) || (!stage.embed && stage.head) {
            return Err(format!("{stage:?} is not a part of this {}-layer model a device can run", cfg.n_layer));
        }
        let tied = cfg.tied_embeddings;
        let mats_list: Vec<_> = model::matrices(&cfg).into_iter().filter(|m| stage.holds(&m.0, tied)).collect();
        let vec_list: Vec<_> = model::vectors(&cfg).into_iter().filter(|v| stage.holds(&v.0, tied)).collect();
        let vec_bytes: u64 = vec_list.iter().map(|v| v.1 as u64 * 4).sum();
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
            if !cfg.tied_embeddings && stage.embed {
                place.insert("token_embd.weight".into(), Place::Cpu);
                placement.cpu_embeddings = true;
            }
            let compact = |name: &str, rows: usize, k: usize| size_in(Layout::Blocks, name, rows, k);
            let mut pool_layers: Vec<usize> = Vec::new();
            if let Some(e) = &cfg.moe {
                for l in (0..cfg.n_layer).filter(|&l| cfg.is_moe(l) && stage.has_layer(l)) {
                    let names = ["gate", "up", "down"].map(|k| format!("blk.{l}.ffn_{k}_exps.weight"));
                    let types = [ty(&names[0]), ty(&names[1]), ty(&names[2])];
                    let w = cfg.expert_in();
                    let shape = [(e.n_ff, w), (e.n_ff, w), (w, e.n_ff)];
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
            let head = mats_list.iter().find(|m| m.0 == head_name && stage.head);
            let head_bytes = head.map_or(0, |h| compact(head_name, h.1 * h.3, h.2));
            let dense_total: u64 = dense.iter().map(|d| d.3).sum();
            if used + head_bytes + dense_total + pool_min <= budget {
                used += head_bytes + dense_total;
                placement.resident_layers = cfg.n_layer;
            } else {
                // Keep the head if it is small next to the budget; stream it otherwise.
                if head_bytes * 4 <= budget.saturating_sub(used) || head.is_none() {
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
        for (name, len) in vec_list {
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
            rope: layout(device, "uuwrrwwrr"),
            attention: layout(device, "uurrrw"),
            topk: layout(device, "uurrw"),
            mla: layout(device, "uurrrw"),
            res_mix: layout(device, "uurrww"),
            fill: layout(device, "uuw"),
            kda_conv: layout(device, "uurww"),
            l2norm: layout(device, "uuw"),
            kda: layout(device, "uurrrrrww"),
            mla_attn: layout(device, "uurrrwrr"),
            idx: layout(device, "uurrrw"),
            idx_topk: layout(device, "uurw"),
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
            recurrent_pos: Cell::new(0),
            snapshot_pos: Cell::new(None),
            all_logits: Cell::new(false),
            batch: Cell::new(FIRST_BATCH),
            rope_delta: Cell::new(0),
            rope_positions: std::cell::RefCell::new(None),
            spec_state: RefCell::new(Vec::new()),
            spec_pos: Cell::new(0),
            stage,
            remote: RefCell::new(None),
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
        let source = match self.cfg.situ {
            Some((b, l)) => source
                .replace("const SITU_B: f32 = 4.0;", &format!("const SITU_B: f32 = {b:?};"))
                .replace("const SITU_L: f32 = 25.0;", &format!("const SITU_L: f32 = {l:?};")),
            None => source,
        };
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
        if let Some(ef) = self.mats.get("token_embd.weight").map(|m| m.fmt) {
            self.pipeline(format!("embed-{}", ef.key()), shaders::embed(ef), &mm);
        }
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
            let ma = self.layouts.mla_attn.clone();
            self.pipeline("mla-store".into(), shaders::mla_store(), &ml);
            self.pipeline("mla-attention".into(), shaders::mla_attention(), &ma);
        }
        if self.cfg.indexer.is_some() {
            let (ix, it) = (self.layouts.idx.clone(), self.layouts.idx_topk.clone());
            self.pipeline("idx-store".into(), shaders::idx_store(), &ix);
            self.pipeline("idx-score".into(), shaders::idx_score(), &ix);
            self.pipeline("idx-topk".into(), shaders::idx_topk(), &it);
        }
        let l = &self.layouts;
        let (rm, fl, kc, l2, kd) = (l.res_mix.clone(), l.fill.clone(), l.kda_conv.clone(), l.l2norm.clone(), l.kda.clone());
        if self.cfg.attn_res_block > 0 {
            self.pipeline("res-mix".into(), shaders::res_mix(), &rm);
        }
        if self.cfg.expert_latent > 0 {
            self.pipeline("fill".into(), shaders::fill(), &fl);
        }
        if self.cfg.kda.is_some() {
            self.pipeline("kda-conv".into(), shaders::kda_conv(), &kc);
            self.pipeline("l2norm".into(), shaders::l2norm(), &l2);
            self.pipeline("kda".into(), shaders::kda(), &kd);
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
        let mut acts = 2 * cfg.n_embd + 2 * cfg.q_dim() + 2 * cfg.kv_dim() + cfg.n_ff.max(cfg.moe.as_ref().map_or(0, |e| e.n_ff_shared));
        if let Some(e) = &cfg.moe {
            acts += e.n_expert + 3 * e.n_used + e.n_used * e.n_ff + e.n_ff_shared + e.n_used * cfg.n_embd;
        }
        if let Some(m) = &cfg.mla {
            acts += m.q_lora + m.kv_lora + m.rope + 2 * cfg.n_head * m.kv_lora + cfg.n_head * m.v_dim;
        }
        let attn_layers = (0..cfg.n_layer).filter(|&l| !cfg.is_kda(l)).count() as u64;
        let kv = attn_layers * ctx as u64 * cfg.cache_dim() as u64 * 2;
        let rope = ctx as u64 * (cfg.rope_dim / 2) as u64 * 8;
        let mut extra = 0u64;
        if let Some(k) = &cfg.kda {
            let d = k.width();
            acts += 5 * d + k.head_dim + k.n_head;
            let layers = k.layers.iter().filter(|&&x| x).count() as u64;
            // State and convolution inputs, twice (the snapshot).
            extra += layers * (k.n_head * k.head_dim * k.head_dim + 3 * d * 3) as u64 * 4 * 2;
        }
        if cfg.attn_res_block > 0 {
            acts += cfg.n_embd * (1 + cfg.n_layer.div_ceil(cfg.attn_res_block));
        }
        acts += 2 * cfg.expert_latent;
        if let Some(i) = &cfg.indexer {
            acts += i.n_head * i.head_dim + i.head_dim + i.n_head + ctx + 1;
            let layers = (0..cfg.n_layer).filter(|&l| i.full[l] && !cfg.is_kda(l)).count() as u64;
            extra += layers * ctx as u64 * i.head_dim as u64 * 2;
        }
        t * acts as u64 * 4 + kv + rope + cfg.n_vocab as u64 * 8 + extra
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
        let kda = c.kda.clone();
        let kd = kda.as_ref().map_or(0, |k| k.width());
        let ein = c.expert_in();
        let latent = c.expert_latent;
        let bs = c.attn_res_block;
        let n_ckpt = if bs > 0 { c.n_layer.div_ceil(bs) } else { 0 };
        // A model that streams reads long prompts layer by layer, up to SUPER_BATCH tokens at once.
        let super_tokens = if self.placement.streaming { SUPER_BATCH.min(ctx.next_multiple_of(t)) } else { 0 };
        let ck_tokens = super_tokens.max(t);
        let act_mode = if c.situ.is_some() { 4 } else { 2 };
        let x = buf("x", t * c.n_embd);
        let xn = buf("xn", t * c.n_embd);
        let q = buf("q", t * c.q_dim());
        let k = buf("k", t * c.kv_dim());
        let v = buf("v", t * c.kv_dim());
        let att = buf("att", t * c.q_dim().max(v_width).max(kd));
        // Residuals across blocks: the mixed sub-layer input and the banked checkpoints.
        let xm = buf("x-mix", if bs > 0 { t * c.n_embd } else { 1 });
        // Indexed by the token's offset in a layer-by-layer prompt (step.chunk0), so every chunk
        // of it keeps its own checkpoints.
        let ck = buf("checkpoints", ck_tokens * n_ckpt.max(1) * c.n_embd);
        // Linear attention: q, k, v per token, the decay gate and its low-rank input, beta.
        let kqv = buf("kda-qkv", t * 3 * kd);
        let gk = buf("kda-gate", t * kd);
        let fa = buf("kda-gate-a", t * kda.as_ref().map_or(1, |k| k.head_dim));
        let betab = buf("kda-beta", t * kda.as_ref().map_or(1, |k| k.n_head));
        // Latent experts: their input and their summed output.
        let xl = buf("expert-in", t * latent.max(1));
        let ml = buf("expert-out", t * latent.max(1));
        // DSA indexer: queries, keys, head weights, per-layer key caches (f16), the scores of
        // every cached position and each token's selection threshold.
        let ix = c.indexer.clone();
        let (iqb, ikb, iwb, iscore, ithr) = match &ix {
            Some(i) => (
                buf("idx-q", t * i.n_head * i.head_dim),
                buf("idx-k", t * i.head_dim),
                buf("idx-w", t * i.n_head),
                buf("idx-score", t * ctx),
                buf("idx-thr", t),
            ),
            None => (buf("unused", 4), buf("unused", 4), buf("unused", 4), buf("unused", 4), buf("unused", 4)),
        };
        let icache: Vec<Option<wgpu::Buffer>> = (0..c.n_layer)
            .map(|l| ix.as_ref().filter(|i| i.full[l] && !c.is_kda(l) && self.stage.has_layer(l)).map(|i| buf("idx-cache", ctx * i.head_dim / 2)))
            .collect();
        let gate = buf("gate", t * ff_width);
        let xf = buf("xf", c.n_embd);
        let logits = buf("logits", c.n_vocab);
        let tokens = buf("tokens", t);
        let (kc, vc): (Vec<_>, Vec<_>) = (0..c.n_layer)
            .map(|l| match &mla {
                _ if c.is_kda(l) || !self.stage.has_layer(l) => (buf("unused", 4), buf("unused", 4)),
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
                buf("expert out", t * e.n_used * c.n_embd.max(ein)),
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
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Per-token (time, height, width) rotary positions of a chunk, when given (images).
        let mpos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multimodal positions"),
            size: (SUPER_BATCH.max(MAX_BATCH) * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        let mut kda_state = Vec::new();
        let mut kda_snap = Vec::new();
        if let Some(kk) = &kda {
            for (l, _) in kk.layers.iter().enumerate().filter(|(_, &x)| x) {
                // Layers another device runs keep their state there.
                let here = self.stage.has_layer(l);
                let pair = || match here {
                    true => (buf("kda-state", kk.n_head * kk.head_dim * kk.head_dim), buf("kda-conv", 3 * kd * 3)),
                    false => (buf("unused", 4), buf("unused", 4)),
                };
                kda_state.push(pair());
                kda_snap.push(pair());
            }
        }
        let mut ops: Vec<Op> = Vec::new();
        let dummy = &self.dummy;
        let tables = &self.tables;
        // Writable placeholders: a buffer cannot be bound read-only and writable in one dispatch.
        let (spare_a, spare_b) = (buf("unused", 4), buf("unused", 4));
        let route_w = (t * moe.as_ref().map_or(0, |e| e.n_used)) as u32;

        // y (+)= W x for every shard of a matrix; `out` places the rows in a wider output
        // (row stride per token, first row).
        #[allow(clippy::too_many_arguments)]
        let matmul_at = |ops: &mut Vec<Op>,
                         name: &str,
                         x: &wgpu::Buffer,
                         y: &wgpu::Buffer,
                         mode: u32,
                         bias: Option<&wgpu::Buffer>,
                         gemv: bool,
                         input: Input,
                         out: Option<(usize, usize)>| {
            let m = &self.mats[name];
            let (ostride, ooff) = out.unwrap_or((m.total_rows(), 0));
            for s in &m.shards {
                let p = params(&words(&[
                    m.k as u32,
                    s.rows as u32,
                    ostride as u32,
                    (ooff + s.row0) as u32,
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
        #[allow(clippy::too_many_arguments)]
        let matmul = |ops: &mut Vec<Op>,
                      name: &str,
                      x: &wgpu::Buffer,
                      y: &wgpu::Buffer,
                      mode: u32,
                      bias: Option<&wgpu::Buffer>,
                      gemv: bool,
                      input: Input| matmul_at(ops, name, x, y, mode, bias, gemv, input, None);
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
                        if down { ein } else { m.rows } as u32,
                        0,
                        mode,
                        0,
                        0,
                        if down { m.k } else { ein } as u32,
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

        // The embedding on the GPU (when this stage reads tokens and the table is resident).
        let emb = self.mats.get("token_embd.weight").filter(|m| self.stage.embed && m.place == Place::Resident);
        for (emb, s) in emb.iter().flat_map(|m| m.shards.iter().map(move |s| (*m, s))) {
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
            c.rope_sections.map_or(half, |s| s[0]) as u32,
            c.rope_sections.map_or(0, |s| s[1]) as u32,
            c.rope_sections.map_or(0, |s| s[2]) as u32,
            c.rope_interleaved as u32,
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
                sd.extend(words(&[!m.rotate as u32, 0, 0]));
                let mut ad = words(&[
                    c.n_head as u32,
                    m.kv_lora as u32,
                    m.rope as u32,
                    c.head_dim as u32,
                    m.nope as u32,
                    (m.kv_lora + m.rope) as u32,
                ]);
                ad.extend(c.kq_scale.to_le_bytes());
                ad.extend(words(&[mla_heads(&c) as u32, c.indexer.as_ref().map_or(0, |i| i.top_k) as u32, ctx as u32, 0, 0]));
                (params(&sd), params(&ad))
            }
            None => (params(&[0u8; 32]), params(&[0u8; 48])),
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
        // Mix the residual stream with the banked block checkpoints into `dst`, optionally
        // banking the stream as checkpoint `n_ckpt`.
        let res_mix = |ops: &mut Vec<Op>, w: &wgpu::Buffer, n_ckpt: usize, bank: bool, dst: &wgpu::Buffer| {
            let mut d = words(&[c.n_embd as u32, n_ckpt as u32, bank as u32]);
            d.extend(c.rms_eps.to_le_bytes());
            d.extend(words(&[ck_tokens as u32, 0, 0, 0]));
            ops.push(Op {
                when: When::Always,
                kernel: Kernel::ResMix,
                grid: Grid::Tokens,
                bind: bind(&self.layouts.res_mix, "res-mix", &[&step, &params(&d), &x, w, &ck, dst]),
            });
        };
        let embed_end = ops.len();
        let mut layers = Vec::with_capacity(c.n_layer);
        let mut kda_index = 0;
        for l in 0..c.n_layer {
            if !self.stage.has_layer(l) {
                // Another device runs this layer.
                if c.is_kda(l) {
                    kda_index += 1;
                }
                let at = ops.len();
                layers.push(LayerOps { start: at, route_end: at, experts_end: at, end: at });
                continue;
            }
            let n = |s: &str| format!("blk.{l}.{s}");
            let start = ops.len();
            let (mut route_end, mut experts_end) = (0, 0);
            // A layer that starts a block banks its input and restarts the stream from its
            // attention output.
            let banked = bs > 0 && l % bs == 0;
            let out_mode = if banked { 0 } else { 1 };
            if bs > 0 {
                let before = if l == 0 { 0 } else { (l - 1) / bs + 1 };
                res_mix(&mut ops, &self.vecs[&n("attn_res_score.weight")], before, banked, &xm);
                norm(&mut ops, &xm, &self.vecs[&n("attn_norm.weight")], &xn, c.n_embd, false);
            } else {
                norm(&mut ops, &x, &self.vecs[&n("attn_norm.weight")], &xn, c.n_embd, false);
            }
            if c.is_kda(l) {
                let kk = kda.as_ref().unwrap();
                let input = Input::rows(c.n_embd);
                for (j, w) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
                    matmul_at(&mut ops, &n(&format!("{w}.weight")), &xn, &kqv, 0, None, false, input, Some((3 * kd, j * kd)));
                }
                let (state, conv) = &kda_state[kda_index];
                kda_index += 1;
                for (j, w) in ["q", "k", "v"].iter().enumerate() {
                    let p = params(&words(&[kd as u32, (3 * kd) as u32, (j * kd) as u32, (j * kd) as u32]));
                    ops.push(Op {
                        when: When::Always,
                        kernel: Kernel::KdaConv,
                        grid: Grid::Groups(kd.div_ceil(256) as u32),
                        bind: bind(&self.layouts.kda_conv, "kda-conv", &[&step, &p, &self.vecs[&n(&format!("ssm_conv1d_{w}.weight"))], &kqv, conv]),
                    });
                }
                let mut d = words(&[kk.head_dim as u32, (2 * kk.n_head) as u32, (3 * kd) as u32]);
                d.extend(c.rms_eps.to_le_bytes());
                ops.push(Op {
                    when: When::Always,
                    kernel: Kernel::L2Norm,
                    grid: Grid::Heads((2 * kk.n_head) as u32),
                    bind: bind(&self.layouts.l2norm, "l2norm", &[&step, &params(&d), &kqv]),
                });
                matmul(&mut ops, &n("ssm_f_a.weight"), &xn, &fa, 0, None, false, input);
                matmul(&mut ops, &n("ssm_f_b.weight"), &fa, &gk, 0, None, false, Input::rows(kk.head_dim));
                matmul(&mut ops, &n("ssm_beta.weight"), &xn, &betab, 0, None, false, input);
                let mut d = words(&[kk.n_head as u32, kd as u32, (3 * kd) as u32]);
                d.extend((1.0 / (kk.head_dim as f32).sqrt()).to_le_bytes());
                d.extend(kk.gate_lower_bound.to_le_bytes());
                d.extend(words(&[0, 0, 0]));
                ops.push(Op {
                    when: When::Always,
                    kernel: Kernel::Kda,
                    grid: Grid::Groups((2 * kk.n_head) as u32),
                    bind: bind(
                        &self.layouts.kda,
                        "kda",
                        &[&step, &params(&d), &kqv, &gk, &betab, &self.vecs[&n("ssm_dt.bias")], &self.vecs[&n("ssm_a")], state, &att],
                    ),
                });
                // Per-head RMSNorm, then the sigmoid output gate.
                norm_inplace(&mut ops, &att, &self.vecs[&n("ssm_norm.weight")], kk.head_dim, kk.n_head);
                matmul(&mut ops, &n("ssm_g.weight"), &xn, &att, 5, None, false, input);
                matmul(&mut ops, &n("attn_output.weight"), &att, &x, out_mode, None, false, Input::rows(kd));
            } else if let Some(m) = &mla {
                if m.q_lora > 0 {
                    matmul(&mut ops, &n("attn_q_a.weight"), &xn, &qa, 0, None, false, Input::rows(c.n_embd));
                    norm_inplace(&mut ops, &qa, &self.vecs[&n("attn_q_a_norm.weight")], m.q_lora, 1);
                    matmul(&mut ops, &n("attn_q_b.weight"), &qa, &q, 0, None, false, Input::rows(m.q_lora));
                } else {
                    matmul(&mut ops, &n("attn_q.weight"), &xn, &q, 0, None, false, Input::rows(c.n_embd));
                }
                matmul(&mut ops, &n("attn_kv_a_mqa.weight"), &xn, &kva, 0, None, false, Input::rows(c.n_embd));
                // DSA: score every cached position and pick each token's top_k (layers without
                // their own indexer reuse the last selection).
                if let (Some(i), Some(cache)) = (&ix, &icache[l]) {
                    matmul(&mut ops, &n("indexer.attn_q_b.weight"), &qa, &iqb, 0, None, false, Input::rows(m.q_lora));
                    matmul(&mut ops, &n("indexer.attn_k.weight"), &xn, &ikb, 0, None, false, Input::rows(c.n_embd));
                    matmul(&mut ops, &n("indexer.proj.weight"), &xn, &iwb, 0, None, false, Input::rows(c.n_embd));
                    let mut d = words(&[i.head_dim as u32]);
                    d.extend(i.eps.to_le_bytes());
                    d.extend(words(&[0, 0]));
                    ops.push(Op {
                        when: When::Always,
                        kernel: Kernel::IdxStore,
                        grid: Grid::Tokens,
                        bind: bind(&self.layouts.idx, "idx-store", &[&step, &params(&d), &ikb, &self.vecs[&n("indexer.k_norm.weight")], &self.vecs[&n("indexer.k_norm.bias")], cache]),
                    });
                    let mut d = words(&[i.n_head as u32, i.head_dim as u32, ctx as u32, i.top_k as u32]);
                    d.extend((1.0 / ((i.head_dim * i.n_head) as f32).sqrt()).to_le_bytes());
                    d.extend(words(&[0, 0, 0]));
                    ops.push(Op {
                        when: When::Always,
                        kernel: Kernel::IdxScore,
                        grid: Grid::KeyBlocks(ctx.div_ceil(256) as u32),
                        bind: bind(&self.layouts.idx, "idx-score", &[&step, &params(&d), &iqb, &iwb, cache, &iscore]),
                    });
                    ops.push(Op {
                        when: When::Always,
                        kernel: Kernel::IdxTopK,
                        grid: Grid::Tokens,
                        bind: bind(&self.layouts.idx_topk, "idx-topk", &[&step, &params(&words(&[ctx as u32, i.top_k as u32, 0, 0])), &iscore, &ithr]),
                    });
                }
                // Rotate the tail of every query head (the key's rotated part goes to the cache).
                if m.rotate {
                    ops.push(Op {
                        when: When::Always,
                        kernel: Kernel::Rope,
                        grid: Grid::Heads(c.n_head as u32),
                        bind: bind(&self.layouts.rope, "rope-q", &[&step, &rope_p, &q, dummy, dummy, &spare_a, &spare_b, &rope_table, &mpos]),
                    });
                }
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
                    bind: bind(&self.layouts.mla_attn, "mla-attention", &[&step, &mla_attn_p, &qlat, &q, &kc[l], &olat, &iscore, &ithr]),
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
                if c.attn_gate {
                    matmul(&mut ops, &n("attn_gate.weight"), &xn, &att, 5, None, false, Input::rows(c.n_embd));
                }
                matmul(&mut ops, &n("attn_output.weight"), &att, &x, out_mode, None, false, Input::rows(v_width));
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
                    bind: bind(&self.layouts.rope, "rope", &[&step, &rope_p, &q, &k, &v, &kc[l], &vc[l], &rope_table, &mpos]),
                });
                ops.push(Op {
                    when: When::Always,
                    kernel: Kernel::Attention,
                    grid: Grid::Heads(c.n_head as u32),
                    bind: bind(&self.layouts.attention, "attention", &[&step, &attn_p, &q, &kc[l], &vc[l], &att]),
                });
                matmul(&mut ops, &n("attn_output.weight"), &att, &x, out_mode, None, false, Input::rows(c.q_dim()));
            }
            if bs > 0 {
                res_mix(&mut ops, &self.vecs[&n("ffn_res_score.weight")], l / bs + 1, false, &xm);
                norm(&mut ops, &xm, &self.vecs[&n("ffn_norm.weight")], &xn, c.n_embd, false);
            } else {
                norm(&mut ops, &x, &self.vecs[&n("ffn_norm.weight")], &xn, c.n_embd, false);
            }
            let input = Input::rows(c.n_embd);
            match moe.as_ref().filter(|_| c.is_moe(l)) {
                Some(e) => {
                    // Latent experts read a down-projection of the input and write a sum that is
                    // normalised and projected back up.
                    if latent > 0 {
                        matmul(&mut ops, &n("ffn_routed_down.weight"), &xn, &xl, 0, None, false, input);
                        ops.push(Op {
                            when: When::Always,
                            kernel: Kernel::Fill,
                            grid: Grid::Width(latent as u32),
                            bind: bind(&self.layouts.fill, "fill", &[&step, &params(&words(&[latent as u32, 0, 0, 0])), &ml]),
                        });
                    }
                    let (xin, target) = if latent > 0 { (&xl, &ml) } else { (&xn, &x) };
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
                    matmul_id(&mut ops, l, &n("ffn_gate_exps.weight"), xin, &hexp, 0, false);
                    matmul_id(&mut ops, l, &n("ffn_up_exps.weight"), xin, &hexp, act_mode, false);
                    matmul_id(&mut ops, l, &n("ffn_down_exps.weight"), &hexp, target, 1, true);
                    let sum_p = params(&words(&[ein as u32, e.n_used as u32, route_w, (l * e.n_expert) as u32]));
                    ops.push(Op {
                        when: When::Prompt,
                        kernel: Kernel::MoeSum,
                        grid: Grid::Width(ein as u32),
                        bind: bind(&self.layouts.moe_sum, "moe-sum", &[&step, &sum_p, &route, &emap, &dout, target]),
                    });
                    experts_end = ops.len();
                    if latent > 0 {
                        if let Some(w) = self.vecs.get(&n("ffn_routed_norm.weight")) {
                            norm_inplace(&mut ops, &ml, w, latent, 1);
                        }
                        matmul(&mut ops, &n("ffn_routed_up.weight"), &ml, &x, 1, None, false, Input::rows(latent));
                    }
                    if e.n_ff_shared > 0 {
                        matmul(&mut ops, &n("ffn_gate_shexp.weight"), &xn, &gate, 0, None, false, input);
                        matmul(&mut ops, &n("ffn_up_shexp.weight"), &xn, &gate, act_mode, None, false, input);
                        matmul(&mut ops, &n("ffn_down_shexp.weight"), &gate, &x, 1, None, false, Input::rows(e.n_ff_shared));
                    }
                }
                None => {
                    matmul(&mut ops, &n("ffn_gate.weight"), &xn, &gate, 0, None, false, input);
                    matmul(&mut ops, &n("ffn_up.weight"), &xn, &gate, act_mode, None, false, input);
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
        let head = if c.tied_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        // Only the stage that makes logits has the head.
        if self.stage.head {
            if bs > 0 {
                res_mix(&mut ops, &self.vecs["output_res_score.weight"], n_ckpt, false, &xm);
                norm(&mut ops, &xm, &self.vecs["output_norm.weight"], &xf, c.n_embd, true);
            } else {
                norm(&mut ops, &x, &self.vecs["output_norm.weight"], &xf, c.n_embd, true);
            }
            matmul(&mut ops, head, &xf, &logits, 0, None, true, Input::rows(c.n_embd));
        }
        // The same head for every token of a short batch: verifying a speculative draft needs
        // the logits after each of its tokens.
        let head_end = ops.len();
        let xfa = buf("xf-all", t * c.n_embd);
        let logits_all = buf("logits-all", SPEC_MAX * c.n_vocab);
        let staging_all = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-all"),
            size: (SPEC_MAX * c.n_vocab * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if self.stage.head {
            if bs > 0 {
                res_mix(&mut ops, &self.vecs["output_res_score.weight"], n_ckpt, false, &xm);
                norm(&mut ops, &xm, &self.vecs["output_norm.weight"], &xfa, c.n_embd, false);
            } else {
                norm(&mut ops, &x, &self.vecs["output_norm.weight"], &xfa, c.n_embd, false);
            }
            matmul(&mut ops, head, &xfa, &logits_all, 0, None, false, Input::rows(c.n_embd));
        }

        let route_bytes = route.size();
        let sup = (super_tokens > 0).then(|| {
            let chunks = super_tokens / t;
            Super {
                tokens: super_tokens,
                x: buf("prompt x", super_tokens * c.n_embd),
                xn: buf("prompt xn", super_tokens * c.n_embd),
                xl: buf("prompt expert-in", super_tokens * latent.max(1)),
                ml: buf("prompt expert-out", super_tokens * latent.max(1)),
                route: buf("prompt routes", chunks * route_bytes as usize / 4),
                route_staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("prompt routes staging"),
                    size: chunks as u64 * route_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            }
        });
        let mut bytes = Self::state_bytes(&c, ctx);
        if let Some(s) = &sup {
            bytes += s.x.size() + s.xn.size() + s.xl.size() + s.ml.size() + s.route.size() + (ck_tokens - t) as u64 * (n_ckpt * c.n_embd * 4) as u64;
        }
        let route_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("route staging"),
            size: route.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A stage hands its hidden state on a chunk at a time: x and every checkpoint.
        let hidden_staging = (self.stage != Stage::whole(c.n_layer)).then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hidden staging"),
                size: (t * c.n_embd * (1 + n_ckpt) * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
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
            kda: kda_state,
            kda_snap,
            head_end,
            logits_all,
            staging_all,
            xn,
            xl,
            ml,
            sup,
            ck,
            ck_tokens,
            hidden_staging,
            mpos,
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
            (Kernel::ResMix, Grid::Tokens) => ("res-mix".into(), (t, 1)),
            (Kernel::Fill, Grid::Width(w)) => ("fill".into(), (w.div_ceil(256), t)),
            (Kernel::KdaConv, Grid::Groups(n)) => ("kda-conv".into(), (n, 1)),
            (Kernel::L2Norm, Grid::Heads(h)) => ("l2norm".into(), (h, t)),
            (Kernel::Kda, Grid::Groups(n)) => ("kda".into(), (n, 1)),
            (Kernel::IdxStore, Grid::Tokens) => ("idx-store".into(), (t, 1)),
            (Kernel::IdxScore, Grid::KeyBlocks(n)) => ("idx-score".into(), (n, t)),
            (Kernel::IdxTopK, Grid::Tokens) => ("idx-topk".into(), (t, 1)),
            _ => unreachable!(),
        }
    }

    fn encode_chunk(&self, n_tokens: usize, with_head: bool) -> wgpu::CommandBuffer {
        let st = self.state.as_ref().unwrap();
        if !with_head {
            return self.encode_range(0..st.head_start, n_tokens, false);
        }
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.record(&mut enc, 0..st.head_start, n_tokens);
        self.record_head(&mut enc, n_tokens);
        enc.finish()
    }

    /// The output head after the last layer, and the copy of its logits for reading back: the
    /// last token's, or every token's while verifying a draft.
    fn record_head(&self, enc: &mut wgpu::CommandEncoder, n_tokens: usize) {
        let st = self.state.as_ref().unwrap();
        if self.all_logits.get() {
            self.record(enc, st.head_end..st.ops.len(), n_tokens);
            enc.copy_buffer_to_buffer(&st.logits_all, 0, &st.staging_all, 0, (n_tokens * self.cfg.n_vocab * 4) as u64);
        } else {
            self.record(enc, st.head_start..st.head_end, n_tokens);
            enc.copy_buffer_to_buffer(&st.logits, 0, &st.staging, 0, st.staging.size());
        }
    }

    /// Logits after each of `tokens` (at most SPEC_MAX), for verifying a speculative draft.
    pub async fn forward_all(&self, tokens: &[u32], pos0: usize) -> Result<Vec<Vec<f32>>, String> {
        if tokens.is_empty() || tokens.len() > SPEC_MAX {
            return Err(format!("a verification batch holds 1 to {SPEC_MAX} tokens"));
        }
        self.all_logits.set(true);
        let out = self.forward(tokens, pos0).await;
        self.all_logits.set(false);
        out?;
        let st = self.state.as_ref().unwrap();
        let v = self.cfg.n_vocab;
        let bytes = read_back(&self.gpu, &st.staging_all, (tokens.len() * v * 4) as u64).await?;
        let all: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        Ok(all.chunks(v).map(|r| r.to_vec()).collect())
    }

    /// Keep a copy of the recurrent state before verifying a draft.
    pub fn spec_save(&self) {
        let Some(st) = self.state.as_ref() else { return };
        let mut spare = self.spec_state.borrow_mut();
        if spare.is_empty() {
            let make = |b: &wgpu::Buffer| {
                self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("kda-spec"),
                    size: b.size(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            *spare = st.kda.iter().map(|(a, b)| (make(a), make(b))).collect();
        }
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        for (live, copy) in st.kda.iter().zip(spare.iter()) {
            enc.copy_buffer_to_buffer(&live.0, 0, &copy.0, 0, live.0.size());
            enc.copy_buffer_to_buffer(&live.1, 0, &copy.1, 0, live.1.size());
        }
        self.gpu.queue.submit([enc.finish()]);
        self.spec_pos.set(self.recurrent_pos.get());
    }

    /// Go back to the state `spec_save` kept (a draft was only partly accepted).
    pub fn spec_restore(&self) {
        let Some(st) = self.state.as_ref() else { return };
        let spare = self.spec_state.borrow();
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        for (live, copy) in st.kda.iter().zip(spare.iter()) {
            enc.copy_buffer_to_buffer(&copy.0, 0, &live.0, 0, live.0.size());
            enc.copy_buffer_to_buffer(&copy.1, 0, &live.1, 0, live.1.size());
        }
        self.gpu.queue.submit([enc.finish()]);
        self.recurrent_pos.set(self.spec_pos.get());
    }

    /// True when weights are read from storage (or the network) while the model runs.
    pub fn streaming(&self) -> bool {
        self.placement.streaming
    }

    /// The per-step uniform: tokens in this chunk, their first position, and the chunk's first
    /// token within a prompt read layer by layer.
    fn write_step(&self, n: usize, pos: usize, chunk0: usize) {
        let st = self.state.as_ref().unwrap();
        // Multimodal rotary positions for these tokens, when this pass was given them.
        let mut mode = 0u32;
        if let Some((p0, ref list)) = *self.rope_positions.borrow() {
            if pos >= p0 && (pos - p0 + n) * 3 <= list.len() {
                let i = (pos - p0) * 3;
                self.gpu.queue.write_buffer(&st.mpos, 0, &words(&list[i..i + n * 3]));
                mode = 1;
            }
        }
        let n = n as u32;
        let pos = pos as u32;
        self.gpu.queue.write_buffer(&st.step, 0, &words(&[n, pos, pos + n, n - 1, chunk0 as u32, mode, self.rope_delta.get() as u32, 0]));
    }

    /// Rotary positions minus cache slots for the tokens that follow (Qwen2-VL: an image of n
    /// tokens fills n slots but moves positions on by its larger side only).
    pub fn set_rope_delta(&self, delta: i32) {
        self.rope_delta.set(delta);
    }

    /// Like `forward_hidden`, with each token's (time, height, width) rotary positions.
    pub async fn forward_hidden_positioned(&self, hidden: &[f32], n: usize, pos0: usize, positions: &[[u32; 3]]) -> Result<Vec<f32>, String> {
        if positions.len() != n {
            return Err(format!("{} positions for {n} tokens", positions.len()));
        }
        *self.rope_positions.borrow_mut() = Some((pos0, positions.iter().flatten().copied().collect()));
        let out = self.forward_hidden(hidden, n, pos0).await;
        *self.rope_positions.borrow_mut() = None;
        out
    }

    /// Record ops into an encoder (with copies around them, for the layer-by-layer prompt).
    fn record(&self, enc: &mut wgpu::CommandEncoder, range: std::ops::Range<usize>, n_tokens: usize) {
        let st = self.state.as_ref().unwrap();
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

    fn encode_range(&self, range: std::ops::Range<usize>, n_tokens: usize, copy_logits: bool) -> wgpu::CommandBuffer {
        let st = self.state.as_ref().unwrap();
        let with_head = copy_logits;
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        self.record(&mut enc, range, n_tokens);
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
        self.write_step(n_tokens, pos0, 0);
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
    /// True when the model keeps a recurrent state, which can only move forward.
    pub fn is_recurrent(&self) -> bool {
        self.cfg.kda.is_some()
    }

    /// Copy the recurrent state, to come back to this position later.
    pub fn snapshot(&self) {
        self.copy_state(true);
        self.snapshot_pos.set(Some(self.recurrent_pos.get()));
        if let Some(r) = self.remote.borrow().clone() {
            r.snapshot();
        }
    }

    /// Return to the snapshot; the position it was taken at, if there is one.
    pub fn restore(&self) -> Option<usize> {
        let pos = self.snapshot_pos.get()?;
        self.copy_state(false);
        self.recurrent_pos.set(pos);
        if let Some(r) = self.remote.borrow().clone() {
            r.restore();
        }
        Some(pos)
    }

    fn copy_state(&self, save: bool) {
        let Some(st) = self.state.as_ref() else { return };
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        for (live, snap) in st.kda.iter().zip(&st.kda_snap) {
            let (from, to) = if save { (live, snap) } else { (snap, live) };
            enc.copy_buffer_to_buffer(&from.0, 0, &to.0, 0, from.0.size());
            enc.copy_buffer_to_buffer(&from.1, 0, &to.1, 0, from.1.size());
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    /// Logits after reading `tokens` at positions from `pos0`. A model with a recurrent state
    /// must continue where it stopped, or start over at 0. A stage without the head returns
    /// the hidden state after its last layer instead (see [`Engine::hidden_width`]).
    pub async fn forward(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, String> {
        if !self.stage.embed {
            return Err("this device runs layers in the middle of the model; it reads hidden states, not tokens".into());
        }
        self.forward_input(Feed::Tokens(tokens), pos0).await
    }

    /// For a stage that runs layers after another device's: read the hidden state that device
    /// handed on (`n` tokens from `pos0`) and return the one after this stage's last layer.
    pub async fn forward_hidden(&self, hidden: &[f32], n: usize, pos0: usize) -> Result<Vec<f32>, String> {
        if self.stage.embed {
            return Err("this device reads tokens, not hidden states".into());
        }
        if n == 0 || hidden.len() != n * self.hidden_width(self.stage.first) {
            return Err(format!("expected the hidden state of {n} tokens before layer {}", self.stage.first));
        }
        self.forward_input(Feed::Hidden(hidden, n), pos0).await
    }

    async fn forward_input(&self, input: Feed<'_>, pos0: usize) -> Result<Vec<f32>, String> {
        if self.is_recurrent() && pos0 != 0 && pos0 != self.recurrent_pos.get() {
            return Err(format!(
                "this model's recurrent state is at position {}, not {pos0}",
                self.recurrent_pos.get()
            ));
        }
        let n = input.len();
        let out = self.forward_at(input, pos0).await;
        match &out {
            Ok(_) => self.recurrent_pos.set(pos0 + n),
            Err(_) => {
                // A pass that stopped part way (a cancelled or failed read) leaves some layers'
                // recurrent state ahead of the others, and expert slots that may not hold what
                // the cache thinks: start both again.
                self.recurrent_pos.set(usize::MAX);
                let mut st = self.stream.borrow_mut();
                for lru in &mut st.lru {
                    lru.owner.iter_mut().for_each(|o| *o = None);
                    lru.used.iter_mut().for_each(|u| *u = 0);
                    lru.slot_of.clear();
                }
            }
        }
        out
    }

    /// Tokens the recurrent state has read (usize::MAX after a pass that did not finish).
    pub fn recurrent_position(&self) -> usize {
        self.recurrent_pos.get()
    }

    /// Block checkpoints that exist before layer `l` (models with residuals across blocks).
    fn ckpts_before(&self, l: usize) -> usize {
        match self.cfg.attn_res_block {
            0 => 0,
            bs => l.div_ceil(bs),
        }
    }

    /// Floats per token of the hidden state handed from one device to the next before layer
    /// `l`: the residual stream, then each block checkpoint made so far. Tokens follow each
    /// other, so a batch can be cut or joined anywhere.
    pub fn hidden_width(&self, l: usize) -> usize {
        self.cfg.n_embd * (1 + self.ckpts_before(l))
    }

    /// Who runs the layers after this engine's, when it holds the head but not every layer.
    pub fn set_remote(&self, remote: Option<std::rc::Rc<dyn Remote>>) {
        *self.remote.borrow_mut() = remote;
    }

    pub fn has_remote(&self) -> bool {
        self.remote.borrow().is_some()
    }

    /// Write `n` tokens of a hidden state with `k` checkpoints: the residual stream into `xbuf`
    /// from token `x_tok`, the checkpoints at chunk offset `c0`.
    fn write_hidden(&self, h: &[f32], n: usize, k: usize, xbuf: &wgpu::Buffer, x_tok: usize, c0: usize) {
        let st = self.state.as_ref().unwrap();
        let ne = self.cfg.n_embd;
        let w = ne * (1 + k);
        for j in 0..=k {
            let mut part = Vec::with_capacity(n * ne);
            for t in 0..n {
                part.extend_from_slice(&h[t * w + j * ne..t * w + (j + 1) * ne]);
            }
            let (buf, at) = match j {
                0 => (xbuf, x_tok),
                _ => (&st.ck, (j - 1) * st.ck_tokens + c0),
            };
            self.gpu.queue.write_buffer(buf, (at * ne * 4) as u64, &floats(&part));
        }
    }

    /// Read `n` tokens of the hidden state back with `k` checkpoints (the reverse of
    /// `write_hidden`), a staging buffer's worth at a time.
    async fn read_hidden(&self, n: usize, k: usize, xbuf: &wgpu::Buffer, x_tok: usize, c0: usize) -> Result<Vec<f32>, String> {
        let st = self.state.as_ref().unwrap();
        let staging = st.hidden_staging.as_ref().ok_or("this engine runs the whole model")?;
        let ne = self.cfg.n_embd;
        let w = ne * (1 + k);
        let cap = (staging.size() as usize / 4 / ne / (1 + self.ckpts_before(self.cfg.n_layer))).max(1);
        let mut out = vec![0f32; n * w];
        let mut s = 0;
        while s < n {
            let m = cap.min(n - s);
            let mut e = self.gpu.device.create_command_encoder(&Default::default());
            let len = (m * ne * 4) as u64;
            e.copy_buffer_to_buffer(xbuf, ((x_tok + s) * ne * 4) as u64, staging, 0, len);
            for j in 0..k {
                e.copy_buffer_to_buffer(&st.ck, ((j * st.ck_tokens + c0 + s) * ne * 4) as u64, staging, (j + 1) as u64 * len, len);
            }
            self.gpu.queue.submit([e.finish()]);
            let bytes = read_back(&self.gpu, staging, (1 + k) as u64 * len).await?;
            for j in 0..=k {
                for t in 0..m {
                    let src = ((j * m + t) * ne) * 4;
                    let dst = (s + t) * w + j * ne;
                    for (o, b) in out[dst..dst + ne].iter_mut().zip(bytes[src..src + ne * 4].chunks_exact(4)) {
                        *o = f32::from_le_bytes(b.try_into().unwrap());
                    }
                }
            }
            s += m;
        }
        Ok(out)
    }

    /// The remote for a stage that holds the head but not every layer, or an error when none
    /// is connected.
    fn remote_for_pass(&self) -> Result<Option<std::rc::Rc<dyn Remote>>, String> {
        let remote = self.remote.borrow().clone();
        if self.stage.head && self.stage.end < self.cfg.n_layer && remote.is_none() {
            return Err(format!(
                "layers {} to {} run on another device, and none is connected",
                self.stage.end + 1,
                self.cfg.n_layer
            ));
        }
        Ok(remote)
    }

    /// The other devices' layers for the whole batch, then the hidden state of its last chunk
    /// (`n` tokens at batch offset `s`, checkpoints at chunk offset `c0`) back in place for the
    /// head.
    async fn through_remote(&self, remote: &dyn Remote, hidden: Vec<f32>, total: usize, pos0: usize, s: usize, n: usize, c0: usize) -> Result<(), String> {
        let st = self.state.as_ref().unwrap();
        let back = remote.forward(hidden, total, pos0).await?;
        let w = self.hidden_width(self.cfg.n_layer);
        if back.len() != total * w {
            return Err("another device returned a hidden state of the wrong size".into());
        }
        self.write_hidden(&back[s * w..(s + n) * w], n, self.ckpts_before(self.cfg.n_layer), &st.x, 0, c0);
        Ok(())
    }

    async fn forward_at(&self, input: Feed<'_>, pos0: usize) -> Result<Vec<f32>, String> {
        if self.placement.streaming {
            return self.forward_streamed(input, pos0).await;
        }
        let st = self.state.as_ref().ok_or("model is not loaded")?;
        let total = input.len();
        if total == 0 {
            return Err("nothing to process".into());
        }
        if pos0 + total > st.ctx {
            return Err(format!(
                "the conversation is longer than the {}-token context",
                st.ctx
            ));
        }
        let remote = self.remote_for_pass()?;
        let hand_on = remote.is_some() || !self.stage.head;
        if hand_on && self.all_logits.get() {
            return Err("drafts are not verified while other devices run part of the model".into());
        }
        let (k_in, k_out) = (self.ckpts_before(self.stage.first), self.ckpts_before(self.stage.end));
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut hidden = Vec::new();
        // With later layers on another device (or this machine's other GPU), each chunk goes on
        // as soon as this GPU has it, so both work at once: this GPU reads chunk i+1 while the
        // other runs chunk i. The other device takes chunks in the order sent.
        let mut sent: Vec<std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>, String>> + '_>>> = Vec::new();
        // Each pass starts small (a smaller size a slow device found is kept), as it may begin far
        // deeper into the context than the last pass ended.
        self.batch.set(self.batch.get().min(FIRST_BATCH));
        let mut s = 0;
        while s < total {
            let n = self.batch.get().clamp(MIN_BATCH, MAX_BATCH).min(total - s);
            let last = s + n == total;
            chunks.push((s, n));
            let started = super::now_ms();
            self.write_step(n, pos0 + s, 0);
            match input.slice(s, n) {
                Feed::Tokens(t) => self.gpu.queue.write_buffer(&st.tokens, 0, &words(t)),
                Feed::Hidden(h, _) => self.write_hidden(h, n, k_in, &st.x, 0, 0),
            }
            let cmd = self.encode_chunk(n, last && !hand_on);
            self.gpu.queue.submit([cmd]);
            if hand_on {
                let h = self.read_hidden(n, k_out, &st.x, 0, 0).await?;
                match &remote {
                    Some(r) if self.stage.head => {
                        let mut call = r.forward(h, n, pos0 + s);
                        // Started now (a call begins when first polled), awaited after the loop. A call
                        // that finishes on that first poll (a remote in this page) keeps its result: a
                        // finished future must not be polled again.
                        match call.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
                            std::task::Poll::Ready(v) => sent.push(Box::pin(std::future::ready(v))),
                            std::task::Poll::Pending => sent.push(call),
                        }
                    }
                    _ => hidden.extend(h),
                }
            } else if !last {
                // One chunk at a time: drivers cancel queued work that waits too long (Linux i915
                // after about 10 s), and a long prompt on a slow GPU queues far more than that.
                super::wait_idle(&self.gpu).await?;
            }
            // The next chunk's size from this one's time (each chunk sees a longer context).
            if !last && n == self.batch.get().min(MAX_BATCH) {
                let ms = super::now_ms() - started;
                if ms > SLOW_CHUNK_MS && n > MIN_BATCH {
                    self.batch.set((n / 2).max(MIN_BATCH));
                } else if ms < SLOW_CHUNK_MS / 3.0 && n < MAX_BATCH {
                    self.batch.set((n * 2).min(MAX_BATCH));
                }
            }
            s += n;
        }
        if !self.stage.head {
            return Ok(hidden);
        }
        if remote.is_some() {
            let &(s, n) = chunks.last().unwrap();
            // The last chunk's state after every later layer comes back for the head.
            let mut back = Vec::new();
            for call in sent {
                back = call.await?;
            }
            if back.len() != n * self.hidden_width(self.cfg.n_layer) {
                return Err("another device returned a hidden state of the wrong size".into());
            }
            self.write_hidden(&back, n, self.ckpts_before(self.cfg.n_layer), &st.x, 0, 0);
            self.write_step(n, pos0 + s, 0);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.record_head(&mut enc, n);
            self.gpu.queue.submit([enc.finish()]);
        }
        if self.all_logits.get() {
            return Ok(Vec::new()); // forward_all reads every token's logits
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
        *self.source.borrow_mut() = Some(source.into());
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

    /// Let the weight source fetch what the next reads need, together (it matters when the
    /// weights come over the network: many requests in flight instead of one at a time).
    async fn prefetch(&self, spans: &[Span]) -> Result<(), String> {
        if spans.is_empty() {
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let src = self.source.borrow().clone();
        let src = src.ok_or("this model streams weights but has no weight source")?;
        src.prefetch(spans).await?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.stream.borrow_mut().stats.read_seconds += t0.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Row slices of a ring matrix, as `upload_rings` reads them.
    fn ring_slices(&self, name: &str) -> Vec<(usize, usize)> {
        let tp = self.tensor(name);
        // About 32 MB each, so a large head never needs one huge buffer.
        let step = ((32u64 << 20) / tp.row_bytes).max(1) as usize;
        let total = self.mats[name].total_rows();
        (0..total).step_by(step).map(|r| (r, step.min(total - r))).collect()
    }

    fn ring_spans(&self, names: &[String]) -> Vec<Span> {
        let mut out = Vec::new();
        for name in names {
            let tp = self.tensor(name);
            for (r, n) in self.ring_slices(name) {
                out.push(Span {
                    file: tp.file,
                    offset: tp.offset + r as u64 * tp.row_bytes,
                    len: n * tp.row_bytes as usize,
                    every_token: true,
                });
            }
        }
        out
    }

    /// What `load_expert` reads for each expert of layer `l`.
    fn expert_spans(&self, p: usize, l: usize, experts: &[u32]) -> Vec<Span> {
        let pool = &self.pools[p];
        let mut out = Vec::new();
        for &e in experts {
            for (i, kind) in ["gate", "up", "down"].iter().enumerate() {
                let tp = self.tensor(&format!("blk.{l}.ffn_{kind}_exps.weight"));
                let len = pool.shape[i].0 * tp.row_bytes as usize;
                out.push(Span {
                    file: tp.file,
                    offset: tp.offset + e as u64 * len as u64,
                    len,
                    every_token: false,
                });
            }
        }
        out
    }

    fn embed_spans(&self, tokens: &[u32]) -> Vec<Span> {
        let tp = self.tensor("token_embd.weight");
        let mut seen = std::collections::HashSet::new();
        tokens
            .iter()
            .filter(|&&t| (t as u64) < tp.rows && seen.insert(t))
            .map(|&t| Span {
                file: tp.file,
                offset: tp.offset + t as u64 * tp.row_bytes,
                len: tp.row_bytes as usize,
                every_token: false,
            })
            .collect()
    }

    /// Read whole ring matrices into their slots before the ops that use them.
    fn upload_rings(&self, names: &[String]) -> Result<u64, String> {
        let mut bytes = 0;
        for name in names {
            let m = &self.mats[name];
            let tp = self.tensor(name).clone();
            for (r, n) in self.ring_slices(name) {
                let raw = self.read(&tp, r as u64 * tp.row_bytes, n * tp.row_bytes as usize)?;
                #[cfg(not(target_arch = "wasm32"))]
                let t0 = std::time::Instant::now();
                let w = quant::to_gpu_bytes(m.ggml_type, &raw, m.k, n)?;
                self.gpu.queue.write_buffer(&m.shards[0].codes, m.fmt.bytes(m.k, r), &w);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.stream.borrow_mut().stats.upload_seconds += t0.elapsed().as_secs_f64();
                }
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

    /// A prompt of several chunks for a model that streams, read layer by layer: each streamed
    /// weight is read once for all of its chunks rather than once per chunk. In expert layers,
    /// every chunk's attention and routing run first; then each expert-cache pass loads its
    /// experts once and runs every chunk that chose them; then the rest of the layer. Chunk
    /// activations wait in `Super` buffers between phases, and block checkpoints are indexed by
    /// the chunk's offset (`step.chunk0`). With `head`, the last token's logits go to staging.
    /// A stage without the head returns the hidden state after its last layer; one with a
    /// remote hands it to the other devices first.
    async fn forward_layers(&self, input: Feed<'_>, pos0: usize, head: bool, remote: Option<&dyn Remote>) -> Result<Vec<f32>, String> {
        let st = self.state.as_ref().unwrap();
        let sup = st.sup.as_ref().unwrap();
        let q = &self.gpu.queue;
        let c = &self.cfg;
        let n_expert = c.moe.as_ref().map_or(0, |e| e.n_expert);
        let n_used = c.moe.as_ref().map_or(0, |e| e.n_used);
        let latent = c.expert_latent;
        let total = input.len();
        let chunks: Vec<(usize, usize)> = (0..total).step_by(MAX_BATCH).map(|s| (s, MAX_BATCH.min(total - s))).collect();
        let route_bytes = st.route.size();
        let enc = || self.gpu.device.create_command_encoder(&Default::default());
        // Byte ranges of `n` tokens from token `s`, `w` values wide.
        let at = |s: usize, w: usize| (s * w * 4) as u64;
        let len = |n: usize, w: usize| (n * w * 4) as u64;
        // Park or restore a chunk's activations.
        let park = |e: &mut wgpu::CommandEncoder, s: usize, n: usize, out: bool, which: &[(&wgpu::Buffer, &wgpu::Buffer, usize)]| {
            for &(work, all, w) in which {
                if out {
                    e.copy_buffer_to_buffer(work, 0, all, at(s, w), len(n, w));
                } else {
                    e.copy_buffer_to_buffer(all, at(s, w), work, 0, len(n, w));
                }
            }
        };
        let ne = c.n_embd;
        let (k_in, k_out) = (self.ckpts_before(self.stage.first), self.ckpts_before(self.stage.end));
        for &(s, n) in &chunks {
            self.write_step(n, pos0 + s, s);
            match input.slice(s, n) {
                Feed::Tokens(tokens) => {
                    let mut e = enc();
                    if self.placement.cpu_embeddings || st.embed_end == 0 {
                        self.prefetch(&self.embed_spans(tokens)).await?;
                        q.write_buffer(&st.x, 0, &floats(&self.embed_cpu(tokens)?));
                    } else {
                        q.write_buffer(&st.tokens, 0, &words(tokens));
                        self.record(&mut e, 0..st.embed_end, n);
                    }
                    park(&mut e, s, n, true, &[(&st.x, &sup.x, ne)]);
                    q.submit([e.finish()]);
                }
                // The previous device's residual stream goes straight to its parked place.
                Feed::Hidden(h, _) => self.write_hidden(h, n, k_in, &sup.x, s, s),
            }
        }
        let mut queued = 0u64;
        for (l, lo) in st.layers.iter().enumerate().filter(|(l, _)| self.stage.has_layer(*l)) {
            self.prefetch(&self.ring_spans(&self.ring_layers[l])).await?;
            queued += self.upload_rings(&self.ring_layers[l])?;
            let Some(pool) = self.layer_pool[l] else {
                for &(s, n) in &chunks {
                    self.write_step(n, pos0 + s, s);
                    let mut e = enc();
                    park(&mut e, s, n, false, &[(&st.x, &sup.x, ne)]);
                    self.record(&mut e, lo.start..lo.end, n);
                    park(&mut e, s, n, true, &[(&st.x, &sup.x, ne)]);
                    q.submit([e.finish()]);
                }
                if queued > 512 << 20 {
                    super::wait_idle(&self.gpu).await?;
                    queued = 0;
                }
                continue;
            };
            // What the experts read and add to, as (working buffer, parked copy, width).
            let expert_io: Vec<(&wgpu::Buffer, &wgpu::Buffer, usize)> = if latent > 0 {
                vec![(&st.xl, &sup.xl, latent), (&st.ml, &sup.ml, latent)]
            } else {
                vec![(&st.xn, &sup.xn, ne), (&st.x, &sup.x, ne)]
            };
            // 1. Attention and routing for every chunk.
            for (ci, &(s, n)) in chunks.iter().enumerate() {
                self.write_step(n, pos0 + s, s);
                let mut e = enc();
                park(&mut e, s, n, false, &[(&st.x, &sup.x, ne)]);
                self.record(&mut e, lo.start..lo.route_end, n);
                park(&mut e, s, n, true, &[(&st.x, &sup.x, ne), (&st.xn, &sup.xn, ne)]);
                if latent > 0 {
                    park(&mut e, s, n, true, &expert_io);
                }
                e.copy_buffer_to_buffer(&st.route, 0, &sup.route, ci as u64 * route_bytes, route_bytes);
                q.submit([e.finish()]);
            }
            let mut e = enc();
            e.copy_buffer_to_buffer(&sup.route, 0, &sup.route_staging, 0, chunks.len() as u64 * route_bytes);
            q.submit([e.finish()]);
            let raw = read_back(&self.gpu, &sup.route_staging, chunks.len() as u64 * route_bytes).await?;
            let ids: Vec<Vec<u32>> = chunks
                .iter()
                .enumerate()
                .map(|(ci, &(_, n))| {
                    let b = &raw[ci * route_bytes as usize..];
                    (0..n * n_used).map(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())).collect()
                })
                .collect();
            if ids.iter().flatten().any(|&e| e as usize >= n_expert) {
                return Err("expert routing produced an invalid expert".into());
            }
            let all: Vec<u32> = ids.iter().flatten().copied().collect();
            // 2. Each pass loads its experts once and runs every chunk that chose one of them.
            for (pass, loads) in self.plan_experts(l, &all)? {
                let missing: Vec<u32> = loads.iter().map(|&(_, e)| e).collect();
                self.prefetch(&self.expert_spans(pool, l, &missing)).await?;
                for (slot, e) in loads {
                    self.load_expert(pool, slot, l, e)?;
                }
                let mut map = vec![ABSENT; n_expert];
                for &(e, slot) in &pass {
                    map[e as usize] = slot as u32;
                }
                q.write_buffer(&st.emap, (l * n_expert * 4) as u64, &words(&map));
                for (ci, &(s, n)) in chunks.iter().enumerate() {
                    if !ids[ci].iter().any(|&e| map[e as usize] != ABSENT) {
                        continue;
                    }
                    self.write_step(n, pos0 + s, s);
                    let mut e = enc();
                    park(&mut e, s, n, false, &expert_io);
                    if latent == 0 {
                        // Gate and up read xn; the down projection adds into x.
                        park(&mut e, s, n, false, &[(&st.xn, &sup.xn, ne)]);
                    }
                    e.copy_buffer_to_buffer(&sup.route, ci as u64 * route_bytes, &st.route, 0, route_bytes);
                    self.record(&mut e, lo.route_end..lo.experts_end, n);
                    park(&mut e, s, n, true, &expert_io[1..]);
                    q.submit([e.finish()]);
                }
            }
            // 3. The rest of the layer: shared experts, and latent experts' way back.
            for &(s, n) in &chunks {
                self.write_step(n, pos0 + s, s);
                let mut e = enc();
                park(&mut e, s, n, false, &[(&st.x, &sup.x, ne), (&st.xn, &sup.xn, ne)]);
                if latent > 0 {
                    park(&mut e, s, n, false, &[(&st.ml, &sup.ml, latent)]);
                }
                self.record(&mut e, lo.experts_end..lo.end, n);
                park(&mut e, s, n, true, &[(&st.x, &sup.x, ne)]);
                q.submit([e.finish()]);
            }
            queued = 0;
        }
        let mut hidden = Vec::new();
        if remote.is_some() || !self.stage.head {
            for &(s, n) in &chunks {
                hidden.extend(self.read_hidden(n, k_out, &sup.x, s, s).await?);
            }
        }
        if !self.stage.head {
            return Ok(hidden);
        }
        if head {
            let &(s, n) = chunks.last().unwrap();
            self.prefetch(&self.ring_spans(&self.ring_head)).await?;
            self.upload_rings(&self.ring_head)?;
            match remote {
                Some(r) => self.through_remote(r, hidden, total, pos0, s, n, s).await?,
                None => {
                    let mut e = enc();
                    park(&mut e, s, n, false, &[(&st.x, &sup.x, ne)]);
                    q.submit([e.finish()]);
                }
            }
            self.write_step(n, pos0 + s, s);
            let mut e = enc();
            self.record_head(&mut e, n);
            q.submit([e.finish()]);
        } else if let Some(r) = remote {
            // Not the last part of the prompt: the other devices' layers still read it.
            r.forward(hidden, total, pos0).await?;
        }
        Ok(Vec::new())
    }

    /// Forward pass for models that do not fit: layer by layer, reading streamed weights and
    /// the experts each token chose.
    async fn forward_streamed(&self, input: Feed<'_>, pos0: usize) -> Result<Vec<f32>, String> {
        let st = self.state.as_ref().ok_or("model is not loaded")?;
        let total = input.len();
        if total == 0 {
            return Err("nothing to process".into());
        }
        if pos0 + total > st.ctx {
            return Err(format!("the conversation is longer than the {}-token context", st.ctx));
        }
        let remote = self.remote_for_pass()?;
        let hand_on = remote.is_some() || !self.stage.head;
        if hand_on && self.all_logits.get() {
            return Err("drafts are not verified while other devices run part of the model".into());
        }
        if let Some(sup) = st.sup.as_ref().filter(|_| total > MAX_BATCH) {
            let mut hidden = Vec::new();
            let mut s = 0;
            while s < total {
                let n = sup.tokens.min(total - s);
                hidden.extend(self.forward_layers(input.slice(s, n), pos0 + s, s + n == total, remote.as_deref()).await?);
                s += n;
            }
            if !self.stage.head {
                return Ok(hidden);
            }
            if self.all_logits.get() {
                return Ok(Vec::new()); // forward_all reads every token's logits
            }
            let bytes = read_back(&self.gpu, &st.staging, st.staging.size()).await?;
            return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
        }
        let q = &self.gpu.queue;
        let n_expert = self.cfg.moe.as_ref().map_or(0, |e| e.n_expert);
        let n_used = self.cfg.moe.as_ref().map_or(0, |e| e.n_used);
        let (k_in, k_out) = (self.ckpts_before(self.stage.first), self.ckpts_before(self.stage.end));
        let chunks: Vec<(usize, usize)> = (0..total).step_by(MAX_BATCH).map(|s| (s, MAX_BATCH.min(total - s))).collect();
        let mut hidden = Vec::new();
        for (ci, &(s, n)) in chunks.iter().enumerate() {
            let last = ci + 1 == chunks.len();
            self.write_step(n, pos0 + s, 0);
            match input.slice(s, n) {
                Feed::Tokens(chunk) => {
                    q.write_buffer(&st.tokens, 0, &words(chunk));
                    if self.placement.cpu_embeddings || st.embed_end == 0 {
                        self.prefetch(&self.embed_spans(chunk)).await?;
                        q.write_buffer(&st.x, 0, &floats(&self.embed_cpu(chunk)?));
                    } else {
                        q.submit([self.encode_range(0..st.embed_end, n, false)]);
                    }
                }
                Feed::Hidden(h, _) => self.write_hidden(h, n, k_in, &st.x, 0, 0),
            }
            let mut queued = 0u64;
            for (l, lo) in st.layers.iter().enumerate().filter(|(l, _)| self.stage.has_layer(*l)) {
                self.prefetch(&self.ring_spans(&self.ring_layers[l])).await?;
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
                            // One pass at a time, so the fetched experts never exceed the cache.
                            let missing: Vec<u32> = loads.iter().map(|&(_, e)| e).collect();
                            self.prefetch(&self.expert_spans(pool, l, &missing)).await?;
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
            if hand_on {
                hidden.extend(self.read_hidden(n, k_out, &st.x, 0, 0).await?);
            } else if last {
                self.prefetch(&self.ring_spans(&self.ring_head)).await?;
                self.upload_rings(&self.ring_head)?;
                let mut enc = self.gpu.device.create_command_encoder(&Default::default());
                self.record_head(&mut enc, n);
                q.submit([enc.finish()]);
            }
        }
        if !self.stage.head {
            return Ok(hidden);
        }
        if let Some(r) = &remote {
            let &(s, n) = chunks.last().unwrap();
            self.through_remote(r.as_ref(), hidden, total, pos0, s, n, 0).await?;
            self.prefetch(&self.ring_spans(&self.ring_head)).await?;
            self.upload_rings(&self.ring_head)?;
            self.write_step(n, pos0 + s, 0);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.record_head(&mut enc, n);
            q.submit([enc.finish()]);
        }
        if self.all_logits.get() {
            return Ok(Vec::new()); // forward_all reads every token's logits
        }
        let bytes = read_back(&self.gpu, &st.staging, st.staging.size()).await?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
