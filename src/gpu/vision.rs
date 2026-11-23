//! The Qwen2.5-VL vision encoder on the GPU (see `crate::vision` for the model and the CPU
//! reference it must match). Small self-contained kernels: a tiled matrix product over f16
//! weights, RMSNorm, 2D rotary positions, attention within spans (windows or the whole image)
//! with an online softmax, and the activations. Weights go to the GPU from the file's f16 bytes,
//! never expanded on the CPU. Window order, rotary angles and the final reordering are computed
//! on the CPU (they are small); each block is its own submission, so no single GPU job runs long.

use super::{read_back, wait_idle, Gpu};
use crate::gguf::{parse, Gguf};
use crate::vision::{Encoded, Patches, VisionConfig};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

const MATMUL: &str = r#"
struct P { n: u32, k: u32, m: u32, flags: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> xs: array<array<f32, 16>, 16>;
var<workgroup> ws: array<array<f32, 16>, 16>;
fn wt(o: u32, j: u32) -> f32 {
    let e = o * p.k + j;
    return unpack2x16float(w[e >> 1u])[e & 1u];
}
// y[n, m] = x[n, k] . w[m, k]^T (+ b) (+ y): bit 0 of flags adds the bias, bit 1 adds into y.
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let row = g.y * 16u + l.y;
    let col = g.x * 16u + l.x;
    let o = g.x * 16u + l.y;
    var acc = 0.0;
    for (var t = 0u; t < p.k; t += 16u) {
        let c = t + l.x;
        var xv = 0.0;
        if (row < p.n && c < p.k) { xv = x[row * p.k + c]; }
        xs[l.y][l.x] = xv;
        var wv = 0.0;
        if (o < p.m && c < p.k) { wv = wt(o, c); }
        ws[l.y][l.x] = wv;
        workgroupBarrier();
        for (var i = 0u; i < 16u; i++) { acc += xs[l.y][i] * ws[l.x][i]; }
        workgroupBarrier();
    }
    if (row < p.n && col < p.m) {
        var v = acc;
        if ((p.flags & 1u) != 0u) { v += b[col]; }
        let idx = row * p.m + col;
        if ((p.flags & 2u) != 0u) { v += y[idx]; }
        y[idx] = v;
    }
}
"#;

const RMS_NORM: &str = r#"
struct P { n: u32, d: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = g.x;
    var s = 0.0;
    for (var j = li; j < p.d; j += 64u) { let v = x[r * p.d + j]; s += v * v; }
    part[li] = s;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(part[0] / f32(p.d) + p.eps);
    for (var j = li; j < p.d; j += 64u) { y[r * p.d + j] = x[r * p.d + j] * inv * w[j]; }
}
"#;

const LAYER_NORM: &str = r#"
struct P { n: u32, d: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, 64>;
fn total(v: f32, li: u32) -> f32 {
    part[li] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if (li < st) { part[li] += part[li + st]; }
        workgroupBarrier();
    }
    let t = part[0];
    workgroupBarrier();
    return t;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = g.x;
    var s = 0.0;
    for (var j = li; j < p.d; j += 64u) { s += x[r * p.d + j]; }
    let mean = total(s, li) / f32(p.d);
    var q = 0.0;
    for (var j = li; j < p.d; j += 64u) { let c = x[r * p.d + j] - mean; q += c * c; }
    let inv = 1.0 / sqrt(total(q, li) / f32(p.d) + p.eps);
    for (var j = li; j < p.d; j += 64u) { y[r * p.d + j] = (x[r * p.d + j] - mean) * inv * w[j] + b[j]; }
}
"#;

const ROTATE: &str = r#"
struct P { n: u32, heads: u32, hd: u32, total: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> ang: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (g.x + g.y * nw.x) * 64u + li;
    if (i >= p.total) { return; }
    let half = p.hd / 2u;
    let t = i / (p.heads * half);
    let rem = i % (p.heads * half);
    let h = rem / half;
    let j = rem % half;
    let a = ang[t * half + j];
    let base = (t * p.heads + h) * p.hd;
    let x0 = q[base + j];
    let x1 = q[base + j + half];
    q[base + j] = x0 * cos(a) - x1 * sin(a);
    q[base + j + half] = x1 * cos(a) + x0 * sin(a);
}
"#;

const ATTEND: &str = r#"
struct P { heads: u32, hd: u32, d: u32, scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> blocks: array<vec4<u32>>;
var<workgroup> ks: array<f32, 2048>;
var<workgroup> vs: array<f32, 2048>;
// One workgroup per (block of up to 64 queries, head); a block is (query start, query count,
// key start, key count): a window, or 64 tokens against the whole image. Keys and values pass
// through shared memory 16 at a time; each thread keeps its query's running softmax.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let blk = blocks[g.x];
    let h = g.y;
    let qi = blk.x + li;
    let inside = li < blk.y;
    var qv: array<f32, 128>;
    var acc: array<f32, 128>;
    if (inside) {
        for (var e = 0u; e < p.hd; e++) { qv[e] = q[qi * p.d + h * p.hd + e] * p.scale; acc[e] = 0.0; }
    }
    var m = -1e30;
    var s = 0.0;
    for (var t = 0u; t < blk.w; t += 16u) {
        let cnt = min(16u, blk.w - t);
        for (var idx = li; idx < 16u * p.hd; idx += 64u) {
            let kk = idx / p.hd;
            let e = idx % p.hd;
            var kv = 0.0;
            var vv = 0.0;
            if (kk < cnt) { let r = (blk.z + t + kk) * p.d + h * p.hd + e; kv = k[r]; vv = v[r]; }
            ks[idx] = kv;
            vs[idx] = vv;
        }
        workgroupBarrier();
        if (inside) {
            for (var kk = 0u; kk < cnt; kk++) {
                var dot = 0.0;
                for (var e = 0u; e < p.hd; e++) { dot += qv[e] * ks[kk * p.hd + e]; }
                let mn = max(m, dot);
                let c = exp(m - mn);
                let pj = exp(dot - mn);
                s = s * c + pj;
                for (var e = 0u; e < p.hd; e++) { acc[e] = acc[e] * c + pj * vs[kk * p.hd + e]; }
                m = mn;
            }
        }
        workgroupBarrier();
    }
    if (inside) {
        for (var e = 0u; e < p.hd; e++) { o[qi * p.d + h * p.hd + e] = acc[e] / s; }
    }
}
"#;

const ACTIVATE: &str = r#"
struct P { total: u32, kind: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
fn erf(z: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * abs(z));
    let y = 1.0 - t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return select(-y, y, z >= 0.0);
}
// kind 0: x = silu(x) * u (SwiGLU); 1: gelu(x) (exact, with erf); 2: gelu(x) (tanh
// approximation); 3: x + u.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (g.x + g.y * nw.x) * 64u + li;
    if (i >= p.total) { return; }
    let v = x[i];
    if (p.kind == 0u) { x[i] = v / (1.0 + exp(-v)) * u[i]; }
    else if (p.kind == 1u) { x[i] = 0.5 * v * (1.0 + erf(v * 0.70710678118654752)); }
    else if (p.kind == 2u) { x[i] = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v))); }
    else { x[i] = v + u[i]; }
}
"#;

struct Mat {
    buf: wgpu::Buffer,
    k: usize,
    m: usize,
}

pub struct VisionGpu {
    /// The device the language model runs on, shared (wgpu handles are reference counted).
    gpu: Gpu,
    pub cfg: VisionConfig,
    mats: HashMap<String, Mat>,
    vecs: HashMap<String, wgpu::Buffer>,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    zero: wgpu::Buffer,
    /// The learned position table (Qwen3-VL), resized per image on the CPU.
    positions: Option<Vec<f32>>,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Elementwise dispatch for `n` items in groups of 64, within the 65535-per-dimension limit.
fn groups(n: usize) -> (u32, u32) {
    let wg = n.div_ceil(64).max(1);
    let x = wg.min(65535);
    (x as u32, wg.div_ceil(x) as u32)
}

impl VisionGpu {
    /// Upload the encoder from a whole mmproj file. Matrices must be F16, vectors F32 (as in
    /// llama.cpp's f16 mmproj files).
    pub fn load(gpu: &Gpu, bytes_all: &[u8]) -> Result<Self, String> {
        let gpu = gpu.clone();
        let g: Gguf = parse(bytes_all).map_err(|e| e.to_string())?;
        let cfg = VisionConfig::read(&g)?;
        let dev = &gpu.device;
        let raw = |name: &str| -> Result<(&[u8], &crate::gguf::TensorInfo), String> {
            let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
            let n = t.elements() as usize;
            let size = match t.ggml_type {
                0 => n * 4,
                1 => n * 2,
                other => return Err(format!("{name} is {}; the vision encoder reads F16 and F32 tensors", crate::quant::type_name(other))),
            };
            let s = t.start() as usize;
            Ok((bytes_all.get(s..s + size).ok_or_else(|| format!("{name} is cut short"))?, t))
        };
        let storage = |label: &str, data: &[u8]| {
            let mut d = data.to_vec();
            d.resize(d.len().div_ceil(4).max(1) * 4, 0);
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &d, usage: wgpu::BufferUsages::STORAGE })
        };
        let mut mats = HashMap::new();
        let mut vecs = HashMap::new();
        let mut positions = None;
        let as_f32 = |data: &[u8], t: u32| -> Vec<f32> {
            if t == 0 { f32s(data) } else { data.chunks_exact(2).map(|c| crate::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect() }
        };
        for t in &g.tensors {
            let name = t.name.as_str();
            if name == "v.patch_embd.weight" || name == "v.patch_embd.weight.1" {
                continue;
            }
            let (data, info) = raw(name)?;
            if name == "v.position_embd.weight" {
                positions = Some(as_f32(data, info.ggml_type));
            } else if info.dims.len() >= 2 {
                if info.ggml_type != 1 {
                    return Err(format!("{name} must be F16"));
                }
                let (k, m) = (info.dims[0] as usize, info.dims[1] as usize);
                // A fused QKV projection becomes three: its rows are Q, then K, then V.
                if let Some(base) = name.strip_suffix("attn_qkv.weight") {
                    for (j, part) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
                        let rows = m / 3;
                        mats.insert(format!("{base}{part}.weight"), Mat { buf: storage(name, &data[j * rows * k * 2..(j + 1) * rows * k * 2]), k, m: rows });
                    }
                } else {
                    mats.insert(name.to_string(), Mat { buf: storage(name, data), k, m });
                }
            } else {
                let v = as_f32(data, info.ggml_type);
                if let Some(base) = name.strip_suffix("attn_qkv.bias") {
                    let n = v.len() / 3;
                    for (j, part) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
                        vecs.insert(format!("{base}{part}.bias"), storage(name, &bytes(&v[j * n..(j + 1) * n])));
                    }
                } else {
                    vecs.insert(name.to_string(), storage(name, &bytes(&v)));
                }
            }
        }
        // The patch embedding as one matrix over the (channel, frame, y, x) patch rows.
        let (w0, t0) = raw("v.patch_embd.weight")?;
        let (w1, _) = raw("v.patch_embd.weight.1")?;
        let pp = cfg.patch * cfg.patch;
        let d = t0.dims[3] as usize;
        let mut wc: Vec<u8> = Vec::with_capacity(d * 3 * 2 * pp * 2);
        for o in 0..d {
            for c in 0..3 {
                for w in [w0, w1] {
                    let s = (o * 3 + c) * pp * 2;
                    wc.extend_from_slice(&w[s..s + pp * 2]);
                }
            }
        }
        mats.insert("patch".into(), Mat { buf: storage("patch", &wc), k: 3 * 2 * pp, m: d });
        let mut pipes = HashMap::new();
        for (key, src) in [("matmul", MATMUL), ("rms", RMS_NORM), ("layer_norm", LAYER_NORM), ("rotate", ROTATE), ("attend", ATTEND), ("act", ACTIVATE)] {
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(key), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(key), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            pipes.insert(key, pipe);
        }
        let zero = storage("zero", &[0u8; 16]);
        if let Some(e) = gpu.take_error() {
            return Err(format!("GPU error while loading the vision encoder: {e}"));
        }
        Ok(Self { gpu, cfg, mats, vecs, pipes, zero, positions })
    }

    fn buffer(&self, label: &str, floats: usize) -> wgpu::Buffer {
        self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (floats.max(1) * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    fn dispatch(&self, enc: &mut wgpu::CommandEncoder, pipe: &str, uniform: [u32; 4], bufs: &[&wgpu::Buffer], groups: (u32, u32)) {
        let dev = &self.gpu.device;
        let u = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &uniform.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), usage: wgpu::BufferUsages::UNIFORM });
        let p = &self.pipes[pipe];
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() });
        }
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// y = x . W^T (+ bias) (+ y when `add`), x of `n` rows.
    fn linear(&self, enc: &mut wgpu::CommandEncoder, name: &str, bias: Option<&str>, x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize, add: bool) {
        let m = &self.mats[name];
        let b = bias.and_then(|b| self.vecs.get(b)).unwrap_or(&self.zero);
        let flags = u32::from(bias.is_some_and(|b| self.vecs.contains_key(b))) | (u32::from(add) << 1);
        self.dispatch(enc, "matmul", [n as u32, m.k as u32, m.m as u32, flags], &[x, &m.buf, b, y], (m.m.div_ceil(16) as u32, n.div_ceil(16) as u32));
    }

    /// LayerNorm when the norm has a bias (Qwen3-VL), else RMSNorm; `name` without `.weight`.
    fn norm(&self, enc: &mut wgpu::CommandEncoder, name: &str, x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize, d: usize) {
        let w = &self.vecs[&format!("{name}.weight")];
        let u = [n as u32, d as u32, self.cfg.eps.to_bits(), 0];
        match self.vecs.get(&format!("{name}.bias")) {
            Some(b) => self.dispatch(enc, "layer_norm", u, &[x, w, b, y], (n as u32, 1)),
            None => self.dispatch(enc, "rms", u, &[x, w, y], (n as u32, 1)),
        }
    }

    /// A merger: `x` (patches) as rows of four patches, normalized before (`pre`, per patch) or
    /// after (`post`, per row of four), then linear, exact GELU, linear into `out`.
    #[allow(clippy::too_many_arguments)]
    fn merger(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, tmp: &wgpu::Buffer, mid: &wgpu::Buffer, out: &wgpu::Buffer, n: usize, pre: Option<&str>, post: Option<&str>, fc: (&str, &str)) {
        let (d, tokens) = (self.cfg.dim, n / 4);
        let src = match (pre, post) {
            (Some(nm), _) => { self.norm(enc, nm, x, tmp, n, d); tmp }
            (None, Some(nm)) => { self.norm(enc, nm, x, tmp, tokens, 4 * d); tmp }
            (None, None) => x,
        };
        let wide = self.mats[&format!("{}.weight", fc.0)].m;
        self.linear(enc, &format!("{}.weight", fc.0), Some(&format!("{}.bias", fc.0)), src, mid, tokens, false);
        self.dispatch(enc, "act", [(tokens * wide) as u32, 1, 0, 0], &[mid, &self.zero], groups(tokens * wide));
        self.linear(enc, &format!("{}.weight", fc.1), Some(&format!("{}.bias", fc.1)), mid, out, tokens, false);
    }

    /// The image's embeddings for the language model, as `crate::vision::Vision::encode` gives them.
    pub async fn encode(&self, px: &Patches) -> Result<Encoded, String> {
        let c = &self.cfg;
        let (d, hd, heads) = (c.dim, c.dim / c.heads, c.heads);
        if hd > 128 {
            return Err(format!("vision heads of {hd} dimensions are more than the GPU kernel's 128"));
        }
        let (order, windows, angles) = crate::vision::layout(c, px);
        let n = px.grid_h * px.grid_w;
        let unit = 4;
        let row = px.data.len() / n;
        let mut patches = vec![0f32; px.data.len()];
        for (k, &u) in order.iter().enumerate() {
            patches[k * unit * row..(k + 1) * unit * row].copy_from_slice(&px.data[u * unit * row..(u + 1) * unit * row]);
        }
        let dev = &self.gpu.device;
        let upload = |label: &str, data: &[u8]| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
        let input = upload("patches", &bytes(&patches));
        let ang = upload("angles", &bytes(&angles));
        let block_list = |spans: &[(usize, usize)]| -> Vec<u32> {
            let mut v = Vec::new();
            for &(s, len) in spans {
                let mut q = s;
                while q < s + len {
                    let qn = 64.min(s + len - q);
                    v.extend([q as u32, qn as u32, s as u32, len as u32]);
                    q += qn;
                }
            }
            v
        };
        let words = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let win_blocks = block_list(&windows);
        let all_blocks = block_list(&[(0, n)]);
        let win_buf = upload("window blocks", &words(&win_blocks));
        let all_buf = upload("image blocks", &words(&all_blocks));
        let x = self.buffer("x", n * d);
        let h = self.buffer("h", n * d);
        let (q, k, v, a) = (self.buffer("q", n * d), self.buffer("k", n * d), self.buffer("v", n * d), self.buffer("a", n * d));
        let (gate, up) = (self.buffer("gate", n * c.ff), self.buffer("up", n * c.ff));
        let mut enc = dev.create_command_encoder(&Default::default());
        let patch_bias = self.vecs.contains_key("v.patch_embd.bias").then_some("v.patch_embd.bias");
        self.linear(&mut enc, "patch", patch_bias, &input, &x, n, false);
        let pos_buf;
        if let Some(table) = &self.positions {
            let pe = crate::vision::position_embeddings(table, d, px.grid_h, px.grid_w);
            let mut permuted = vec![0f32; pe.len()];
            for (k, &u) in order.iter().enumerate() {
                permuted[k * unit * d..(k + 1) * unit * d].copy_from_slice(&pe[u * unit * d..(u + 1) * unit * d]);
            }
            pos_buf = upload("positions", &bytes(&permuted));
            self.dispatch(&mut enc, "act", [(n * d) as u32, 3, 0, 0], &[&x, &pos_buf], groups(n * d));
        }
        self.gpu.queue.submit([enc.finish()]);
        let tokens = n / unit;
        let mid = self.mats["mm.0.weight"].m;
        let out_dim = self.mats["mm.2.weight"].m;
        let wide_tmp = self.buffer("wide", n * d);
        let m0 = self.buffer("merge", tokens * mid);
        let deep_out: Vec<wgpu::Buffer> = c.deepstack.iter().map(|_| self.buffer("deepstack", tokens * out_dim)).collect();
        let rot = (n * heads * hd / 2) as u32;
        for l in 0..c.layers {
            let b = |s: &str| format!("v.blk.{l}.{s}");
            let mut enc = dev.create_command_encoder(&Default::default());
            self.norm(&mut enc, &b("ln1"), &x, &h, n, d);
            for (w, out) in [("attn_q", &q), ("attn_k", &k), ("attn_v", &v)] {
                self.linear(&mut enc, &b(&format!("{w}.weight")), Some(&b(&format!("{w}.bias"))), &h, out, n, false);
            }
            for t in [&q, &k] {
                self.dispatch(&mut enc, "rotate", [n as u32, heads as u32, hd as u32, rot], &[t, &ang], groups(rot as usize));
            }
            let (blocks, list) = if c.full_attention(l) { (&all_buf, &all_blocks) } else { (&win_buf, &win_blocks) };
            let scale = (1.0 / (hd as f32).sqrt()).to_bits();
            self.dispatch(&mut enc, "attend", [heads as u32, hd as u32, d as u32, scale], &[&q, &k, &v, &a, blocks], ((list.len() / 4) as u32, heads as u32));
            self.linear(&mut enc, &b("attn_out.weight"), Some(&b("attn_out.bias")), &a, &x, n, true);
            self.norm(&mut enc, &b("ln2"), &x, &h, n, d);
            if self.mats.contains_key(&b("ffn_gate.weight")) {
                self.linear(&mut enc, &b("ffn_gate.weight"), Some(&b("ffn_gate.bias")), &h, &gate, n, false);
                self.linear(&mut enc, &b("ffn_up.weight"), Some(&b("ffn_up.bias")), &h, &up, n, false);
                self.dispatch(&mut enc, "act", [(n * c.ff) as u32, 0, 0, 0], &[&gate, &up], groups(n * c.ff));
            } else {
                self.linear(&mut enc, &b("ffn_up.weight"), Some(&b("ffn_up.bias")), &h, &gate, n, false);
                self.dispatch(&mut enc, "act", [(n * c.ff) as u32, 2, 0, 0], &[&gate, &self.zero], groups(n * c.ff));
            }
            self.linear(&mut enc, &b("ffn_down.weight"), Some(&b("ffn_down.bias")), &gate, &x, n, true);
            if let Some(j) = c.deepstack.iter().position(|&dl| dl == l) {
                let p = format!("v.deepstack.{l}");
                self.merger(&mut enc, &x, &wide_tmp, &m0, &deep_out[j], n, None, Some(&format!("{p}.norm")), (&format!("{p}.fc1"), &format!("{p}.fc2")));
            }
            self.gpu.queue.submit([enc.finish()]);
            wait_idle(&self.gpu).await?;
        }
        // The merger: normalized patches, four side by side, linear, GELU, linear.
        let out = self.buffer("out", tokens * out_dim);
        let mut enc = dev.create_command_encoder(&Default::default());
        self.merger(&mut enc, &x, &h, &m0, &out, n, Some("v.post_ln"), None, ("mm.0", "mm.2"));
        let size = (tokens * out_dim * 4) as u64;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("vision out"), size: size * (1 + deep_out.len() as u64), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        for (j, b) in std::iter::once(&out).chain(deep_out.iter()).enumerate() {
            enc.copy_buffer_to_buffer(b, 0, &staging, j as u64 * size, size);
        }
        self.gpu.queue.submit([enc.finish()]);
        let all = f32s(&read_back(&self.gpu, &staging, size * (1 + deep_out.len() as u64)).await?);
        let raster = |merged: &[f32]| {
            let mut r = vec![0f32; merged.len()];
            for (k, &u) in order.iter().enumerate() {
                r[u * out_dim..(u + 1) * out_dim].copy_from_slice(&merged[k * out_dim..(k + 1) * out_dim]);
            }
            r
        };
        let per = tokens * out_dim;
        Ok(Encoded { embeddings: raster(&all[..per]), deepstack: (0..deep_out.len()).map(|j| raster(&all[(j + 1) * per..(j + 2) * per])).collect() })
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    #[test]
    fn gpu_encoder_matches_the_cpu_reference() {
        for fixture in ["tiny-qwen25vl-vision.json", "tiny-qwen3vl-vision.json"] {
            check(fixture);
        }
    }

    fn check(fixture: &str) {
        let Ok(gpu) = pollster::block_on(super::super::init()) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join(fixture)).unwrap()).unwrap();
        let bytes = crate::vision::tests_base64(fx["gguf"].as_str().unwrap());
        let rgb = crate::vision::tests_base64(fx["rgb"].as_str().unwrap());
        let cpu = crate::vision::Vision::load(&bytes).unwrap();
        let px = cpu.patches(&rgb, fx["width"].as_u64().unwrap() as usize, fx["height"].as_u64().unwrap() as usize).unwrap();
        let want = cpu.encode(&px).unwrap();
        let v = super::VisionGpu::load(&gpu, &bytes).unwrap();
        let got = pollster::block_on(v.encode(&px)).unwrap();
        assert_eq!(got.deepstack.len(), want.deepstack.len(), "{fixture}");
        for (g, w) in std::iter::once((&got.embeddings, &want.embeddings)).chain(got.deepstack.iter().zip(&want.deepstack)) {
            assert_eq!(g.len(), w.len());
            let scale = w.iter().fold(0f32, |m, x| m.max(x.abs()));
            let worst = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst <= 1e-4 * scale.max(1.0), "{fixture}: GPU differs from the CPU reference by {worst} (values up to {scale})");
        }
    }
}
