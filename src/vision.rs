//! Vision encoders: an image to the embeddings a vision-language model reads in place of its
//! image tokens. Weights come from llama.cpp's mmproj files (`general.architecture = clip`).
//!
//! Qwen2.5-VL first (`clip.projector_type = qwen2.5vl_merger`): it is the base of the strongest
//! small computer-use models (Holo1.5, UI-TARS-1.5, GUI-Actor, OpenCUA). The image is cut into
//! 14x14 patches, each embedded by a 3D convolution over two identical frames; 2x2 neighbouring
//! patches stay together throughout ("merge units"). Transformer blocks use 2D rotary positions
//! (height and width), attention inside 112-pixel windows except every `n_wa_pattern`-th block,
//! which attends over the whole image. A merger turns each 2x2 unit into one token for the
//! language model. This module is the reference implementation on the CPU (exact, checked
//! against Hugging Face transformers in tests); the GPU path must match it.

use crate::gguf::Gguf;
use crate::quant;
use std::collections::HashMap;

/// Window size in pixels (Qwen2.5-VL's `window_size`).
const WINDOW: usize = 112;
/// Patches per side of a merge unit.
const MERGE: usize = 2;
/// Frames per patch: a still image is two identical frames.
const TEMPORAL: usize = 2;

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub dim: usize,
    pub ff: usize,
    pub layers: usize,
    pub heads: usize,
    pub patch: usize,
    pub out_dim: usize,
    pub eps: f32,
    /// Every `wa_pattern`-th block attends over the whole image; the rest within windows.
    pub wa_pattern: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl VisionConfig {
    pub fn read(g: &Gguf) -> Result<Self, String> {
        if g.str("general.architecture") != Some("clip") {
            return Err("not a vision encoder file (general.architecture is not clip)".into());
        }
        match g.str("clip.projector_type") {
            Some("qwen2.5vl_merger") => {}
            Some(other) => return Err(format!("the {other} vision encoder is not supported yet (supported: qwen2.5vl_merger)")),
            None => return Err("missing clip.projector_type".into()),
        }
        let need = |k: &str| g.u64(k).map(|v| v as usize).ok_or_else(|| format!("missing {k}"));
        let three = |k: &str, d: [f32; 3]| -> [f32; 3] {
            g.get(k)
                .and_then(|v| v.as_array())
                .map(|a| {
                    let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                    if v.len() == 3 { [v[0], v[1], v[2]] } else { d }
                })
                .unwrap_or(d)
        };
        Ok(Self {
            dim: need("clip.vision.embedding_length")?,
            ff: need("clip.vision.feed_forward_length")?,
            layers: need("clip.vision.block_count")?,
            heads: need("clip.vision.attention.head_count")?,
            patch: need("clip.vision.patch_size")?,
            out_dim: need("clip.vision.projection_dim")?,
            eps: g.f64("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6) as f32,
            wa_pattern: g.u64("clip.vision.n_wa_pattern").map(|v| v as usize).unwrap_or(0),
            mean: three("clip.vision.image_mean", [0.481_454_7, 0.457_827_5, 0.408_210_7]),
            std: three("clip.vision.image_std", [0.268_629_5, 0.261_302_6, 0.275_777_1]),
        })
    }

    /// Image side lengths must be whole merge units: multiples of this many pixels.
    pub fn unit(&self) -> usize {
        self.patch * MERGE
    }

    /// Whether block `i` attends over the whole image (else within windows).
    pub fn full_attention(&self, i: usize) -> bool {
        self.wa_pattern == 0 || (i + 1).is_multiple_of(self.wa_pattern)
    }
}

pub struct Vision {
    pub cfg: VisionConfig,
    weights: HashMap<String, Vec<f32>>,
}

/// The patches of one image, in merge-unit order: each row is (channel, frame, y, x).
pub struct Patches {
    pub data: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Patches {
    /// Image tokens the language model reads: one per 2x2 merge unit.
    pub fn tokens(&self) -> usize {
        self.grid_h * self.grid_w / (MERGE * MERGE)
    }
}

impl Vision {
    /// Load from a whole mmproj file.
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let g = crate::gguf::parse(bytes).map_err(|e| e.to_string())?;
        let cfg = VisionConfig::read(&g)?;
        let mut weights = HashMap::new();
        for t in &g.tensors {
            let n = t.elements() as usize;
            let start = t.start() as usize;
            let len = quant::row_bytes(t.ggml_type, t.dims[0] as usize)? * (n / t.dims[0] as usize);
            let raw = bytes.get(start..start + len).ok_or_else(|| format!("{} is cut short", t.name))?;
            let mut out = vec![0f32; n];
            quant::dequantize(t.ggml_type, raw, &mut out)?;
            weights.insert(t.name.clone(), out);
        }
        let v = Self { cfg, weights };
        for name in ["v.patch_embd.weight", "v.patch_embd.weight.1", "v.post_ln.weight", "mm.0.weight", "mm.2.weight"] {
            v.w(name)?;
        }
        Ok(v)
    }

    fn w(&self, name: &str) -> Result<&[f32], String> {
        self.weights.get(name).map(|v| v.as_slice()).ok_or_else(|| format!("missing tensor {name}"))
    }

    /// RGB bytes (row-major, 3 per pixel) to normalized patches. Width and height must be
    /// multiples of `cfg.unit()` (28 for Qwen2.5-VL); resize before.
    pub fn patches(&self, rgb: &[u8], width: usize, height: usize) -> Result<Patches, String> {
        let (p, u) = (self.cfg.patch, self.cfg.unit());
        if !width.is_multiple_of(u) || !height.is_multiple_of(u) || width == 0 || height == 0 {
            return Err(format!("image is {width}x{height}; both sides must be multiples of {u}"));
        }
        if rgb.len() != width * height * 3 {
            return Err("RGB data does not match the image size".into());
        }
        let (gh, gw) = (height / p, width / p);
        let row = 3 * TEMPORAL * p * p;
        let mut data = Vec::with_capacity(gh * gw * row);
        for bh in 0..gh / MERGE {
            for bw in 0..gw / MERGE {
                for mh in 0..MERGE {
                    for mw in 0..MERGE {
                        let (y0, x0) = ((bh * MERGE + mh) * p, (bw * MERGE + mw) * p);
                        for c in 0..3 {
                            for _frame in 0..TEMPORAL {
                                for y in 0..p {
                                    for x in 0..p {
                                        let v = rgb[((y0 + y) * width + x0 + x) * 3 + c] as f32 / 255.0;
                                        data.push((v - self.cfg.mean[c]) / self.cfg.std[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Patches { data, grid_h: gh, grid_w: gw })
    }

    /// Image embeddings for the language model: `tokens() x out_dim`, in raster order of the
    /// merge units.
    pub fn encode(&self, px: &Patches) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, hd) = (c.dim, c.dim / c.heads);
        let n = px.grid_h * px.grid_w;
        let unit = MERGE * MERGE;
        // Patch embedding: a 3D convolution with a kernel as large as the patch is one linear map
        // per frame slice.
        let row = 3 * TEMPORAL * c.patch * c.patch;
        let pp = c.patch * c.patch;
        let (w0, w1) = (self.w("v.patch_embd.weight")?, self.w("v.patch_embd.weight.1")?);
        let mut x = vec![0f32; n * d];
        for i in 0..n {
            let src = &px.data[i * row..(i + 1) * row];
            for o in 0..d {
                let mut s = 0f32;
                for ch in 0..3 {
                    let k = &w0[(o * 3 + ch) * pp..(o * 3 + ch + 1) * pp];
                    let k1 = &w1[(o * 3 + ch) * pp..(o * 3 + ch + 1) * pp];
                    let f0 = &src[(ch * TEMPORAL) * pp..(ch * TEMPORAL + 1) * pp];
                    let f1 = &src[(ch * TEMPORAL + 1) * pp..(ch * TEMPORAL + 2) * pp];
                    for j in 0..pp {
                        s += k[j] * f0[j] + k1[j] * f1[j];
                    }
                }
                x[i * d + o] = s;
            }
        }
        let (order, windows, angles) = layout(c, px);
        let permute = |v: &[f32], width: usize| {
            let mut out = vec![0f32; v.len()];
            for (k, &u) in order.iter().enumerate() {
                out[k * unit * width..(k + 1) * unit * width].copy_from_slice(&v[u * unit * width..(u + 1) * unit * width]);
            }
            out
        };
        x = permute(&x, d);
        let whole = [(0usize, n)];
        for l in 0..c.layers {
            let b = |s: &str| format!("v.blk.{l}.{s}");
            let spans: &[(usize, usize)] = if c.full_attention(l) { &whole } else { &windows };
            let h = rms_norm(&x, self.w(&b("ln1.weight"))?, d, c.eps);
            let mut q = linear(&h, self.w(&b("attn_q.weight"))?, self.weights.get(&b("attn_q.bias")).map(|v| v.as_slice()), d, d);
            let mut k = linear(&h, self.w(&b("attn_k.weight"))?, self.weights.get(&b("attn_k.bias")).map(|v| v.as_slice()), d, d);
            let v = linear(&h, self.w(&b("attn_v.weight"))?, self.weights.get(&b("attn_v.bias")).map(|v| v.as_slice()), d, d);
            rotate(&mut q, &angles, c.heads, hd);
            rotate(&mut k, &angles, c.heads, hd);
            let a = attention(&q, &k, &v, spans, c.heads, hd);
            let o = linear(&a, self.w(&b("attn_out.weight"))?, self.weights.get(&b("attn_out.bias")).map(|v| v.as_slice()), d, d);
            add(&mut x, &o);
            let h = rms_norm(&x, self.w(&b("ln2.weight"))?, d, c.eps);
            let gate = linear(&h, self.w(&b("ffn_gate.weight"))?, self.weights.get(&b("ffn_gate.bias")).map(|v| v.as_slice()), d, c.ff);
            let up = linear(&h, self.w(&b("ffn_up.weight"))?, self.weights.get(&b("ffn_up.bias")).map(|v| v.as_slice()), d, c.ff);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let down = linear(&act, self.w(&b("ffn_down.weight"))?, self.weights.get(&b("ffn_down.bias")).map(|v| v.as_slice()), c.ff, d);
            add(&mut x, &down);
        }
        // Merger: each merge unit's four patches side by side, then a two-layer MLP.
        let h = rms_norm(&x, self.w("v.post_ln.weight")?, d, c.eps);
        let wide = unit * d;
        let m = linear(&h, self.w("mm.0.weight")?, self.weights.get("mm.0.bias").map(|v| v.as_slice()), wide, wide);
        let m: Vec<f32> = m.iter().map(|&v| gelu(v)).collect();
        let merged = linear(&m, self.w("mm.2.weight")?, self.weights.get("mm.2.bias").map(|v| v.as_slice()), wide, c.out_dim);
        // Back to raster order.
        let mut out = vec![0f32; merged.len()];
        for (k, &u) in order.iter().enumerate() {
            out[u * c.out_dim..(u + 1) * c.out_dim].copy_from_slice(&merged[k * c.out_dim..(k + 1) * c.out_dim]);
        }
        Ok(out)
    }
}

/// The order the encoder works in, for an image's patches: the merge units grouped into
/// windows of 4x4 units (row by row), the windows as (start, len) in patches, and each patch's
/// rotary angles (`head_dim / 2`: by row, then by column) in that order.
pub fn layout(c: &VisionConfig, px: &Patches) -> (Vec<usize>, Vec<(usize, usize)>, Vec<f32>) {
    let hd = c.dim / c.heads;
    let n = px.grid_h * px.grid_w;
    let unit = MERGE * MERGE;
    // Rotary angles: the first quarter of each head by patch row, the second by column.
    let quarter = hd / 4;
    let inv: Vec<f32> = (0..quarter).map(|j| 1.0 / 10000f32.powf((2 * j) as f32 / (hd / 2) as f32)).collect();
    let mut angles = vec![0f32; n * hd / 2];
    let (gh, gw) = (px.grid_h, px.grid_w);
    let mut i = 0;
    for bh in 0..gh / MERGE {
        for bw in 0..gw / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    let (r, col) = ((bh * MERGE + mh) as f32, (bw * MERGE + mw) as f32);
                    for j in 0..quarter {
                        angles[i * hd / 2 + j] = r * inv[j];
                        angles[i * hd / 2 + quarter + j] = col * inv[j];
                    }
                    i += 1;
                }
            }
        }
    }
    // Window order: merge units grouped into windows of `ws x ws` units, row by row.
    let (lh, lw) = (gh / MERGE, gw / MERGE);
    let ws = WINDOW / MERGE / c.patch;
    let (nh, nw) = (lh.div_ceil(ws), lw.div_ceil(ws));
    let mut order = Vec::with_capacity(lh * lw);
    let mut windows = Vec::new(); // (start, len) in patches
    for wy in 0..nh {
        for wx in 0..nw {
            let start = order.len();
            for y in wy * ws..((wy + 1) * ws).min(lh) {
                for x in wx * ws..((wx + 1) * ws).min(lw) {
                    order.push(y * lw + x);
                }
            }
            if order.len() > start {
                windows.push((start * unit, (order.len() - start) * unit));
            }
        }
    }
    let mut permuted = vec![0f32; angles.len()];
    let w = hd / 2;
    for (k, &u) in order.iter().enumerate() {
        permuted[k * unit * w..(k + 1) * unit * w].copy_from_slice(&angles[u * unit * w..(u + 1) * unit * w]);
    }
    (order, windows, permuted)
}

/// `y = x W^T + b` for `x` of rows of `k`; `w` is GGUF `[k, m]` (rows of `k` per output).
fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, k: usize, m: usize) -> Vec<f32> {
    let rows = x.len() / k;
    let mut y = vec![0f32; rows * m];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for o in 0..m {
            let wr = &w[o * k..(o + 1) * k];
            let mut s = 0f32;
            for j in 0..k {
                s += xr[j] * wr[j];
            }
            y[r * m + o] = s + b.map_or(0.0, |b| b[o]);
        }
    }
    y
}

fn rms_norm(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (xr, yr) in x.chunks(d).zip(y.chunks_mut(d)) {
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let s = 1.0 / (ms + eps).sqrt();
        for j in 0..d {
            yr[j] = xr[j] * s * w[j];
        }
    }
    y
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Rotate each head's halves (NEOX style) by the token's angles (`hd / 2` per token).
fn rotate(x: &mut [f32], angles: &[f32], heads: usize, hd: usize) {
    let half = hd / 2;
    for t in 0..x.len() / (heads * hd) {
        let a = &angles[t * half..(t + 1) * half];
        for h in 0..heads {
            let base = (t * heads + h) * hd;
            for j in 0..half {
                let (s, c) = a[j].sin_cos();
                let (x0, x1) = (x[base + j], x[base + j + half]);
                x[base + j] = x0 * c - x1 * s;
                x[base + j + half] = x1 * c + x0 * s;
            }
        }
    }
}

/// Attention within each span of tokens (no causal mask).
fn attention(q: &[f32], k: &[f32], v: &[f32], spans: &[(usize, usize)], heads: usize, hd: usize) -> Vec<f32> {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; q.len()];
    let mut scores = Vec::new();
    for &(start, len) in spans {
        for h in 0..heads {
            for i in start..start + len {
                let qi = &q[i * d + h * hd..i * d + (h + 1) * hd];
                scores.clear();
                let mut max = f32::NEG_INFINITY;
                for j in start..start + len {
                    let kj = &k[j * d + h * hd..j * d + (h + 1) * hd];
                    let s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                    max = max.max(s);
                    scores.push(s);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let o = &mut out[i * d + h * hd..i * d + (h + 1) * hd];
                for (jj, j) in (start..start + len).enumerate() {
                    let p = scores[jj] / sum;
                    let vj = &v[j * d + h * hd..j * d + (h + 1) * hd];
                    for e in 0..hd {
                        o[e] += p * vj[e];
                    }
                }
            }
        }
    }
    out
}

/// GELU with the error function, as PyTorch's nn.GELU().
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// The error function (Abramowitz and Stegun 7.1.26 refined to about 1e-7 with f64).
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

#[cfg(test)]
pub(crate) fn tests_base64(s: &str) -> Vec<u8> {
    tests::base64(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn base64(s: &str) -> Vec<u8> {
        let val = |c: u8| match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            _ => 63,
        } as u32;
        let mut out = Vec::new();
        for chunk in s.trim_end_matches('=').as_bytes().chunks(4) {
            let n = chunk.iter().fold(0u32, |a, &c| (a << 6) | val(c)) << (6 * (4 - chunk.len()));
            out.extend(&n.to_be_bytes()[1..chunk.len()]);
        }
        out
    }

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [(0.0f32, 0.0f32), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (-2.0, -0.995_322_3), (3.0, 0.999_977_9)] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {} want {want}", erf(x));
        }
    }

    #[test]
    fn tiny_qwen25vl_encoder_matches_transformers() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("tiny-qwen25vl-vision.json")).unwrap()).unwrap();
        let bytes = base64(want["gguf"].as_str().unwrap());
        let v = Vision::load(&bytes).unwrap();
        let rgb = base64(want["rgb"].as_str().unwrap());
        let px = v.patches(&rgb, want["width"].as_u64().unwrap() as usize, want["height"].as_u64().unwrap() as usize).unwrap();
        assert_eq!(px.tokens(), want["tokens"].as_u64().unwrap() as usize);
        let got = v.encode(&px).unwrap();
        let exp: Vec<f32> = want["embeddings"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        assert_eq!(got.len(), exp.len());
        let scale = exp.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = got.iter().zip(&exp).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // About 6e-7 when right; a wrong window size gives 1.6e-3.
        assert!(worst <= 1e-5 * scale.max(1.0), "largest difference {worst} (values up to {scale})");
    }
}
