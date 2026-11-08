//! A plain f32 CPU forward pass over dequantized weights. It is slow and only
//! exists to check the GPU kernels and the architecture details.

use crate::gguf::Gguf;
use crate::model::{self, Config};
use crate::quant;

pub struct CpuModel<'a> {
    pub cfg: Config,
    g: &'a Gguf,
    file: &'a [u8],
    rope_factors: Option<Vec<f32>>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scale = 1.0 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    x.iter().zip(w).map(|(&v, &w)| v * scale * w).collect()
}

impl<'a> CpuModel<'a> {
    pub fn new(g: &'a Gguf, file: &'a [u8]) -> Result<Self, String> {
        let cfg = model::config(g)?;
        let mut m = CpuModel {
            cfg,
            g,
            file,
            rope_factors: None,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
        };
        if m.cfg.has_rope_freqs {
            m.rope_factors = Some(m.vector("rope_freqs.weight"));
        }
        m.k_cache = vec![Vec::new(); m.cfg.n_layer];
        m.v_cache = vec![Vec::new(); m.cfg.n_layer];
        Ok(m)
    }

    fn bytes(&self, name: &str) -> (&'a [u8], u32, &[u64]) {
        let t = self
            .g
            .tensor(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        let start = (self.g.data_offset + t.offset) as usize;
        let len = quant::row_bytes(t.ggml_type, t.dims[0] as usize).unwrap()
            * (t.elements() / t.dims[0]) as usize;
        (&self.file[start..start + len], t.ggml_type, &t.dims)
    }

    pub fn vector(&self, name: &str) -> Vec<f32> {
        let (b, ty, dims) = self.bytes(name);
        let mut out = vec![0f32; dims.iter().product::<u64>() as usize];
        quant::dequantize(ty, b, &mut out).unwrap();
        out
    }

    fn row(&self, name: &str, r: usize) -> Vec<f32> {
        let (b, ty, dims) = self.bytes(name);
        let k = dims[0] as usize;
        let rb = quant::row_bytes(ty, k).unwrap();
        let mut out = vec![0f32; k];
        quant::dequantize(ty, &b[r * rb..(r + 1) * rb], &mut out).unwrap();
        out
    }

    /// y = W x, with rows dequantized on the fly across threads.
    fn matvec(&self, name: &str, x: &[f32]) -> Vec<f32> {
        let (b, ty, dims) = self.bytes(name);
        let k = dims[0] as usize;
        let n = dims[1] as usize;
        assert_eq!(x.len(), k, "{name}");
        let rb = quant::row_bytes(ty, k).unwrap();
        let mut y = vec![0f32; n];
        let threads = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(4)
            .min(16);
        let per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, chunk) in y.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    let mut w = vec![0f32; k];
                    for (i, out) in chunk.iter_mut().enumerate() {
                        let r = ci * per + i;
                        quant::dequantize(ty, &b[r * rb..(r + 1) * rb], &mut w).unwrap();
                        let mut acc = 0f64;
                        for j in 0..k {
                            acc += (w[j] * x[j]) as f64;
                        }
                        *out = acc as f32;
                    }
                });
            }
        });
        y
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        let c = &self.cfg;
        let half = c.rope_dim / 2;
        for head in v.chunks_mut(c.head_dim) {
            for i in 0..half {
                let angle = pos as f32 * model::rope_inv_freq(c, i, self.rope_factors.as_deref());
                let (s, co) = (angle as f64).sin_cos();
                let (s, co) = (s as f32, co as f32);
                let (a, b) = if c.rope_neox {
                    (i, i + half)
                } else {
                    (2 * i, 2 * i + 1)
                };
                let (x0, x1) = (head[a], head[b]);
                head[a] = x0 * co - x1 * s;
                head[b] = x0 * s + x1 * co;
            }
        }
    }

    /// Run tokens at positions pos0.. and return the logits of the last one.
    pub fn forward(&mut self, tokens: &[u32], pos0: usize) -> Vec<f32> {
        let c = self.cfg.clone();
        let mut last = Vec::new();
        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = pos0 + ti;
            let mut x = self.row("token_embd.weight", tok as usize);
            for l in 0..c.n_layer {
                let p = |s: &str| format!("blk.{l}.{s}");
                let h = rmsnorm(&x, &self.vector(&p("attn_norm.weight")), c.rms_eps);
                let mut q = self.matvec(&p("attn_q.weight"), &h);
                let mut k = self.matvec(&p("attn_k.weight"), &h);
                let mut v = self.matvec(&p("attn_v.weight"), &h);
                if c.qkv_bias {
                    for (vec, n) in [
                        (&mut q, "attn_q.bias"),
                        (&mut k, "attn_k.bias"),
                        (&mut v, "attn_v.bias"),
                    ] {
                        for (a, b) in vec.iter_mut().zip(self.vector(&p(n))) {
                            *a += b;
                        }
                    }
                }
                if c.qk_norm {
                    let qn = self.vector(&p("attn_q_norm.weight"));
                    let kn = self.vector(&p("attn_k_norm.weight"));
                    q = q
                        .chunks(c.head_dim)
                        .flat_map(|h| rmsnorm(h, &qn, c.rms_eps))
                        .collect();
                    k = k
                        .chunks(c.head_dim)
                        .flat_map(|h| rmsnorm(h, &kn, c.rms_eps))
                        .collect();
                }
                self.rope(&mut q, pos);
                self.rope(&mut k, pos);
                let kv = c.kv_dim();
                self.k_cache[l].truncate(pos * kv);
                self.v_cache[l].truncate(pos * kv);
                self.k_cache[l].extend(&k);
                self.v_cache[l].extend(&v);
                let group = c.n_head / c.n_head_kv;
                let scale = 1.0 / (c.head_dim as f32).sqrt();
                let mut att = vec![0f32; c.q_dim()];
                for hh in 0..c.n_head {
                    let kvh = hh / group;
                    let qh = &q[hh * c.head_dim..(hh + 1) * c.head_dim];
                    let scores: Vec<f32> = (0..=pos)
                        .map(|j| {
                            let kj = &self.k_cache[l][j * kv + kvh * c.head_dim..][..c.head_dim];
                            qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                        })
                        .collect();
                    let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                    let sum: f32 = e.iter().sum();
                    for (j, p) in e.iter().enumerate() {
                        let vj = &self.v_cache[l][j * kv + kvh * c.head_dim..][..c.head_dim];
                        for d in 0..c.head_dim {
                            att[hh * c.head_dim + d] += p / sum * vj[d];
                        }
                    }
                }
                let o = self.matvec(&p("attn_output.weight"), &att);
                x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
                let h = rmsnorm(&x, &self.vector(&p("ffn_norm.weight")), c.rms_eps);
                let gate = self.matvec(&p("ffn_gate.weight"), &h);
                let up = self.matvec(&p("ffn_up.weight"), &h);
                let act: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let d = self.matvec(&p("ffn_down.weight"), &act);
                x.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
            }
            if ti + 1 == tokens.len() {
                let h = rmsnorm(&x, &self.vector("output_norm.weight"), c.rms_eps);
                let head = if c.tied_embeddings {
                    "token_embd.weight"
                } else {
                    "output.weight"
                };
                last = self.matvec(head, &h);
            }
        }
        last
    }
}
