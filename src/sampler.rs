//! Token sampling: repetition penalty, temperature, top-k, top-p and min-p.

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    state: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Sampler {
        let state = if cfg.seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            cfg.seed
        };
        Sampler { cfg, state }
    }

    pub fn reseed(&mut self, seed: u64) {
        self.state = seed.max(1);
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let v = self.state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pick the next token. `recent` holds previous tokens for the repetition penalty.
    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let mut l = logits.to_vec();
        let c = &self.cfg;
        if c.repeat_penalty != 1.0 && c.repeat_last_n > 0 {
            let start = recent.len().saturating_sub(c.repeat_last_n);
            let mut seen = std::collections::HashSet::new();
            for &t in &recent[start..] {
                if (t as usize) < l.len() && seen.insert(t) {
                    let v = &mut l[t as usize];
                    *v = if *v > 0.0 {
                        *v / c.repeat_penalty
                    } else {
                        *v * c.repeat_penalty
                    };
                }
            }
        }
        if c.temperature <= 0.0 {
            return argmax(&l);
        }
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        let k = if c.top_k == 0 {
            l.len()
        } else {
            c.top_k.min(l.len())
        };
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| l[b as usize].total_cmp(&l[a as usize]));
            idx.truncate(k);
        }
        idx.sort_unstable_by(|&a, &b| l[b as usize].total_cmp(&l[a as usize]));
        let max = l[idx[0] as usize];
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((l[i as usize] - max) / c.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        // min-p keeps tokens at least min_p times as likely as the best one.
        let mut keep = probs.len();
        if c.min_p > 0.0 {
            let floor = probs[0] * c.min_p;
            keep = probs.iter().take_while(|&&p| p >= floor).count().max(1);
        }
        if c.top_p < 1.0 {
            let mut cum = 0.0;
            for (i, p) in probs[..keep].iter().enumerate() {
                cum += p;
                if cum >= c.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[keep - 1]
    }
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max_and_penalty_applies() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&[0.1, 3.0, 2.0], &[]), 1);
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&[0.1, 3.0, 2.0], &[1]), 2);
    }

    #[test]
    fn sampling_respects_top_k() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 7,
            ..Default::default()
        });
        for _ in 0..200 {
            let t = s.sample(&[5.0, 0.0, 4.9, -1.0], &[]);
            assert!(t == 0 || t == 2);
        }
    }
}
