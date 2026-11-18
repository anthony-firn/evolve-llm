//! A conversation with one loaded model: prompt formatting, prefix reuse of
//! the KV cache across turns, and sampling.

use crate::chat::{ChatTemplate, Message, Options};
use crate::gguf::Gguf;
use crate::gpu::engine::{Engine, SPEC_MAX};
use std::collections::VecDeque;

/// Speculative decoding for models that stream their weights, where a pass costs about the same
/// for a few tokens as for one: the continuation of the latest earlier occurrence of the last
/// few tokens is drafted (prompt lookup) and verified in one pass. Drafting pauses when recent
/// drafts were mostly rejected.
#[derive(Default)]
struct Speculation {
    on: bool,
    /// (drafted, accepted) of recent verification passes.
    recent: VecDeque<(usize, usize)>,
    /// Tokens to decode one at a time before drafting again.
    pause: usize,
}

impl Speculation {
    fn draft_len(&mut self) -> usize {
        if !self.on {
            return 0;
        }
        if self.pause > 0 {
            self.pause -= 1;
            return 0;
        }
        let (d, a) = self.recent.iter().fold((0, 0), |(d, a), &(x, y)| (d + x, a + y));
        if self.recent.len() >= 4 && a * 10 < d * 3 {
            self.pause = 16;
            self.recent.clear();
            return 0;
        }
        SPEC_MAX - 2
    }

    fn record(&mut self, drafted: usize, accepted: usize) {
        self.recent.push_back((drafted, accepted));
        if self.recent.len() > 8 {
            self.recent.pop_front();
        }
    }
}
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

pub struct Session {
    pub engine: Engine,
    pub tok: Tokenizer,
    pub template: ChatTemplate,
    pub sampler: Sampler,
    /// Tokens whose keys and values are in the cache, in order.
    cached: Vec<u32>,
    logits: Vec<f32>,
    /// Tokens the engine's recurrent-state snapshot has read (recurrent models only).
    snapshot: Option<usize>,
    /// A token already chosen (a draft's first mismatch) that the next step emits.
    pending: Option<u32>,
    spec: Speculation,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Prefill {
    /// Prompt tokens already in the cache from the previous turn.
    pub reused: usize,
    /// Prompt tokens computed now.
    pub computed: usize,
}

impl Session {
    pub fn new(engine: Engine, g: &Gguf) -> Result<Session, String> {
        let tok = Tokenizer::from_gguf(g)?;
        let template = ChatTemplate::from_gguf(g, &tok);
        let mut s = Session {
            engine,
            tok,
            template,
            sampler: Sampler::new(SamplerConfig::default()),
            cached: Vec::new(),
            logits: Vec::new(),
            snapshot: None,
            pending: None,
            spec: Speculation::default(),
        };
        s.spec.on = s.engine.streaming();
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(v) = std::env::var("EVOLVE_LLM_SPECULATE") {
            s.spec.on = v == "1";
        }
        Ok(s)
    }

    pub fn render(&self, messages: &[Message], opts: &Options) -> Result<String, String> {
        self.template.render(messages, true, opts)
    }

    /// Prompt tokens for a conversation, ending with the assistant's turn.
    pub fn prompt_tokens(&self, messages: &[Message], opts: &Options) -> Result<Vec<u32>, String> {
        let text = self.render(messages, opts)?;
        Ok(self.encode_prompt(&text))
    }

    /// Tokenize formatted prompt text, adding the start token when the model expects one.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut t = self.tok.encode(text, false, true);
        if self.tok.add_bos {
            if let Some(b) = self.tok.bos {
                if t.first() != Some(&b) {
                    t.insert(0, b);
                }
            }
        }
        t
    }

    pub fn context_used(&self) -> usize {
        self.cached.len()
    }

    pub fn reset(&mut self) {
        self.cached.clear();
        self.logits.clear();
        self.snapshot = None;
        self.pending = None;
    }

    /// Turn speculative decoding on or off (on by default when the model streams).
    pub fn set_speculation(&mut self, on: bool) {
        self.spec.on = on;
    }

    /// Bring the cache to exactly `tokens`, computing only the part that differs.
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<Prefill, String> {
        if tokens.is_empty() {
            return Err("empty prompt".into());
        }
        self.pending = None;
        let ctx = self.engine.context_length();
        if tokens.len() >= ctx {
            return Err(format!(
                "the prompt has {} tokens; the context holds {ctx}",
                tokens.len()
            ));
        }
        let mut common = self
            .cached
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        // The last prompt token always runs, because its logits pick the first reply token.
        common = common.min(tokens.len() - 1);
        // A recurrent state cannot be rewound. When this prompt leaves the cache, or a pass
        // stopped part way (the state no longer matches the cache), go back to the snapshot
        // taken at the end of the last prompt if it is still a prefix, or start over.
        let stale = self.engine.is_recurrent() && self.engine.recurrent_position() != self.cached.len();
        if self.engine.is_recurrent() && (stale || common < self.cached.len()) {
            common = match self.snapshot.filter(|&p| p <= common) {
                Some(p) if self.engine.restore() == Some(p) => p,
                _ => 0,
            };
        }
        self.cached.truncate(common);
        let rest = &tokens[common..];
        self.logits = self.engine.forward(rest, common).await?;
        self.cached.extend_from_slice(rest);
        if self.engine.is_recurrent() {
            self.engine.snapshot();
            self.snapshot = Some(self.cached.len());
        }
        Ok(Prefill {
            reused: common,
            computed: rest.len(),
        })
    }

    /// Sample the next token and, unless it ends the turn, advance the model past it.
    pub async fn next(&mut self) -> Result<u32, String> {
        if self.logits.is_empty() && self.pending.is_none() {
            return Err("prefill a prompt first".into());
        }
        let id = match self.pending.take() {
            Some(p) => p,
            None => self.sampler.sample(&self.logits, &self.cached),
        };
        if self.tok.is_end_of_generation(id) {
            self.logits.clear();
            return Ok(id);
        }
        if self.cached.len() + 1 >= self.engine.context_length() {
            return Err("the context is full".into());
        }
        let pos = self.cached.len();
        self.logits = self.engine.forward(&[id], pos).await?;
        self.cached.push(id);
        Ok(id)
    }

    /// The continuation of the latest earlier occurrence of the context's last few tokens.
    fn draft(&self, last: u32, k: usize) -> Vec<u32> {
        let mut h = self.cached.clone();
        h.push(last);
        let from = h.len().saturating_sub(8192);
        for n in [4, 3, 2] {
            if h.len() < from + n + 1 {
                continue;
            }
            let tail = &h[h.len() - n..];
            if let Some(j) = (from..h.len() - n).rev().find(|&j| &h[j..j + n] == tail) {
                let end = (j + n + k).min(h.len());
                return h[j + n..end].to_vec();
            }
        }
        Vec::new()
    }

    /// Like `next`, but when speculation is on, verify a drafted continuation in the same pass:
    /// returns the tokens the model has now read (the next token, then the accepted part of the
    /// draft). Each position samples from the model's own logits and a draft token is accepted
    /// only when it is what was sampled, so the text is what one-at-a-time decoding gives
    /// (identical when greedy). The first mismatch becomes the next token.
    pub async fn next_many(&mut self) -> Result<Vec<u32>, String> {
        let k = self.spec.draft_len();
        if k == 0 {
            return Ok(vec![self.next().await?]);
        }
        if self.logits.is_empty() && self.pending.is_none() {
            return Err("prefill a prompt first".into());
        }
        let id = match self.pending.take() {
            Some(p) => p,
            None => self.sampler.sample(&self.logits, &self.cached),
        };
        if self.tok.is_end_of_generation(id) {
            self.logits.clear();
            return Ok(vec![id]);
        }
        let pos = self.cached.len();
        let ctx = self.engine.context_length();
        if pos + 1 >= ctx {
            return Err("the context is full".into());
        }
        let room = ctx - pos - 2;
        let draft = self.draft(id, k.min(room));
        if draft.is_empty() {
            self.logits = self.engine.forward(&[id], pos).await?;
            self.cached.push(id);
            return Ok(vec![id]);
        }
        let recurrent = self.engine.is_recurrent();
        if recurrent {
            self.engine.spec_save();
        }
        let mut batch = vec![id];
        batch.extend_from_slice(&draft);
        let rows = self.engine.forward_all(&batch, pos).await?;
        let mut history = self.cached.clone();
        history.push(id);
        let mut accepted = 0;
        let mut next = None;
        for (i, row) in rows.iter().enumerate() {
            let s = self.sampler.sample(row, &history);
            if i < draft.len() && s == draft[i] && !self.tok.is_end_of_generation(s) {
                history.push(s);
                accepted += 1;
            } else {
                next = Some(s);
                break;
            }
        }
        if recurrent && accepted < draft.len() {
            // The recurrent state read the whole draft: go back and read only what was kept.
            self.engine.spec_restore();
            self.engine.forward(&batch[..accepted + 1], pos).await?;
        }
        self.cached.extend_from_slice(&batch[..accepted + 1]);
        self.logits = rows[accepted].clone();
        self.pending = next;
        self.spec.record(draft.len(), accepted);
        Ok(batch[..accepted + 1].to_vec())
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn set_sampler(&mut self, cfg: SamplerConfig) {
        let seed = cfg.seed;
        self.sampler = Sampler::new(cfg);
        if seed != 0 {
            self.sampler.reseed(seed);
        }
    }
}
