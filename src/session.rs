//! A conversation with one loaded model: prompt formatting, prefix reuse of
//! the KV cache across turns, and sampling.

use crate::chat::{ChatTemplate, Message, Options};
use crate::gguf::Gguf;
use crate::gpu::engine::Engine;
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
        Ok(Session {
            engine,
            tok,
            template,
            sampler: Sampler::new(SamplerConfig::default()),
            cached: Vec::new(),
            logits: Vec::new(),
            snapshot: None,
        })
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
    }

    /// Bring the cache to exactly `tokens`, computing only the part that differs.
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<Prefill, String> {
        if tokens.is_empty() {
            return Err("empty prompt".into());
        }
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
        if self.logits.is_empty() {
            return Err("prefill a prompt first".into());
        }
        let id = self.sampler.sample(&self.logits, &self.cached);
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
