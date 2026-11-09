//! WebAssembly bindings used by the browser worker.

use crate::chat::{Message, Options};
use crate::gguf::{self, ParseError};
use crate::gpu::{self, engine::Engine};
use crate::model;
use crate::quant;
use crate::sampler::SamplerConfig;
use crate::session::Session;
use crate::tokenizer::StreamDecoder;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn json<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".into())
}

/// Open the GPU once to report what the browser offers.
#[wasm_bindgen]
pub async fn gpu_info() -> Result<String, JsValue> {
    let g = gpu::init().await.map_err(err)?;
    Ok(json(&g.info))
}

#[derive(Serialize)]
struct Inspection {
    need: Option<u64>,
    error: Option<String>,
    model: Option<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    config: model::Config,
    parameters: u64,
    weight_types: BTreeMap<String, u64>,
    gpu_weight_bytes: u64,
    file_bytes: u64,
    chat_template: bool,
    tokenizer: String,
    license: Option<String>,
}

fn describe(g: &gguf::Gguf) -> Result<ModelInfo, String> {
    let config = model::config(g)?;
    let mut weight_types = BTreeMap::new();
    let mut parameters = 0;
    let mut gpu_weight_bytes = 0;
    let mut file_bytes = 0;
    for (name, rows, k) in model::matrices(&config) {
        let t = g.tensor(&name).unwrap();
        *weight_types
            .entry(quant::type_name(t.ggml_type).to_string())
            .or_insert(0) += (rows * k) as u64;
        parameters += (rows * k) as u64;
        let (c, s) = quant::GpuFormat::for_type(t.ggml_type)
            .unwrap()
            .sizes(k, rows);
        gpu_weight_bytes += c + s;
        file_bytes += quant::row_bytes(t.ggml_type, k)? as u64 * rows as u64;
    }
    for (_, len) in model::vectors(&config) {
        parameters += len as u64;
        gpu_weight_bytes += len as u64 * 4;
    }
    let tokenizer = g
        .str("tokenizer.ggml.model")
        .unwrap_or("unknown")
        .to_string();
    if !matches!(tokenizer.as_str(), "gpt2" | "llama") {
        return Err(format!("the {tokenizer} tokenizer is not supported yet"));
    }
    Ok(ModelInfo {
        config,
        parameters,
        weight_types,
        gpu_weight_bytes,
        file_bytes,
        chat_template: g.str("tokenizer.chat_template").is_some(),
        tokenizer,
        license: g.str("general.license").map(str::to_string),
    })
}

/// Parse a file prefix. Returns `{need}` when more bytes are required,
/// `{error}` when the model cannot run, otherwise `{model}`.
#[wasm_bindgen]
pub fn inspect(header: &[u8]) -> String {
    let r = match gguf::parse(header) {
        Err(ParseError::NeedMore(n)) => Inspection {
            need: Some(n),
            error: None,
            model: None,
        },
        Err(e) => Inspection {
            need: None,
            error: Some(e.to_string()),
            model: None,
        },
        Ok(g) => match describe(&g) {
            Ok(m) => Inspection {
                need: None,
                error: None,
                model: Some(m),
            },
            Err(e) => Inspection {
                need: None,
                error: Some(e),
                model: None,
            },
        },
    };
    json(&r)
}

/// Estimated GPU bytes for activations and the KV cache at a context length.
#[wasm_bindgen]
pub fn state_bytes(header: &[u8], ctx: u32) -> Result<f64, JsValue> {
    let g = gguf::parse(header).map_err(err)?;
    let cfg = model::config(&g).map_err(err)?;
    Ok(Engine::state_bytes(&cfg, ctx as usize) as f64)
}

#[wasm_bindgen]
pub struct Loader {
    engine: Option<Engine>,
    header: gguf::Gguf,
}

/// Open the GPU and allocate weight buffers for the model whose header is given.
#[wasm_bindgen]
pub async fn create_loader(header: Vec<u8>, low_power: bool) -> Result<Loader, JsValue> {
    let g = gguf::parse(&header).map_err(err)?;
    let preference = if low_power {
        wgpu::PowerPreference::LowPower
    } else {
        wgpu::PowerPreference::HighPerformance
    };
    let gpu = gpu::init_with(preference).await.map_err(err)?;
    let engine = Engine::new(gpu, &g).map_err(err)?;
    Ok(Loader {
        engine: Some(engine),
        header: g,
    })
}

#[wasm_bindgen]
impl Loader {
    /// Tensors to upload, sorted by file offset, as JSON.
    pub fn plan(&self) -> String {
        json(&self.engine.as_ref().unwrap().plan())
    }

    pub fn device(&self) -> String {
        json(&self.engine.as_ref().unwrap().gpu.info)
    }

    pub fn upload(&mut self, name: &str, row0: u32, bytes: &[u8]) -> Result<(), JsValue> {
        self.engine
            .as_mut()
            .ok_or("finished")?
            .upload(name, row0 as usize, bytes)
            .map_err(err)
    }

    /// Allocate the context and return a chat session.
    pub fn finish(&mut self, ctx: u32) -> Result<Chat, JsValue> {
        let mut engine = self.engine.take().ok_or("finished")?;
        engine.finish(ctx as usize).map_err(err)?;
        let session = Session::new(engine, &self.header).map_err(err)?;
        Ok(Chat {
            inner: Rc::new(RefCell::new(session)),
            decoder: RefCell::new(StreamDecoder::default()),
        })
    }
}

#[wasm_bindgen]
pub struct Chat {
    inner: Rc<RefCell<Session>>,
    decoder: RefCell<StreamDecoder>,
}

#[derive(Serialize)]
struct ChatInfo<'a> {
    name: &'a str,
    context: usize,
    state_bytes: u64,
    weight_bytes: u64,
    template_from_model: bool,
    end_tokens: Vec<u32>,
    device: &'a gpu::DeviceInfo,
}

fn parse_messages(messages: &str, options: &str) -> Result<(Vec<Message>, Options), JsValue> {
    let m: Vec<Message> = serde_json::from_str(messages).map_err(err)?;
    let o: Options = if options.is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(err)?
    };
    Ok((m, o))
}

#[wasm_bindgen]
impl Chat {
    pub fn info(&self) -> String {
        let s = self.inner.borrow();
        json(&ChatInfo {
            name: &s.engine.cfg.name,
            context: s.engine.context_length(),
            state_bytes: s.engine.state_memory(),
            weight_bytes: s.engine.weight_bytes,
            template_from_model: s.template.from_model,
            end_tokens: s.tok.end_tokens(),
            device: &s.engine.gpu.info,
        })
    }

    /// The formatted prompt text for a conversation.
    pub fn render(&self, messages: &str, options: &str) -> Result<String, JsValue> {
        let (m, o) = parse_messages(messages, options)?;
        self.inner.borrow().render(&m, &o).map_err(err)
    }

    /// Prompt tokens for a conversation, ready for `prefill`.
    pub fn prompt(&self, messages: &str, options: &str) -> Result<Vec<u32>, JsValue> {
        let (m, o) = parse_messages(messages, options)?;
        self.inner.borrow().prompt_tokens(&m, &o).map_err(err)
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.borrow().tok.encode(text, false, true)
    }

    /// Resolves to JSON `{reused, computed}`.
    pub fn prefill(&self, tokens: Vec<u32>) -> js_sys::Promise {
        let inner = self.inner.clone();
        self.decoder.borrow_mut().flush();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut s = inner
                .try_borrow_mut()
                .map_err(|_| err("the model is busy"))?;
            let p = s.prefill(&tokens).await.map_err(err)?;
            Ok(JsValue::from_str(&json(&p)))
        })
    }

    /// Resolves to the next token id.
    pub fn next(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut s = inner
                .try_borrow_mut()
                .map_err(|_| err("the model is busy"))?;
            let id = s.next().await.map_err(err)?;
            Ok(JsValue::from(id))
        })
    }

    /// Text for a generated token; incomplete UTF-8 is held until it completes.
    pub fn piece(&self, id: u32) -> String {
        let s = self.inner.borrow();
        self.decoder.borrow_mut().push(&s.tok, id)
    }

    pub fn flush(&self) -> String {
        self.decoder.borrow_mut().flush()
    }

    pub fn is_end(&self, id: u32) -> bool {
        self.inner.borrow().tok.is_end_of_generation(id)
    }

    pub fn set_sampler(&self, config: &str) -> Result<(), JsValue> {
        let cfg: SamplerConfig = serde_json::from_str(config).map_err(err)?;
        self.inner.borrow_mut().set_sampler(cfg);
        Ok(())
    }

    pub fn reset(&self) {
        self.inner.borrow_mut().reset();
        self.decoder.borrow_mut().flush();
    }

    pub fn context_used(&self) -> u32 {
        self.inner.borrow().context_used() as u32
    }

    pub fn context_length(&self) -> u32 {
        self.inner.borrow().engine.context_length() as u32
    }
}
