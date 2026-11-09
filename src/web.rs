//! WebAssembly bindings used by the browser worker.

use crate::chat::{Message, Options};
use crate::gguf::{self, ParseError};
use crate::gpu::{self, engine::{Engine, Options as EngineOptions}};
use crate::source::WeightSource;
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
    /// Parts of a split model; the other parts' headers are needed before it can be described.
    split: Option<usize>,
}

#[derive(Serialize)]
struct ModelInfo {
    config: model::Config,
    parameters: u64,
    /// Parameters each token uses (fewer than `parameters` for mixture-of-experts models).
    active_parameters: u64,
    /// GPU bytes when every weight is resident, in the fast and in the file-sized layout.
    fast_bytes: u64,
    compact_bytes: u64,
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
    for (name, rows, k, experts) in model::matrices(&config) {
        let t = g.tensor(&name).unwrap();
        let rows = rows * experts;
        *weight_types
            .entry(quant::type_name(t.ggml_type).to_string())
            .or_insert(0) += (rows * k) as u64;
        parameters += (rows * k) as u64;
        gpu_weight_bytes += quant::GpuFormat::for_type(t.ggml_type)
            .unwrap()
            .bytes(k, rows);
        file_bytes += quant::row_bytes(t.ggml_type, k)? as u64 * rows as u64;
    }
    let mut compact_bytes = 0;
    for (name, rows, k, experts) in model::matrices(&config) {
        let t = g.tensor(&name).unwrap();
        compact_bytes += quant::GpuFormat::choose(t.ggml_type, quant::Layout::Compact)
            .unwrap()
            .bytes(k, rows * experts);
    }
    for (_, len) in model::vectors(&config) {
        parameters += len as u64;
        gpu_weight_bytes += len as u64 * 4;
        compact_bytes += len as u64 * 4;
    }
    let active_parameters = config.parameters().1;
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
        active_parameters,
        fast_bytes: gpu_weight_bytes,
        compact_bytes,
        weight_types,
        gpu_weight_bytes,
        file_bytes,
        chat_template: g.str("tokenizer.chat_template").is_some(),
        tokenizer,
        license: g.str("general.license").map(str::to_string),
    })
}

fn inspection(r: Result<gguf::Gguf, ParseError>) -> Inspection {
    let none = Inspection {
        need: None,
        error: None,
        model: None,
        split: None,
    };
    match r {
        Err(ParseError::NeedMore(n)) => Inspection { need: Some(n), ..none },
        Err(e) => Inspection { error: Some(e.to_string()), ..none },
        Ok(g) => match describe(&g) {
            Ok(m) => Inspection { model: Some(m), ..none },
            Err(e) => Inspection { error: Some(e), ..none },
        },
    }
}

/// Parse a file prefix. Returns `{need}` when more bytes are required, `{split}` for the first
/// part of a split model, `{error}` when the model cannot run, otherwise `{model}`.
#[wasm_bindgen]
pub fn inspect(header: &[u8]) -> String {
    match gguf::parse(header) {
        Ok(g) if g.split_count() > 1 => json(&Inspection {
            need: None,
            error: None,
            model: None,
            split: Some(g.split_count()),
        }),
        r => json(&inspection(r)),
    }
}

fn merged(headers: &js_sys::Array) -> Result<gguf::Gguf, String> {
    let parts = headers
        .iter()
        .map(|h| gguf::parse(&js_sys::Uint8Array::new(&h).to_vec()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    gguf::Gguf::merge(parts)
}

/// Describe a split model from the headers of all its parts, in order.
#[wasm_bindgen]
pub fn inspect_parts(headers: js_sys::Array) -> String {
    json(&inspection(merged(&headers).map_err(ParseError::Invalid)))
}

/// Estimated GPU bytes for activations and the KV cache at a context length.
#[wasm_bindgen]
pub fn state_bytes(headers: js_sys::Array, ctx: u32) -> Result<f64, JsValue> {
    let g = merged(&headers).map_err(err)?;
    let cfg = model::config(&g).map_err(err)?;
    Ok(Engine::state_bytes(&cfg, ctx as usize) as f64)
}

/// Reads weights through a synchronous JavaScript function `(file, offset, target)` that fills
/// `target` (a view of WebAssembly memory) from an OPFS access handle or a `File`.
struct JsSource(js_sys::Function);

impl WeightSource for JsSource {
    fn read(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; len];
        // The view stays valid because the callback does not call back into WebAssembly.
        let view = unsafe { js_sys::Uint8Array::view_mut_raw(buf.as_mut_ptr(), len) };
        self.0
            .call3(&JsValue::NULL, &JsValue::from(file as f64), &JsValue::from(offset as f64), &view)
            .map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}")))?;
        Ok(buf)
    }
}

#[derive(serde::Deserialize, Default)]
struct LoadOptions {
    #[serde(default)]
    low_power: bool,
    /// GPU bytes for weights; larger models stream the rest from storage.
    #[serde(default)]
    budget: Option<f64>,
}

#[wasm_bindgen]
pub struct Loader {
    engine: Option<Engine>,
    header: gguf::Gguf,
}

/// Open the GPU and allocate weight buffers for the model whose headers are given (one per file
/// of a split model). When the GPU runs out of memory, retries keeping less of the model on it.
#[wasm_bindgen]
pub async fn create_loader(headers: js_sys::Array, options: String) -> Result<Loader, JsValue> {
    let g = merged(&headers).map_err(err)?;
    let o: LoadOptions = serde_json::from_str(&options).unwrap_or_default();
    let preference = if o.low_power {
        wgpu::PowerPreference::LowPower
    } else {
        wgpu::PowerPreference::HighPerformance
    };
    let mut gpu = gpu::init_with(preference).await.map_err(err)?;
    let mut budget = o.budget.map(|b| b as u64);
    for _ in 0..6 {
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let opts = EngineOptions {
            binding_limit: None,
            layout: crate::quant::Layout::Fast,
            budget,
        };
        let engine = Engine::with_options(gpu, &g, opts).map_err(err)?;
        match scope.pop().await {
            None => {
                return Ok(Loader {
                    engine: Some(engine),
                    header: g,
                })
            }
            Some(_) => {
                // Give the memory back, then try again with 60% of what was asked for.
                let tried = engine.weight_bytes;
                gpu = engine.into_gpu();
                budget = Some((tried as f64 * 0.6) as u64);
                if budget.unwrap() < 64 << 20 {
                    break;
                }
            }
        }
    }
    Err(err("the GPU ran out of memory even for a small part of this model"))
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

    /// Which weights stay on the GPU and which are read while the model runs, as JSON.
    pub fn placement(&self) -> String {
        json(&self.engine.as_ref().unwrap().placement)
    }

    /// The reader for weights that are not resident.
    pub fn set_source(&self, read: js_sys::Function) {
        self.engine.as_ref().unwrap().set_source(Box::new(JsSource(read)));
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
    placement: &'a crate::gpu::engine::Placement,
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
            placement: &s.engine.placement,
        })
    }

    /// Streaming counters since the last call, as JSON.
    pub fn stats(&self) -> String {
        json(&self.inner.borrow().engine.take_stream_stats())
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
