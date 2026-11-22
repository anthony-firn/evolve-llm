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
/// `target` (a view of WebAssembly memory) from an OPFS access handle, a `File`, or bytes a
/// network reader fetched beforehand. The optional `prefetch(spans)` receives the spans the
/// next reads will ask for, as a Float64Array of (file, offset, length, every_token) quads, and
/// returns a Promise that settles when they can be read.
struct JsSource {
    read: js_sys::Function,
    prefetch: Option<js_sys::Function>,
}

fn js_error(e: JsValue) -> String {
    e.as_string()
        .or_else(|| js_sys::Reflect::get(&e, &"message".into()).ok().and_then(|m| m.as_string()))
        .unwrap_or_else(|| format!("{e:?}"))
}

impl WeightSource for JsSource {
    fn read(&self, file: usize, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; len];
        // The view stays valid because the callback does not call back into WebAssembly.
        let view = unsafe { js_sys::Uint8Array::view_mut_raw(buf.as_mut_ptr(), len) };
        self.read
            .call3(&JsValue::NULL, &JsValue::from(file as f64), &JsValue::from(offset as f64), &view)
            .map_err(js_error)?;
        Ok(buf)
    }

    fn prefetch<'a>(&'a self, spans: &'a [crate::source::Span]) -> crate::source::Ready<'a> {
        Box::pin(async move {
            let Some(f) = &self.prefetch else { return Ok(()) };
            let quads: Vec<f64> = spans
                .iter()
                .flat_map(|s| [s.file as f64, s.offset as f64, s.len as f64, s.every_token as u8 as f64])
                .collect();
            let arr = js_sys::Float64Array::from(&quads[..]);
            let promise = f.call1(&JsValue::NULL, &arr).map_err(js_error)?;
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise))
                .await
                .map_err(js_error)?;
            Ok(())
        })
    }
}

/// The layers after this device's, run by the person's other devices: `forward(hidden, n,
/// pos0)` (a Float32Array) resolves to the hidden state after the model's last layer;
/// `control(op)` passes on "snapshot" and "restore".
struct JsRemote {
    forward: js_sys::Function,
    control: js_sys::Function,
}

impl crate::gpu::engine::Remote for JsRemote {
    fn forward<'a>(&'a self, hidden: Vec<f32>, n: usize, pos0: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>, String>> + 'a>> {
        Box::pin(async move {
            let arr = js_sys::Float32Array::from(&hidden[..]);
            let p = self
                .forward
                .call3(&JsValue::NULL, &arr, &JsValue::from(n as f64), &JsValue::from(pos0 as f64))
                .map_err(js_error)?;
            let v = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(p)).await.map_err(js_error)?;
            Ok(js_sys::Float32Array::new(&v).to_vec())
        })
    }

    fn snapshot(&self) {
        let _ = self.control.call1(&JsValue::NULL, &JsValue::from_str("snapshot"));
    }

    fn restore(&self) {
        let _ = self.control.call1(&JsValue::NULL, &JsValue::from_str("restore"));
    }
}

/// What each part of a model weighs, for sharing it among devices, as JSON: `n_layer`,
/// `embed` and `head` bytes, `layers` (bytes of each), and `hidden` (floats per token handed
/// from one device to the next before each layer, 0 to n_layer).
#[wasm_bindgen]
pub fn layer_bytes(headers: js_sys::Array) -> Result<String, JsValue> {
    let g = merged(&headers).map_err(err)?;
    let cfg = model::config(&g).map_err(err)?;
    let mut layers = vec![0u64; cfg.n_layer];
    let (mut embed, mut head) = (0u64, 0u64);
    for t in &g.tensors {
        let rows: u64 = t.dims[1..].iter().product();
        let bytes = crate::quant::row_bytes(t.ggml_type, t.dims[0] as usize).map_or(0, |b| b as u64 * rows);
        let layer = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok());
        match layer {
            Some(l) if l < cfg.n_layer => layers[l] += bytes,
            Some(_) => {}
            None if t.name == "token_embd.weight" => embed += bytes,
            None => head += bytes,
        }
    }
    let bs = cfg.attn_res_block;
    let hidden: Vec<usize> = (0..=cfg.n_layer).map(|l| cfg.n_embd * (1 + if bs > 0 { l.div_ceil(bs) } else { 0 })).collect();
    Ok(json(&serde_json::json!({
        "n_layer": cfg.n_layer,
        "embed": embed,
        "head": head,
        "layers": layers,
        "hidden": hidden,
        "tied": cfg.tied_embeddings,
        // The attention cache a layer keeps per token (keys and values in f16): with the context
        // length, what a layer needs besides its weights, for planning where layers fit.
        "kv_per_token": 2 * cfg.kv_dim() * 2,
    })))
}

#[derive(serde::Deserialize, Default)]
struct LoadOptions {
    #[serde(default)]
    low_power: bool,
    /// GPU bytes for weights; larger models stream the rest from storage.
    #[serde(default)]
    budget: Option<f64>,
    /// The layers this device runs when the model is shared with the person's other devices.
    #[serde(default)]
    stage: Option<crate::gpu::engine::Stage>,
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
            stage: o.stage,
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

    /// The reader for weights that are not resident, and optionally a prefetch function for
    /// readers that are slow to start a read (the network).
    pub fn set_source(&self, read: js_sys::Function, prefetch: Option<js_sys::Function>) {
        self.engine.as_ref().unwrap().set_source(Box::new(JsSource { read, prefetch }));
    }

    pub fn upload(&mut self, name: &str, row0: u32, bytes: &[u8]) -> Result<(), JsValue> {
        self.engine
            .as_mut()
            .ok_or("finished")?
            .upload(name, row0 as usize, bytes)
            .map_err(err)
    }

    /// For the device the chat runs on, when the person's other devices run the layers after
    /// its own (see `JsRemote`). Call before `finish`.
    pub fn set_remote(&self, forward: js_sys::Function, control: js_sys::Function) {
        self.engine.as_ref().unwrap().set_remote(Some(Rc::new(JsRemote { forward, control })));
    }

    /// For a device that runs layers in the middle or at the end of a model for another device:
    /// allocate the context and return the stage.
    pub fn finish_stage(&mut self, ctx: u32) -> Result<Layers, JsValue> {
        let mut engine = self.engine.take().ok_or("finished")?;
        engine.finish(ctx as usize).map_err(err)?;
        Ok(Layers { engine: Rc::new(engine) })
    }

    /// Allocate the context and return a chat session.
    pub fn finish(&mut self, ctx: u32) -> Result<Chat, JsValue> {
        let mut engine = self.engine.take().ok_or("finished")?;
        engine.finish(ctx as usize).map_err(err)?;
        let session = Session::new(engine, &self.header).map_err(err)?;
        Ok(Chat {
            inner: Rc::new(RefCell::new(session)),
            decoder: RefCell::new(StreamDecoder::default()),
            vision: RefCell::new(None),
        })
    }
}

/// Layers this device runs for another device's chat.
#[wasm_bindgen]
pub struct Layers {
    engine: Rc<Engine>,
}

#[wasm_bindgen]
impl Layers {
    /// Resolves to the hidden state after this stage's last layer (a Float32Array), given the
    /// one before its first layer for `n` tokens from `pos0`. Calls must not overlap.
    pub fn forward(&self, hidden: js_sys::Float32Array, n: u32, pos0: u32) -> js_sys::Promise {
        let engine = self.engine.clone();
        let h = hidden.to_vec();
        wasm_bindgen_futures::future_to_promise(async move {
            let out = engine.forward_hidden(&h, n as usize, pos0 as usize).await.map_err(err)?;
            Ok(js_sys::Float32Array::from(&out[..]).into())
        })
    }

    pub fn snapshot(&self) {
        self.engine.snapshot();
    }

    pub fn restore(&self) {
        self.engine.restore();
    }

    /// Weights read while running since the last call, as JSON.
    pub fn stats(&self) -> String {
        json(&self.engine.take_stream_stats())
    }

    pub fn placement(&self) -> String {
        json(&self.engine.placement)
    }

    pub fn weight_bytes(&self) -> f64 {
        self.engine.weight_bytes as f64
    }

    pub fn context_length(&self) -> u32 {
        self.engine.context_length() as u32
    }
}

#[wasm_bindgen]
pub struct Chat {
    inner: Rc<RefCell<Session>>,
    decoder: RefCell<StreamDecoder>,
    /// The image encoder (an mmproj file), when one is attached.
    vision: RefCell<Option<Rc<crate::gpu::vision::VisionGpu>>>,
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

    /// Load an image encoder (llama.cpp mmproj GGUF) on the chat's GPU. Returns the size images'
    /// sides must be multiples of.
    pub fn attach_vision(&self, bytes: &[u8]) -> Result<u32, JsValue> {
        let s = self.inner.borrow();
        if s.image_pad().is_none() {
            return Err(err("this model does not read images"));
        }
        let v = crate::gpu::vision::VisionGpu::load(&s.engine.gpu, bytes).map_err(err)?;
        let unit = v.cfg.unit() as u32;
        drop(s);
        *self.vision.borrow_mut() = Some(Rc::new(v));
        Ok(unit)
    }

    /// `prefill` for a prompt with one <|image_pad|> per image: `rgb` holds the images' pixels
    /// one after another (3 bytes each), `sizes` each one's width and height. Resolves to JSON
    /// `{reused, computed}`.
    pub fn prefill_images(&self, tokens: Vec<u32>, rgb: Vec<u8>, sizes: Vec<u32>) -> js_sys::Promise {
        let inner = self.inner.clone();
        let vision = self.vision.borrow().clone();
        self.decoder.borrow_mut().flush();
        wasm_bindgen_futures::future_to_promise(async move {
            let vision = vision.ok_or_else(|| err("no image encoder is attached"))?;
            let mut images = Vec::new();
            let mut at = 0usize;
            for wh in sizes.chunks_exact(2) {
                let n = wh[0] as usize * wh[1] as usize * 3;
                let px = rgb.get(at..at + n).ok_or_else(|| err("the pixels do not match the image sizes"))?;
                images.push((px, wh[0] as usize, wh[1] as usize));
                at += n;
            }
            let mut s = inner
                .try_borrow_mut()
                .map_err(|_| err("the model is busy"))?;
            let p = s.prefill_rgb(&tokens, &vision, &images).await.map_err(err)?;
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

    /// Resolves to the token ids the model read this step (a Uint32Array): one, or more when it
    /// streams its weights and a draft from earlier text was verified in the same pass.
    pub fn next_many(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut s = inner
                .try_borrow_mut()
                .map_err(|_| err("the model is busy"))?;
            let ids = s.next_many().await.map_err(err)?;
            Ok(js_sys::Uint32Array::from(&ids[..]).into())
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

    /// The text of a control token that `piece` hides (empty for every other token), so the
    /// page can read markup such as tool calls that models write with control tokens.
    pub fn control_text(&self, id: u32) -> String {
        self.inner.borrow().tok.control_text(id).unwrap_or("").to_string()
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
