# Evolve LLM

Run open language models on your own GPU from a web page, including models far larger than your
GPU's memory. Evolve LLM is an inference engine written in Rust. It is compiled to WebAssembly
and computes every token with WebGPU compute shaders. Open the page, pick a model, and chat with
it. There is nothing to install, no account, and no server that sees your messages.

**Try it:** [evolvetech.dev/play/evolve-llm](https://www.evolvetech.dev/play/evolve-llm/)

**Status: early release.** The engine runs dense models (Llama, Mistral, Qwen2, Qwen3, SmolLM)
and mixture-of-experts models (Qwen3-MoE, Mixtral, OLMoE, DeepSeek V2 and V3, Kimi K2) from
GGUF files, single or split. It is tested in Chrome on Linux with NVIDIA and Intel GPUs and
against reference implementations (see [Verification](#verification)). The largest models in
the catalog have not been run end to end yet; see [Limits](#limits).

## Browser workspace and Pi

Open **Terminal** for a persistent WASM text workspace with pipes and file redirects. Load a
tool-capable model, then enter `pi <task>` to run Pi agent core with sandbox file and research
tools. Web access is explicit; there is no host shell or server inference. See
[commands, limits, permissions and self-hosting](docs/workbench.md).

## What it does

- **Downloads a model once.** Choose from the catalog, browse any Hugging Face repository, or
  paste a GGUF file's path. Files stream into the browser's private storage and resume after a
  dropped connection; split models download part by part. You can also open GGUF files already
  on your device; they are read in place.
- **Runs it on your GPU.** Weights upload in 16 MB slices, so loading never needs the whole
  file in memory.
- **Runs models that do not fit.** Give the model a GPU memory budget in Settings. What fits
  stays on the GPU; the rest is read from storage while the model runs. Mixture-of-experts
  models keep their attention weights on the GPU and read only the experts each token chooses,
  through an expert cache. The operating system's file cache keeps recently read weights in RAM.
- **Searches the web when you let it.** With Internet access on, models whose chat templates
  take tools can call `web_search` and `open_url`; other models get search results with each
  question. Searches go from the browser to Wikipedia and DuckDuckGo, and pages are read through
  Jina Reader. Each search and page shows above the answer with its sources.
- **Chats with the model's own template.** Prompts are formatted by the Jinja chat template in
  the GGUF file. Each new turn only computes the part of the conversation that is not already
  in the KV cache.
- **Shows its reasoning.** Qwen3 thinking is shown in a collapsible block and can be turned off.

With web access off, nothing leaves the device except model downloads from Hugging Face. There
are no analytics.

## Supported models

| | Supported |
|---|---|
| Dense architectures | `llama` (Llama 2 and 3, Mistral, SmolLM2, TinyLlama), `qwen2` (Qwen2 and 2.5, Coder), `qwen3` |
| Mixture of experts | `qwen3moe` (Qwen3 30B-A3B to Coder 480B-A35B), `llama` with experts (Mixtral), `olmoe`, `deepseek2` (DeepSeek V2, V3, V3.1 and R1, Kimi K2, Moonlight) |
| Attention | Grouped-query attention, multi-head latent attention (DeepSeek), YaRN and linear position scaling, Llama 3 rope factors |
| Weight formats | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, BF16, F32 |
| Tokenizers | Byte-level BPE (GPT-2, Llama 3, Qwen2, SmolLM, DeepSeek V3 and Kimi K2 pre-tokenizers) and SentencePiece-style BPE |
| Files | Single GGUF files and split sets (`name-00001-of-00004.gguf`) |
| Not yet | Gemma, Phi, gpt-oss, hybrid linear-attention models (Qwen3.5 and 3.6, Kimi K3, DeepSeek V4, GLM-5), fused gate-up expert tensors, images |

## Browser support

WebGPU is required: Chrome or Edge 113 or later on Windows, macOS, ChromeOS and Android; Safari
26 or later; Firefox 141 or later on Windows. On Linux, Chrome needs
`chrome://flags/#enable-unsafe-webgpu` and Vulkan enabled. The page explains what to do when
WebGPU is missing. Large models also need browser storage: Chrome lets a site use a large share
of free disk space.

## How it works

The Rust crate owns everything that touches the model: GGUF parsing, weight layouts, the WGSL
kernels, the placement of weights between GPU and storage, tokenizers, chat templates, the KV
cache and sampling. JavaScript handles the page, downloads, file reads and web tools. The same
engine runs natively through [wgpu](https://wgpu.rs) on Vulkan, Metal or Direct3D 12, which is
how it is tested. See [docs/architecture.md](docs/architecture.md) for the kernels, layouts and
streaming.

```
src/gguf.rs        GGUF header parsing from a file prefix, split-file merging
src/quant.rs       Dequantization and the GPU layouts of every weight format
src/iq_tables.rs   IQ code books from ggml (MIT, see NOTICE)
src/model.rs       Architecture configuration and tensor names
src/tokenizer.rs   BPE and SentencePiece tokenizers, streaming UTF-8 decoding
src/chat.rs        Chat templates through MiniJinja, with tools
src/gpu/           wgpu device, WGSL kernels, the forward pass and weight streaming
src/source.rs      Where streamed weights are read from
src/session.rs     Prompt-prefix reuse across turns and sampling
src/cpu.rs         A slow f32 reference forward pass for tests
src/web.rs         WebAssembly bindings
web/               The page, the worker, the model catalog, web tools and the Markdown renderer
```

## Verification

Reference results come from independent implementations (`tools/make_fixtures.py`,
`tools/make_tiny_models.py`, `tools/make_header_fixtures.py`):

| Check | Reference | Result |
|---|---|---|
| Dequantization, 24 formats | gguf-py NumPy dequantizers | Equal within 1e-6 |
| Tokenization, 17 hard strings × 7 tokenizers | Hugging Face `tokenizers`; for Kimi K2 an independent tiktoken-rank encoder | Identical token ids |
| Chat templates, 10 conversations including tool calls | transformers `apply_chat_template` | Identical text |
| Final logits of five real models | transformers float32 over the same GGUF | Same top token; top-20 logits within 0.2% of the logit range |
| Qwen3-MoE, Mixtral, OLMoE, DeepSeek V3 and DeepSeek V2 Lite layouts (tiny random models converted by llama.cpp) | transformers float32 | Within 0.03% of the logit range, resident and streamed with a 4-expert cache |
| All 25 weight formats on the GPU, in both layouts and fully streamed | `src/cpu.rs` | Within 0.05% of the logit range |
| Streamed against resident runs of Llama 3.2 1B and OLMoE 1B-7B | The same engine with everything on the GPU | Identical greedy replies |
| Browser reply, greedy | The native build on Vulkan | Identical text for SmolLM2 135M, Qwen2.5 0.5B, Qwen3 0.6B, Llama 3.2 1B and OLMoE 1B-7B (resident and streamed) |

The tiny-model tests build a random model of each mixture-of-experts layout with
transformers, convert it with llama.cpp's `convert_hf_to_gguf.py`, and require the engine to
reproduce PyTorch's logits for a whole prompt at once and token by token. They cover sigmoid
routing with a selection bias, group-limited choice, shared experts, leading dense layers,
query compression, latent attention and YaRN. The tokenizer and template checks for DeepSeek
V3.1 and Kimi K2 read only the header of the first file, fetched with a range request.

## Measured speed

Measured on one laptop with its battery power profile active: an Intel Raptor Lake integrated
GPU with 15 GB of shared memory, and an RTX 3050 Laptop GPU that the profile held at its lowest
clocks. Expect more from a desktop GPU. "Prompt" is tokens read per second for a 256-token
prompt; "reply" is tokens generated per second.

| Model | Intel GPU, native | Intel GPU, Chrome |
|---|---|---|
| SmolLM2 135M Q8_0 | prompt 830 to 1,060/s, reply 70 to 82/s | reply 45 to 74/s |
| Qwen2.5 0.5B Q4_K_M | prompt 310 to 415/s, reply 28 to 40/s | reply 33 to 36/s |
| Llama 3.2 1B Q4_K_M | prompt 105 to 166/s, reply 14 to 25/s | reply 22/s |
| OLMoE 1B-7B Q2_K (7B parameters, 2.6 GB on the GPU) | prompt 67 to 92/s, reply 15/s | reply 16/s |
| OLMoE 1B-7B Q2_K with a 1 GB GPU budget | reply 6/s, 57% of experts cached | reply 1.7 to 1.9/s |
| Llama 3.2 1B Q4_K_M with a 300 MB GPU budget (every layer read per token) | reply 1.9/s | |

Speeds vary by up to a third between runs on this power-limited machine. Streamed speed depends
on how much of the model the file cache holds and on how fast storage is. Chrome reads opened
files through `FileReaderSync`, and each expert layer waits for the GPU to report its choices,
so streaming in the browser is slower than native.

## Limits

- The largest catalog models (Qwen3 235B-A22B, Qwen3 Coder 480B-A35B, DeepSeek V3.1, Kimi K2)
  use architectures, weight formats, tokenizers and templates that are verified here, but they
  have not been run end to end: the test machine lacks the disk space. Their speed is unknown;
  each token reads several gigabytes from storage, so expect seconds per token.
- Hybrid linear-attention models, the newest generation of the largest open models, are not
  supported yet.
- Small models use tools unreliably and often answer wrongly even with good search results.
- Keeping weights file-sized on the GPU (the layout streaming uses) costs up to a third of the
  decoding speed of the expanded layout used when a model fits.
- Speed has not been tuned against llama.cpp or WebLLM.

## Build and run

You need Rust with the `wasm32-unknown-unknown` target, `wasm-bindgen-cli` matching
`Cargo.lock`, Python 3 and Node.js 20 or later. `wasm-opt` (binaryen) is used when installed.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
cargo install wasm-opt --locked      # optional: a 24% smaller engine
python3 tools/build_web.py           # writes dist/
python3 -m http.server 8080 --directory dist
```

Open http://localhost:8080. Any static host works. The app needs these content security policy
sources and no inline scripts (the last four only for web access):

```
script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co
  https://*.wikipedia.org https://api.duckduckgo.com https://r.jina.ai https://s.jina.ai
```

The native command line runs the same engine, reading weights from disk as it needs them:

```sh
cargo build --release
./target/release/evolve-llm run model.gguf --chat --prompt "Hello" -n 128
./target/release/evolve-llm run model-00001-of-00004.gguf --gpu-mem 4000 --chat --prompt "Hello"
./target/release/evolve-llm check model.gguf --prompt "The capital of France is"   # GPU vs CPU
./target/release/evolve-llm bench model.gguf
./target/release/evolve-llm profile model.gguf                                     # time per kernel
```

`--gpu-mem` is a weight budget in MB; the rest streams. `EVOLVE_LLM_LAYOUT=compact` keeps
weights file-sized on the GPU.

## Tests

```sh
cargo test --release                 # units, fixtures, and GPU tests when models are present
npm test                             # browser modules
python3 tests/browser_test.py --model path/to/model.gguf [--gpu-mem 1] [--download]
```

GPU and model tests look for GGUF files in `$EVOLVE_LLM_MODELS` (default
`~/.cache/evolve-llm/models`) and skip themselves when the files or a GPU adapter are missing.
Regenerating fixtures needs Python with numpy, torch, transformers 4.57, accelerate,
tokenizers, sentencepiece and gguf, and a llama.cpp checkout for its converter.

## License

Apache 2.0. The IQ code books and block decoders derive from ggml (MIT); see NOTICE. Models are
not part of this repository; each one keeps its own license, shown in the model picker.

## Browser embedding API

[Try the developer quickstart](https://www.evolvetech.dev/developers/llm/) or read [the embedding contract](docs/embedding.md). The versioned runtime bundle includes `createEngine`, TypeScript declarations and an executable example. Its offline protocol fixture needs no model download; a separate button runs real inference. Stream cancellation, timeouts and worker failures release the model, and concurrent calls are rejected. The engine and its measured results are unchanged by this SDK release.
