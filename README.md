# Evolve LLM

Run open language models on your own GPU from a web page. Evolve LLM is an inference engine
written in Rust. It is compiled to WebAssembly and computes every token with WebGPU compute
shaders. Open the page, pick a model, and chat with it. There is nothing to install, no account,
and no server that sees your messages.

**Try it:** [evolvetech.dev/play/evolve-llm](https://www.evolvetech.dev/play/evolve-llm/)

**Status: early release.** The engine runs Llama, Mistral, Qwen2, Qwen3 and SmolLM models
from single-file GGUF downloads. It is tested in Chrome on Linux with NVIDIA and Intel GPUs and
against reference implementations (see [Verification](#verification)). Speed has not yet been
tuned against llama.cpp or WebLLM.

## What it does

- **Downloads a model once.** Choose from a catalog of small open models or paste any
  single-file GGUF on Hugging Face. The file streams into the browser's private storage and
  resumes after a dropped connection. You can also open a GGUF file already on your device;
  it is read in place.
- **Runs it on your GPU.** The weights are repacked for WGSL as they upload, in 16 MB slices,
  so loading never needs the whole file in memory.
- **Chats with the model's own template.** Prompts are formatted by the Jinja chat template in
  the GGUF file. Each new turn only computes the part of the conversation that is not already
  in the KV cache.
- **Shows its reasoning.** Qwen3 thinking is shown in a collapsible block and can be turned off.

Nothing leaves the device except the model download from Hugging Face. There are no analytics.

## Supported models

| | Supported |
|---|---|
| Architectures | `llama` (Llama 2 and 3, Mistral, SmolLM2, TinyLlama), `qwen2` (Qwen2 and 2.5, Coder), `qwen3` (dense) |
| Weight formats | Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, IQ4_NL, IQ4_XS, F16, BF16, F32 |
| Tokenizers | Byte-level BPE (GPT-2, Llama 3, Qwen2, SmolLM pre-tokenizers) and SentencePiece-style BPE |
| Not yet | Gemma, Phi, mixture-of-experts, IQ1 to IQ3 formats, split GGUF files, YaRN scaling, images |

Model size is limited by GPU memory. Models up to about 4 billion parameters at 4-bit fit on
GPUs with 4 GB or more. Larger matrices are split across buffers when a device has a small
binding limit.

## Browser support

WebGPU is required: Chrome or Edge 113 or later on Windows, macOS, ChromeOS and Android; Safari
26 or later; Firefox 141 or later on Windows. On Linux, Chrome needs
`chrome://flags/#enable-unsafe-webgpu` and Vulkan enabled. The page explains what to do when
WebGPU is missing.

## How it works

The Rust crate owns everything that touches the model: GGUF parsing, weight repacking, the
WGSL kernels, tokenizers, chat templates, the KV cache and sampling. JavaScript only handles the
page, downloads and file reads. The same engine runs natively through
[wgpu](https://wgpu.rs) on Vulkan, Metal or Direct3D 12, which is how it is tested against a CPU
reference. See [docs/architecture.md](docs/architecture.md) for the kernels and data layout.

```
src/gguf.rs        GGUF header parsing from a file prefix
src/quant.rs       Dequantization and GPU repacking for every weight format
src/model.rs       Architecture configuration and tensor names
src/tokenizer.rs   BPE and SentencePiece tokenizers, streaming UTF-8 decoding
src/chat.rs        Chat templates through MiniJinja
src/gpu/           wgpu device, WGSL kernels and the forward pass
src/session.rs     Prompt-prefix reuse across turns and sampling
src/cpu.rs         A slow f32 reference forward pass for tests
src/web.rs         WebAssembly bindings
web/               The page, the worker, the model catalog and the Markdown renderer
```

## Verification

Reference fixtures come from independent implementations (`tools/make_fixtures.py`):

| Check | Reference | Result |
|---|---|---|
| Dequantization, 14 formats | gguf-py NumPy dequantizers | Equal within 1e-6 |
| Tokenization, 17 hard strings × 5 models | Hugging Face `tokenizers` with each model's `tokenizer.json` | Identical token ids |
| Chat templates, 5 conversations | transformers `apply_chat_template` | Identical text |
| Final logits, 50–70 token chat prompts | transformers float32 forward pass over the same GGUF | Same top token; top-20 logits within 0.2% of the logit range |
| Every prompt position of the five models, one-token and batched kernels | `src/cpu.rs` | Within 0.06% of the logit range |
| All 15 weight formats on the GPU, using a generated two-layer model | `src/cpu.rs` | Within 0.05% of the logit range |
| Browser reply, greedy | The native build on Vulkan | Identical text for SmolLM2 135M, Qwen2.5 0.5B, Qwen3 0.6B and Llama 3.2 1B |

The logit fixtures use SmolLM2 135M, Qwen2.5 0.5B Q4_K_M, Qwen3 0.6B Q8_0, Llama 3.2 1B Q4_K_M
and llama2.c stories15M, which together exercise every architecture flag: biased projections,
query and key norms, tied and separate output heads, Llama 3 rope factors and grouped-query
attention. `tests/synthetic.rs` writes a small random model in each weight format, so formats
that none of those files use still run through the GPU kernels.

## Measured speed

Measured on a laptop whose power profile held the RTX 3050 Laptop GPU in its lowest performance
state (210 MHz core, 405 MHz memory), so these are floor numbers rather than what the GPU can do.
"Prompt" is tokens read per second for a 256-token prompt; "reply" is tokens generated per second.

| Model | Native, RTX 3050 (capped) | Chrome WebGPU, RTX 3050 (capped) |
|---|---|---|
| SmolLM2 135M Q8_0 | prompt 661/s, reply 24/s | reply 26/s |
| Qwen2.5 0.5B Q4_K_M | prompt 268/s, reply 11/s | reply 10.5/s |
| Qwen3 0.6B Q8_0 | prompt 184/s, reply 8/s | reply 9/s |
| Llama 3.2 1B Q4_K_M | prompt 109/s, reply 6/s | reply 6/s |

At these clocks the one-token kernel reads weights at the memory bandwidth the GPU was allowed.
Browser and native speeds match, so WebAssembly and WebGPU add little overhead.

## Build and run

You need Rust with the `wasm32-unknown-unknown` target, `wasm-bindgen-cli` matching
`Cargo.lock`, Python 3 and Node.js 20 or later.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
python3 tools/build_web.py          # writes dist/
python3 -m http.server 8080 --directory dist
```

Open http://localhost:8080. Any static host works. The app needs these content security policy
sources and no inline scripts:

```
script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co
```

The native command line runs the same engine:

```sh
cargo build --release
./target/release/evolve-llm run model.gguf --chat --prompt "Hello" -n 128
./target/release/evolve-llm check model.gguf --prompt "The capital of France is"   # GPU vs CPU
./target/release/evolve-llm bench model.gguf
./target/release/evolve-llm profile model.gguf                                     # time per kernel
```

## Tests

```sh
cargo test --release                 # units, fixtures, and GPU tests when models are present
npm test                             # browser modules
python3 tests/browser_test.py --model path/to/model.gguf [--download]
```

GPU and model tests look for GGUF files in `$EVOLVE_LLM_MODELS` (default
`~/.cache/evolve-llm/models`) and skip themselves when the files or a GPU adapter are missing.
Regenerating fixtures needs Python with numpy, torch, transformers 4.57, accelerate,
tokenizers and gguf.

## License

Apache 2.0. Models are not part of this repository; each one keeps its own license, shown in
the model picker.
