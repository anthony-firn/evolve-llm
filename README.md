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

## One chat that can also do things

Ask the main chat anything, or ask it to make or change something: an app or game (it plays in the
conversation), files, or work on your GitHub repositories. The model decides when to use tools, and
its work shows as one collapsed line of plain steps under the reply. Behind it is Pi's agent core
with the loaded model, working in the Evolve VM: a Rust kernel in WebAssembly with a POSIX-style
shell, utilities, and `git` and `gh` that work on GitHub through its API (clone, commit, push,
pull requests, CI results and merging). Pushing follows the autonomy you choose in Settings, and
merging always asks you. **Terminal** opens the same files in a Linux terminal, where `pi <task>`
runs the same agent. Browsers without WebAssembly JSPI (Firefox, Safari) use the older browser
sandbox instead, without `git`. Web access is explicit; there is no host shell or server
inference. See [commands, limits, permissions and self-hosting](docs/workbench.md).

## Web browser

**Browser** opens a web browser engine compiled to WebAssembly: Blitz with Stylo CSS and Taffy
layout, Boa JavaScript, and rustls TLS, all in a worker. It loads real sites, keeps cookies and
logins on this device, and you and Pi can use the same live page at once; Pi never types
passwords. Nothing is installed or hosted for it: the engine contains a Tor client (Arti's
`tor-proto`) and reaches sites through the Tor network over Snowflake, with TLS end to end inside
the engine. For speed you can run `evolve-relay` on your computer, or use a community relay
announced on Nostr. See [docs/browser.md](docs/browser.md).

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
- **Runs models larger than your device.** The largest open models (Kimi K3.1 Pro is 593 GB at
  its smallest) do not fit on most laptops' disks. These run straight from Hugging Face: each
  forward pass fetches the byte ranges it needs, many requests at once, and a capped cache in
  browser storage keeps what it can for the next token and the next visit. Weights that every
  token reads are kept first. Nothing is downloaded up front; it is slow, but it runs. A long
  prompt is read layer by layer, so each weight is read once per 1,024 prompt tokens rather than
  once per 128, and each reply step also checks a continuation drafted from earlier text, so text
  the model copies from the conversation (code, paths) comes several tokens per pass.
- **Works from your phone.** In Settings on a computer, "Use this computer from your phone" shows
  a QR code; scan it and the phone becomes the screen for that computer: chat and Pi on the phone,
  the model on the computer's GPU (the phone needs no GPU of its own). The link stays the same,
  so a bookmark reconnects, to the model still loaded. A phone can also run small models itself
  with WebGPU (Chrome on Android, Safari 26 on iOS), or open a pairing link to lend its GPU to a
  computer; it keeps its screen on while it works, and asks for less memory than a computer.
- **Runs one model across your devices.** Pair your other computers in Settings ("Pair another
  device", then open the link on each): the device you chat on keeps the embedding, the first
  layers and the output; each paired device runs a later run of layers, sized to its GPU memory,
  and reads only its share of the model from Hugging Face. The model's working state goes device
  to device over an encrypted WebRTC connection, found through public Nostr relays with a code
  only your devices know. Replies are the same as on one device.
- **Shares model pieces between visitors when you let them.** With sharing on (Settings, off by
  default), a model read from Hugging Face comes from other visitors' browsers when they have the
  pieces, over WebRTC, found through public Nostr relays; whatever no visitor has comes from
  Hugging Face. Every 256 KiB piece is checked against hashes published in
  [evolve-llm-manifests](https://github.com/anthony-firn/evolve-llm-manifests) before use, and a
  visitor that sends a wrong piece is dropped. Pieces your device stores go to visitors who need
  them, within an upload limit. Peers see your IP address and which model files you read; your
  conversations never leave the device.
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
| Hybrid linear attention | `kimi-k31` and `kimi-k3` (Kimi K3.1 Pro and Kimi K3): Kimi delta attention in 69 of 93 layers, latent attention without rotation and with an output gate, DeepSeek sparse attention's lightning indexer, residuals that attend across blocks, experts in a latent space, the situ activation |
| Attention | Grouped-query attention, multi-head latent attention (DeepSeek), DeepSeek sparse attention (top-k positions from a lightning indexer), YaRN and linear position scaling, Llama 3 rope factors |
| Weight formats | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, BF16, F32 |
| Tokenizers | Byte-level BPE (GPT-2, Llama 3, Qwen2, SmolLM, DeepSeek V3 and Kimi K2 pre-tokenizers) and SentencePiece-style BPE |
| Files | Single GGUF files and split sets (`name-00001-of-00004.gguf`) |
| Not yet | Gemma, Phi, gpt-oss, the other new hybrid and sparse layouts (Qwen3.5 to 3.8, DeepSeek V4 and V4.1, GLM-5.x, MiniMax M3, MiMo V2.6), fused gate-up expert tensors, images |

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
| Kimi K3.1 layout (tiny random model, akopytko/tiny-kimi-k3.1-pro-GGUF), dense and with the indexer's top-k at 8 | `tools/ref_kimi_k31.py`, float64 NumPy written from llama.cpp's graph | Within 0.06% of the logit range, resident and streamed with a 16-slot expert cache |
| Llama 3.2 1B read from Hugging Face (`hf:` path, 300 MB GPU budget) | The same model from a local file | Identical greedy replies |
| The same tiny Kimi K3.1 in llama.cpp 2cbfc3c (`kimi_k31_matches_llama_cpp`), dense on 7 tokens and with sparse attention selecting on 22 | llama.cpp's CPU logits (all 163,840 compared once; the test keeps its top 256) | Within 0.17% of the logit range, correlation 0.999995 and up, the same top 10 and most likely token, resident and streamed; llama.cpp rounds activations to bf16, which accounts for the difference |
| Kimi K3.1 Pro IQ1_S headers (14 parts, 2,693 tensors) | The engine's shape and format checks, over the network | Every tensor accepted |
| Kimi K3.1, DeepSeek V3, Qwen3 MoE (tiny) and Llama 3.2 1B split across three engines (`split_across_devices_matches_one_device`) | The model on one engine | Bit-identical logits, resident and streamed (layer-by-layer prompts included; 1e-5 where a stage's share fits its budget and uses the faster layout); the same greedy text over two turns |
| A phone using a computer (`tests/remote_browser_test.py`): a Pixel 7 profile in a browser with no GPU adapter, SmolLM2 135M loaded onto the computer from the phone | The model run directly on the computer | Identical replies, also after the phone reloads and reconnects to the model still loaded; with `--pi-hf` Qwen2.5 0.5B, Pi on the phone wrote a file with a tool call |
| A phone as the helper (`tests/devices_browser_test.py --helper-device "Pixel 7"`) | The model alone | Identical reply; the phone offered 1.6 GB (its default) and ran layers 23–30 |
| A chat device and a helper (`tests/devices_browser_test.py`), SmolLM2 135M from Hugging Face | The model alone in a third browser | Identical reply, with layers 1–15 on the chat device and 16–30 on the helper, each reading only its share (127 MB instead of 214 MB); the same with both streaming their layers |
| Two visitors sharing pieces (`tests/swarm_browser_test.py`), SmolLM2 135M read from Hugging Face | The first visitor's reply | Identical; the second received 219 MB from the first over WebRTC and 4 KB from Hugging Face, loading and replying in 11 s instead of 71 s; the same on evolvetech.dev through public Nostr relays |
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

Llama 3.2 1B with a 300 MB GPU budget repeating a sentence went from 0.6 to 1.4 tokens/s with
drafts from earlier text (13 passes over the weights for 30 tokens).

Read from Hugging Face on the RTX 3050 laptop over a busy home Wi-Fi link (0.5 to 4 MB/s):
Llama 3.2 1B with a 300 MB GPU budget took 18 minutes for its first pass (812 MB fetched), then
1.1 to 1.5 s per token from the device cache, as fast as the local file. Kimi K3.1 Pro on the
same laptop (`evolve-llm plan`): 4.3 GB of weights and 1.5 GB of state on the 6 GB GPU at an 8K
context, and each token reads about 47 GB of attention and shared weights plus up to 9.7 GB of
experts. That is roughly a minute per token once a fast SSD cache holds it, and hours per token
over that connection. Cold byte ranges of rarely downloaded repositories also start slowly on
Hugging Face's CDN (seconds to minutes per request).

Speeds vary by up to a third between runs on this power-limited machine. Streamed speed depends
on how much of the model the file cache holds and on how fast storage is. Chrome reads opened
files through `FileReaderSync`, and each expert layer waits for the GPU to report its choices,
so streaming in the browser is slower than native.

## Limits

- The largest catalog models (Qwen3 235B-A22B, Qwen3 Coder 480B-A35B, DeepSeek V3.1, Kimi K2,
  Kimi K3.1 Pro) use architectures, weight formats, tokenizers and templates that are verified
  here, but they have not produced a reply end to end: over the test laptop's connection one
  token of Kimi K3.1 Pro takes hours. Their speed depends on the connection and on how much
  of each token's reads the device cache holds.
- Kimi K3.1 is checked against llama.cpp and a NumPy transcription of its graph on a tiny random
  model, not against the original weights (no reply from Kimi K3.1 Pro yet, see above). Its
  linear-attention state cannot be rewound, so when a new
  turn changes earlier text the prompt is read again from the snapshot taken at the end of the
  previous prompt, or from the start.
- The indexer's keys are cached in f16; where two positions score within that precision of the
  top-k cut, the choice can differ from a float32 implementation.
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
sources and no inline scripts (the Jina, DuckDuckGo and Wikipedia sources only for web access;
`https://raw.githubusercontent.com` and `wss:` for the browser's Tor route, Nostr lookups and community
relays; the `ws://` sources only for a relay on this computer):

```
script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co
  https://*.wikipedia.org https://api.duckduckgo.com https://r.jina.ai https://s.jina.ai
  https://raw.githubusercontent.com wss: ws://127.0.0.1:* ws://localhost:*
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

Models that stay on Hugging Face take an `hf:` path (or a `huggingface.co` URL); any part of a
split model stands for the whole:

```sh
./target/release/evolve-llm plan hf:KarLexAI/Kimi-K3.1-Pro-GGUF/KLX-IQ1_S/Kimi-K3.1-Pro-KLX-IQ1_S-00001-of-00014.gguf --gpu-mem 4000
./target/release/evolve-llm run hf:owner/repo/model-00001-of-00014.gguf --gpu-mem 4000 --cache-gb 50 --chat --prompt "Hello" -n 64
./target/release/evolve-llm render model.gguf --prompt "Hello"      # the chat template's text
```

`plan` reads only the headers and reports where the weights would live and what each token reads.
`--cache-gb` caps the device cache (default 8, in `~/.cache/evolve-llm/stream/`, or
`--cache-dir`), and `--parallel` sets the requests in flight (default 8).

## Tests

```sh
cargo test --release                 # units, fixtures, and GPU tests when models are present
npm test                             # browser modules
python3 tests/browser_test.py --model path/to/model.gguf [--gpu-mem 1] [--download]
python3 tests/stream_browser_test.py --model path/to/model.gguf --hf owner/repo/model.gguf   # read from Hugging Face
python3 tests/pi_real_model_browser_test.py --runtime dist model.gguf [...]                  # Pi tool calls, real models
python3 tests/swarm_browser_test.py                                                         # two visitors sharing pieces
python3 tests/devices_browser_test.py [--memory 0.05]                                       # one model on two devices
python3 tests/remote_browser_test.py [--pi-hf owner/repo/file.gguf]                         # a phone using a computer
python3 tools/build_vm.py && python3 tests/vm_browser_test.py                               # the Evolve VM
python3 tests/vm_terminal_browser_test.py                                                   # its terminal, typed into
python3 tests/vm_github_browser_test.py && python3 tests/vm_pi_browser_test.py               # git, gh and Pi, fake GitHub
python3 tests/chat_github_real_model_test.py --runtime dist model.gguf                     # a real model in the chat: clone to merge
```

GPU and model tests look for GGUF files in `$EVOLVE_LLM_MODELS` (default
`~/.cache/evolve-llm/models`) and skip themselves when the files or a GPU adapter are missing.
Regenerating fixtures needs Python with numpy, torch, transformers 4.57, accelerate,
tokenizers, sentencepiece and gguf, and a llama.cpp checkout for its converter. The Kimi K3.1
fixtures need only NumPy and the tiny model `tiny-k31-bf16.gguf` from
akopytko/tiny-kimi-k3.1-pro-GGUF:

```sh
python3 tools/ref_kimi_k31.py $EVOLVE_LLM_MODELS/tiny-k31-bf16.gguf tests/fixtures/tiny-kimik31.json
python3 tools/ref_kimi_k31.py $EVOLVE_LLM_MODELS/tiny-k31-bf16.gguf tests/fixtures/tiny-kimik31-topk8.json --top-k 8
```

`tests/fixtures/k31-llama-cpp.json` holds llama.cpp's largest logits for the same model and two
prompts, from `llama-eval-callback -m <model> -p <prompt> -n 1` with its tensor printer patched
to print every value at full precision (the fixture records the prompts and tokens).

## License

Apache 2.0. The IQ code books and block decoders derive from ggml (MIT); see NOTICE. Models are
not part of this repository; each one keeps its own license, shown in the model picker.

## Browser embedding API

[Try the developer quickstart](https://www.evolvetech.dev/developers/llm/) or read [the embedding contract](docs/embedding.md). The versioned runtime bundle includes `createEngine`, TypeScript declarations and an executable example. Its offline protocol fixture needs no model download; a separate button runs real inference. Stream cancellation, timeouts and worker failures release the model, and concurrent calls are rejected. The engine and its measured results are unchanged by this SDK release.
