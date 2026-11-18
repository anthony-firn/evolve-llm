# How Evolve LLM works

Evolve LLM is a transformer inference engine written in Rust. The same source compiles to a
native command-line program, which uses Vulkan, Metal or Direct3D 12, and to WebAssembly, which
uses the browser's WebGPU. Every token is computed by WGSL compute shaders. The CPU only
tokenizes, samples and schedules work.

## From file to GPU

1. **Header.** `src/gguf.rs` parses a GGUF prefix. When the prefix is too short, the parser
   reports how many bytes it needs, so the browser reads only the header (often 2 to 10 MB)
   before deciding whether the model can run. A split model's headers are merged into one
   tensor directory whose tensors remember their file.
2. **Plan.** `src/model.rs` validates the architecture, shapes and weight formats. The engine
   decides where each weight matrix lives (see [Models larger than the GPU](#models-larger-than-the-gpu)),
   allocates GPU buffers for the resident ones and returns an upload plan: each tensor's file,
   offset, row size and row count, and whether it is resident.
3. **Upload.** The worker reads 16 MB slices of the resident tensors and passes them to
   `Engine::upload`, which converts each slice to its GPU layout before `queue.write_buffer`.
   The model never has to fit in WebAssembly memory at once.
4. **Shards.** A matrix larger than the device's storage-binding limit is split into row shards
   on 64-row boundaries (whole experts for expert tensors), so devices with small limits can
   still hold large vocabularies.

## Weight layouts

GGUF blocks pack scales, minimums and bit planes into odd sizes (18, 22, 34, 66 or 210 bytes),
and WGSL reads storage buffers as 32-bit words. The engine has two layouts and chooses per
format:

- **Expanded**, for speed when a model fits. Codes become a plane of 4-bit nibbles (plus a plane
  of one or two extra bits for 5- and 6-bit formats, or bytes for Q8_0) and each group gets an
  f32 scale and bias. A 32-value step is one 16-byte load. This costs up to a third more memory
  than the file.
- **Compact**, file-sized. Each GGUF block keeps its bit layout; fields are moved so arrays start
  on word boundaries, and blocks of 32-value formats are paired so 18-, 22- and 34-byte blocks
  fill whole words. Shaders decode the original format: K-quant sub-scales, IQ code books
  (`src/iq_tables.rs`, from ggml) with their sign bits, ternary digits and MXFP4's shared
  exponents. For Q2_K, Q4_K, Q5_K, IQ4_XS, IQ1_M, Q4_1 and Q5_1 the compact layout is the file
  layout byte for byte, so streamed weights go to the GPU without being rewritten.

| Layout | Formats | GPU memory |
|---|---|---|
| Expanded | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K | 6% to 33% more than the file |
| Compact | Every quantized format, including all IQ, TQ and MXFP4 formats | The file size (plus up to 2 bytes per block) |
| Half floats | F16, BF16, F32 | 16 bits per weight |

When a model fits, Q2_K, Q3_K and the IQ, TQ and MXFP4 formats use the compact layout and the
others the expanded one. When it does not, everything uses the compact layout, except formats
whose expanded form costs under 15% more. Llama 3.2 1B Q4_K_M takes 1,028 MB expanded and
803 MB compact; the compact kernels decoded 30 to 40% slower on the test laptop.

## Kernels

`src/gpu/shaders.rs` and `src/gpu/blocks.rs` generate WGSL per weight layout.

- **Matrix-vector product** for one token. Four lanes cooperate on each output row; compact
  K-quant kernels decode one super-block's scales once and read its codes as 16-byte vectors.
- **Tiled matrix product** for prompts: 64 rows by 32 tokens per workgroup through shared memory,
  with weights dequantized while the tile loads. Matrices with per-head groups (MLA's key and
  value expansions) read a different input slice per group of rows.
- **Expert routing**: per token, softmax or sigmoid probabilities, DeepSeek V3's selection bias
  and group limit (the best groups by the sum of their two best experts), top-k selection, and
  renormalized, scaled weights.
- **Expert products.** One token runs matrix-vector kernels per chosen (token, expert) pair; the
  down projection sums the weighted experts into the residual. A prompt first sorts its pairs by
  expert, then runs one tiled product per expert over its tokens, so each expert's weights are
  read once per 32 tokens instead of once per token, then sums each token's experts. Kernels
  find an expert's weights through an expert-to-slot map, which is what lets experts live in a
  cache.
- **RMSNorm**, per row, per head (Qwen3) or over a whole projection (OLMoE).
- **Rotary embeddings** in both pairing conventions, with Llama 3 frequency factors and YaRN
  blending from a precomputed table. The same kernel writes keys and values into an f16 cache.
- **Attention** with a streaming softmax: one workgroup per head and token walks the cache in
  blocks of 256 keys. Grouped-query attention maps several query heads to each cached head.
- **Multi-head latent attention** (DeepSeek, Kimi): the cache holds one normalized latent vector
  and one rotated key per token (576 values instead of 40,960 for DeepSeek V3). Queries absorb
  the key expansion, attention runs over the latent cache with up to four heads sharing each
  read, and the value expansion follows.

Matrix kernels end with an epilogue that stores the result, adds it to the residual stream, or
multiplies it into the SiLU-gated feed-forward activation. When the whole model is resident, one
token's forward pass is a single command buffer. Prompts run in chunks of 128 tokens. Only the
final token's logits are read back.

## Models larger than the GPU

Given a GPU memory budget, `Engine::with_options` places each matrix:

1. Vectors (norms, biases) always stay on the GPU.
2. If the whole model fits, everything is resident. If it fits in the compact layout, that
   layout is used. Otherwise the model streams.
3. Token embeddings are looked up on the CPU, one row per token.
4. Each layer's attention, router and shared-expert weights are needed by every token, so they
   come next: whole layers stay resident while they fit. The rest are read before each use into
   one of two ring buffers per matrix kind, so one layer's weights upload while the previous
   layer runs. The output head stays resident when it is small next to the budget.
5. Routed experts share an expert cache, one per combination of weight formats, sized from what
   is left. Each MoE layer runs attention and routing, reads back the chosen experts, loads the
   missing ones into the least recently used slots, rewrites the expert-to-slot map and runs the
   expert kernels. When a prompt needs more experts than the cache holds, the layer runs in
   several passes; each pass's loads are queued after the previous pass's work, so reused slots
   are safe.

A prompt longer than one 128-token chunk is read layer by layer when the model streams, up to
1,024 tokens at once: each layer's streamed weights are read once for all of its chunks. In an
expert layer, every chunk's attention and routing run first; each expert-cache pass then loads
its experts once and runs every chunk that chose one of them; then the rest of the layer runs
for each chunk. Chunk activations wait in buffers between these phases, and block checkpoints
are indexed by the chunk's offset. Reading a chunk at a time instead would read almost every
expert of a large model again for every 128 tokens. The results match the chunk-at-a-time
resident pass (`tests/tiny.rs`).

While a model that streams generates, a pass reading a few tokens costs about as much as one
reading a single token, because the reads are the cost. So each step drafts up to six tokens: the
continuation of the latest earlier occurrence of the last four, three or two tokens (prompt
lookup; coding agents copy paths, code and JSON from their context). One pass returns logits after
every drafted token; each position samples from the model's own logits, and a drafted token is kept
only when it is what was sampled, so the text is what one-at-a-time decoding gives (identical when
greedy, `tests/tiny.rs`). The first mismatch becomes the next token. A model with a recurrent state
copies it before the pass and, when a draft is only partly kept, restores it and reads the kept
tokens again. Drafting pauses for 16 tokens when fewer than 30% of recent drafted tokens were kept.

Weights are read through `WeightSource`: positioned reads of the files natively, and OPFS access
handles or `FileReaderSync` in the browser, filling WebAssembly memory directly. The operating
system's file cache keeps recently read weights in RAM, which is the middle tier between the GPU
and the disk; the engine does not keep its own RAM copy. Streaming results are bit-identical to
resident runs in the tests.

## Models larger than the device

The largest open models are bigger than most laptops' free storage (Kimi K3.1 Pro: 593 GB at
IQ1_S). Those read their weights from Hugging Face while they run, with the device's storage as
a cache in front of the network:

1. **Addresses.** Each part's `resolve` URL is requested once with `Range: bytes=0-0`; the
   response gives the part's size and the signed CDN address it redirects to, which is reused
   until shortly before its `Expires=` time. Hugging Face's API therefore sees one request per
   file and hour, not one per read. The CDN answers byte ranges with
   `access-control-allow-origin: *`, so the page can read them directly under the same content
   security policy (`https://*.hf.co`).
2. **Prefetch.** `WeightSource::prefetch` takes the spans the next reads will ask for. The
   streamed forward pass calls it before each layer's attention and shared weights, and again
   for each expert-cache pass's missing experts once routing is known, so a long prompt never
   holds more experts than the GPU cache. A network source splits the spans into requests of
   up to 16 MB and keeps 6 (browser) or 8 (native) in flight; the reads that follow are served
   from memory. A read nothing prefetched falls back to one synchronous request, which a worker
   may make.
3. **Cache.** Fetched ranges are kept on the device up to a cap: files named
   `{part}-{offset}-{length}.{t|e}` in the origin-private file system (browser) or in
   `~/.cache/evolve-llm/stream/` (native), written under a temporary name and moved into place.
   Ranges every token reads (`t`) are kept first and never evicted by experts (`e`), which share
   what is left, least recently used first. Plain least-recently-used eviction would get no hits
   at all from a cache smaller than one token's reads, because every token reads the same
   weights in the same order.

`web/remote-reader.mjs` is the browser side, `src/remote.rs` the native one.

### One model across a person's devices

A model too large for one device can run across several of the same person's
(`web/devices.mjs`, `Stage` and `Remote` in `src/gpu/engine.rs`):

- **Stages.** An engine built with `Options.stage` holds only layers `first..end`, plus the
  embedding and output head when it is the device the chat runs on. Tensors outside the stage
  are never allocated or read, and placement (resident, streamed, expert caches) applies to the
  stage's share, so each device keeps as much of its layers on its GPU as its budget allows.
  `web/split.mjs` gives each device a run of layers in proportion to its memory; the chat
  device's share counts the embedding and head.
- **Passes.** The chat device embeds the tokens and runs its layers, then hands the hidden state
  to a `Remote` (the page, which sends it through each helper in layer order) and runs the head
  on what comes back. A helper's `forward_hidden` reads the hidden state and returns its own.
  The state is token-major: the residual stream, then (Kimi K3.1) each block checkpoint made so
  far, so a batch can be cut or joined anywhere; it is about 28 KB per token for most models.
  The whole batch goes through once per pass, so a long prompt costs one round trip per
  layer-by-layer pass, and a helper that streams reads its weights once per prompt as well.
  Recurrent snapshots and restores are mirrored to the helpers, which process messages in
  order; speculative decoding is off (verifying a draft needs every token's logits from one
  pass).
- **Pairing.** The chat device shows a link with a random code. Pages announce themselves on a
  Nostr topic derived from the code, with an HMAC of their role and per-visit key, so a relay
  observer cannot join, and only a chat device and a helper connect. Offers and answers are
  encrypted to their recipient, as for the swarm; hidden states travel over the WebRTC data
  channel in 64 KiB frames. A helper receives the model's internal state for each step (derived from the conversation),
  not the conversation's text; pair only devices you trust.

### Sharing pieces between visitors

With sharing on, visitors running the same model file form a swarm and pass pieces browser to
browser (`web/swarm*.mjs`):

- **Pieces and trust.** Files are cut into 256 KiB chunks. The browser reads and caches whole
  chunks, so what it stores can be served. Every chunk a peer sends is checked against a manifest
  of chunk hashes (`tools/chunk_manifest.py`), published in
  [evolve-llm-manifests](https://github.com/anthony-firn/evolve-llm-manifests) by a GitHub Actions
  workflow that streams each file from Hugging Face once and requires it to hash to the SHA-256
  Hugging Face lists. The page trusts `index.json` (TLS from GitHub), checks each 128 KB page of
  hashes against it, and each chunk against its page; a peer that sends a wrong chunk is dropped
  for the visit. Files are named by their SHA-256, so every copy of a file shares one swarm.
- **Finding peers.** Each page makes a fresh Nostr key per visit and announces the files it shares
  as ephemeral events (kind 25075) on public relays; relays forward them without keeping them.
  The page with the smaller key makes the WebRTC offer; offers and answers are encrypted to their
  recipient (ECDH, AES-GCM), so relays never see addresses. STUN finds public addresses; there is
  no TURN relay, so some network pairs cannot connect and simply use Hugging Face.
- **Moving pieces.** Peers exchange which chunk ranges they hold, then request up to 4 MiB at a
  time in 64 KiB frames, from the least busy peer that has them. Serving reads the device cache
  and is capped by the upload limit. WebRTC is not available in workers, so the swarm runs on the
  page; the engine's worker asks it for each piece and falls back to Hugging Face for what no peer
  delivers.

## Kimi K3.1

Kimi K3.1 Pro (`kimi-k31`, 93 layers) differs from DeepSeek V3 in five ways, each with its own
kernels:

- **Kimi delta attention** in 69 layers. Queries, keys and values pass through a short causal
  convolution (width 4, its last three inputs kept between calls) and SiLU; queries and keys are
  L2-normalised per head. Each head keeps a 128 x 128 state that decays per key channel by
  `exp(-5 * sigmoid(...))` and moves toward each token's value along its key (the gated delta
  rule); the query reads it out. One thread owns one value channel and walks the tokens in
  order, so there is no cross-thread reduction. The output is RMS-normalised per head and gated
  by a sigmoid. The state is 434 MB for the whole model and cannot be rewound, so the session
  keeps a copy taken at the end of each prompt and returns to it when a new turn changes later
  text.
- **Latent attention without rotation** in the other 24 layers, with a sigmoid gate on its
  output before the projection.
- **DeepSeek sparse attention.** Past 2048 cached positions, each query attends only to the 2048
  that a lightning indexer scores highest: per head, `relu(q . k)` against LayerNorm'd keys cached
  in f16, weighted per head and summed. An exact radix select (four 8-bit passes) finds each
  token's 2048th-best score, and the attention kernel skips positions below it.
- **Residuals across blocks.** Every 12 layers the residual stream is banked as a checkpoint and
  restarted; each sub-layer's input is a softmax-weighted mix of the stream and the checkpoints,
  scored by an RMS-normalised dot product with a learned vector.
- **Latent experts.** 896 experts, 16 per token, work in a 3584-wide projection of the 7168-wide
  hidden state; their weighted sum is normalised and projected back, and two shared experts
  read the full width. Feed-forward blocks use the situ gate,
  `4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)`, an epilogue of the matrix kernels.

## Text in, text out

- `src/tokenizer.rs` implements byte-level BPE with each model family's pre-tokenizer
  (GPT-2, Llama 3, Qwen2 with NFC normalization, SmolLM's digit splitting, DeepSeek V3's digit,
  CJK and word passes, and Kimi K2's Han-aware tiktoken pattern) and the
  SentencePiece-style merge algorithm with byte fallback for Llama 2 vocabularies. Special
  tokens in the prompt text are matched first.
- `src/chat.rs` renders the chat template stored in the GGUF file with MiniJinja, configured as
  Hugging Face configures Jinja (`trim_blocks`, `lstrip_blocks`, Python string methods,
  `raise_exception`, `strftime_now`, insertion-ordered JSON), and passes tool schemas to
  templates that take them.
- `src/session.rs` keeps the tokens whose keys and values are cached. A new turn computes only
  the part of the prompt that differs from the cached prefix, so follow-up questions start
  quickly.
- `src/sampler.rs` applies repetition penalty, temperature, top-k, top-p and min-p.

## In the browser

`web/worker.js` runs the engine in a module worker so the page stays responsive. Downloads go
straight from Hugging Face into the origin-private file system with `createSyncAccessHandle`,
part by part for split models, and resume with HTTP range requests after a dropped connection.
While a model runs, the worker keeps a synchronous reader per file for streamed weights. The
GPU memory setting becomes the engine's budget; Auto keeps models up to 4 GB fully resident and
gives larger ones 4 GB. If allocation runs out of GPU memory, the loader retries with 60% of
the budget.

With Internet access on, `web/tools.js` gives tool-capable templates `web_search` and
`open_url`. The worker runs up to three tool rounds per reply, skips repeated calls and then asks
for an answer without tools. Models without tool support get search results prepended to the
question. Searches go to Wikipedia and DuckDuckGo, pages through Jina Reader (or Wikipedia's
API for articles), all from the browser. The page needs only this content security policy:

```
script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co
  https://*.wikipedia.org https://api.duckduckgo.com https://r.jina.ai https://s.jina.ai
  https://raw.githubusercontent.com wss: ws://127.0.0.1:* ws://localhost:*
```

The browser build uses a whole-program-optimized release profile, 128-bit WebAssembly SIMD and
binaryen's `wasm-opt`, for a 2.5 MB engine.

## Verification

`tools/make_fixtures.py` records reference results from independent implementations, and
`tests/reference.rs` checks the engine against them:

- dequantization of random blocks in every supported format against gguf-py's NumPy code;
- tokenization of 17 difficult strings (Unicode, emoji, code, numbers, whitespace runs,
  special tokens) against each model's original Hugging Face `tokenizer.json`;
- chat templates rendered by transformers from the template stored in each GGUF file;
- final-token logits against transformers' float32 PyTorch forward pass over the same GGUF
  weights, for a 50 to 70 token chat prompt.

`tests/synthetic.rs` generates a two-layer model in each of the 25 weight formats and compares
the GPU with the CPU reference in the expanded and compact layouts, sharded, and fully streamed,
so every kernel variant runs even when no public model uses it. `tests/tiny.rs` checks each
mixture-of-experts layout against transformers with tiny random models converted by llama.cpp,
resident and streamed through a four-slot expert cache. The DeepSeek V3.1 and Kimi K2
tokenizers and templates are checked from their GGUF headers alone.
The `check` command compares every position of a prompt with the slow CPU reference in
`src/cpu.rs`, through both the one-token and the batched kernels. `tests/browser_test.py`
runs the built page in Chrome with WebGPU under the production content security policy. It
requires the browser's greedy reply to match the native build's reply.
