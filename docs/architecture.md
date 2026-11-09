# How Evolve LLM works

Evolve LLM is a transformer inference engine written in Rust. The same source compiles to a
native command-line program, which uses Vulkan, Metal or Direct3D 12, and to WebAssembly, which
uses the browser's WebGPU. Every token is computed by WGSL compute shaders. The CPU only
tokenizes, samples and schedules work.

## From file to GPU

1. **Header.** `src/gguf.rs` parses a GGUF prefix. When the prefix is too short, the parser
   reports how many bytes it needs, so the browser reads only the header (often 2 to 10 MB)
   before deciding whether the model can run.
2. **Plan.** `src/model.rs` validates the architecture, shapes and weight formats. The engine
   allocates one GPU buffer pair per weight matrix and returns an upload plan: each tensor's
   file offset, row size and row count.
3. **Upload.** The worker reads 16 MB slices of the file and passes them to
   `Engine::upload`. `src/quant.rs` repacks each block into a word-aligned layout before
   `queue.write_buffer`. The model never has to fit in WebAssembly memory at once.
4. **Shards.** A matrix larger than the device's storage-binding limit is split into
   row shards on 64-row boundaries, so phones and integrated GPUs with small limits can still
   hold large vocabularies.

## Weight layouts

GGUF blocks pack scales, minimums and bit planes into odd sizes (18, 22, 34 and 210 bytes).
WGSL reads storage buffers as 32-bit words, so the engine decodes each block into integer codes
plus a scale and an optional bias per group, with `value = scale * (code - offset) + bias`. The
arithmetic reproduces ggml's reference dequantization exactly.

Codes of up to six bits are stored as a plane of 4-bit nibbles plus, for 5- and 6-bit formats,
a second plane holding the extra one or two bits per value. Each matrix keeps both planes in one
buffer, so a 32-value step reads one 16-byte nibble load and one or two extra words.

| GPU layout | Source formats | Bits per weight on the GPU |
|---|---|---|
| 4-bit nibbles, groups of 32 | Q4_0, Q4_1, Q4_K | 5 to 6 |
| Nibbles plus 1 extra bit, groups of 32 | Q5_0, Q5_1, Q5_K | 6 to 7 |
| Nibbles plus 2 extra bits, groups of 16 | Q6_K | 8 |
| Nibbles, groups of 16 | Q3_K, Q2_K | 6 to 8 |
| 8-bit codes, groups of 32 | Q8_0, IQ4_NL, IQ4_XS (codebook values) | 9 |
| Half floats | F16, BF16, F32 | 16 |

Scales stay in f32 so the GPU values equal the reference values. Storing them as f16 would save
about one more bit per weight.

## Kernels

`src/gpu/shaders.rs` generates WGSL per weight layout.

- **Matrix-vector product** for one token. Four lanes cooperate on each output row and each lane
  issues one 16-byte weight load per step. Four lanes per row measured 2.6 times faster than 16
  on the Intel GPU and tied with every alternative on the power-capped NVIDIA GPU, where memory
  bandwidth was the limit.
- **Tiled matrix product** for prompts: 64 rows by 32 tokens per workgroup through shared memory,
  with weights dequantized while the tile loads.
- **RMSNorm**, including per-head normalization of queries and keys for Qwen3.
- **Rotary embeddings** in both pairing conventions (adjacent pairs for Llama, half-split for
  Qwen), with Llama 3 frequency factors from a precomputed table. The same kernel writes keys and
  values into an f16 cache.
- **Attention** with a streaming softmax: one workgroup per head and token walks the cache in
  blocks of 256 keys, so context length is limited only by cache memory. Grouped-query attention
  maps several query heads to each cached head.

Matrix kernels end with an epilogue that stores the result, adds it to the residual stream, or
multiplies it into the SiLU-gated feed-forward activation. Each layer needs about a dozen
dispatches, and one token's forward pass is a single command buffer. Prompts run in chunks of
128 tokens. Only the final token's logits are read back.

## Text in, text out

- `src/tokenizer.rs` implements byte-level BPE with each model family's pre-tokenizer
  (GPT-2, Llama 3, Qwen2 with NFC normalization, and SmolLM's digit splitting) and the
  SentencePiece-style merge algorithm with byte fallback for Llama 2 vocabularies. Special
  tokens in the prompt text are matched first.
- `src/chat.rs` renders the chat template stored in the GGUF file with MiniJinja, configured as
  Hugging Face configures Jinja (`trim_blocks`, `lstrip_blocks`, Python string methods,
  `raise_exception`, `strftime_now`).
- `src/session.rs` keeps the tokens whose keys and values are cached. A new turn computes only
  the part of the prompt that differs from the cached prefix, so follow-up questions start
  quickly.
- `src/sampler.rs` applies repetition penalty, temperature, top-k, top-p and min-p.

## In the browser

`web/worker.js` runs the engine in a module worker so the page stays responsive. Downloads go
straight from Hugging Face into the origin-private file system with `createSyncAccessHandle`.
They resume with HTTP range requests after a dropped connection. Loading reads that stored file,
or a file the person opens, in slices. The page needs only this content security policy:

```
script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co
```

## Verification

`tools/make_fixtures.py` records reference results from independent implementations, and
`tests/reference.rs` checks the engine against them:

- dequantization of random blocks in every supported format against gguf-py's NumPy code;
- tokenization of 17 difficult strings (Unicode, emoji, code, numbers, whitespace runs,
  special tokens) against each model's original Hugging Face `tokenizer.json`;
- chat templates rendered by transformers from the template stored in each GGUF file;
- final-token logits against transformers' float32 PyTorch forward pass over the same GGUF
  weights, for a 50 to 70 token chat prompt.

`tests/synthetic.rs` generates a two-layer model in each of the 15 weight formats and compares
the GPU with the CPU reference, so every kernel variant runs even when no public model uses it.
The `check` command compares every position of a prompt with the slow CPU reference in
`src/cpu.rs`, through both the one-token and the batched kernels. `tests/browser_test.py`
runs the built page in Chrome with WebGPU under the production content security policy. It
requires the browser's greedy reply to match the native build's reply.
