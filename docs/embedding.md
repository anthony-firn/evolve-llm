# Embed Evolve LLM

The browser SDK owns a dedicated module worker. It streams events from the existing Rust/WASM
engine and releases the worker on disposal, cancellation or failure. It has no dependency on
the chat UI. Internet tools default to off; opt in per generation with `internet: { enabled: true }`.
Tool events appear in the stream. See [public web permissions and providers](workbench.md).

Download the runtime bundle from https://www.evolvetech.dev/developers/llm/. It contains the SDK,
TypeScript declarations, worker, model catalog, WebAssembly engine, license and an example.
Serve the files from the **same origin** as your application: browsers reject a cross-origin
Worker even when an imported JavaScript module itself is accessible.

```sh
unzip evolve-llm-browser-0.3.0.zip -d evolve-llm
python3 -m http.server 8080 --directory evolve-llm
# Open http://localhost:8080/examples/
```

The protocol fixture works without WebGPU, model weights or external requests. It emits scripted
text and timing and is not an inference benchmark. The separate real-model button explicitly
downloads about 145 MB from Hugging Face, then generates text using the visitor's GPU. No prompt
is sent to a provider. WebGPU, disk space, browser support and model licenses still apply.

```js
import { createEngine, EngineError } from './evolve-llm/sdk.mjs';
const engine = createEngine();
const abort = new AbortController();
try {
  await engine.initialize({ signal: abort.signal });
  await engine.load('smollm2-135m', {
    contextSize: 2048,
    signal: abort.signal,
    onProgress: event => console.log(event.type),
  });
  for await (const event of engine.generate([
    { role: 'user', content: 'Explain WebGPU in one sentence.' },
  ], { signal: abort.signal, maxTokens: 64 })) {
    if (event.type === 'token') document.querySelector('#answer').textContent += event.text;
  }
} catch (error) {
  if (!(error instanceof EngineError)) throw error;
  console.error(error.code); // handle a stable code; no prompt in the error
} finally {
  engine.dispose();
}
// Bind abort.abort() to a Cancel control while the operation is running.
```

## Contract

`initialize()` checks the GPU adapter and initializes the WASM runtime. `load(id, options)`
uses an exact catalog ID, such as `smollm2-135m`, `qwen2.5-0.5b` or `llama3.2-1b`.
Use `load({ files: [...fileInput.files] })` for local GGUF files. Remote catalog files use the
catalog's Hugging Face URL; its current revision is `main`, not an immutable model artifact.
Pin and retain your own model files if you need reproducible weights.

`generate(messages, options)` returns an async iterator of `prefill`, `token` and `done` events.
Token events contain incremental `text`; `done` includes engine counters and timing. These
counters are measurements of that run, not a browser-wide performance guarantee. Messages
accept `system`, `user` and `assistant` roles. Settings include `maxTokens`, `temperature`,
`seed` and `thinking`. `reset()` clears chat state; `dispose()` permanently closes that client.

`contextSize` accepts 256–32,768 tokens and is capped by the model’s training context. `gpuMemoryMB` accepts a positive integer MB budget or `auto`; the SDK converts MB to the worker’s GB setting.

One operation may run per client. A concurrent call fails with `busy` instead of mixing work.
Cancellation and breaking out of an unfinished iterator terminate the worker and unload the
model. Reload before generating again. A completed iterator retains the model. A new client
is required after `dispose()`. Events from a terminated worker cannot complete a later load.
The SDK buffers at most 1,024 stream events and fails with `backpressure` if the consumer falls
behind. Loading progress is delivered to the callback without accumulating an event history.

The default operation deadline is 120 seconds; set `createEngine({ timeoutMs })` from 1 ms to
one hour for slow downloads or longer generations. A timeout unloads the model. The SDK never
automatically retries a model download or a generation. The runtime's existing download cache
can resume a partial file when the caller explicitly loads it again.

`EngineError.code` is one of `busy`, `not_loaded`, `unsupported_gpu`, `aborted`, `timeout`,
`worker`, `protocol`, `engine`, `disposed` or `backpressure`. Argument errors are `TypeError`.
Worker errors use fixed messages, so raw engine exceptions and prompt text are not reflected.

## Verified scope

SmolLM2 135M, Qwen2.5 0.5B, Qwen3 0.6B, Llama 3.2 1B and OLMoE 1B-7B have browser/native
reply comparisons in the engine's existing verification record. The largest catalog models
have architecture and layout tests, but have not been run end to end. Their speed is unmeasured.
See [the engine's verification and measured results](https://github.com/anthony-firn/evolve-llm#verification).

The SDK protocol tests use synthetic workers to check stream ordering, cancellation, bounded
buffers, timeouts, late messages, failure cleanup and concurrent calls. The browser fixture
checks the actual module Worker and async-iterator path without claiming to test GPU inference.

The download also includes the complete chat app and browser Pi/WASM terminal. Open its root
for the workbench or `examples/` for the SDK demo. See the included `WORKBENCH.md`.
