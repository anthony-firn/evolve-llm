# Browser terminal and Pi

Open [Evolve LLM](https://www.evolvetech.dev/play/evolve-llm/) and select **Terminal**.
The shell works before loading a model. It runs a Rust kernel compiled to WebAssembly in a
separate worker. It has no WASI imports, host filesystem, operating system processes or sockets.

```sh
printf "pear\napple\npear\n" | sort | uniq > fruit.txt
cat fruit.txt
mkdir -p notes
echo "My research" > notes/start.txt
```

Quotes, pipes, `;`, `&&`, `||`, `<`, `>` and `>>` work. Run `help` for the supported commands:
`pwd cd ls find mkdir touch cat echo printf head tail grep rg wc sort uniq cp mv rm tee true false`.
This is a shell subset: no GNU Bash runtime, scripting, globbing, environment variables, command
substitution, Python, Node, npm, package installation or native Pi CLI. `rm` removes files only.
Text utilities support the options shown by their help/errors, rather than all GNU flags.

Files live under `/workspace`, persist in this browser's IndexedDB, and are never synchronized
to a server. Limits are 4 MiB across 128 text files and 128 directories, 32 commands per invocation,
128 KiB output, and 20 seconds per workspace operation. A failed write or invalid restore keeps
the previous state. Stop or Ctrl+C terminates the sandbox worker and retains its last checkpoint.
Use **Import text**, **Export**, and **Restore** to move files between devices. Browser storage can
be cleared or evicted, so export important work. Models and files remain on the visitor's device;
static hosting can be mirrored or served locally without an inference server.

## Pi with the loaded model

Load a tool-capable GGUF such as Qwen, wait for Ready, then enter:

```text
pi Read README.txt and write a short summary to summary.txt
```

This integrates the real MIT-licensed **Pi agent core 1.5.1** from pi-mono, with a custom transport
to the existing Rust/WASM/WebGPU model. It reuses the loaded weights. The Pi bundle and shell
kernel load on demand when opening Terminal, with no second model or provider account.
Pi gets sequential `bash`, `read`, `write`, `edit`, `web_search`, and `open_url` tools.
The UI displays tool arguments, actual results, and completion counts. Each task allows 6 model
turns, 12 tools, 3 tools per model response, and 3 minutes total; tool output sent to the model is
bounded. Stop cancels inference and outstanding reads. If inference will not stop, its worker is
restarted and the model must be reloaded. Chat and model loading wait while Pi owns the engine.

Tool reliability depends on the model. Small models can choose the wrong arguments, ignore
parts of a task or finish without executing tools. Inspect the displayed results and saved files;
a confident answer alone does not establish that a task succeeded. This is the browser agent
integration, not the complete Node-based Pi coding-agent CLI.

## Explicit public internet reads

Enable **Web access** to allow:

```sh
search "WebGPU"
curl https://example.com
```

Search uses Wikipedia and DuckDuckGo instant answers. An optional user-supplied Jina key enables
Jina search. Page reads use Wikipedia or Jina Reader. This is bounded read-only research, not a
full browser or a universal search index. Queries and requested public URLs go to those services;
provider availability, rate limits and CORS apply. Authentication cookies are omitted. Local and
private addresses, URL credentials, redirects, and IPv6 literals are unsupported. The application
uses a fixed public-provider CSP and does not proxy requests through EvolveTech. Web off blocks
new reads, even cache hits; changing the setting cancels current work. Successful reads cache in
memory for 60 seconds (32 entries). Pages and search results are untrusted input, not instructions.
Never put passwords or private tokens in a query or public URL.

The SDK also supports explicit `internet: { enabled: true }` in `generate()`; the default is off.
To self-host the complete app, unzip browser runtime 0.3.0 and serve its directory over localhost
or HTTPS, then open its root. No server-side execution is required.

## Build and check

Install Node 20+, Rust with the wasm32-unknown-unknown target, and wasm-bindgen-cli 0.2.130:

```sh
npm ci --ignore-scripts
cargo test --locked --manifest-path sandbox/Cargo.toml
node tools/build_workbench.mjs
npm test
npm run typecheck
python3 tests/workbench_browser_test.py --runtime web
```

The browser test requires Python Playwright and its Chromium installation. It uses production
CSP, the actual compiled shell and actual Pi core with a scripted model transport, so it does
not download weights. A local Chrome/Qwen3 0.6B GPU smoke test separately verified a two-tool
write/read loop; this is functional evidence, not a model accuracy benchmark.
Pi's upstream license is preserved in `vendor/PI-LICENSE.txt` and distributed beside the bundle.
