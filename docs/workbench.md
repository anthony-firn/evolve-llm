# The main chat, the terminal and Pi

There is one chat. Ask it anything: **“What is WebGPU?”**, **“Make me a Snake game I can play”**,
or **“Clone anthony-firn/evolve-llm, fix the typo in the README and open a pull request.”**
The model decides whether a message needs tools. A question gets an answer; a request to make or
change something runs tools and shows one collapsed line under the reply (**Worked · 4 steps**),
each step in plain words (“Ran git status”, “Edited README.md”) with its output inside. An app or
game it opens plays in the conversation. Follow-up messages see the conversation and the same
files. Stop ends the current reply. New chat clears the conversation and previews; files stay.

Behind the chat is Pi's agent core (below) with the loaded model and one prompt and set of tools:
`bash`, `read`, `write`, `edit`, `preview` and `browser_test`, `web_search` and `open_url` when Web
is on, and the shared browser for requests about a live website. In Chrome and Edge it works in the
Evolve VM's Linux system, the same files, `git` and `gh` as **Terminal** (see “Linux terminal”
below); its `cd` carries over between commands and messages. Browsers without WebAssembly JSPI
(Firefox, Safari) cannot run the VM, so there it works in the browser sandbox's `/workspace`, with
the same files and previews but no `git`. Models whose chat template takes no tools (SmolLM2,
OLMoE) answer as plain chat.

**Chats** lists earlier conversations, saved in this browser only (IndexedDB) after every reply.
Opening one shows its replies and steps, reopens an app it made from the files, and continues where
it stopped; Delete removes it.

### Watching the assistant work, on your own computer

```sh
python3 tools/build_web.py
python3 tools/serve_local.py            # http://localhost:8080/play/evolve-llm/
python3 tools/trace_view.py --follow    # in another terminal
```

`serve_local.py` serves the app to this computer only and appends a trace of every chat to
`~/.cache/evolve-llm/traces/YYYY-MM-DD.jsonl`: each run's request and settings (never the GitHub
token or API keys), each model turn with its raw output, token counts and time, each tool call and
result, and each approval. `trace_view.py` prints one line per event (`--full` for whole outputs),
which shows where a model goes wrong: an invalid tool call, a wrong path, a refused command. The
page traces only when it was served from this computer by that server; the public site and plain
static hosting never send it. `tests/chat_github_real_model_test.py` prints the same trace live.

Making a new Snake game starts from the included working Snake game, saved under an unused name
and opened at once; the model then only applies the requested changes. Other apps rely on the
loaded model. Small models still make wrong tool choices: the steps show what actually ran, and a
runtime smoke check does not prove gameplay logic.

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

The **Browser** button opens a real web browser engine that Pi can share with you; see
[browser.md](browser.md).

## Pi with the loaded model

Load a tool-capable GGUF such as Qwen, wait for Ready, then enter:

```text
pi Read README.txt and write a short summary to summary.txt
```

This integrates the real MIT-licensed **Pi agent core 1.5.1** from pi-mono, with a custom transport
to the existing Rust/WASM/WebGPU model. It reuses the loaded weights. The Pi bundle and shell
kernel load on demand when opening Terminal, with no second model or provider account.
Pi gets sequential `bash`, `read`, `write`, `edit`, `web_search`, `open_url`, `preview`, and `browser_test` tools.
The UI displays tool arguments, actual results, and completion counts. Each task allows 10 model
turns, 12 tool calls and up to 3 tools per model response; tool output sent to the model is
bounded. A task may take 3 minutes when the model fits on the GPU, 2 hours (30 minutes per model
turn) when it streams from this device's storage, and as long as it needs when it is read from
Hugging Face, where a token can take minutes or hours. Up to two format retries count toward the
turn limit. Invalid or truncated calls never execute; failed tool calls are shown alongside the
actual file-change count. Edits require a prior read and a nonempty exact substring from that read.
Network tools are offered only with Web on. Stop cancels inference and outstanding reads,
including weights still arriving from Hugging Face. If inference will not stop, its worker is
restarted and the model must be reloaded. Chat and model loading wait while Pi owns the engine.

Pi reads tool calls in each model's own syntax: Qwen and Hermes `<tool_call>` JSON, Qwen3 Coder's
`<function=…>` parameters, Llama 3's JSON object, Mistral's `[TOOL_CALLS]`, DeepSeek V3 and V3.1,
Kimi K2, and Kimi K3 and K3.1, whose tool markers are control tokens the page keeps for this. Earlier
calls and results go back to the model as structured messages, so its template shows them the way
it was trained to see them. Models whose templates have no tools (SmolLM2, OLMoE) get JSON
instructions instead. `tests/tool_templates.rs` renders a call with each family's template and
`tests/tool-formats.test.mjs` parses it back.

Tool reliability depends on the model. Small models can choose the wrong arguments, ignore
parts of a task or finish without executing tools. Inspect the displayed results and saved files;
a confident answer alone does not establish that a task succeeded. This is the browser agent
integration, not the complete Node-based Pi coding-agent CLI.

## Linux terminal: Pi on your repositories

**Terminal** opens the Linux terminal on the Evolve VM (in Chrome and Edge): a Rust kernel compiled to WebAssembly that
runs WASI programs as processes, with files, pipes, terminals and a POSIX-style shell, in a
GNOME-style terminal (tabs, Ctrl+Shift+T, Find, themes) drawn by ghostty-web. Files under
`/home/user` are saved in this browser.

`git` and `gh` work on GitHub repositories through GitHub's API (the VM has no sockets, so this
is not Git's own protocol):

```text
gh repo clone anthony-firn/evolve-llm      # every file is checked against GitHub's blob hash
cd evolve-llm && git checkout -b fix-typo
git status; git diff; git commit -m "Fix a typo"   # every change is included; no staging area
git push                                    # creates the branch on GitHub
gh pr create --title "Fix a typo" --body "…"
gh run list --branch fix-typo; gh run view RUN_ID  # CI results: failed steps and annotations
```

`python3` runs Python there: RustPython 0.6.0 (a Python 3 interpreter written in Rust, with its
standard library inside), built from its crates.io release by `tools/build_vm.py` and loaded the
first time it runs (33 MB). Scripts, the standard library and `unittest` work; there is no pip,
and C extensions such as numpy are not available. Bytecode caches are not written, so `git`
does not pick up `__pycache__`.

`pi TASK` in the Linux terminal gives the task to Pi with the loaded model. Pi works in the
current directory with the VM's shell, files, `git` and `gh`, writes its progress to the
terminal, and stops on Ctrl+C. A task allows 40 model turns and 80 tool calls. The VM has no
compiler, Node or Python yet, so builds and tests run in the repository's GitHub Actions: Pi pushes
a branch and reads its CI run.

Merging works on GitHub, and always asks the person first, whatever the autonomy setting:

```text
gh pr merge                  # this branch's pull request (or gh pr merge 12, --squash, --rebase)
git checkout main && git pull
git merge feature            # feature into the current branch, as a merge commit on GitHub
```

The question names the pull request or branches, the commits and files, and the checks. A declined
or conflicting merge changes nothing and says so. `git checkout BRANCH` switches to a branch on
GitHub, `git branch` lists them, and every refusal names the command to run instead (“3 changed
files not committed yet. First run: git commit -m "MESSAGE"”), so a small model can recover. `git
help` and `gh help` print a short recipe. Switching, pulling and merging need everything committed
and pushed first.

Small models make the same slips; the tools catch them. `git commit` refuses when a file of 20 or
more lines loses most of them (it looks overwritten, for example with `>` instead of `>>`) and
says how to undo it; `--allow-shrink` commits it anyway. Every commit prints each file's added and
removed lines, and the question before a push says the same ("README.md +1 -0"). `edit FILE OLD
NEW` is a command in the VM: OLD becomes NEW where it occurs exactly once (`\n` is a new line).
A failed shell command comes back with notes on what to run instead.

Settings → **GitHub** sets what the assistant (in the chat or `pi`) may do:

- **Commit locally only**: `git push`, `gh pr create`, merging and write calls through `gh api`
  are refused.
- **Ask me before pushing, opening a pull request or merging** (default): the page asks each time.
- **Push branches and open pull requests freely**: no question, except for a repository's
  default branch and for merging, which always ask.

A GitHub token is optional; without one, Pi can clone public repositories (GitHub allows 60 API
requests an hour without a token). Use a fine-grained token limited to the repositories Pi may
change, with Contents and Pull requests (read and write) and Actions (read). The token stays in this
browser and in the page's VM host: programs in the VM ask the host to make a request, the host
adds the token only for `api.github.com`, and only `api.github.com` and
`raw.githubusercontent.com` are reachable. `tests/vm_github_browser_test.py` checks clone, status,
diff, commit, push (the pushed tree, its parent, the question), pull requests, CI, switching
branches, `git merge` and `gh pr merge` (asked even when pushing is free; declined and conflicting
merges change nothing), the three autonomy levels and the token's destination against a fake
GitHub; `tests/vm_pi_browser_test.py` runs Pi's agent core with a scripted model through a clone, a
`cd` in its own command, an edit, a push, a pull request and an approved merge.
`tests/chat_github_real_model_test.py --runtime dist model.gguf` sends the same work to a real
model through the main chat and prints its steps and what the fake GitHub then holds.

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
To self-host the complete app, unzip browser runtime 0.4.0 and serve its directory over localhost
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
python3 tests/chat_development_browser_test.py --runtime web   # the main chat, Chrome (VM) and Firefox (sandbox)
```

The browser test requires Python Playwright and its Chromium installation. It uses production
CSP, the actual compiled shell and actual Pi core with a scripted model transport, so it does
not download weights. A local Chrome/Qwen3 0.6B GPU smoke test separately verified a two-tool
write/read loop; this is functional evidence, not a model accuracy benchmark.
For a real-GGUF main-chat check, run `python3 tests/chat_local_model_browser_test.py --runtime dist --model /path/to/model.gguf`. This checks the included Snake game and cancellation with the actual WebGPU model; it does not claim that a tiny model reliably customizes games.
Pi's upstream license is preserved in `vendor/PI-LICENSE.txt` and distributed beside the bundle.

## Run and inspect HTML apps

Enter `preview snake.html` to run a saved single-file HTML app, or `test snake.html` for a
runtime smoke check. `demo snake` copies the bundled Snake example to snake-example.html and
opens it, with no model needed. This is a labeled example, not generated model output.
Use arrow keys or its on-screen controls; Reload, Test, Full size and Show/hide log stay available.
Pi automatically opens and checks the last HTML file it successfully writes or edits.
The preview tool is the final action: a successful preview ends the task without another model
turn. Use browser_test for intermediate checks and repairs, then preview when finished.

Try `pi Make snake-example.html use a blue background, then browser_test it and preview it`.
The Pi tokens selector defaults to 2,048 output tokens, with options for 1,024 or 4,096; the
loaded model context must also have room for the prompt and response. A compact game or
small edits work better than generating a large app in one call. Truncated or invalid tool
calls report failure and never execute partially. For models with a broken double-braced JSON
example in their GGUF tool template, Pi supplies valid tool instructions through the ordinary
chat template. Complete JSON-only Markdown blocks are also accepted; prose examples and
incomplete JSON are rejected. Qwen2.5 0.5B can still invent edits or ignore feedback; use a coding model and
inspect the file-change count before trusting a completion claim.

The preview runs inline HTML/CSS/JavaScript in an opaque-origin iframe, with no same-origin
permission, host execution, workspace API, storage, external assets, network requests, forms,
popups or top navigation. Web access in the assistant does not grant network to preview apps.
No npm, server programs or multi-file asset resolution is provided. Stop preview removes the
frame. The runtime check waits about one second, collects exceptions, rejected promises,
console errors and CSP violations, checks canvas dimensions/paint, dispatches two arrow keys
and clicks a Restart button if present. Later script errors appear in the log. These checks
and their in-frame reports are limited smoke evidence, not proof of scoring or collision logic.
The frame executes DOM JavaScript through the browser; the shell and model use WASM.

The app's own CSP stays strict. Only preview/ permits inline scripts, with HTTP sandbox
allow-scripts and no connect or frame sources. The bundled preview also sets a restrictive
meta CSP for localhost hosting; preserve the iframe sandbox attribute. Production host
headers are in evolvetech.dev/public/_headers and mirrored by its Bunny deployment.
