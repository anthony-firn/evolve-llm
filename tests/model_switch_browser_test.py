"""Requests climbing the device's frontier, in the real app with a scripted model worker.

The worker says a 6 GB GPU has Qwen3 4B Instruct 2507, Qwen3.5 4B and Qwen3 Coder 30B-A3B saved,
answers with the name of the model loaded last, and fails a turn on Qwen3.5 4B whose request
contains FAIL. Checks: an everyday request runs on Qwen3 4B 2507 and a coding request on Qwen3.5
4B, each answered after the switch with the conversation kept; a turn that fails is tried again
on the next rung (Qwen3 Coder) and answered there; the trace records each switch; a model a
developer opened from a file is never replaced.

    python3 tests/model_switch_browser_test.py [--runtime web]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import Handler, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "unsloth__Qwen3-4B-Instruct-2507-GGUF__main__Qwen3-4B-Instruct-2507-Q4_K_M.gguf": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120),
    "unsloth__Qwen3.5-4B-GGUF__main__Qwen3.5-4B-Q4_K_M.gguf": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2740937888),
    "unsloth__Qwen3-Coder-30B-A3B-Instruct-GGUF__main__Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf": (
        "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf", 12848766112),
}
WORKER = """
const emit = (type, data = {}) => postMessage({ type, ...data });
const SAVED = %s;
let current = 'none';
self.onmessage = ({ data: m }) => {
  if (m.type === 'hello' || m.type === 'listCache') {
    if (m.type === 'hello') emit('adapter', { adapter: { ok: true, description: 'Scripted', maxBufferSize: 1073741824 } });
    emit('cache', { supported: true, usage: 20e9, quota: 100e9, models: Object.entries(SAVED).map(([key, [repo, file, bytes]]) => ({ key, repo, file, revision: 'main', bytes, stored: bytes, complete: true })) });
  } else if (m.type === 'probe') {
    emit('probed', { gpu: m.gpu, bytes: 6e9 });
  } else if (m.type === 'load') {
    current = m.source?.files?.[0]?.name || m.source?.file || 'unknown';
    emit('loaded', { source: { key: m.source?.key ?? null, id: m.source?.id ?? null }, model: { parameters: 4e9, active_parameters: 4e9, config: { n_layer: 36 }, weight_types: { Q4_K: 1 } },
      chat: { name: current, context: 8192, weight_bytes: 100, state_bytes: 100, placement: { streaming: false } }, tools: true });
  } else if (m.type === 'agent_turn') {
    if (current.startsWith('Qwen3.5-4B') && JSON.stringify(m).includes('FAIL')) emit('error', { requestId: m.requestId, message: 'scripted failure' });
    else emit('agent_done', { requestId: m.requestId, text: `Answered by ${current}.`, reason: 'end', n: 5 });
  }
};
""" % json.dumps(MODELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=str(ROOT / "web"))
    a = ap.parse_args()
    trace = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    Handler.trace_file = str(trace)
    server = serve(a.runtime)
    problems = []

    def check(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            problems.append(what)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.route("**/worker.js", lambda r: r.fulfill(status=200, content_type="text/javascript", body=WORKER))
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/")
            page.wait_for_selector("#gpu-chip[data-state=ok]")
            page.wait_for_selector("#ready:not([hidden])", timeout=15000)

            def send(text):
                page.locator("#prompt").fill(text)
                page.locator("#send").click()
                page.wait_for_function("() => document.querySelector('#app').dataset.phase === 'ready' && document.querySelectorAll('.msg.user').length === document.querySelectorAll('.msg.assistant').length", timeout=30000)
                return page.locator(".msg.assistant").last.inner_text()

            def status():
                return page.locator("#status-line").inner_text()

            reply = send("Tell me a short joke about cats")
            check("Answered by Qwen3-4B-Instruct-2507" in reply, f"an everyday request runs on Qwen3 4B Instruct 2507 ({reply!r})")
            reply = send("Fix the bug in calc.py and run its tests")
            check("Answered by Qwen3.5-4B" in reply, f"a coding request climbs to Qwen3.5 4B ({reply!r})")
            check("for real work" in status(), f"the status line says why ({status()!r})")
            reply = send("Add a --verbose flag to calc.py FAIL")
            check("Answered by Qwen3-Coder-30B-A3B" in reply, f"a failed turn is tried again on the next rung and answered there ({reply!r})")
            check("is trying again" in status(), f"the status line says so ({status()!r})")
            check(page.locator(".msg.user").count() == 3 and page.locator(".msg.assistant").count() == 3, "the conversation is kept, with the failed reply replaced")
            switches = [json.loads(l) for l in trace.read_text().splitlines() if '"switch"' in l]
            check([(x["to"], x.get("retry", False)) for x in switches] == [("Qwen3 4B Instruct 2507", False), ("Qwen3.5 4B", False), ("Qwen3 Coder 30B-A3B", True)],
                  f"the trace records each switch ({[(x['to'], x.get('retry')) for x in switches]})")
            # A developer's own file stays, whatever the request.
            page.set_input_files("#file-input", {"name": "my-model.gguf", "mimeType": "application/octet-stream", "buffer": b"x"})
            page.wait_for_timeout(500)
            page.wait_for_function("() => document.querySelector('#app').dataset.phase === 'ready'", timeout=15000)
            reply = send("Refactor calc.py into a package and debug its race condition")
            check("Answered by my-model.gguf" in reply, f"a model opened from a file is not replaced ({reply!r})")
            check(not errors, f"no page errors ({errors})")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("model switch browser test passed")


if __name__ == "__main__":
    main()
