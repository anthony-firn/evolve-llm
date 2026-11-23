"""Switching models by kind of request, in the real app with a scripted model worker.

The worker says Qwen3 4B and Nanbeige4.1 3B are both saved on this device and answers with the
name of the model loaded last. Of the two, Nanbeige ranks higher for coding and Qwen3 4B for chat
(BEST_FOR in web/models.js). Checks: a coding request on Qwen3 4B switches to Nanbeige, keeps the
conversation and is answered after the switch, with a switch line in the trace; a chat request
switches back; with the setting off, nothing switches.

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
    "Qwen__Qwen3-4B-GGUF__main__Qwen3-4B-Q4_K_M.gguf": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    "mradermacher__Nanbeige4.1-3B-GGUF__main__Nanbeige4.1-3B.Q4_K_M.gguf": ("mradermacher/Nanbeige4.1-3B-GGUF", "Nanbeige4.1-3B.Q4_K_M.gguf"),
}
WORKER = """
const emit = (type, data = {}) => postMessage({ type, ...data });
const SAVED = %s;
let current = 'none';
self.onmessage = ({ data: m }) => {
  if (m.type === 'hello' || m.type === 'listCache') {
    if (m.type === 'hello') emit('adapter', { adapter: { ok: true, description: 'Scripted', maxBufferSize: 1073741824 } });
    emit('cache', { supported: true, models: Object.entries(SAVED).map(([key, [repo, file]]) => ({ key, repo, file, revision: 'main', bytes: 2500000000, stored: 2500000000, complete: true })) });
  } else if (m.type === 'load') {
    current = m.source?.files?.[0]?.name || m.source?.file || 'unknown';
    (self.loads ??= []).push(current);
    emit('loaded', { source: { key: m.source?.key ?? null, id: m.source?.id ?? null }, model: { parameters: 4e9, active_parameters: 4e9, config: { n_layer: 36 }, weight_types: { Q4_K: 1 } },
      chat: { name: current, context: 8192, weight_bytes: 100, state_bytes: 100, placement: { streaming: false } }, tools: true });
  } else if (m.type === 'agent_turn') {
    emit('agent_done', { requestId: m.requestId, text: `Answered by ${current}.`, reason: 'end', n: 5 });
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
            page.set_input_files("#file-input", {"name": "Qwen3-4B-Q4_K_M.gguf", "mimeType": "application/octet-stream", "buffer": b"x"})
            page.wait_for_selector("#ready:not([hidden])")

            def send(text):
                page.locator("#prompt").fill(text)
                page.locator("#send").click()
                page.wait_for_function("() => document.querySelector('#app').dataset.phase === 'ready' && document.querySelectorAll('.msg.user').length === document.querySelectorAll('.msg.assistant').length", timeout=30000)
                return page.locator(".msg.assistant").last.inner_text()

            reply = send("Fix the bug in calc.py and run its tests")
            check("Answered by Nanbeige4.1-3B.Q4_K_M.gguf" in reply, f"a coding request switches to Nanbeige and is answered there ({reply!r})")
            check("Switched to Nanbeige4.1 3B" in page.locator("#status-line").inner_text(), f"the status line says why ({page.locator('#status-line').inner_text()!r})")
            reply = send("Tell me a short joke about cats")
            check("Answered by Qwen3-4B-Q4_K_M.gguf" in reply, f"a chat request switches back to Qwen3 4B ({reply!r})")
            check(page.locator(".msg.user").count() == 2, "the conversation is kept across the switches")
            switches = [json.loads(l) for l in trace.read_text().splitlines() if '"switch"' in l]
            check([(x["kind"], x["to"]) for x in switches] == [("coding", "Nanbeige4.1 3B"), ("chat", "Qwen3 4B")], f"the trace records both switches ({switches})")
            page.evaluate("""() => { const s = JSON.parse(localStorage.getItem('evolve-llm-settings') || '{}'); s.autoSwitch = false; localStorage.setItem('evolve-llm-settings', JSON.stringify(s)); }""")
            page.locator("#open-settings").click()
            box = page.locator("input[name=autoSwitch]")
            if box.is_checked():
                box.uncheck()
            page.keyboard.press("Escape")
            reply = send("Fix the bug in calc.py again")
            check("Answered by Qwen3-4B-Q4_K_M.gguf" in reply, f"with switching off, the coding request stays on Qwen3 4B ({reply!r})")
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
