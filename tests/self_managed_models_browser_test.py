"""Models that manage themselves, in the real app with a scripted model worker.

The worker pretends to be this device: how much GPU memory a probe finds, which models are saved,
and it reports what the app asks of it (loads, background downloads, removals) on a
BroadcastChannel the test reads. Each visit is a page load with the saved models the earlier
steps left. Checks:
1. First visit, a 6 GB GPU: the page shows only the message box and the conversation list, says
   the first message downloads Qwen3.5 4B, and the first message loads it and is answered.
2. Next visit: the saved model loads by itself.
3. A 2 GB GPU: the 4B (saved, last used) loads; when idle, Qwen3.5 2B downloads in the
   background, and the next request runs on it (the 4B does not fit this GPU's frontier).
4. A model unused for 100 days is removed when idle; the running one and the pick are kept.
5. Each request runs on the rung of the frontier its difficulty needs: an everyday request on
   Qwen3 4B Instruct 2507, a coding request on Qwen3.5 4B; "that did not work" climbs a level,
   whose model (Qwen3 Coder 30B-A3B) downloads in the background and then answers; a new
   conversation starts again from the request's own level. The model list is not a picker.
6. With "Choose models for me" off, nothing loads by itself.

    python3 tests/self_managed_models_browser_test.py [--runtime web]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GB = 10**9
KEY = {
    "qwen3.5-4b": ("unsloth__Qwen3.5-4B-GGUF__main__Qwen3.5-4B-Q4_K_M.gguf", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2740937888, "Qwen3.5 4B"),
    "qwen3.5-2b": ("unsloth__Qwen3.5-2B-GGUF__main__Qwen3.5-2B-Q4_K_M.gguf", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1280835840, "Qwen3.5 2B"),
    "qwen3-4b-2507": ("unsloth__Qwen3-4B-Instruct-2507-GGUF__main__Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                      "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120, "Qwen3 4B Instruct 2507"),
}

WORKER = """
const emit = (type, data = {}) => postMessage({ type, ...data });
const log = new BroadcastChannel('scripted-worker');
const STATE = %s;
const cache = () => ({ supported: true, usage: 5e9, quota: 100e9, models: Object.values(STATE.saved) });
self.onmessage = ({ data: m }) => {
  if (m.type === 'hello') { emit('adapter', { adapter: { ok: true, description: 'Scripted GPU', maxBufferSize: 1073741824 } }); emit('cache', cache()); }
  else if (m.type === 'listCache') emit('cache', cache());
  else if (m.type === 'probe') { log.postMessage({ type: 'probe', gpu: m.gpu }); emit('probed', { gpu: m.gpu, bytes: STATE.gpuBytes }); }
  else if (m.type === 'load') {
    log.postMessage({ type: 'load', id: m.source.id, kind: m.source.kind });
    emit('loaded', { source: { kind: m.source.kind, key: m.source.key ?? null, id: m.source.id ?? null }, model: { parameters: 4e9, active_parameters: 4e9, config: { n_layer: 36 }, weight_types: { Q4_K: 1 } },
      chat: { name: m.source.id || 'file', context: 8192, weight_bytes: 100, state_bytes: 100, placement: { streaming: false } }, tools: true });
  } else if (m.type === 'fetch') {
    log.postMessage({ type: 'fetch', id: m.source.id });
    emit('fetch-progress', { key: m.source.key, loaded: m.source.bytes / 2, total: m.source.bytes });
    STATE.saved[m.source.key] = { key: m.source.key, repo: m.source.repo, file: m.source.file, name: m.source.name, bytes: m.source.bytes, stored: m.source.bytes, complete: true };
    emit('cache', cache());
    emit('fetched', { key: m.source.key });
  } else if (m.type === 'agent_turn') {
    emit('agent_done', { requestId: m.requestId, text: 'Done.', reason: 'end', n: 2 });
  } else if (m.type === 'removeCache') {
    log.postMessage({ type: 'remove', key: m.key });
    delete STATE.saved[m.key];
    emit('cache', cache());
  }
};
"""


def saved(*ids):
    out = {}
    for i in ids:
        key, repo, file, size, name = KEY[i]
        out[key] = {"key": key, "repo": repo, "file": file, "name": name, "bytes": size, "stored": size, "complete": True}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=str(ROOT / "web"))
    a = ap.parse_args()
    server = serve(a.runtime)
    url = f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
    problems = []

    def check(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            problems.append(what)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context()
            # The worker's messages to the test, kept across page loads in this tab.
            ctx.add_init_script("window.__log = []; new BroadcastChannel('scripted-worker').onmessage = (e) => window.__log.push(e.data);")
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            state = {"gpuBytes": 6 * GB, "saved": {}}
            page.route("**/worker.js", lambda r: r.fulfill(status=200, content_type="text/javascript", body=WORKER % json.dumps(state)))

            def visit():
                page.goto(url)
                page.wait_for_selector("#gpu-chip[data-state=ok]")

            def log(kind):
                return [e for e in page.evaluate("window.__log") if e["type"] == kind]

            def ask(text, replies):
                page.locator("#prompt").fill(text)
                page.locator("#send").click()
                page.wait_for_function(f"() => document.querySelector('#app').dataset.phase === 'ready' && document.querySelectorAll('.msg.assistant').length === {replies}", timeout=20000)

            def wait_log(kind, seconds):
                end = time.time() + seconds
                while time.time() < end and not log(kind):
                    time.sleep(0.5)
                return log(kind)

            # 1. First visit on a 6 GB GPU.
            visit()
            first = page.locator("#first-run")
            first.wait_for(state="visible")
            check("Qwen3.5 4B (2.74 GB)" in first.inner_text(), f"the page says what the first message downloads ({first.inner_text()!r})")
            controls = page.eval_on_selector_all("button", "bs => bs.filter(b => b.offsetParent).map(b => b.getAttribute('aria-label') || b.textContent.trim())")
            check(controls == ["New chat", "Settings", "Send"], f"the page has no other controls ({controls})")
            check(len(log("load")) == 0, "nothing loads before the first message")
            ask("Hello", 1)
            check([e["id"] for e in log("load")] == ["qwen3.5-4b"], f"the first message loads Qwen3.5 4B and is answered ({log('load')})")

            # 2. Next visit, the 4B saved: it loads by itself.
            state["saved"] = saved("qwen3.5-4b")
            visit()
            page.wait_for_selector("#ready:not([hidden])", timeout=15000)
            check([e["id"] for e in log("load")] == ["qwen3.5-4b"], f"the saved model loads by itself ({log('load')})")
            check(len(log("probe")) == 0, "the GPU is measured once, not on every visit")

            # 3. A smaller GPU (measured again): the 4B loads; the 2B downloads when idle and, with
            # a conversation under way, takes over at the next new conversation.
            page.evaluate("() => { for (const k of Object.keys(localStorage)) if (k.startsWith('evolve-llm-gpu-')) localStorage.removeItem(k); }")
            state["gpuBytes"] = 2 * GB
            visit()
            page.wait_for_selector("#ready:not([hidden])", timeout=15000)
            check([e["id"] for e in log("load")] == ["qwen3.5-4b"], f"the saved model still loads first ({log('load')})")
            page.locator("#prompt").fill("Hello")
            page.locator("#send").click()
            page.wait_for_function("() => document.querySelector('#app').dataset.phase === 'ready' && document.querySelectorAll('.msg.assistant').length === 1", timeout=15000)
            fetched = wait_log("fetch", 45)
            check([e["id"] for e in fetched] == ["qwen3.5-2b"], f"when idle, the device's pick downloads in the background ({fetched})")
            page.wait_for_function("() => /answers the requests it suits/.test(document.querySelector('#status-line').textContent)", timeout=5000)
            ask("Hello again", 2)
            check([e["id"] for e in log("load")] == ["qwen3.5-4b", "qwen3.5-2b"], f"the next request runs on it ({log('load')})")
            check("Switched to Qwen3.5 2B" in page.locator("#status-line").inner_text(), f"the status line says so ({page.locator('#status-line').inner_text()!r})")

            # 3b. The last model is gone, another is saved: that one starts, the pick downloads,
            # and with no conversation under way it takes over at once.
            state["saved"] = saved("qwen3.5-4b")
            visit()
            page.wait_for_selector("#ready:not([hidden])", timeout=15000)
            wait_log("fetch", 45)
            page.wait_for_function("() => /Switched to Qwen3\\.5 2B/.test(document.querySelector('#status-line').textContent)", timeout=10000)
            check([e["id"] for e in log("load")] == ["qwen3.5-4b", "qwen3.5-2b"], f"an empty conversation switches as soon as the pick is saved ({log('load')})")

            # 4. The 4B unused for 100 days: removed when idle; the running 2B is kept.
            state["saved"] = saved("qwen3.5-4b", "qwen3.5-2b")
            page.evaluate("""() => { const k = 'unsloth__Qwen3.5-4B-GGUF__main__Qwen3.5-4B-Q4_K_M.gguf'; const u = JSON.parse(localStorage.getItem('evolve-llm-used') || '{}'); u[k] = Date.now() - 100 * 864e5; localStorage.setItem('evolve-llm-used', JSON.stringify(u)); }""")
            visit()
            page.wait_for_selector("#ready:not([hidden])", timeout=15000)
            check([e["id"] for e in log("load")] == ["qwen3.5-2b"], f"the pick, now saved, loads by itself ({log('load')})")
            removed = wait_log("remove", 45)
            check([e["key"] for e in removed] == [KEY["qwen3.5-4b"][0]], f"the model unused for 100 days is removed ({removed})")
            check("Removed Qwen3.5 4B" in page.locator("#status-line").inner_text(), f"the status line says what was removed ({page.locator('#status-line').inner_text()!r})")

            # 5. Each request on its rung, a 6 GB GPU with both 4B models saved.
            page.evaluate("() => { for (const k of Object.keys(localStorage)) if (k.startsWith('evolve-llm-gpu-')) localStorage.removeItem(k); }")
            state["gpuBytes"] = 6 * GB
            state["saved"] = saved("qwen3-4b-2507", "qwen3.5-4b")
            visit()
            page.wait_for_selector("#ready:not([hidden])", timeout=15000)
            check([e["id"] for e in log("load")] == ["qwen3.5-4b"], f"the model kept ready for real work loads first ({log('load')})")
            # The model list is for developers, in Settings, and is not a picker.
            page.locator("#open-settings").click()
            page.locator("#dev-models > summary").click()
            check(page.locator("#model-list button.model").count() == 0, "the model list is not a picker")
            uses = page.locator("#model-list .model .tag.rec").all_inner_texts()
            check(uses == ["Everyday requests", "Real work", "Hard requests", "The hardest requests, thinking", "Pictures and screenshots"], f"it says what each model is for ({uses})")
            page.keyboard.press("Escape")
            ask("Hello there", 1)
            check([e["id"] for e in log("load")][-1] == "qwen3-4b-2507", f"an everyday request runs on the fastest rung ({log('load')})")
            check("for an everyday request" in page.locator("#status-line").inner_text(), f"and says why ({page.locator('#status-line').inner_text()!r})")
            ask("Fix the bug in calc.py so the test passes", 2)
            check([e["id"] for e in log("load")][-1] == "qwen3.5-4b", f"a coding request climbs to Qwen3.5 4B ({log('load')})")
            ask("That did not work", 3)
            fetched = wait_log("fetch", 45)
            check([e["id"] for e in fetched] == ["qwen3-coder-30b-a3b"], f"a complaint asks for the next rung, which downloads in the background ({fetched})")
            check([e["id"] for e in log("load")][-1] == "qwen3.5-4b", "until it is saved, the best saved rung below it answers")
            ask("Look at calc.py again", 4)
            check([e["id"] for e in log("load")][-1] == "qwen3-coder-30b-a3b", f"once saved, it answers the conversation's next request ({log('load')})")
            page.locator("#new-chat").click()
            page.wait_for_function("() => document.querySelector('#app').dataset.phase === 'ready'", timeout=10000)
            ask("Hello", 1)
            check([e["id"] for e in log("load")][-1] == "qwen3-4b-2507", f"a new conversation starts again from the request's own level ({log('load')})")

            # 6. Off: nothing loads by itself.
            page.evaluate("() => { const s = JSON.parse(localStorage.getItem('evolve-llm-settings') || '{}'); s.manageModels = false; localStorage.setItem('evolve-llm-settings', JSON.stringify(s)); }")
            visit()
            time.sleep(2)
            check(len(log("load")) == 0 and len(log("probe")) == 0, f"with managing off, nothing loads or is measured ({log('load')}, {log('probe')})")
            check(not errors, f"no page errors ({errors})")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("self-managed models browser test passed")


if __name__ == "__main__":
    main()
