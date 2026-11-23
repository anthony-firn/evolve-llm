"""Pi with a real model in the app's Linux terminal, on a repository of the fake GitHub.

Loads a GGUF in the app, opens the Terminal (the Linux terminal), clones demo/app from the fake
GitHub (tests/vm_github_browser_test.py) and runs `pi` there with a small real task: retitle the
README, commit on a new branch and push. Reports the transcript and whether the fake GitHub then
holds the change. Small models often fail multi-step tool tasks; this fails only on page errors
unless --require is given.

    python3 tools/build_web.py
    python3 tests/vm_pi_real_model_browser_test.py --runtime dist model.gguf [--require]
"""
import argparse
import http.server
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vm_github_browser_test as gh  # noqa: E402
from browser_test import FLAGS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

TASK = ('Change the first line of README.md to "# Demo app". Then create a branch named readme-title, '
        'commit with the message "Retitle the README", and push it.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("model", type=Path)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--require", action="store_true", help="fail unless the model completes the task")
    a = ap.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: gh.Handler(*x, directory=str(a.runtime), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=FLAGS)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        settings = {"temperature": 0, "maxTokens": 1024, "ctx": a.ctx, "thinking": False, "gpu": "high-performance", "gpuMemory": "auto", "autonomy": 2}
        context.add_init_script(f"if(window===window.top)localStorage.setItem('evolve-llm-settings',JSON.stringify({json.dumps(settings)}))")
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(origin + "/play/evolve-llm/")
        page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
        t0 = time.time()
        page.set_input_files("#file-input", str(a.model))
        page.wait_for_selector("#ready:not([hidden])", timeout=600000)
        print(f"{a.model.name} loaded in {time.time() - t0:.0f}s", flush=True)
        page.dispatch_event("#open-terminal", "click")
        page.wait_for_selector(".vm-console .vt-term:not([hidden])", timeout=30000)
        vm = "(await import('./vm/client.mjs')).sharedVm()"
        page.evaluate(f"async (o) => {{ ({vm}).configure({{ github: {{ api: o + '/mockgh/api', raw: o + '/mockgh/raw', token: 't' }} }}); }}", origin)
        r = page.evaluate(f"async () => ({vm}).exec('gh repo clone demo/app', {{ timeoutMs: 60000 }})")
        assert r["code"] == 0, r
        command = "cd app && pi '" + TASK.replace("'", "") + "' > /tmp/pi.log 2>&1; echo $? > /tmp/pi.done"
        t0 = time.time()
        page.evaluate(f"async (c) => {{ ({vm}).exec(c, {{ timeoutMs: {int(a.minutes * 60000)} }}); }}", command)
        read = lambda path: page.evaluate(f"async (p) => {{ const r = await ({vm}).read(p); return r.ok ? r.data : null; }}", path)
        while read("/tmp/pi.done") is None and time.time() - t0 < a.minutes * 60 + 30:
            time.sleep(5)
        elapsed = time.time() - t0
        log = read("/tmp/pi.log") or ""
        status = (read("/tmp/pi.done") or "?").strip()
        browser.close()
    print("---- transcript ----\n" + log.replace("\x1b[2m", "").replace("\x1b[0m", "").replace("\x1b[1;36m", "").replace("\x1b[31m", "").replace("\x1b[33m", "") + "\n--------------------")
    head = gh.FAKE.refs.get("readme-title")
    tree = gh.FAKE.trees[gh.FAKE.commits[head]["tree"]] if head else {}
    readme = gh.FAKE.blobs[tree["README.md"]].decode() if head else None
    branches = sorted(b for b in gh.FAKE.refs if b != "main")
    done = bool(head) and readme.splitlines()[:1] == ["# Demo app"] and gh.FAKE.refs["main"] == gh.C0
    print(f"{a.model.name}: pi exit {status} after {elapsed:.0f}s; branches pushed: {branches}; "
          f"README on readme-title: {readme!r}; task complete: {done}")
    if errors:
        sys.exit(f"page errors: {errors}")
    if a.require and not done:
        sys.exit("the model did not complete the task")


if __name__ == "__main__":
    main()
