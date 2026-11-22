"""A real model in the main chat, working on a repository of the fake GitHub.

Loads a GGUF in the app, points the VM's git and gh at the fake GitHub
(tests/vm_github_browser_test.py) and sends one request through the chat box: clone, edit, commit
on a branch, push, open a pull request and merge it. The approval dialogs (push, pull request,
merge) are accepted as a person would. Prints the reply, the steps and the questions, then checks
what the fake GitHub holds. Small models often fail multi-step tasks; this fails only on page
errors unless --require is given.

    python3 tools/build_web.py
    python3 tests/chat_github_real_model_test.py --runtime dist model.gguf [--require] [--thinking]
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from trace_view import line as trace_line  # noqa: E402
from browser_test import FLAGS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

TASK = ('Clone the GitHub repository demo/app. Change the first line of README.md to "# Demo app". '
        'Commit that on a new branch named readme-title, push it, open a pull request, and merge it.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("model", type=Path)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--gpu", default="low-power")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--require", action="store_true", help="fail unless the model completes the task")
    ap.add_argument("--trace", type=Path, default=Path("/tmp/evolve-llm-chat-trace.jsonl"), help="the page's live trace (tools/trace_view.py reads it)")
    a = ap.parse_args()
    a.trace.write_text("")
    gh.Handler.trace_file = str(a.trace)
    print(f"trace: {a.trace}", flush=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: gh.Handler(*x, directory=str(a.runtime), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    errors, questions = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=FLAGS)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        settings = {"temperature": 0, "maxTokens": 2048, "ctx": a.ctx, "thinking": a.thinking, "gpu": a.gpu, "gpuMemory": "auto", "autonomy": 1}
        context.add_init_script(f"if(window===window.top)localStorage.setItem('evolve-llm-settings',JSON.stringify({json.dumps(settings)}))")
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("dialog", lambda d: (questions.append(d.message), d.accept()))
        page.goto(origin + "/play/evolve-llm/")
        page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
        t0 = time.time()
        page.set_input_files("#file-input", str(a.model))
        page.wait_for_selector("#ready:not([hidden])", timeout=600000)
        print(f"{a.model.name} loaded in {time.time() - t0:.0f}s", flush=True)
        vm = "(await import('./vm/client.mjs')).sharedVm()"
        page.evaluate(f"async (o) => {{ ({vm}).configure({{ github: {{ api: o + '/mockgh/api', raw: o + '/mockgh/raw', token: 't' }} }}); }}", origin)
        t0 = time.time()
        page.locator("#prompt").fill(a.task)
        page.locator("#send").click()
        page.wait_for_selector("#app[data-phase=agent]", timeout=30000)
        # The page's trace as it arrives: every model turn, tool call, result and question.
        with a.trace.open(encoding="utf-8") as live:
            def drain():
                for raw in live.readlines():
                    try:
                        print(trace_line(json.loads(raw), False), flush=True)
                    except ValueError:
                        pass
            shown, last_peek = "", 0.0
            while page.get_attribute("#app", "data-phase") != "ready":
                if time.time() - t0 > a.minutes * 60:
                    drain()
                    sys.exit(f"no reply after {a.minutes} minutes")
                drain()
                # The reply being written, every 15 s, so a slow turn shows its progress too.
                if time.time() - last_peek > 15:
                    last_peek = time.time()
                    reply = page.evaluate("document.querySelector('.msg.assistant:last-child .bubble')?.innerText || ''")
                    if reply != shown:
                        shown = reply
                        print(f"  … {time.time() - t0:4.0f}s, {len(reply)} characters so far: {reply[-160:]!r}", flush=True)
                time.sleep(2)
            time.sleep(1)
            drain()
        elapsed = time.time() - t0
        message = page.locator(".msg.assistant").last
        steps = message.locator(".work-steps")
        if steps.is_visible():
            steps.locator("summary").first.click()
            for item in message.locator(".work-steps li > details").all():
                item.locator("summary").click()
        transcript = message.inner_text()
        page.screenshot(path="/tmp/llm-chat-github-real-model.png", full_page=False)
        browser.close()
    print("---- the reply and its steps ----\n" + transcript + "\n---------------------------------")
    print("questions asked:", json.dumps(questions, indent=1))
    head = gh.FAKE.refs.get("readme-title")
    tree = gh.FAKE.trees[gh.FAKE.commits[head]["tree"]] if head else {}
    readme = gh.FAKE.blobs[tree["README.md"]].decode() if head else None
    main_tree = gh.FAKE.trees[gh.FAKE.commits[gh.FAKE.refs["main"]]["tree"]]
    main_readme = gh.FAKE.blobs[main_tree["README.md"]].decode()
    pulls = [{"number": x["number"], "title": x["title"], "head": x["head"], "merged": x.get("merged", False)} for x in gh.FAKE.pulls]
    done = bool(head) and readme.splitlines()[:1] == ["# Demo app"] and main_readme.splitlines()[:1] == ["# Demo app"] and any(x["merged"] for x in pulls)
    print(f"{a.model.name}: {elapsed:.0f}s; branches: {sorted(gh.FAKE.refs)}; README on readme-title: {readme!r}; "
          f"README on main: {main_readme!r}; pull requests: {pulls}; task complete: {done}")
    if errors:
        sys.exit(f"page errors: {errors}")
    if a.require and not done:
        sys.exit("the model did not complete the task")


if __name__ == "__main__":
    main()
