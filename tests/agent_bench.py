"""How good a real local model is as a coding agent in the app: fixed tasks, scored.

Each task gets a fresh browser profile (an empty Linux home) and a fresh fake GitHub repository
(tests/vm_github_browser_test.py), loads the model, sends one chat message as a person would,
answers the approval questions as a careful person would (never a push that removes many lines;
a merge only when the task asks for one), and checks the result on the fake GitHub or in the
page. Prints one row per task (success, time, model turns, tool calls, failed calls, format
repairs) from the app's own trace, and writes them as JSON.

    python3 tools/build_web.py
    python3 tests/agent_bench.py --runtime dist MODEL.gguf [--tasks docs-line,bugfix] [--gpu pooled]
"""
import argparse
import http.server
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vm_github_browser_test as gh  # noqa: E402
from browser_test import FLAGS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

TARGET = "python3 -m http.server 8080 --directory dist"
LONG_README = "\n".join([f"Line {i} of a long README that a small model should not read in full." for i in range(1, 200)] + [TARGET] + [f"Line {i} after the command." for i in range(201, 300)]) + "\n"
NEW_LINE = "python3 tools/serve_local.py   # serves the app with a live trace"
CALC = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"


def tree(fake, ref):
    c = fake.refs.get(ref)
    return {p: fake.blobs[s] for p, s in fake.trees[fake.commits[c]["tree"]].items()} if c else None


def pr_from(fake, branch):
    return next((p for p in fake.pulls if p["head"] == branch), None)


def unchanged_except(fake, branch, path):
    """Every file but `path` the same on `branch` as on main."""
    a, b = tree(fake, "main"), tree(fake, branch)
    return {k: v for k, v in a.items() if k != path} == {k: v for k, v in b.items() if k != path}


def check_docs_line(fake, page, run):
    t = tree(fake, "docs-serve")
    if not t:
        return False, "no docs-serve branch on GitHub"
    want = LONG_README.replace(TARGET + "\n", TARGET + "\n" + NEW_LINE + "\n")
    got = t.get("README.md", b"").decode()
    if got != want:
        return False, f"README.md on docs-serve differs ({len(got.splitlines())} lines, want {len(want.splitlines())})"
    if not unchanged_except(fake, "docs-serve", "README.md"):
        return False, "other files changed"
    return (True, "one line inserted, pull request open") if pr_from(fake, "docs-serve") else (False, "no pull request")


def check_bugfix(fake, page, run):
    t = tree(fake, "fix-add")
    if not t:
        return False, "no fix-add branch on GitHub"
    got = t.get("calc.py", b"").decode()
    if got != CALC.replace("return a - b", "return a + b", 1):
        return False, f"calc.py on fix-add is {got!r}"
    return (True, "add() fixed, nothing else touched, pull request open") if pr_from(fake, "fix-add") and unchanged_except(fake, "fix-add", "calc.py") else (False, "no pull request or other files changed")


def check_new_file(fake, page, run):
    t = tree(fake, "contributing")
    if not t:
        return False, "no contributing branch on GitHub"
    text = t.get("CONTRIBUTING.md", b"").decode()
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3 or "pull request" not in text.lower():
        return False, f"CONTRIBUTING.md has {len(lines)} lines{'' if 'pull request' in text.lower() else ', no mention of pull requests'}"
    return (True, f"CONTRIBUTING.md, {len(lines)} lines, pull request open") if pr_from(fake, "contributing") and unchanged_except(fake, "contributing", "CONTRIBUTING.md") else (False, "no pull request or other files changed")


def check_question(fake, page, run):
    reply = run["reply"].lower()
    if run["calls"]:
        return False, f"used {run['calls']} tools for a question"
    return ("branch" in reply and len(reply) > 20), f"reply: {run['reply'][:80]!r}"


def setup_merge():
    fake = gh.Fake({"README.md": b"# Demo\n", "calc.py": CALC.encode()})
    t = dict(fake.trees[fake.commits[gh.C0]["tree"]])
    t["README.md"] = fake.put(b"# Demo\n\nA small calculator.\n")
    fake.commits["f" * 40] = {"tree": fake.tree_id(t), "parents": [gh.C0], "message": "Describe the project"}
    fake.refs["feature"] = "f" * 40
    fake.pulls.append({"number": 1, "title": "Describe the project", "head": "feature", "base": "main", "body": ""})
    return fake


def check_merge(fake, page, run):
    if not fake.pulls[0].get("merged"):
        return False, "pull request #1 not merged"
    return (tree(fake, "main")["README.md"] == b"# Demo\n\nA small calculator.\n"), "merged; main has the change"


def check_html(fake, page, run):
    html = page.evaluate("async () => { for (const p of ['/home/user/counter.html']) { const r = await (await import('./vm/client.mjs')).sharedVm().read(p); if (r.ok) return r.data; } return ''; }")
    if "<button" not in html.lower():
        return False, "no counter.html with a button in the home folder"
    shown = page.locator(".msg.assistant").last.locator(".chat-preview-host iframe").count()
    return (shown > 0), ("counter.html made and shown" if shown else "counter.html made but not shown")


TASKS = {
    "docs-line": (lambda: gh.Fake({"README.md": LONG_README.encode(), "main.py": b"print('hi')\n"}),
                  f'Clone the GitHub repository demo/app. In README.md, right after the line "{TARGET}", add the line "{NEW_LINE}". Commit it on a new branch named docs-serve, push it and open a pull request.',
                  check_docs_line),
    "bugfix": (lambda: gh.Fake({"README.md": b"# Calculator\n", "calc.py": CALC.encode()}),
               "In the GitHub repository demo/app, calc.py has a bug: add() subtracts instead of adding. Fix it, commit the fix on a new branch named fix-add, push it and open a pull request.",
               check_bugfix),
    "new-file": (lambda: gh.Fake({"README.md": b"# Demo\n"}),
                 "In the GitHub repository demo/app, add a file CONTRIBUTING.md that explains in a few lines how to open a pull request. Commit it on a new branch named contributing, push it and open a pull request.",
                 check_new_file),
    "question": (lambda: gh.Fake(), "What does git checkout -b NAME do? Answer in one sentence.", check_question),
    "merge": (setup_merge, "Merge pull request #1 in the GitHub repository demo/app.", check_merge),
    "html-app": (lambda: gh.Fake(), "Make a small web page counter.html with a button that counts how many times it is clicked. Test it and show it to me.", check_html),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("model", type=Path)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--gpu", choices=["pooled", "high", "low"], default="pooled")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--minutes", type=float, default=20, help="per task")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    trace = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    gh.Handler.trace_file = str(trace)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: gh.Handler(*x, directory=str(a.runtime), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    settings = {"temperature": 0, "maxTokens": 2048, "ctx": a.ctx, "thinking": False, "gpuMemory": "auto", "autonomy": 1, "githubToken": "t",
                "gpu": "low-power" if a.gpu == "low" else "high-performance", "poolGpus": a.gpu == "pooled"}
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=FLAGS)
        for name in a.tasks.split(","):
            setup, prompt, check = TASKS[name]
            gh.FAKE = setup()
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
            ctx.add_init_script(f"if(window===window.top)localStorage.setItem('evolve-llm-settings',JSON.stringify({json.dumps(settings)}))")
            page = ctx.new_page()
            errors, questions = [], []

            def answer(d, name=name):
                q = d.message.split("\n\n", 1)[-1]
                allow = (not q.startswith("Merge") or name == "merge") and not any(int(x) > 20 for x in re.findall(r" -(\d+)", q))
                questions.append((q, allow))
                d.accept() if allow else d.dismiss()
            page.on("dialog", answer)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(origin + "/play/evolve-llm/")
            page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
            page.set_input_files("#file-input", str(a.model))
            page.wait_for_selector("#ready:not([hidden])", timeout=900000)
            page.evaluate("async (o) => (await import('./vm/client.mjs')).sharedVm().configure({ github: { api: o + '/mockgh/api', raw: o + '/mockgh/raw', token: 't' } })", origin)
            seen = len(trace.read_text().splitlines())
            t0 = time.time()
            page.locator("#prompt").fill(prompt)
            page.locator("#send").click()
            page.wait_for_selector("#app[data-phase=agent]", timeout=30000)
            timed_out = False
            try:
                page.wait_for_selector("#app[data-phase=ready]", timeout=int(a.minutes * 60000))
            except Exception:
                timed_out = True
            time.sleep(1)
            events = [json.loads(l) for l in trace.read_text().splitlines()[seen:]]
            run = {
                "task": name, "seconds": round(time.time() - t0), "timed_out": timed_out,
                "turns": sum(e["type"] == "turn" for e in events), "calls": sum(e["type"] == "tool" for e in events),
                "failed": sum(e["type"] == "result" and e.get("error") for e in events), "repairs": sum(e["type"] == "retry" for e in events),
                "reply": page.locator(".msg.assistant .bubble").last.inner_text() if page.locator(".msg.assistant").count() else "",
                "questions": questions, "page_errors": errors,
            }
            try:
                ok, detail = (False, "timed out") if timed_out else check(gh.FAKE, page, run)
            except Exception as e:  # a check that cannot read the result counts as a failure
                ok, detail = False, f"check failed: {e}"
            run.update(ok=bool(ok), detail=detail)
            rows.append(run)
            print(f"{'PASS' if ok else 'FAIL'} {name:10} {run['seconds']:5}s  turns {run['turns']:2}  calls {run['calls']:2}  failed {run['failed']:2}  repairs {run['repairs']}  {detail}", flush=True)
            ctx.close()
        browser.close()
    passed = sum(r["ok"] for r in rows)
    print(f"\n{a.model.name} ({a.gpu}): {passed}/{len(rows)} tasks, {sum(r['seconds'] for r in rows)}s in all")
    if a.out:
        a.out.write_text(json.dumps({"model": a.model.name, "gpu": a.gpu, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
