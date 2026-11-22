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


TEST_CALC = "import unittest\nfrom calc import add, sub\n\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_sub(self):\n        self.assertEqual(sub(5, 3), 2)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n"


def setup_ci():
    fake = gh.Fake({"README.md": b"# Calculator\n", "calc.py": CALC.encode(), "test_calc.py": TEST_CALC.encode()})
    fake.runs = [{"id": 41, "status": "completed", "conclusion": "failure", "name": "CI", "head_branch": "main", "jobs": [
        {"id": 4101, "name": "test", "conclusion": "failure", "steps": [{"name": "Run python3 test_calc.py", "conclusion": "failure"}],
         "annotations": [{"path": "test_calc.py", "start_line": 7, "message": "AssertionError: -1 != 5 (add(2, 3) returned -1)"}]}]}]
    return fake


def check_ci(fake, page, run):
    t = tree(fake, "fix-ci")
    if not t:
        return False, "no fix-ci branch on GitHub"
    if t.get("calc.py", b"").decode() != CALC.replace("return a - b", "return a + b", 1):
        return False, f"calc.py on fix-ci is {t.get('calc.py', b'').decode()!r}"
    tested = " (ran python3)" if run.get("ran_python") else " (did not run python3)"
    return (True, "add() fixed, pull request open" + tested) if pr_from(fake, "fix-ci") and unchanged_except(fake, "fix-ci", "calc.py") else (False, "no pull request or other files changed")


def check_html(fake, page, run):
    html = page.evaluate("async () => { for (const p of ['/home/user/counter.html']) { const r = await (await import('./vm/client.mjs')).sharedVm().read(p); if (r.ok) return r.data; } return ''; }")
    if "<button" not in html.lower():
        return False, "no counter.html with a button in the home folder"
    shown = page.locator(".msg.assistant").last.locator(".chat-preview-host iframe").count()
    return (shown > 0), ("counter.html made and shown" if shown else "counter.html made but not shown")


# ---- research: a fake web, so answers come from reading it, not from memory ----
# Made-up facts, served for the three services the web tools use (Wikipedia, DuckDuckGo, and the
# r.jina.ai page reader), with pages that are close but wrong beside the right one.
WEB = {
    "https://en.wikipedia.org/wiki/Marrowfield_Lighthouse": ("Marrowfield Lighthouse",
        "The Marrowfield Lighthouse is a stone lighthouse on the northern headland of Marrowfield Bay. It was designed by the "
        "engineer Elsbeth Varga and first lit on 14 March 1887. Its light was automated in 1962, and the keepers' cottages are now a museum."),
    "https://en.wikipedia.org/wiki/Marrowfield_Bay": ("Marrowfield Bay",
        "Marrowfield Bay is a sheltered bay on the north coast. The bay's first harbour light was a wooden beacon built in 1841 by the "
        "harbour board; it was replaced decades later by the stone lighthouse on the headland."),
    "https://www.sunvale.gov/code/title-12/12-4": ("Sunvale City Code § 12-4 Security deposits",
        "Sunvale City Code, Title 12 (Housing), § 12-4 Security deposits. (a) A landlord shall return a tenant's security deposit, less "
        "any lawful deductions itemized in writing, within 21 days after the tenant vacates the premises. (b) A landlord who does not "
        "comply with subsection (a) is liable to the tenant for twice the amount of the deposit wrongfully withheld."),
    "https://www.sunvale.gov/code/title-12/12-3": ("Sunvale City Code § 12-3 Rent increases",
        "Sunvale City Code, Title 12 (Housing), § 12-3 Rent increases. A landlord shall give a tenant at least 30 days' written notice "
        "before any increase in rent takes effect."),
    "https://www.renters-help.example.org/deposits": ("How long does a landlord have to return a deposit?",
        "Many cities give landlords 30 days to return a security deposit, but the rule depends on where you live: check your own "
        "city's code for the exact number of days."),
}


def web_matches(query, wiki):
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
    scored = []
    for url, (title, text) in WEB.items():
        if ("wikipedia.org" in url) != wiki:
            continue
        hay = (title + " " + text).lower()
        score = sum(w in hay for w in words)
        if score >= max(1, len(words) // 2):
            scored.append((-score, url, title, text))
    return [x[1:] for x in sorted(scored)]


def fake_web(route):
    from urllib.parse import urlparse, parse_qs, unquote
    url = route.request.url
    u = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    if u.hostname == "api.duckduckgo.com":
        hits = web_matches(q.get("q", ""), wiki=False)
        body = {"Heading": "", "AbstractText": "", "AbstractURL": "", "RelatedTopics": [{"FirstURL": h[0], "Text": f"{h[1]} - {h[2][:200]}"} for h in hits]}
    elif u.hostname.endswith("wikipedia.org") and q.get("list") == "search":
        body = {"query": {"search": [{"title": h[1], "snippet": h[2][:200]} for h in web_matches(q.get("srsearch", ""), wiki=True)]}}
    elif u.hostname.endswith("wikipedia.org") and q.get("titles"):
        hit = WEB.get("https://en.wikipedia.org/wiki/" + q["titles"].replace(" ", "_"))
        body = {"query": {"pages": {"1": {"title": hit[0], "extract": hit[1]}} if hit else {"-1": {"missing": ""}}}}
    elif u.hostname == "r.jina.ai":
        page = WEB.get(unquote(url.split("r.jina.ai/", 1)[1]))
        if not page:
            return route.fulfill(status=404, body="Not found")
        return route.fulfill(status=200, content_type="text/plain", body=f"Title: {page[0]}\n\nURL Source: {url.split('r.jina.ai/', 1)[1]}\n\nMarkdown Content:\n{page[1]}\n")
    else:
        return route.fulfill(status=404, body="Not found")
    route.fulfill(status=200, content_type="application/json", headers={"Access-Control-Allow-Origin": "*"}, body=json.dumps(body))


def used_web(run):
    return run["calls"] > 0 and run.get("web_calls", 0) > 0


def check_research(fake, page, run):
    reply = run["reply"]
    if not used_web(run):
        return False, f"answered without searching: {reply[:80]!r}"
    found = {"1887": "1887" in reply, "Varga": "Varga" in reply, "link": "wikipedia.org/wiki/Marrowfield_Lighthouse" in reply}
    missing = [k for k, ok in found.items() if not ok]
    return (not missing), ("year, designer and link" if not missing else f"missing {', '.join(missing)}: {reply[:120]!r}")


def check_law(fake, page, run):
    reply = run["reply"]
    if not used_web(run):
        return False, f"answered without searching: {reply[:80]!r}"
    found = {"21 days": re.search(r"\b21\b", reply) is not None, "§ 12-4": "12-4" in reply, "link": "sunvale.gov/code/title-12/12-4" in reply}
    missing = [k for k, ok in found.items() if not ok]
    return (not missing), ("21 days, § 12-4 and link" if not missing else f"missing {', '.join(missing)}: {reply[:120]!r}")


# ---- browser: a small shop on this computer, reached through the in-app browser's relay ----
SHOP_ITEMS = [("red-kettle", "Red Kettle", "19.00"), ("blue-kettle", "Blue Kettle", "24.50"), ("green-teapot", "Green Teapot", "31.00")]


class Shop(http.server.BaseHTTPRequestHandler):
    added = []  # item ids added to the cart, in order

    def page(self, body, title="Kettle Shop"):
        html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body><h1>Kettle Shop</h1>{body}</body></html>"
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        rows = "".join(f"<li><h2>{name}</h2><p>Price: ${price}</p><form method='post' action='/cart'><input type='hidden' name='item' value='{key}'>"
                       f"<button type='submit'>Add {name} to cart</button></form></li>" for key, name, price in SHOP_ITEMS)
        self.page(f"<p>Cart: {len(Shop.added)} item(s)</p><ul>{rows}</ul>")

    def do_POST(self):
        from urllib.parse import parse_qs
        n = int(self.headers.get("Content-Length") or 0)
        item = parse_qs(self.rfile.read(n).decode()).get("item", [""])[0]
        Shop.added.append(item)
        name = next((nm for key, nm, _ in SHOP_ITEMS if key == item), item)
        self.page(f"<p>Added {name} to your cart. Cart: {len(Shop.added)} item(s)</p><p><a href='/'>Keep shopping</a></p>", title="Cart")

    def log_message(self, *args):
        pass


def check_browser(fake, page, run):
    if Shop.added != ["blue-kettle"]:
        return False, f"cart has {Shop.added or 'nothing'}"
    if "24.50" not in run["reply"] and "24.5" not in run["reply"]:
        return False, f"Blue Kettle in the cart, but the reply has no price: {run['reply'][:100]!r}"
    return True, "Blue Kettle added and its price reported"


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
    "ci-fix": (setup_ci, "CI is failing for the GitHub repository demo/app. Find out why, fix it on a new branch named fix-ci, test the fix, push it and open a pull request.", check_ci),
    "research": (lambda: gh.Fake(), "Look up when the Marrowfield Lighthouse was first lit and who designed it. Answer briefly and include the link you used.", check_research),
    "law": (lambda: gh.Fake(), "Under the Sunvale City Code, how many days does a landlord have to return a tenant's security deposit after the tenant moves out? Cite the code section and link to it.", check_law),
    "browser": (lambda: gh.Fake(), "Go to {shop} in the browser. Find the price of the Blue Kettle, add it to the cart, and tell me the price.", check_browser),
    "html-app": (lambda: gh.Fake(), "Make a small web page counter.html with a button that counts how many times it is clicked. Test it and show it to me.", check_html),
}

# The same tasks said another way (--wording 2), so a harness change is judged on more than one
# phrasing: same repository, branch names and checks.
REWORDED = {
    "docs-line": f'In demo/app on GitHub, README.md has the line "{TARGET}". Put "{NEW_LINE}" on the line below it. Use a branch called docs-serve and open a PR for it.',
    "bugfix": "calc.py in demo/app (GitHub) is broken: add(2, 3) gives -1. Please fix it on a branch named fix-add and send me a pull request.",
    "new-file": "demo/app on GitHub needs a CONTRIBUTING.md with a few lines on how to open a pull request. Add it on a branch called contributing and open a PR.",
    "question": 'Explain in one sentence what "git checkout -b NAME" does.',
    "merge": "Please merge PR #1 on demo/app (it is on GitHub).",
    "ci-fix": "The checks on demo/app's main branch are red. Figure out what broke, fix it on a branch fix-ci, run the tests, then push and open a pull request.",
    "html-app": "Build me counter.html: a page with a button and a number that goes up by one on each click. Check that it works and show it.",
    "research": "Who designed the Marrowfield Lighthouse, and in what year was it first lit? Give me the source too.",
    "browser": "Open the shop at {shop} and put a Blue Kettle in my cart. How much is it?",
    "law": "My landlord in Sunvale still has my deposit. What's the deadline in the city code for returning it? Please cite the section with a link.",
}

# What kind of work each task is, for scores per kind (and choosing a model per kind).
KIND = {"docs-line": "coding", "bugfix": "coding", "new-file": "coding", "merge": "coding", "ci-fix": "coding", "html-app": "coding",
        "question": "chat", "research": "research", "law": "law", "browser": "browser"}
# Settings a task needs beyond the defaults, and whether it reads the fake web.
NEEDS = {"research": {"settings": {"internet": True}, "web": True}, "law": {"settings": {"internet": True}, "web": True}, "browser": {"browser": True}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("model", type=Path)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--gpu", choices=["pooled", "high", "low"], default="pooled")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--minutes", type=float, default=20, help="per task")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wording", type=int, choices=[1, 2], default=1, help="2: the tasks said another way")
    a = ap.parse_args()
    trace = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    gh.Handler.trace_file = str(trace)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: gh.Handler(*x, directory=str(a.runtime), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    settings = {"temperature": 0, "maxTokens": 2048, "ctx": a.ctx, "thinking": False, "gpuMemory": "auto", "autonomy": 1, "githubToken": "t",
                "gpu": "low-power" if a.gpu == "low" else "high-performance", "poolGpus": a.gpu == "pooled"}
    rows = []
    # The browser task: the shop, and a relay that may reach it (skipped when the relay or the
    # browser engine is not built).
    shop = relay = relay_url = None
    if any(NEEDS.get(t, {}).get("browser") for t in a.tasks.split(",")):
        from browser_view_browser_test import start_relay
        shop = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Shop)
        threading.Thread(target=shop.serve_forever, daemon=True).start()
        if (a.runtime / "browser-pkg/browser.js").exists():
            relay, relay_url = start_relay(origin, shop.server_address[1])
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=FLAGS)
        for name in a.tasks.split(","):
            setup, prompt, check = TASKS[name]
            prompt = REWORDED[name] if a.wording == 2 else prompt
            need = NEEDS.get(name, {})
            if need.get("browser"):
                if not relay_url:
                    print(f"SKIP {name:10} build the browser engine (node tools/build_workbench.mjs) and the relay (cargo build --release --manifest-path browser/relay/Cargo.toml)", flush=True)
                    continue
                Shop.added = []
                prompt = prompt.replace("{shop}", f"http://127.0.0.1:{shop.server_address[1]}/")
            gh.FAKE = setup()
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
            if need.get("browser"):
                ctx.route("**/browser-config.json", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps({"relayUrl": relay_url})))
            ctx.add_init_script(f"if(window===window.top)localStorage.setItem('evolve-llm-settings',JSON.stringify({json.dumps({**settings, **need.get('settings', {})})}))")
            if need.get("web"):
                ctx.route(re.compile(r"^https://(api\.duckduckgo\.com|[a-z-]+\.wikipedia\.org|r\.jina\.ai|s\.jina\.ai)/"), fake_web)
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
                "ran_python": any(e["type"] == "tool" and "python3" in e.get("args", "") for e in events),
                "web_calls": sum(e["type"] == "tool" and e.get("name") in ("web_search", "open_url") for e in events),
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
    if relay:
        relay.kill()
    passed = sum(r["ok"] for r in rows)
    kinds = {}
    for r in rows:
        k = kinds.setdefault(KIND.get(r["task"], "other"), [0, 0])
        k[0] += r["ok"]; k[1] += 1
    print(f"\n{a.model.name} ({a.gpu}): {passed}/{len(rows)} tasks, {sum(r['seconds'] for r in rows)}s in all")
    print("by kind: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(kinds.items())))
    if a.out:
        a.out.write_text(json.dumps({"model": a.model.name, "gpu": a.gpu, "wording": a.wording, "kinds": kinds, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
