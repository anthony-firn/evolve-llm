"""`gh` and `git` in the Evolve VM against a fake GitHub served by the test server.

The fake implements the API calls the VM uses (repository, branch, commit and tree listings,
raw files, blobs, trees, commits, refs, pull requests, Actions runs) for one repository, so the
test can push without touching GitHub. In Chrome, through the VM worker:

clone (every file checked against its blob hash), status, diff, commit, log, a push to a new
branch (asked and allowed; the fake must then hold exactly the edited tree, parented on the
original commit), a pull request, CI runs, a refused push (autonomy: local only), a declined push,
a hash mismatch refusing a clone, and the token reaching only the API.

    python3 tools/build_vm.py && python3 tests/vm_github_browser_test.py
"""
import base64
import hashlib
import http.server
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import browser_test  # noqa: E402
from browser_test import FLAGS, ROOT  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


C0 = "0f3c1a7e9b2d4c6e8a0b1c2d3e4f5a6b7c8d9e0f"  # the original commit


class Fake:
    """One repository, demo/app, with branches, trees (path -> blob sha) and blobs."""

    def __init__(self):
        self.blobs = {}
        self.trees = {}
        self.commits = {}
        files = {"README.md": b"# Demo\n", "src/main.rs": b"fn main() {}\n", "docs/notes.txt": b"a\nb\n"}
        tree = {p: self.put(d) for p, d in files.items()}
        self.commits[C0] = {"tree": self.tree_id(tree), "parents": [], "message": "Initial commit"}
        self.refs = {"main": C0}
        self.pulls = []
        self.corrupt = None
        self.auth = []

    def put(self, data):
        sha = blob_sha(data)
        self.blobs[sha] = data
        return sha

    def tree_id(self, tree):
        tid = "t" + hashlib.sha1(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:12]
        self.trees[tid] = dict(tree)
        return tid

    def handle(self, method, path, query, body, headers):
        """(status, json-or-bytes) for a request under /mockgh/."""
        self.auth.append((path, headers.get("Authorization")))
        if path.startswith("/raw/demo/app/"):
            commit, _, p = path[len("/raw/demo/app/"):].partition("/")
            sha = self.trees[self.commits[commit]["tree"]].get(p)
            if sha is None:
                return 404, b"not found"
            data = self.blobs[sha]
            return 200, (b"tampered" if self.corrupt == p else data)
        if path == "/api/user":
            return (200, {"login": "tester"}) if headers.get("Authorization") else (401, {"message": "Requires authentication"})
        prefix = "/api/repos/demo/app"
        if not path.startswith(prefix):
            return 404, {"message": "Not Found"}
        rest = path[len(prefix):]
        if rest == "" and method == "GET":
            return 200, {"default_branch": "main", "private": False}
        if rest.startswith("/branches/"):
            b = rest[len("/branches/"):]
            return (200, {"commit": {"sha": self.refs[b]}}) if b in self.refs else (404, {"message": "Branch not found"})
        if rest.startswith("/git/commits/") and method == "GET":
            c = self.commits.get(rest.split("/")[-1])
            return (200, {"tree": {"sha": c["tree"]}}) if c else (404, {"message": "No commit"})
        if rest.startswith("/git/trees/") and method == "GET":
            t = self.trees[rest.split("/")[-1]]
            return 200, {"truncated": False, "tree": [{"path": p, "mode": "100644", "type": "blob", "sha": s, "size": len(self.blobs[s])} for p, s in sorted(t.items())]}
        if rest == "/git/blobs" and method == "POST":
            return 201, {"sha": self.put(base64.b64decode(body["content"]))}
        if rest == "/git/trees" and method == "POST":
            tree = dict(self.trees[body["base_tree"]])
            for e in body["tree"]:
                if e["sha"] is None:
                    tree.pop(e["path"], None)
                else:
                    tree[e["path"]] = e["sha"]
            return 201, {"sha": self.tree_id(tree)}
        if rest == "/git/commits" and method == "POST":
            cid = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
            self.commits[cid] = {"tree": body["tree"], "parents": body["parents"], "message": body["message"]}
            return 201, {"sha": cid}
        if rest.startswith("/git/ref/heads/"):
            b = rest[len("/git/ref/heads/"):]
            return (200, {"object": {"sha": self.refs[b]}}) if b in self.refs else (404, {"message": "Not Found"})
        if rest.startswith("/git/refs/heads/") and method == "PATCH":
            self.refs[rest[len("/git/refs/heads/"):]] = body["sha"]
            return 200, {}
        if rest == "/git/refs" and method == "POST":
            self.refs[body["ref"][len("refs/heads/"):]] = body["sha"]
            return 201, {}
        if rest == "/pulls" and method == "POST":
            n = len(self.pulls) + 1
            self.pulls.append({**body, "number": n})
            return 201, {"number": n, "html_url": f"https://github.com/demo/app/pull/{n}"}
        if rest == "/pulls":
            return 200, [{"number": p["number"], "title": p["title"], "head": {"ref": p["head"]}, "base": {"ref": p["base"]}} for p in self.pulls]
        if rest == "/actions/runs":
            return 200, {"workflow_runs": [{"id": 7, "status": "completed", "conclusion": "failure", "name": "CI", "head_branch": "feature"}]}
        if rest.startswith("/commits"):
            out, c = [], query.get("sha", C0)
            while c and len(out) < 10:
                out.append({"sha": c, "commit": {"message": self.commits[c]["message"], "author": {"name": "Demo", "date": "2026-01-01"}}})
                c = (self.commits[c]["parents"] or [None])[0]
            return 200, out
        return 404, {"message": f"fake GitHub has no {method} {rest}"}


FAKE = Fake()


class Handler(browser_test.Handler):
    def route(self, method):
        if not self.path.startswith("/mockgh/"):
            return False
        u = urlparse(self.path[len("/mockgh"):])
        query = dict(p.split("=", 1) for p in u.query.split("&") if "=" in p)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"null") if n else None
        status, out = FAKE.handle(method, u.path, query, body, dict(self.headers))
        data = out if isinstance(out, bytes) else json.dumps(out).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream" if isinstance(out, bytes) else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        if not self.route("GET"):
            super().do_GET()

    def do_POST(self):
        self.route("POST")

    def do_PATCH(self):
        self.route("PATCH")


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: Handler(*a, directory=str(ROOT / "web"), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    problems = []

    def check(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            problems.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=FLAGS)
        page = browser.new_page()
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(origin + "/play/evolve-llm/vm/harness.html")
        for _ in range(200):
            if page.evaluate("window.vmReady || window.vmError || false"):
                break
            time.sleep(0.05)

        def config(autonomy, token="test-token"):
            page.evaluate("(c) => vmConfig(c)", {"github": {"token": token, "autonomy": autonomy, "api": origin + "/mockgh/api", "raw": origin + "/mockgh/raw"}})

        def sh(cmd):
            r = page.evaluate("(c) => vmExec(c, { timeoutMs: 30000 })", cmd)
            return r["code"], r["stdout"], r["stderr"]

        config(1)
        code, out, err = sh("gh repo clone demo/app && cd app && find . -type f -not -path './.evolve-git/*' | sort")
        check(code == 0 and out.endswith("./README.md\n./docs/notes.txt\n./src/main.rs\n"), f"clone downloads the tree ({out!r} {err!r})")
        code, out, _ = sh("cd app && git status")
        check("nothing to commit, working tree clean" in out, "a fresh clone is clean")
        code, out, _ = sh("cd app && echo more >> README.md && printf 'new\\n' > new.txt && rm docs/notes.txt && git status && git diff README.md")
        check("modified:   README.md" in out and "new file:   new.txt" in out and "deleted:    docs/notes.txt" in out, "status lists the changes")
        check("+more" in out and "--- a/README.md" in out, "diff shows the edit")
        code, out, _ = sh("cd app && git checkout -b feature && git commit -m 'Change things' && git log --oneline -n 2")
        check(code == 0 and "(not pushed) Change things" in out and "Initial commit" in out, f"commit and log ({out!r})")
        code, out, err = sh("cd app && git push")
        asked = page.evaluate("window.vmQuestions || []")
        check(code == 0 and "-> feature (new branch)" in out, f"push creates the branch ({out!r} {err!r})")
        check(asked == ["Push 1 commit to demo/app branch feature?"], f"push asked first ({asked})")
        head = FAKE.refs.get("feature")
        tree = FAKE.trees[FAKE.commits[head]["tree"]] if head else {}
        check(sorted(tree) == ["README.md", "new.txt", "src/main.rs"] and FAKE.blobs[tree["README.md"]] == b"# Demo\nmore\n", "the pushed tree holds the edits")
        check(head and FAKE.commits[head]["parents"] == [C0] and FAKE.refs["main"] == C0, "parented on the original commit; main untouched")
        code, out, _ = sh("cd app && gh pr create --title 'Change things' --body 'Done' && gh pr list && gh run list")
        check(code == 0 and "pull/1" in out and "#1\tChange things\tfeature -> main" in out and "failure\tCI\tfeature" in out, f"pull request and CI runs ({out!r})")
        config(0)
        code, out, err = sh("cd app && echo again >> README.md && git commit -m 'Again' && git push")
        check(code != 0 and "not allowed (autonomy: local commits only" in err, f"local-only refuses to push ({err!r})")
        config(1)
        page.evaluate("window.vmAnswer = false")
        code, out, err = sh("cd app && git push")
        check(code != 0 and "declined" in err and FAKE.commits[FAKE.refs['feature']]['message'] == "Change things", "a declined push leaves GitHub as it was")
        page.evaluate("window.vmAnswer = true")
        FAKE.corrupt = "src/main.rs"
        code, out, err = sh("gh repo clone demo/app bad")
        check(code != 0 and "does not match GitHub's hash" in err, f"a tampered file stops the clone ({err!r})")
        FAKE.corrupt = None
        raw_with_token = [p for p, a in FAKE.auth if p.startswith("/raw/") and a]
        api_with_token = [p for p, a in FAKE.auth if p.startswith("/api/") and a == "Bearer test-token"]
        check(not raw_with_token and api_with_token, "the token goes only to the API")
        browser.close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("vm github browser test passed")


if __name__ == "__main__":
    main()
