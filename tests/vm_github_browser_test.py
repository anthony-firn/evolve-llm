"""`gh` and `git` in the Evolve VM against a fake GitHub served by the test server.

The fake implements the API calls the VM uses (repository, branch, commit and tree listings,
raw files, blobs, trees, commits, refs, pull requests, Actions runs) for one repository, so the
test can push without touching GitHub. In Chrome, through the VM worker:

clone (every file checked against its blob hash), status, diff, commit, log, a push to a new
branch (asked and allowed; the fake must then hold exactly the edited tree, parented on the
original commit), a pull request, CI runs, a refused push (autonomy: local only), a declined push,
a hash mismatch refusing a clone, switching to a branch on GitHub, merging (git merge and gh pr
merge, which ask even when pushing is free; declined and conflicting merges change nothing), and
the token reaching only the API.

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

    def ancestors(self, c):
        seen, todo = [], [c]
        while todo:
            x = todo.pop(0)
            if x not in seen:
                seen.append(x)
                todo.extend(self.commits[x]["parents"])
        return seen

    def merged_tree(self, base, head):
        """The tree of merging head into base, or None on a conflict (file-level, like a test needs)."""
        mb = next(c for c in self.ancestors(base) if c in self.ancestors(head))
        t0, tb, th = (self.trees[self.commits[c]["tree"]] for c in (mb, base, head))
        out = {}
        for p in set(t0) | set(tb) | set(th):
            o, b, h = t0.get(p), tb.get(p), th.get(p)
            if b == h or h == o:
                v = b
            elif b == o:
                v = h
            else:
                return None
            if v is not None:
                out[p] = v
        return out

    def merge(self, base, head, message):
        """Merge branch head into branch base: (status, sha)."""
        b, h = self.refs[base], self.refs[head]
        if h in self.ancestors(b):
            return 204, None
        tree = self.merged_tree(b, h)
        if tree is None:
            return 409, None
        cid = hashlib.sha1(f"{b}{h}{message}".encode()).hexdigest()
        self.commits[cid] = {"tree": self.tree_id(tree), "parents": [b, h], "message": message}
        self.refs[base] = cid
        return 201, cid

    def pull_json(self, p):
        merged = p.get("merged", False)
        mergeable = None if merged or p["head"] not in self.refs else self.merged_tree(self.refs[p["base"]], self.refs[p["head"]]) is not None
        return {"number": p["number"], "title": p["title"], "body": p.get("body", ""), "state": "closed" if merged else "open", "merged": merged, "mergeable": mergeable,
                "head": {"ref": p["head"], "sha": self.refs.get(p["head"])}, "base": {"ref": p["base"]}, "html_url": f"https://github.com/demo/app/pull/{p['number']}"}

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
        if rest == "/branches":
            return 200, [{"name": b, "commit": {"sha": c}} for b, c in sorted(self.refs.items())]
        if rest.startswith("/compare/"):
            base, head = rest[len("/compare/"):].split("...")
            mine = self.ancestors(self.refs[base])
            new = [c for c in self.ancestors(self.refs[head]) if c not in mine]
            mb = next(c for c in self.ancestors(self.refs[head]) if c in mine)
            t0, th = (self.trees[self.commits[c]["tree"]] for c in (mb, self.refs[head]))
            return 200, {"ahead_by": len(new), "files": [{"filename": p} for p in sorted(set(t0) | set(th)) if t0.get(p) != th.get(p)]}
        if rest == "/merges" and method == "POST":
            status, sha = self.merge(body["base"], body["head"], body.get("commit_message", "Merge"))
            return status, ({"sha": sha} if sha else {"message": "Merge conflict"} if status == 409 else b"")
        if "/check-runs" in rest:
            return 200, {"check_runs": [{"name": "CI", "status": "completed", "conclusion": "success"}]}
        if rest.startswith("/pulls/") and rest.endswith("/merge") and method == "PUT":
            p = self.pulls[int(rest.split("/")[2]) - 1]
            status, sha = self.merge(p["base"], p["head"], f"Merge pull request #{p['number']}")
            if status == 409:
                return 405, {"message": "Pull Request is not mergeable"}
            p["merged"] = True
            return 200, {"sha": sha or self.refs[p["base"]], "merged": True}
        if rest.startswith("/pulls/") and method == "GET":
            return 200, self.pull_json(self.pulls[int(rest.split("/")[2]) - 1])
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
            if body["head"] not in self.refs:
                return 422, {"message": "Validation Failed"}
            n = len(self.pulls) + 1
            self.pulls.append({**body, "number": n})
            return 201, {"number": n, "html_url": f"https://github.com/demo/app/pull/{n}"}
        if rest == "/pulls":
            head = query.get("head", "").partition(":")[2]
            return 200, [self.pull_json(p) for p in self.pulls if not p.get("merged") and head in ("", p["head"])]
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
        if status == 204:
            data = b""
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
        if not self.route("POST"):
            super().do_POST()

    def do_PATCH(self):
        self.route("PATCH")

    def do_PUT(self):
        self.route("PUT")


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
        code, out, _ = sh("gh repo clone demo/app")
        check(code == 0 and "app/ already holds demo/app (next: cd app)" in out, f"cloning again is harmless ({out!r})")
        code, out, _ = sh("cd app && echo more >> README.md && printf 'new\\n' > new.txt && rm docs/notes.txt && git status && git diff README.md")
        check("modified:   README.md" in out and "new file:   new.txt" in out and "deleted:    docs/notes.txt" in out, "status lists the changes")
        check("+more" in out and "--- a/README.md" in out, "diff shows the edit")
        code, out, _ = sh("cd app && git checkout -b feature && git commit -m 'Change things' && git log --oneline -n 2")
        check(code == 0 and "(not pushed) Change things" in out and "Initial commit" in out, f"commit and log ({out!r})")
        code, out, err = sh("cd app && git push")
        asked = page.evaluate("window.vmQuestions || []")
        check(code == 0 and "-> feature (new branch)" in out, f"push creates the branch ({out!r} {err!r})")
        check(asked == ["Push 1 commit to demo/app branch feature (README.md +1 -0, new.txt new, 1 line, docs/notes.txt deleted)?"], f"push asked first, saying what changes ({asked})")
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

        # Switching branches needs everything pushed; the error names the command to run.
        code, out, err = sh("cd app && git checkout main")
        check(code != 0 and "not pushed yet. First run: git push" in err, f"switching with unpushed commits says to push ({err!r})")
        code, out, err = sh("cd app && git push && git checkout main && cat README.md && git branch")
        check(code == 0 and "Switched to branch 'main'" in out and "# Demo\n" in out and "* main\n" in out and "  feature\n" in out, f"checkout of a branch on GitHub ({out!r} {err!r})")
        code, out, err = sh("cd app && git pull origin feature")
        check(code != 0 and "run: git merge feature" in err, f"pulling another branch points to git merge ({err!r})")

        # git merge: refused when local-only, asks every time, declined changes nothing.
        config(0)
        code, out, err = sh("cd app && git merge feature")
        check(code != 0 and "not allowed" in err and FAKE.refs["main"] == C0, f"local-only refuses to merge ({err!r})")
        config(2)
        page.evaluate("window.vmAnswer = false; window.vmQuestions = []")
        code, out, err = sh("cd app && git merge feature")
        asked = page.evaluate("window.vmQuestions")
        check(code != 0 and "declined by the person" in err and FAKE.refs["main"] == C0, f"a declined merge changes nothing ({err!r})")
        check(asked == ["Merge 2 commits from feature into main (the default branch) on demo/app, 3 files changed?"], f"merging asks even with push freely ({asked})")
        page.evaluate("window.vmAnswer = true")
        code, out, err = sh("cd app && git merge feature && cat README.md && ls && git status")
        main = FAKE.commits[FAKE.refs["main"]]
        check(code == 0 and "Merged feature into main" in out and main["parents"] == [C0, FAKE.refs["feature"]], f"git merge makes a merge commit on GitHub ({out!r} {err!r})")
        check("# Demo\nmore\nagain\n" in out and "new.txt" in out and "notes.txt" not in out and "working tree clean" in out, "and updates the files")
        code, out, _ = sh("cd app && git merge feature")
        check(code == 0 and "Already up to date" in out, "merging again has nothing to do")

        # gh pr merge: the current branch's pull request, asked even with push freely.
        page.evaluate("window.vmQuestions = []")
        code, out, err = sh("cd app && git checkout -b fix && echo fixed > fix.txt && git commit -m 'Fix' && git push && gh pr create --title 'Fix it' --body 'x' && gh pr merge")
        asked = page.evaluate("window.vmQuestions")
        fixed = FAKE.trees[FAKE.commits[FAKE.refs["main"]]["tree"]]
        check(code == 0 and "Merged pull request #2 into main" in out and "git checkout main" in out, f"gh pr merge merges this branch's pull request ({out!r} {err!r})")
        check(asked == ['Merge pull request #2 "Fix it" (fix into main) on demo/app, checks: 1 passed?'], f"only the merge asked ({asked})")
        check("fix.txt" in fixed and FAKE.pulls[1].get("merged"), "main on GitHub holds the fix")
        code, out, _ = sh("cd app && gh pr merge 2")
        check(code == 0 and "already merged" in out, "merging a merged pull request says so")

        # A conflict: nothing changes, and the message says what to do.
        c = FAKE.refs["main"]
        tree = dict(FAKE.trees[FAKE.commits[c]["tree"]])
        tree["README.md"] = FAKE.put(b"# Changed on main\n")
        cid = "c" * 40
        FAKE.commits[cid] = {"tree": FAKE.tree_id(tree), "parents": [c], "message": "Retitle on main"}
        FAKE.refs["main"] = cid
        code, out, err = sh("cd app && echo other > README.md && git commit -m 'Retitle here' && git push && gh pr create --title 'Retitle' && gh pr merge")
        check(code != 0 and "has merge conflicts with main" in err and FAKE.refs["main"] == cid, f"a conflicting pull request is not merged ({err!r})")
        code, out, err = sh("cd app && git checkout main && git merge fix")
        check(code != 0 and "merge conflict" in err and "Tell the person" in err and FAKE.refs["main"] == cid, f"a conflicting git merge changes nothing ({out!r} {err!r})")
        code, out, err = sh("cd app && git checkout -b empty && git commit -m 'Nothing'")
        check(code != 0 and "nothing to commit: no file has changed" in err, f"committing nothing fails, so a chain stops ({err!r})")
        code, out, err = sh("cd app && git push")
        check(code != 0 and "has no commits to push yet" in err, f"pushing a new branch without commits says to commit ({err!r})")
        page.evaluate("window.vmQuestions = []")
        code, out, err = sh("cd app && gh pr create --title 'Empty'")
        check(code != 0 and "is not on GitHub yet" in err and page.evaluate("window.vmQuestions") == [], f"an empty pull request is refused before asking ({err!r})")
        code, out, err = sh("cd app && git checkout fix && git checkout -b fix")
        check(code != 0 and "already exists on GitHub. Switch to it with: git checkout fix" in err, f"a new branch may not reuse a name on GitHub ({err!r})")
        # A file that loses most of its lines looks overwritten (> instead of >>): no commit.
        code, out, err = sh("gh repo clone demo/app guard >/dev/null && cd guard && seq 1 40 > long.txt && git commit -m Long && echo one > long.txt && git commit -m Oops")
        check(code != 0 and "Long" in out and "long.txt would go from 40 lines to 1, so nothing was committed" in err and "git restore long.txt" in err, f"an overwritten file is not committed ({out!r} {err!r})")
        code, out, err = sh("cd guard && git commit -m Oops --allow-shrink && git log --oneline -n 1")
        check(code == 0 and "long.txt +1 -40" in out, f"unless asked, and the commit says what changed ({out!r} {err!r})")
        code, out, _ = sh("git help && gh help")
        check("git merge BRANCH" in out and "gh pr merge" in out and "always asks the person" in out, "help lists merging")

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
