"""The Linux VM building on a paired computer, end to end, without a model.

A fake GitHub holds demo/app (calc.py whose add() subtracts, and its test). The companion runs in
project mode on 127.0.0.1:8791 with its own copy of demo/app, accepting this test's origin. A page
with the real local-computer.mjs and VM client pairs with it, clones demo/app in the VM, and runs:
cargo before pairing (not available, with the CI hint); native python3 test_calc.py on the
computer (fails: the bug, streamed back); the same after sed fixes add() in the VM (passes, and only
calc.py is sent); node; and native outside a clone (refused). The companion must not be running.

    python3 tests/native_build_browser_test.py [--runtime web]
"""
import argparse
import importlib.util
import http.server
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vm_github_browser_test as gh  # noqa: E402
from browser_test import FLAGS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local', ROOT / 'companion' / 'local.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)

CALC = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
TEST = "import unittest\nfrom calc import add, sub\n\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_sub(self):\n        self.assertEqual(sub(5, 3), 2)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n"
FILES = {"README.md": "# Calculator\n", "calc.py": CALC, "test_calc.py": TEST}

PAGE = """<!doctype html><button id="open-settings">Settings</button><div id="dev-tools"></div><script type="module">
import {mountLocalComputer} from './local-computer.mjs';
import {sharedVm, provideConfig} from './vm/client.mjs';
mountLocalComputer({ready: () => false, lease: () => () => {}, modelName: () => 'none', turn: async () => ({text: '', reason: 'stop'}),
  onConnectionChange: (connected, building) => { window.building = building; }});
provideConfig(() => ({github: {token: 'test-token', autonomy: 2, api: location.origin + '/mockgh/api', raw: location.origin + '/mockgh/raw'}}));
const vm = sharedVm();
vm.onAsk = () => true;
window.vmExec = (c) => vm.exec(c, {timeoutMs: 600000});
vm.ready.then(() => { window.vmReady = true; }, (e) => { window.vmError = String(e); });
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=ROOT / "web")
    a = ap.parse_args()
    problems = []

    def check(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            problems.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        origin_repo = Path(tmp) / "origin"
        origin_repo.mkdir()
        for name, text in FILES.items():
            (origin_repo / name).write_text(text)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        for cmd in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "first"]):
            subprocess.run(["git", "-C", str(origin_repo), *cmd], check=True, env=env)
        asked = []
        project = local.Project("demo/app", approve=lambda q: asked.append(q) or True, url=str(origin_repo), base=Path(tmp) / "projects")
        project.prepare()
        connector = local.Connector(project.copy, approve=lambda _: True)
        connector.project = project

        gh.FAKE = gh.Fake({k: v.encode() for k, v in FILES.items()})
        site = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *x, **k: gh.Handler(*x, directory=str(a.runtime), **k))
        threading.Thread(target=site.serve_forever, daemon=True).start()
        origin = f"http://127.0.0.1:{site.server_address[1]}"
        companion = http.server.ThreadingHTTPServer(("127.0.0.1", local.PORT), local.handler(connector, (origin,)))
        threading.Thread(target=companion.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                # A person allows "connect to apps on this device" when the page first asks; Chrome
                # 151's prompt is not one Playwright can answer, so this test turns the check off.
                browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=FLAGS + ["--disable-features=LocalNetworkAccessChecks"])
                page = browser.new_context().new_page()
                page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
                page.on("console", lambda m: m.type == "error" and print("console:", m.text[:300], flush=True))
                page.route(origin + "/play/evolve-llm/native-test.html", lambda r: r.fulfill(status=200, content_type="text/html", body=PAGE))

                def load(hash=""):
                    page.goto("about:blank")  # a hash alone would not reload the page
                    page.goto(origin + "/play/evolve-llm/native-test.html" + hash)
                    for _ in range(400):
                        if page.evaluate("window.vmReady || window.vmError || false"):
                            break
                        time.sleep(0.05)
                    if not page.evaluate("window.vmReady || false"):
                        raise SystemExit(f"the VM did not start: {page.evaluate('window.vmError || null')}")

                def sh(command):
                    r = page.evaluate("(c) => vmExec(c)", command)
                    return r["code"], r["stdout"], r["stderr"]

                load()
                code, out, err = sh("cargo --version")
                check(code == 127 and "not available in the browser" in err and "gh run watch" in err, f"unpaired, cargo says to use CI ({code} {err!r})")

                load("#local=" + connector.token)
                for _ in range(200):
                    if page.evaluate("window.building || null"):
                        break
                    time.sleep(0.05)
                check(page.evaluate("window.building") == "demo/app", f"the page pairs for building demo/app ({page.locator('#local-status').inner_text()!r})")
                code, out, err = sh("gh repo clone demo/app >/dev/null && cd app && native python3 test_calc.py")
                check(code == 1 and "AssertionError: -1 != 5" in out and "FAILED" in out, f"the test runs on the computer and fails on the bug ({code} {out[-300:]!r} {err!r})")
                check((project.copy / "test_calc.py").read_text() == TEST, "the computer's copy holds the clone's files")
                before = (project.copy / "README.md").stat().st_mtime_ns
                code, out, err = sh("cd app && sed -i '2s/a - b/a + b/' calc.py && native python3 test_calc.py")
                check(code == 0 and "OK" in out, f"after the fix in the VM it passes there ({code} {out[-200:]!r} {err!r})")
                check((project.copy / "calc.py").read_text() == CALC.replace("a - b", "a + b", 1), "the fixed calc.py reached the copy")
                check((project.copy / "README.md").stat().st_mtime_ns == before, "unchanged files are not rewritten")
                if any("node" in b for b in project.tools.get("binds", [])) or Path("/usr/bin/node").exists():
                    code, out, err = sh("cd app && node -e 'console.log(6 * 7)'")
                    check(code == 0 and out.strip() == "42", f"node runs on the computer ({code} {out!r} {err!r})")
                code, out, err = sh("cd / && native true")
                check(code == 127 and "from a clone of demo/app" in err, f"outside a clone, native says where to run it ({err!r})")
                check(len(asked) == 1, f"one approval for the session ({len(asked)})")
                browser.close()
        finally:
            companion.shutdown(); companion.server_close(); site.shutdown(); site.server_close()
            project.kill(); project.close(); os_close(connector)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("native build browser test passed")


def os_close(connector):
    import os
    os.close(connector.fd)


if __name__ == "__main__":
    main()
