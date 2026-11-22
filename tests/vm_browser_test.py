"""The Evolve VM in Chrome: its kernel, shell and utilities, run through the worker.

Each command runs with `sh -c` in a fresh VM (tests/../web/vm/harness.html) and must give exactly
the expected output and status: pipes, redirections, variables, wildcards, scripts, symlinks,
`cd`, waiting on endless producers (`yes | head`), sleeping, and command-not-found.

    python3 tools/build_vm.py && python3 tests/vm_browser_test.py

Needs Python Playwright and Google Chrome (JSPI: Chrome 137 or later).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import FLAGS, ROOT, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CASES = [
    ("echo hello world | tr a-z A-Z", 0, "HELLO WORLD\n"),
    ("mkdir -p proj/src && cd proj && printf 'fn main() {}\\n' > src/main.rs && pwd && cat src/main.rs", 0, "/home/user/proj\nfn main() {}\n"),
    ("cat missing.txt 2>/dev/null; echo status=$?", 0, "status=1\n"),
    ("no_such_program 2>/dev/null || echo status=$?", 0, "status=127\n"),
    ("seq 1 5 | sort -r | head -3 | tr '\\n' ' '", 0, "5 4 3 "),
    ("cd proj && grep -rn main . && find . -name '*.rs' -type f && wc -l src/main.rs", 0, "src/main.rs:1:fn main() {}\n./src/main.rs\n      1 src/main.rs\n"),
    ("X=42; echo \"x is $X\" 'and $X' ~", 0, "x is 42 and $X /home/user\n"),
    ("echo a > f1 && echo b > f2 && cat f? && diff f1 f2; echo diff=$?", 0, "a\nb\n--- f1\n+++ f2\n@@ -1,1 +1,1 @@\n-a\n+b\ndiff=1\n"),
    ("sleep 0.2 && echo slept", 0, "slept\n"),
    ("test -f /bin/ls && [ 3 -gt 2 ] && echo yes; cd /tmp/../home && pwd", 0, "yes\n/home\n"),
    ("printf 'a\\nb\\n' | sh -c 'cat; echo done' 2>&1", 0, "a\nb\ndone\n"),
    ("ln -s proj/src/main.rs link.rs && cat link.rs && readlink link.rs", 0, "fn main() {}\nproj/src/main.rs\n"),
    ("printf '#!/bin/sh\\necho args: $1 $2\\nexit 3\\n' > s.sh && sh s.sh one two", 3, "args: one two\n"),
    ("yes | head -3", 0, "y\ny\ny\n"),
    ("seq 1 100000 | wc -l", 0, " 100000\n"),
    ("echo out; echo err >&2", 0, "out\n"),
    # edit FILE OLD NEW: one exact match, \\n a new line; a miss or two matches change nothing.
    ("printf 'a\\nb\\nb2\\n' > e.txt && edit e.txt a 'a\\nx' && edit e.txt zz q 2>&1; edit e.txt b q 2>&1; cat e.txt && rm e.txt", 0,
     "edited e.txt\nedit: e.txt: the text to replace is not there. Find the exact line with: grep -n \"SOME WORDS\" e.txt\nedit: e.txt: the text to replace is in 2 places; include more of the surrounding text\na\nx\nb\nb2\n"),
    # echo reads backslash escapes as dash does (-E does not); edit takes an optional line number.
    ("echo 'a\\nb' && echo -E 'c\\nd' && echo -n x && echo 'y\\cz'", 0, "a\nb\nc\\nd\nxy"),
    ("printf 'one\\ntwo\\n' > l.txt && edit l.txt 2 two 2 && cat l.txt && rm l.txt", 0, "edited l.txt\none\n2\n"),
    ("printf 'one\\n' > s.txt; edit s.txt one one 2>&1; echo status=$?; rm s.txt", 0, "edit: OLD and NEW are the same, so nothing would change. Show the lines with grep -n \"WORDS\" -A3 s.txt, then edit s.txt 'exact old text' 'corrected text'\nstatus=1\n"),
    ("read notes.txt 2>&1; echo status=$?", 0, "read: notes.txt is not a variable name (read takes a line of input; to show a file, use: cat notes.txt)\nstatus=2\n"),
    # python3: RustPython, loaded on first use; scripts, the standard library and unittest.
    ("python3 -c 'import json; print(json.dumps({\"sum\": 2 + 3}))'", 0, '{"sum": 5}\n'),
    ("printf 'def add(a, b):\\n    return a + b\\n' > calc.py && printf 'import unittest\\nfrom calc import add\\nclass T(unittest.TestCase):\\n    def test_add(self):\\n        self.assertEqual(add(2, 3), 5)\\nunittest.main()\\n' > test_calc.py && python3 test_calc.py 2>&1 | tail -1; rm calc.py test_calc.py", 0, "OK\n"),
    ("rm -r proj f1 f2 link.rs s.sh && ls", 0, "README\n"),
]


def main():
    url = f"http://127.0.0.1:{serve(ROOT / 'web').server_address[1]}/play/evolve-llm/vm/harness.html"
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=FLAGS)
        page = browser.new_page()
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(url)
        for _ in range(200):
            if page.evaluate("window.vmReady || window.vmError || false"):
                break
            time.sleep(0.05)
        if not page.evaluate("window.vmReady || false"):
            raise SystemExit(f"the VM did not start: {page.evaluate('window.vmError || null')}")
        for command, code, want in CASES:
            t0 = time.time()
            r = page.evaluate("(c) => vmExec(c, { timeoutMs: 20000 })", command)
            ms = (time.time() - t0) * 1000
            ok = r["code"] == code and r["stdout"] == want
            print(f"{'ok  ' if ok else 'FAIL'} {ms:5.0f} ms  $ {command}")
            if not ok:
                problems.append(f"$ {command}\n  want [{code}] {want!r}\n  got  [{r['code']}] {r['stdout']!r} stderr {r['stderr']!r}")
        browser.close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"vm browser test passed ({len(CASES)} commands)")


if __name__ == "__main__":
    main()
