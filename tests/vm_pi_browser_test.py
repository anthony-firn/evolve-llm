"""Pi on the Evolve VM: the `pi` command in the terminal, and Pi's Linux tools on a repository.

1. In the terminal (web/vm/terminal.html), `pi TASK` reaches the page's agent handler with the
   task and directory, its progress appears in the terminal, and Ctrl+C cancels the agent.
2. Pi's real agent core in Linux mode, with a scripted model, works on a repository of the fake
   GitHub (tests/vm_github_browser_test.py) through the VM: clone, cd (carried to the next calls),
   read, edit, commit on a branch (one command per line), push, open a pull request and merge it
   (which asks the person). The fake must then hold exactly that change on main.

    python3 tools/build_web.py && python3 tests/vm_pi_browser_test.py
"""
import http.server
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vm_github_browser_test as gh  # noqa: E402
from browser_test import FLAGS, ROOT  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

# The model's replies, one per turn: tool calls in Qwen's syntax, then an answer.
SCRIPT = [
    {"name": "bash", "arguments": {"command": "gh repo clone demo/app"}},
    {"name": "bash", "arguments": {"command": "cd app"}},  # carries over to the next calls
    {"name": "read", "arguments": {"path": "README.md"}},
    {"name": "edit", "arguments": {"path": "README.md", "oldText": "# Demo", "newText": "# Demo\n\nBuilt by Pi."}},
    {"name": "bash", "arguments": {"command": "git checkout -b pi-readme\ngit commit -m 'Describe the project'\ngit push"}},  # one command per line
    {"name": "bash", "arguments": {"command": "gh pr create --title 'Describe the project' --body 'From Pi' && gh pr merge"}},
]


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: gh.Handler(*a, directory=str(ROOT / "web"), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    problems = []

    def check(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            problems.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=FLAGS)
        page = browser.new_page(viewport={"width": 1000, "height": 640})
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(origin + "/play/evolve-llm/vm/terminal.html")
        text = lambda: page.evaluate("window.vmTerminal ? vmTerminal.text() : ''")
        for _ in range(200):
            if "user@evolve" in text():
                break
            time.sleep(0.05)

        # 1. The pi command and the page's agent handler.
        page.evaluate("""() => {
          window.agentCalls = [];
          vmClient.onAgent = (task, cwd, write, signal) => new Promise((resolve) => {
            agentCalls.push({ task, cwd });
            write('working on it\\n');
            signal.addEventListener('abort', () => { agentCalls.push({ cancelled: true }); resolve(130); });
          });
        }""")
        page.click(".vt-term:not([hidden])")
        page.keyboard.type("mkdir -p proj && cd proj && pi add a README")
        page.keyboard.press("Enter")
        for _ in range(100):
            if "working on it" in text():
                break
            time.sleep(0.05)
        calls = page.evaluate("agentCalls")
        check(calls[:1] == [{"task": "add a README", "cwd": "/home/user/proj"}], f"pi passes the task and directory ({calls})")
        check("working on it" in text(), "the agent's progress appears in the terminal")
        page.keyboard.press("Control+c")
        for _ in range(100):
            if {"cancelled": True} in page.evaluate("agentCalls") and text().count("user@evolve:") >= 2:
                break
            time.sleep(0.05)
        check({"cancelled": True} in page.evaluate("agentCalls"), "Ctrl+C cancels the agent")
        check(text().rstrip().endswith("$"), "and the shell is back at its prompt")

        # 2. Pi's agent core in Linux mode, scripted, on a repository.
        result = page.evaluate("""async ({ origin, script }) => {
          const { createPiTask } = await import('../pi-agent.bundle.mjs');
          const { VmWorkspace } = await import('./workspace.mjs');
          vmClient.configure({ github: { token: 't', autonomy: 2, api: origin + '/mockgh/api', raw: origin + '/mockgh/raw' } });
          const asked = [];
          vmClient.onAsk = (q) => { asked.push(q); return true; };
          let turn = 0;
          const events = [];
          const task = createPiTask({
            workspace: new VmWorkspace(vmClient, { cwd: '/home/user' }),
            network: { options: () => ({ enabled: false }) },
            linux: { cwd: '/home/user' },
            onEvent: (e) => { if (e.type === 'tool-result') events.push([e.isError, JSON.stringify(e.result).slice(0, 300)]); },
            turn: async () => {
              const call = script[turn++];
              return call ? { text: `<tool_call>${JSON.stringify(call)}</tool_call>`, reason: 'end' } : { text: 'Opened a pull request.', reason: 'end' };
            },
          });
          const r = await task.run('Describe the project in its README and open a pull request.');
          return { calls: r.calls, events, asked };
        }""", {"origin": origin, "script": SCRIPT})
        failed = [e for e in result["events"] if e[0]]
        check(result["calls"] == len(SCRIPT) and not failed, f"every tool call ran ({result['calls']} calls; failures {failed})")
        head = gh.FAKE.refs.get("pi-readme")
        tree = gh.FAKE.trees[gh.FAKE.commits[head]["tree"]] if head else {}
        check(head and gh.FAKE.blobs[tree["README.md"]] == b"# Demo\n\nBuilt by Pi.\n", "the branch on GitHub holds Pi's edit")
        check(head and gh.FAKE.commits[head]["message"] == "Describe the project" and gh.FAKE.commits[head]["parents"] == [gh.C0], "one commit on the new branch")
        check(gh.FAKE.pulls and gh.FAKE.pulls[-1]["head"] == "pi-readme" and gh.FAKE.pulls[-1]["base"] == "main", "Pi opened the pull request")
        main = gh.FAKE.trees[gh.FAKE.commits[gh.FAKE.refs["main"]]["tree"]]
        check(gh.FAKE.pulls and gh.FAKE.pulls[-1].get("merged") and gh.FAKE.blobs[main["README.md"]] == b"# Demo\n\nBuilt by Pi.\n", "Pi merged it once approved")
        check(len(result["asked"]) == 1 and result["asked"][0].startswith("Merge pull request #1"), f"only the merge asked the person ({result['asked']})")
        shell = [e[1] for e in result["events"] if "exit code" in e[1]]
        check(shell and all("in /home/user" in t for t in shell) and any("in /home/user/app" in t for t in shell), "shell results read as text with the directory")
        browser.close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("vm pi browser test passed")


if __name__ == "__main__":
    main()
