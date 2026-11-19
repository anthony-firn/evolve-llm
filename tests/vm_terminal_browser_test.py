"""The Evolve VM terminal in Chrome, driven with real keystrokes.

In the full-page terminal (web/vm/terminal.html): a pipeline's output, the tab title the shell
sets, Ctrl+C ending a running command without ending the shell, Ctrl+Shift+T and `exit`
opening and closing tabs, and Find. Then in the app: Terminal, Linux terminal, a command, the
autonomy setting reaching `gh`, and `pi` reaching Pi.

    python3 tools/build_web.py && python3 tests/vm_terminal_browser_test.py [--runtime dist]

Needs Python Playwright and Google Chrome.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import FLAGS, ROOT, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def wait_for(fn, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=ROOT / "dist")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{serve(a.runtime).server_address[1]}/play/evolve-llm/"
    problems = []

    def expect(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            problems.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=FLAGS)
        page = browser.new_page(viewport={"width": 1000, "height": 640})
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(base + "vm/terminal.html")
        text = lambda: page.evaluate("window.vmTerminal ? vmTerminal.text() : ''")
        expect(wait_for(lambda: "user@evolve" in text()), "the shell's prompt appears")
        page.click(".vt-term:not([hidden])")
        page.keyboard.type("echo hi there | tr a-z A-Z")
        page.keyboard.press("Enter")
        expect(wait_for(lambda: "HI THERE" in text()), "a pipeline typed at the prompt runs")
        expect(page.text_content(".vt-tab.active .vt-tab-title") == "user@evolve: ~", "the shell sets the tab title")
        page.keyboard.type("sleep 30")
        page.keyboard.press("Enter")
        time.sleep(0.3)
        t0 = time.time()
        page.keyboard.press("Control+c")
        back = wait_for(lambda: text().count("user@evolve:") >= 3, 5)
        expect(back and time.time() - t0 < 2, f"Ctrl+C ends sleep and returns to the prompt ({time.time() - t0:.2f}s)")
        page.keyboard.press("Control+Shift+t")
        expect(wait_for(lambda: page.locator(".vt-tab").count() == 2), "Ctrl+Shift+T opens a tab")
        new_ready = "vmTerminal.active === vmTerminal.tabs[1] && vmTerminal.text().includes('user@evolve') && !vmTerminal.text().includes('HI THERE')"
        expect(wait_for(lambda: page.evaluate(new_ready)), "the new tab is active with its own shell")
        page.keyboard.type("exit")
        page.keyboard.press("Enter")
        expect(wait_for(lambda: page.locator(".vt-tab").count() == 1), "exit closes the tab")
        page.keyboard.press("Control+Shift+f")
        page.keyboard.type("HI THERE")
        expect(wait_for(lambda: (page.text_content(".vt-find-count") or "").endswith("of 2")), "Find counts the matches")
        page.keyboard.press("Escape")

        # In the app: Terminal, then Linux terminal.
        app = browser.new_page(viewport={"width": 1280, "height": 900})
        app.on("pageerror", lambda e: problems.append(f"app page error: {e}"))
        app.goto(base)
        app.wait_for_selector("#gpu-chip[data-state=ok]", state="attached", timeout=30000)
        app.click("#open-terminal")
        app.click("#linux-terminal-toggle")
        app_text = lambda: app.evaluate("document.querySelector('.vm-console .vt-term:not([hidden])') ? 'ready' : ''")
        expect(wait_for(app_text), "the app opens the Linux terminal")
        time.sleep(1)
        app.click(".vm-console .vt-term:not([hidden])")
        app.keyboard.type("uname && ls /bin | wc -l")
        app.keyboard.press("Enter")
        time.sleep(1)
        shown = app.evaluate("[...document.querySelectorAll('.vm-console canvas')].length")
        expect(shown >= 1, "the app's terminal draws")
        # Settings reach the VM, and `pi` reaches Pi (which needs a model first).
        app.evaluate("""() => {
          const f = document.getElementById('settings-form');
          f.elements.autonomy.value = '0';
          f.elements.autonomy.dispatchEvent(new Event('input', { bubbles: true }));
        }""")
        app.keyboard.type("gh auth status > /tmp/auth.txt; pi add a README > /tmp/pi.txt; echo done")
        app.keyboard.press("Enter")
        read = lambda path: app.evaluate("(p) => import('./vm/client.mjs').then(({ sharedVm }) => sharedVm().read(p)).then((r) => r.ok ? r.data : '')", path)
        expect(wait_for(lambda: "local commits only" in read("/tmp/auth.txt"), 15), f"the autonomy setting reaches gh ({read('/tmp/auth.txt')!r})")
        expect(wait_for(lambda: "load a tool-capable model first" in read("/tmp/pi.txt"), 15), f"pi in the app reaches Pi ({read('/tmp/pi.txt')!r})")
        browser.close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("vm terminal browser test passed")


if __name__ == "__main__":
    main()
