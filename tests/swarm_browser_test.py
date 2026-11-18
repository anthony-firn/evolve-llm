"""Two visitors share model pieces browser to browser.

Two Chrome profiles (separate storage) run the same model from Hugging Face with sharing on.
Visitor A reads it first and keeps its pieces; visitor B then receives pieces from A over WebRTC,
found through a local Nostr relay (tests/fixtures/mini-nostr-relay.mjs) and checked against the
published manifests (github.com/anthony-firn/evolve-llm-manifests). B must report pieces from other
visitors and give A's greedy reply.

    python3 tests/swarm_browser_test.py [--hf owner/repo/file.gguf]

Needs: Python Playwright, Google Chrome, Node.js, and a connection to Hugging Face and GitHub.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import FLAGS, ROOT, letters, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def size_mb(text, label):
    m = re.search(r"([\d.]+) (B|KB|MB|GB) " + re.escape(label), text)
    return float(m.group(1)) * {"B": 1e-6, "KB": 1e-3, "MB": 1, "GB": 1e3}[m.group(2)] if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    relay = subprocess.Popen(["node", str(ROOT / "tests/fixtures/mini-nostr-relay.mjs")], stdout=subprocess.PIPE, text=True)
    port = int(relay.stdout.readline())
    server = serve(ROOT / "dist")
    url = f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
    prompt = "Name three planets in our solar system."
    problems = []
    settings = {"temperature": 0, "maxTokens": 12, "ctx": 2048, "thinking": False, "gpu": "high-performance",
                "gpuMemory": "0.05", "cacheGb": 1, "share": True, "shareUpload": 0}
    swarm = {"relays": [f"ws://127.0.0.1:{port}"], "iceServers": []}
    try:
        with sync_playwright() as p:
            # Host candidates as plain addresses, so two profiles on one machine reach each other.
            browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=not a.headed,
                                        args=FLAGS + ["--disable-features=WebRtcHideLocalIpsWithMdns"])

            def visitor(name):
                ctx = browser.new_context(viewport={"width": 1280, "height": 900})
                ctx.add_init_script(f"if(window===window.top){{localStorage.setItem('evolve-llm-settings', {json.dumps(json.dumps(settings))});"
                                    f"localStorage.setItem('evolve-llm-swarm', {json.dumps(json.dumps(swarm))});}}")
                page = ctx.new_page()
                page.on("pageerror", lambda e: problems.append(f"{name}: page error: {e}"))
                page.goto(url)
                page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
                return page

            def run(page, name):
                if not page.evaluate("document.getElementById('other-sources').open"):
                    page.click("#other-sources > summary")
                page.check("#hf-stream")
                page.fill("#hf-input", a.hf)
                t0 = time.time()
                page.press("#hf-input", "Enter")
                page.wait_for_selector("#ready:not([hidden])", timeout=1800000)
                page.fill("#prompt", prompt)
                page.press("#prompt", "Enter")
                page.wait_for_selector(".msg.assistant .msg-stats", timeout=1800000)
                reply = page.inner_text(".msg.assistant .bubble")
                stats = page.inner_text(".msg.assistant .msg-stats")
                print(f"{name}: {time.time() - t0:.0f}s · {stats}", flush=True)
                return reply, stats

            a_page = visitor("A")
            reply_a, _ = run(a_page, "A")
            time.sleep(3)  # A announces the pieces it now holds
            b_page = visitor("B")
            reply_b, stats_b = run(b_page, "B")
            peers = size_mb(stats_b, "from other visitors")
            print(f"B received {peers:.1f} MB from other visitors")
            print("A:", a_page.inner_text("#status-line"))
            if peers <= 0:
                problems.append("visitor B received nothing from visitor A")
            if letters(reply_a) != letters(reply_b):
                problems.append(f"replies differ:\n  A {reply_a!r}\n  B {reply_b!r}")
            browser.close()
    finally:
        relay.kill()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("swarm browser test passed")


if __name__ == "__main__":
    main()
