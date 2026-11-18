"""One model across two devices: the chat device and a helper that opened its pairing link.

Three Chrome profiles (separate storage): a reference visitor runs the model alone; the chat
device pairs with a helper through a local Nostr relay (tests/fixtures/mini-nostr-relay.mjs),
loads the same model split between them (each reading its layers from Hugging Face), and must
give the reference's greedy reply. The helper must report running its layers.

    python3 tests/devices_browser_test.py [--hf owner/repo/file.gguf] [--memory GB] [--url URL]

Needs: Python Playwright, Google Chrome, Node.js, and a connection to Hugging Face.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import FLAGS, ROOT, letters, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf")
    ap.add_argument("--memory", default="auto", help="GPU memory setting on every device (small values make each stream)")
    ap.add_argument("--url", help="a deployed page instead of dist/")
    ap.add_argument("--public-relays", action="store_true")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    relay = subprocess.Popen(["node", str(ROOT / "tests/fixtures/mini-nostr-relay.mjs")], stdout=subprocess.PIPE, text=True)
    port = int(relay.stdout.readline())
    url = a.url or f"http://127.0.0.1:{serve(ROOT / 'dist').server_address[1]}/play/evolve-llm/"
    code = "TEST-PAIR-" + str(int(time.time()))[-6:]
    prompt = "Name three planets in our solar system."
    problems = []
    settings = {"temperature": 0, "maxTokens": 12, "ctx": 2048, "thinking": False, "gpu": "high-performance",
                "gpuMemory": a.memory, "cacheGb": 1}
    network = {"iceServers": []} if a.public_relays else {"relays": [f"ws://127.0.0.1:{port}"], "iceServers": []}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=not a.headed,
                                        args=FLAGS + ["--disable-features=WebRtcHideLocalIpsWithMdns"])

            def visitor(name, extra=None, suffix=""):
                ctx = browser.new_context(viewport={"width": 1280, "height": 900})
                store = {"evolve-llm-settings": json.dumps(settings), "evolve-llm-swarm": json.dumps(network), **(extra or {})}
                ctx.add_init_script("if(window===window.top){" + "".join(
                    f"localStorage.setItem({json.dumps(k)}, {json.dumps(v)});" for k, v in store.items()) + "}")
                page = ctx.new_page()
                page.on("pageerror", lambda e: problems.append(f"{name}: page error: {e}"))
                page.goto(url + suffix)
                page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
                return page

            def run(page, name, stream):
                if not page.evaluate("document.getElementById('other-sources').open"):
                    page.click("#other-sources > summary")
                if stream:
                    page.check("#hf-stream")
                page.fill("#hf-input", a.hf)
                t0 = time.time()
                page.press("#hf-input", "Enter")
                page.wait_for_selector("#ready:not([hidden])", timeout=1800000)
                meta = page.inner_text("#ready-meta")
                page.fill("#prompt", prompt)
                page.press("#prompt", "Enter")
                page.wait_for_selector(".msg.assistant .msg-stats", timeout=1800000)
                reply = page.inner_text(".msg.assistant .bubble")
                print(f"{name}: {time.time() - t0:.0f}s · {meta}\n  {page.inner_text('.msg.assistant .msg-stats')}", flush=True)
                return reply, meta

            ref = visitor("reference")
            reply_ref, _ = run(ref, "reference", True)
            ref.context.close()

            main_page = visitor("chat device", {"evolve-llm-pairing": "1", "evolve-llm-pair-code": code})
            helper = visitor("helper", suffix=f"#pair={code}")
            helper.wait_for_selector("#helping[open]", timeout=10000)
            # The page's CSP forbids evaluated strings, so wait by reading the list.
            for _ in range(120):
                if "Ready" in main_page.text_content("#pair-list"):
                    break
                time.sleep(0.5)
            else:
                raise SystemExit(f"the devices did not pair: {main_page.text_content('#pair-list')!r}")
            print("paired:", main_page.text_content("#pair-list"))
            reply_split, meta = run(main_page, "chat device + helper", False)
            print("helper:", helper.inner_text("#helping-status"))
            if " on " not in meta.split("context")[-1]:
                problems.append(f"the model did not run on the helper: {meta}")
            if "Running layers" not in helper.inner_text("#helping-status"):
                problems.append("the helper does not report running layers")
            if letters(reply_ref) != letters(reply_split):
                problems.append(f"replies differ:\n  alone {reply_ref!r}\n  split {reply_split!r}")
            browser.close()
    finally:
        relay.kill()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("devices browser test passed")


if __name__ == "__main__":
    main()
