"""A phone uses a computer's GPU: the conversation on the phone, the model on the computer.

A computer profile (Chrome with WebGPU) lets a phone use it ("Use this computer from your
phone"); a phone profile (a separate Chrome without WebGPU, emulating a Pixel 7) opens the
#remote= link, loads a model on the computer and chats. Its greedy reply must match the same
model run directly in a third profile. Then the phone reloads: it must reconnect, find the model
still loaded on the computer, and reply again. Pairing goes through a local Nostr relay
(tests/fixtures/mini-nostr-relay.mjs) unless --public-relays.

    python3 tests/remote_browser_test.py [--hf owner/repo/file.gguf] [--url URL] [--public-relays]

Needs: Python Playwright, Google Chrome, Node.js, and a connection to Hugging Face.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import FLAGS, ROOT, letters, open_sandbox, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf")
    ap.add_argument("--url", help="a deployed page instead of dist/")
    ap.add_argument("--public-relays", action="store_true")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--pi-hf", help="then load this model from the phone and give Pi a file task in the phone's terminal")
    a = ap.parse_args()
    relay = subprocess.Popen(["node", str(ROOT / "tests/fixtures/mini-nostr-relay.mjs")], stdout=subprocess.PIPE, text=True)
    port = int(relay.stdout.readline())
    url = a.url or f"http://127.0.0.1:{serve(ROOT / 'dist').server_address[1]}/play/evolve-llm/"
    code = "TEST-HOST-" + str(int(time.time()))[-6:]
    prompts = ["Name three planets in our solar system.", "What is the capital of France?"]
    problems = []
    settings = {"temperature": 0, "maxTokens": 12, "ctx": 2048, "thinking": False, "gpu": "high-performance", "gpuMemory": "auto", "cacheGb": 1}
    network = {"iceServers": []} if a.public_relays else {"relays": [f"ws://127.0.0.1:{port}"], "iceServers": []}

    def init_script(extra=None):
        store = {"evolve-llm-settings": json.dumps(settings), "evolve-llm-swarm": json.dumps(network), **(extra or {})}
        return "if(window===window.top){" + "".join(f"localStorage.setItem({json.dumps(k)}, {json.dumps(v)});" for k, v in store.items()) + "}"

    def ask(page, name, prompt, nth):
        page.fill("#prompt", prompt)
        page.press("#prompt", "Enter")
        page.wait_for_selector(f".msg.assistant >> nth={nth}", timeout=600000)
        page.locator(".msg.assistant").nth(nth).locator(".msg-stats").wait_for(timeout=600000)
        reply = page.locator(".msg.assistant").nth(nth).locator(".bubble").inner_text()
        print(f"{name}: {reply!r}", flush=True)
        return reply

    def load(page):
        # On a phone the model list is a drawer: open it as a person would.
        if page.is_visible("#open-models") and page.get_attribute("#app", "data-drawer") != "open":
            page.click("#open-models")
        if not page.evaluate("document.getElementById('other-sources').open"):
            page.click("#other-sources > summary")
        page.fill("#hf-input", a.hf)
        page.press("#hf-input", "Enter")
        # Wait for this load to start (a model may already be ready), then for it to finish.
        page.wait_for_selector("#ready", state="hidden", timeout=60000)
        t0 = time.time()
        last = 0
        while not page.is_visible("#ready"):
            if time.time() - t0 > 1800:
                raise SystemExit(f"the load did not finish: {page.text_content('#loading-step')!r} {page.text_content('#status-line')!r}")
            if time.time() - last > 30:
                last = time.time()
                print(f"  {time.time() - t0:.0f}s: {page.text_content('#loading-step')} · {page.text_content('#loading-detail') if page.query_selector('#loading-detail') else ''} {page.text_content('#status-line')}", flush=True)
            time.sleep(1)
        return page.inner_text("#ready-meta")

    try:
        with sync_playwright() as p:
            gpu_browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=not a.headed,
                                            args=FLAGS + ["--disable-features=WebRtcHideLocalIpsWithMdns"])
            # The phone: no WebGPU flags, and a phone's screen, touch and user agent.
            phone_browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=not a.headed,
                                              args=["--disable-features=WebRtcHideLocalIpsWithMdns"])

            ref_ctx = gpu_browser.new_context()
            ref_ctx.add_init_script(init_script())
            ref = ref_ctx.new_page()
            ref.goto(url)
            ref.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
            load(ref)
            # Each question in its own conversation (the reloaded phone starts a new one).
            want = [letters(ask(ref, "computer alone", prompts[0], 0))]
            ref.click("#new-chat")
            want.append(letters(ask(ref, "computer alone", prompts[1], 0)))
            ref_ctx.close()

            host_ctx = gpu_browser.new_context()
            host_ctx.add_init_script(init_script({"evolve-llm-hosting": "1", "evolve-llm-host-code": code}))
            host = host_ctx.new_page()
            host.on("pageerror", lambda e: problems.append(f"computer: page error: {e}"))
            host.goto(url)
            host.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)

            phone_ctx = phone_browser.new_context(**p.devices["Pixel 7"])
            phone_ctx.add_init_script(init_script())
            phone = phone_ctx.new_page()
            phone.on("pageerror", lambda e: problems.append(f"phone: page error: {e}"))
            phone.goto(url + f"#remote={code}")
            print("phone has a GPU adapter:", phone.evaluate("navigator.gpu ? navigator.gpu.requestAdapter().then((a) => Boolean(a)) : false"))
            # (The chip is hidden at phone width; its state still says when the computer answered.)
            phone.wait_for_selector("#gpu-chip[data-state=ok]", state="attached", timeout=60000)
            chip = phone.text_content("#gpu-name")
            print("phone shows:", chip)
            if not chip.startswith("Your computer"):
                problems.append(f"the phone does not show the computer's GPU: {chip!r}")
            host.wait_for_selector("#hosting[open]", timeout=10000)
            t0 = time.time()
            meta = load(phone)
            print(f"loaded on the computer from the phone in {time.time() - t0:.0f}s: {meta}")
            got = letters(ask(phone, "phone", prompts[0], 0))
            if got != want[0]:
                problems.append(f"first reply differs: {got!r} vs {want[0]!r}")
            print("computer:", host.inner_text("#hosting-status"))

            # The phone reloads (as after its screen locked): it reconnects to the loaded model.
            phone.reload()
            phone.wait_for_selector("#ready:not([hidden])", timeout=60000)
            got = letters(ask(phone, "phone after reconnecting", prompts[1], 0))
            if got != want[1]:
                problems.append(f"reply after reconnecting differs: {got!r} vs {want[1]!r}")
            if a.pi_hf:
                a.hf = a.pi_hf
                load(phone)
                open_sandbox(phone)
                phone.wait_for_function("() => document.querySelector('#terminal-status')?.textContent.startsWith('Ready')", timeout=60000)
                phone.locator("#terminal-input").fill('pi Write the text "phoneok" to a file named phone.txt')
                phone.locator(".terminal-form button[type=submit]").click()
                phone.wait_for_function("() => !document.querySelector('.terminal-form button[type=submit]').disabled", timeout=900000)
                log = phone.locator(".terminal-log").inner_text()
                phone.locator("#terminal-input").fill("cat phone.txt")
                phone.locator(".terminal-form button[type=submit]").click()
                phone.wait_for_function("() => !document.querySelector('.terminal-form button[type=submit]').disabled", timeout=60000)
                content = phone.locator(".terminal-log").inner_text().rsplit("cat phone.txt", 1)[-1]
                ran = "Pi → write" in log or "Pi → bash" in log
                print(f"Pi on the phone: tool call ran: {ran}; file holds the text: {'phoneok' in content}")
                if "phoneok" not in content:
                    problems.append("Pi on the phone did not write the file")
            gpu_browser.close()
            phone_browser.close()
    finally:
        relay.kill()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("remote browser test passed")


if __name__ == "__main__":
    main()
