"""End-to-end test of the browser app in Chrome with WebGPU.

Serves dist/ at /play/evolve-llm/ with the production Content-Security-Policy, opens a local
GGUF file through the page, chats, stops a reply and checks prompt-cache reuse. With --download
it also fetches the smallest catalog model from Hugging Face into browser storage and reloads it.

    python3 tests/browser_test.py --model ~/.cache/evolve-llm/models/SmolLM2-135M-Instruct-Q8_0.gguf [--download]

Needs: Python Playwright and Google Chrome. On Linux Chrome needs WebGPU flags, which this passes.
"""
import argparse
import functools
import http.server
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--ignore-gpu-blocklist"]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript"}

    def translate_path(self, path):
        prefix = "/play/evolve-llm/"
        path = path.split("?", 1)[0]
        if path.startswith(prefix):
            return str(Path(self.directory) / path[len(prefix):])
        return str(Path(self.directory) / "__missing__")

    def end_headers(self):
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, *args):
        pass


def serve(directory):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def native_reply(model, prompt, n, low_power=False):
    """The same greedy reply from the native build of the engine (Vulkan, Metal or D3D12)."""
    cli = ROOT / "target/release/evolve-llm"
    if not cli.exists():
        return None
    env = {**os.environ, "EVOLVE_LLM_POWER": "low" if low_power else "high"}
    out = subprocess.run([str(cli), "run", str(model), "--chat", "--prompt", prompt, "-n", str(n), "--temp", "0"],
                         capture_output=True, text=True, timeout=600, env=env)
    return out.stdout if out.returncode == 0 else None


def letters(text):
    return "".join(c for c in text if c.isalpha())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--download", action="store_true", help="also test a real Hugging Face download")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--low-power", action="store_true", help="ask the browser for its power-saving GPU")
    ap.add_argument("--screens", type=Path, help="save screenshots here")
    a = ap.parse_args()
    dist = ROOT / "dist"
    server = serve(dist)
    url = f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
    problems = []
    prompt = "Name three planets in our solar system."
    max_tokens = 40
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=not a.headed, args=FLAGS)
        ctx = browser.new_context(viewport={"width": 1360, "height": 900})
        ctx.add_init_script(f"localStorage.setItem('evolve-llm-settings', JSON.stringify({{temperature: 0, maxTokens: {max_tokens}, ctx: 2048, thinking: false, gpu: '{'low-power' if a.low_power else 'high-performance'}'}}))")
        page = ctx.new_page()
        # Fonts come from the evolvetech.dev site and are absent from a standalone build.
        page.on("console", lambda m: m.type in ("error", "warning") and "Failed to load resource" not in m.text and problems.append(f"console {m.type}: {m.text}"))
        page.on("response", lambda r: r.status >= 400 and "/assets/fonts/" not in r.url and problems.append(f"HTTP {r.status} {r.url}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(url)
        page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
        print("adapter:", page.inner_text("#gpu-name"))
        if a.screens:
            a.screens.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(a.screens / "welcome.png"))

        t0 = time.time()
        page.set_input_files("#file-input", str(a.model))
        page.wait_for_selector("#ready:not([hidden])", timeout=180000)
        print(f"loaded local file in {time.time() - t0:.1f}s:", page.inner_text("#ready-meta"))

        page.fill("#prompt", prompt)
        page.press("#prompt", "Enter")
        page.wait_for_selector(".msg.assistant .msg-stats", timeout=120000)
        reply = page.inner_text(".msg.assistant .bubble")
        stats = page.inner_text(".msg.assistant .msg-stats")
        print("reply:", reply.replace("\n", " ")[:200])
        print("stats:", stats)
        assert reply.strip(), "empty reply"

        # Greedy decoding must reproduce the native Vulkan run of the same engine.
        native = native_reply(a.model, prompt, max_tokens, a.low_power)
        if native is not None:
            same = letters(native) == letters(reply)
            print("matches the native greedy reply:", same)
            if not same:
                problems.append(f"browser and native replies differ:\n  browser {reply!r}\n  native  {native!r}")

        # Second turn reuses the cached prompt prefix.
        page.fill("#prompt", "Which of those is largest?")
        page.press("#prompt", "Enter")
        page.locator(".msg.assistant .msg-stats").nth(1).wait_for(timeout=120000)
        stats2 = page.locator(".msg.assistant .msg-stats").nth(1).inner_text()
        print("turn 2:", stats2)
        assert "reused" in stats2, "the second turn did not reuse the cache"

        # Stop a long reply.
        page.evaluate("() => { const s = JSON.parse(localStorage.getItem('evolve-llm-settings')); s.maxTokens = 2048; localStorage.setItem('evolve-llm-settings', JSON.stringify(s)); }")
        page.click("#new-chat")
        page.fill("#prompt", "Count from 1 to 1000, one number per line, with no other text.")
        page.press("#prompt", "Enter")
        page.wait_for_selector(".msg.assistant .bubble .answer", timeout=60000)
        page.click("#stop", timeout=5000)
        page.wait_for_selector(".msg.assistant .msg-stats", timeout=60000)
        stop_stats = page.inner_text(".msg.assistant .msg-stats")
        print("stopped:", stop_stats)
        assert "stopped" in stop_stats, "Stop did not end the reply"
        if a.screens:
            page.screenshot(path=str(a.screens / "chat.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            time.sleep(0.6)
            page.screenshot(path=str(a.screens / "phone.png"))
            page.click("#open-models")
            time.sleep(0.6)
            page.screenshot(path=str(a.screens / "phone-models.png"))
            page.click("#scrim")
            page.set_viewport_size({"width": 1360, "height": 900})

        if a.download:
            page.click("#new-chat")
            page.locator(".model", has_text="SmolLM2 135M").click()
            page.wait_for_selector("#ready:not([hidden])", timeout=600000)
            saved = page.locator(".model", has_text="SmolLM2 135M").inner_text()
            print("downloaded and loaded:", page.inner_text("#ready-meta"))
            assert "Running" in saved
            page.reload()
            page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
            page.wait_for_selector(".model:has-text('SmolLM2 135M') .tag.ok", timeout=30000)
            t0 = time.time()
            page.locator(".model", has_text="SmolLM2 135M").click()
            page.wait_for_selector("#ready:not([hidden])", timeout=120000)
            print(f"reloaded from browser storage in {time.time() - t0:.1f}s")
            steps = page.eval_on_selector_all("#loading-steps li", "els => els.map(e => e.dataset.step + ':' + e.dataset.state)")
            print("steps:", steps)
        browser.close()
    server.shutdown()
    real = [x for x in problems if "Download the React DevTools" not in x]
    if real:
        print("\n".join(real))
        sys.exit(1)
    print("browser test passed")


if __name__ == "__main__":
    main()
