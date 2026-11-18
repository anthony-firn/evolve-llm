"""End-to-end test of running a model from Hugging Face without downloading it, in Chrome.

Serves dist/ with the production Content-Security-Policy, ticks "Run it from Hugging Face
without downloading it first", opens a small model by its Hugging Face path with a GPU budget
far below its size (so every layer is read again for every token), chats, and requires the
greedy reply to match the native build reading the same model from a local file. A second load
must then come mostly from the device cache instead of the network.

    python3 tests/stream_browser_test.py --model ~/.cache/evolve-llm/models/SmolLM2-135M-Instruct-Q8_0.gguf \\
        --hf bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf

Needs: Python Playwright, Google Chrome, and a connection to huggingface.co.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_test import FLAGS, ROOT, letters, native_reply, serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def size_mb(text, label):
    """Megabytes in a stats line like '12.3 MB from Hugging Face'."""
    m = re.search(r"([\d.]+) (B|KB|MB|GB) " + re.escape(label), text)
    if not m:
        return 0.0
    return float(m.group(1)) * {"B": 1e-6, "KB": 1e-3, "MB": 1, "GB": 1e3}[m.group(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="the same model as a local file, for the native reply")
    ap.add_argument("--hf", required=True, help="owner/repo/file.gguf on Hugging Face")
    ap.add_argument("--gpu-mem", default="0.05", help="GPU memory setting in GB")
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    server = serve(ROOT / "dist")
    url = f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
    prompt = "Name three planets in our solar system."
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=not a.headed, args=FLAGS)
        ctx = browser.new_context(viewport={"width": 1360, "height": 900})
        ctx.add_init_script(
            "localStorage.setItem('evolve-llm-settings', JSON.stringify({temperature: 0, maxTokens: %d, ctx: 2048, "
            "thinking: false, gpu: 'high-performance', gpuMemory: '%s', cacheGb: 1}))" % (a.max_tokens, a.gpu_mem)
        )
        page = ctx.new_page()
        page.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and problems.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(url)
        page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)

        def load():
            if page.is_visible("#open-models") and not page.is_visible("#sidebar"):
                page.click("#open-models")
            if not page.evaluate("document.getElementById('other-sources').open"):
                page.click("#other-sources > summary")
            page.check("#hf-stream")
            page.fill("#hf-input", a.hf)
            t0 = time.time()
            page.press("#hf-input", "Enter")
            page.wait_for_selector("#ready:not([hidden])", timeout=1800000)
            print(f"loaded in {time.time() - t0:.1f}s:", page.inner_text("#ready-meta"))

        load()
        page.fill("#prompt", prompt)
        page.press("#prompt", "Enter")
        page.wait_for_selector(".msg.assistant .msg-stats", timeout=1800000)
        reply = page.inner_text(".msg.assistant .bubble")
        stats = page.inner_text(".msg.assistant .msg-stats")
        print("reply:", reply.replace("\n", " ")[:200])
        print("stats:", stats)
        if "from Hugging Face" not in stats and "from this device" not in stats:
            problems.append("the reply's statistics do not show weights read from Hugging Face")
        native = native_reply(a.model, prompt, a.max_tokens)
        if native is not None:
            same = letters(native) == letters(reply)
            print("matches the native greedy reply from the local file:", same)
            if not same:
                problems.append(f"streamed and native replies differ:\n  browser {reply!r}\n  native  {native!r}")

        # Load again: the device cache should now serve most of what the reply reads.
        page.reload()
        page.wait_for_selector("#gpu-chip[data-state=ok]", timeout=30000)
        load()
        page.fill("#prompt", prompt)
        page.press("#prompt", "Enter")
        page.wait_for_selector(".msg.assistant .msg-stats", timeout=1800000)
        stats2 = page.inner_text(".msg.assistant .msg-stats")
        reply2 = page.inner_text(".msg.assistant .bubble")
        print("second load stats:", stats2)
        net, cache = size_mb(stats2, "from Hugging Face"), size_mb(stats2, "from this device")
        print(f"second load: {net:.1f} MB from the network, {cache:.1f} MB from the device cache")
        if letters(reply2) != letters(reply):
            problems.append("the reply changed when read from the cache")
        if cache <= net:
            problems.append("the second load did not come mostly from the device cache")
        browser.close()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("stream browser test passed")


if __name__ == "__main__":
    main()
