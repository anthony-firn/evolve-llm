"""Credential-free Google sign-in UI compatibility test (requires the live Tor network).

python3 -B tests/browser_google_live_test.py --runtime web --screens /tmp/google-ui

Checks real rendering and empty-field validation. It never enters an identifier, password,
or verification code, and does not establish or claim to verify an authenticated session.
"""
import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright
from browser_test import serve
from browser_view_browser_test import client_call, find_ref


def check(runtime, timeout, screens, url, base_url=None):
    server = None if base_url else serve(runtime)
    started = time.monotonic()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            try:
                page = browser.new_page(viewport={"width": 1360, "height": 900})
                page_errors = []
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                base = base_url or f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
                page.goto(base)
                page.locator("#open-browser").click()
                page.wait_for_selector("#browser-panel:not([hidden]) .browser-toolbar")
                client_call(page, "b.ready")
                client_call(page, "b.setRelay('tor:')")
                page.locator('.browser-address-input').fill(url)
                page.locator('.browser-address-input').press('Enter')
                print("Google page requested over Tor.", flush=True)
                deadline = time.monotonic() + timeout
                snap = {}
                while time.monotonic() < deadline:
                    snap = client_call(page, "b.snapshot(6000)")
                    if "Email or phone" in snap.get("text", "") and not snap.get("loading"):
                        opacity = client_call(page, "b.evaluate('getComputedStyle(document.body).opacity')")
                        if json.loads(opacity) != "0":
                            break
                    page.wait_for_timeout(500)
                else:
                    if screens:
                        screens.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(screens / "google-startup-failure.png"))
                    raise AssertionError(f"Google UI did not become visible: {snap}")

                # A DOM snapshot can exist even while the body is invisible. Check actual pixels.
                image = Image.open(io.BytesIO(page.locator('.browser-canvas').screenshot())).convert('RGB')
                visible = sum(count for count, pixel in image.getcolors(image.width * image.height) if min(pixel) < 220)
                assert visible > 1000, f"Google canvas is blank ({visible} contrasting pixels)"
                next_ref = find_ref(snap["text"], 'button "Next"')
                assert client_call(page, f"b.click({next_ref})")["ok"]
                validation_deadline = time.monotonic() + 30
                while time.monotonic() < validation_deadline:
                    snap = client_call(page, "b.snapshot(6000)")
                    if "Enter an email or phone number" in snap["text"]:
                        break
                    page.wait_for_timeout(500)
                else:
                    raise AssertionError(f"Empty-field validation did not respond: {snap}")
                assert not page_errors, page_errors
                if screens:
                    screens.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screens / "google-empty-validation.png"))
                print(f"Google sign-in rendered over Tor and empty-field validation responded in {time.monotonic() - started:.1f}s.")
                print("No account details submitted; authenticated sign-in has not been tested.")
            finally:
                browser.close()
    finally:
        if server:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", type=Path, default=Path("web"))
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--screens", type=Path)
    ap.add_argument("--url", default="https://accounts.google.com/ServiceLogin?hl=en")
    ap.add_argument("--base-url", help="Test an existing deployed workbench instead of serving --runtime locally")
    args = ap.parse_args()
    check(args.runtime, args.timeout, args.screens, args.url, args.base_url)
