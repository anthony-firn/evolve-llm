"""The Browser view under production CSP: panel, layout and, when built, the WASM engine end to end.

    python3 tests/browser_view_browser_test.py --runtime web     # or dist

Always checks the panel (toolbar, start screen, relay help, no horizontal scroll at phone and
desktop widths, no page errors). When <runtime>/browser-pkg/browser.js exists and the relay
binary (browser/relay/target/release/evolve-relay) is built, it also starts a relay that may reach
a local fixture site, loads the site through the engine, and has the agent and the person use the
same page: the agent types and submits a form, the person clicks the field and types, and a
cookie and localStorage value set by page script are visible on the next page.
Otherwise the engine part is skipped with a message.
"""
import argparse
import functools
import http.server
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from browser_test import serve

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests/fixtures/browser-site"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def fixture_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(FIXTURE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def start_relay(app_origin, site_port):
    binary = ROOT / "browser/relay/target/release/evolve-relay"
    if not binary.exists():
        found = shutil.which("evolve-relay")
        if not found:
            return None, None
        binary = Path(found)
    port = free_port()
    proc = subprocess.Popen([str(binary), "--listen", f"127.0.0.1:{port}", "--allow-private", "--allow-port", str(site_port), "--allow-origin", app_origin],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(50):
        if proc.poll() is not None:
            raise SystemExit(f"evolve-relay exited: {proc.stderr.read().decode()[:500]}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return proc, f"ws://127.0.0.1:{port}/"
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("evolve-relay did not start listening")


def client_call(page, expr):
    return page.evaluate(f"async()=>{{const m=await import('./browser-session.mjs');const b=m.getBrowser();return await ({expr});}}")


def find_ref(text, label):
    for line in text.splitlines():
        for part in line.split("  "):
            part = part.strip()
            if part.startswith("[") and label in part:
                return int(part[1:part.index("]")])
    raise AssertionError(f"no ref for {label!r} in snapshot:\n{text}")


def check(runtime):
    server = serve(runtime)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}/play/evolve-llm/"
    engine_built = (Path(runtime) / "browser-pkg/browser.js").exists()
    site = relay = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base)
            page.locator("#open-browser").click()
            page.wait_for_selector("#browser-panel:not([hidden]) .browser-toolbar")
            assert page.locator(".browser-address-input").is_visible()
            assert page.locator(".browser-start").is_visible()
            for width in [390, 1440]:
                page.set_viewport_size({"width": width, "height": 850})
                page.wait_for_timeout(150)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
                assert page.locator(".browser-address-input").is_visible(), width
            # Toggle closes and reopens the same panel.
            page.locator("#open-browser").click()
            page.wait_for_selector("#browser-panel", state="hidden")
            page.locator("#open-browser").click()
            page.wait_for_selector("#browser-panel:not([hidden])")

            # The Google notice offers a normal new tab without copying panel session tokens.
            # Simulate only the URL event; no requests to Google are made by this component check.
            page.evaluate("async()=>{const {getBrowser}=await import('./browser-session.mjs');getBrowser()._apply({type:'url',url:'https://accounts.google.com/v3/signin/rejected?flowName=GlifWebSignIn&service=mail&hl=en&TL=private-panel-token',canBack:false,canForward:false})}")
            notice = page.locator('.browser-signin-help')
            assert notice.is_visible()
            link = notice.get_by_role('link', name='Open Google in new tab')
            assert link.get_attribute('href') == 'https://accounts.google.com/ServiceLogin?service=mail&hl=en'
            assert link.get_attribute('target') == '_blank'
            assert set(link.get_attribute('rel').split()) == {'noopener', 'noreferrer'}
            assert 'does not sign this panel in' in notice.inner_text()
            # Exercise the user click, with a local fixture response instead of an account request.
            external_requests = []
            def external_fixture(route):
                external_requests.append(route.request)
                route.fulfill(status=200, content_type='text/html', body='<p>External sign-in destination</p>')
            page.context.route('https://accounts.google.com/ServiceLogin*', external_fixture)
            try:
                with page.expect_popup() as opened:
                    link.click()
                popup = opened.value
                popup.wait_for_load_state('domcontentloaded')
                assert popup.url == link.get_attribute('href')
                assert popup.evaluate('window.opener === null')
                assert 'referer' not in external_requests[0].headers
                popup.close()
            finally:
                page.context.unroute('https://accounts.google.com/ServiceLogin*', external_fixture)
            page.set_viewport_size({'width': 390, 'height': 850})
            page.wait_for_timeout(150)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert link.is_visible()
            page.set_viewport_size({'width': 1440, 'height': 850})
            page.evaluate("async()=>{const {getBrowser}=await import('./browser-session.mjs');getBrowser()._apply({type:'url',url:'about:blank',canBack:false,canForward:false})}")
            assert notice.is_hidden()
            assert notice.locator('a').get_attribute('href') is None

            if not engine_built:
                page.wait_for_function("() => /not built/.test(document.querySelector('.browser-status-text')?.textContent || '')", timeout=15000)
                assert not errors, errors
                print("Browser panel, toggle, relay chip and mobile layout passed. SKIPPED engine checks: "
                      f"{runtime}/browser-pkg/browser.js is missing (run `node tools/build_workbench.mjs`).")
                browser.close()
                return

            site = fixture_server()
            site_url = f"http://127.0.0.1:{site.server_address[1]}/"
            relay, relay_url = start_relay(f"http://127.0.0.1:{port}", site.server_address[1])
            if not relay:
                print("Browser panel passed. SKIPPED engine checks: build the relay with "
                      "`cargo build --release --manifest-path browser/relay/Cargo.toml`.")
                browser.close()
                return
            client_call(page, f"b.setRelay({json.dumps(relay_url)})")
            page.wait_for_function("() => document.querySelector('.browser-relay')?.dataset.state === 'ok'", timeout=10000)

            # The person navigates with the address bar.
            page.locator(".browser-address-input").fill(site_url)
            page.locator(".browser-address-input").press("Enter")
            page.wait_for_function("async()=>{const m=await import('./browser-session.mjs');return m.getBrowser().state.title==='Fixture start'}", timeout=60000)
            client_call(page, "b.waitForIdle(10000)")
            # A solid-color canvas compresses to a tiny PNG; rendered text and borders do not.
            assert len(page.locator(".browser-canvas").screenshot()) > 4000, "the canvas looks blank"

            # The agent reads the page, types and submits.
            snap = client_call(page, "b.snapshot(4000)")
            assert "Browser fixture" in snap["text"], snap
            name = find_ref(snap["text"], "Name")
            result = client_call(page, f"b.type({name}, 'Pi', {{submit: true}})")
            assert result["ok"], result
            assert {"x", "y", "w", "h"} <= result.keys(), result
            page.wait_for_function("async()=>{const m=await import('./browser-session.mjs');return m.getBrowser().state.title==='Hello Pi'}", timeout=20000)
            assert page.locator(".browser-agent-cursor").count() == 1

            # The person clicks the same field and types; the agent's next action is stale.
            box = page.locator(".browser-stage").bounding_box()
            page.mouse.click(box["x"] + result["x"] + result["w"] / 2, box["y"] + result["y"] + result["h"] / 2)
            page.keyboard.type("Ann")
            page.keyboard.press("Enter")
            page.wait_for_function("async()=>{const m=await import('./browser-session.mjs');return /Ann/.test(m.getBrowser().state.title)}", timeout=20000)
            stale = client_call(page, f"b.click({name})")
            assert stale.get("error") == "stale" or stale.get("ok"), stale

            # The person opens a <select>: the panel draws the menu and the engine applies the choice.
            box_js = "JSON.stringify((r=>[r.x,r.y,r.width,r.height])(document.getElementById('color').getBoundingClientRect()))"
            raw = client_call(page, f"b.evaluate({json.dumps(box_js)})")
            sx, sy, sw, sh = json.loads(json.loads(raw) if raw.startswith('"') else raw)
            page.mouse.click(box["x"] + sx + sw / 2, box["y"] + sy + sh / 2)
            page.wait_for_selector(".browser-menu:not([hidden]) .browser-menu-option", timeout=10000)
            assert page.locator(".browser-menu-group").first.text_content() == "Cool"
            page.locator(".browser-menu-option", has_text="Blue").click()
            page.wait_for_selector(".browser-menu", state="hidden")
            page.wait_for_function("async()=>{const m=await import('./browser-session.mjs');const s=await m.getBrowser().snapshot(4000);return /Picked Blue/.test(s.text)}", timeout=10000)

            # Cookies and localStorage written by page script carry to the next page.
            snap = client_call(page, "b.snapshot(4000)")
            link = find_ref(snap["text"], "Second page")
            assert client_call(page, f"b.click({link})")["ok"]
            page.wait_for_function("async()=>{const m=await import('./browser-session.mjs');return m.getBrowser().state.title==='Second page'}", timeout=20000)
            client_call(page, "b.waitForIdle(10000)")
            text = client_call(page, "b.snapshot(4000)")["text"]
            assert "greeted=" in text and "stored:" in text, text

            assert not errors, errors
            print("Browser panel, address bar, relay, rendering, agent typing and submit, person input on the same page, "
                  "stale agent actions, the select menu, cookies and localStorage across pages passed.")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        if site:
            site.shutdown()
            site.server_close()
        if relay:
            relay.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    check(parser.parse_args().runtime)
