"""Offline WASM regression for nested inline scripts and worker stack exhaustion.

python3 -B tests/browser_script_depth_test.py --runtime web
No relay, external site, account or model download is used.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright
from browser_test import serve
from browser_view_browser_test import client_call


def check(runtime, depth):
    source = 'window.deepResult=' + 'identity(' * depth + '0' + ')' * depth + ';'
    html = ('<!doctype html><p id="out">pending</p><script>'
            'function identity(n){return n+1;}'
            'const nested=document.createElement("script");'
            f'nested.text={json.dumps(source)};'
            'document.body.appendChild(nested);'
            'document.getElementById("out").textContent="parsed:"+deepResult;'
            '</script>')
    server = serve(runtime)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome')
            try:
                context = browser.new_context()
                # Explicit unused local route prevents Tor bootstrapping. data: navigation
                # and the script test are handled entirely by the engine on this device.
                context.add_init_script("localStorage.setItem('evolve-browser-relay','ws://127.0.0.1:9/')")
                page = context.new_page()
                page.goto(f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/')
                page.locator('#open-browser').click()
                page.wait_for_selector('#browser-panel:not([hidden]) .browser-toolbar')
                client_call(page, 'b.ready')
                client_call(page, "(globalThis.__testCrashes=[],b.addEventListener('crashed', e=>globalThis.__testCrashes.push(e.detail.error)),true)")
                url = 'data:text/html;charset=utf-8,' + quote(html, safe='')
                client_call(page, f"b.request('navigate',{{url:{json.dumps(url)}}})")
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    crashes = page.evaluate('globalThis.__testCrashes')
                    assert not crashes, f'WASM worker crashed while parsing depth {depth}: {crashes}'
                    snapshot = client_call(page, 'b.snapshot(2000)')
                    if f'parsed:{depth}' in snapshot.get('text', ''):
                        assert client_call(page, "b.evaluate('deepResult')") == str(depth)
                        print(f'PASS: nested inline script depth {depth}; no worker crash')
                        return
                    page.wait_for_timeout(200)
                raise AssertionError('Nested inline script did not finish')
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('web'))
    parser.add_argument('--depth', type=int, default=128)
    args = parser.parse_args()
    check(args.runtime, args.depth)
