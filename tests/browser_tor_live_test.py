"""The browser's Tor route against the live Tor network (needs internet; not part of the default tests).

    python3 tests/browser_tor_live_test.py --runtime web [--url https://news.ycombinator.com/]

Serves the workbench under the production Content-Security-Policy, opens the Browser panel with
the route set to Tor, waits for the engine to reach the Tor network over Snowflake, loads real
pages through Tor and reads them with the agent snapshot. Prints the measured times.
"""
import argparse
import json
import time

from playwright.sync_api import sync_playwright
from browser_test import serve


def client_call(page, expr):
    return page.evaluate(f"async()=>{{const m=await import('./browser-session.mjs');const b=m.getBrowser();return await ({expr});}}")


def state(page):
    return client_call(page, "b.state")


def check(runtime, urls, timeout_s, route="tor:"):
    server = serve(runtime)
    base = f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        page.locator("#open-browser").dispatch_event("click")
        page.wait_for_selector("#browser-panel:not([hidden]) .browser-toolbar")
        client_call(page, "b.ready")
        client_call(page, f"b.setRelay({json.dumps(route)})")
        started = time.time()
        while True:
            s = state(page)
            if s["relayOk"] is not None or time.time() - started > timeout_s:
                break
            time.sleep(0.5)
        tor_s = time.time() - started
        assert s["relayUrl"] == route, s
        assert s["relayOk"] is True, f"The route did not connect: {s['network']}"
        if route == "tor:":
            print(f"Tor ready in {tor_s:.1f} s: {s['network']['detail']}")
            assert page.locator(".browser-relay span").inner_text() == "Tor"

        seen = 0
        for url in urls:
            t0 = time.time()
            client_call(page, f"b.navigate({json.dumps(url)})")
            client_call(page, f"b.waitForIdle({timeout_s * 1000})")
            s = state(page)
            snap = client_call(page, "b.snapshot(4000)")
            took = time.time() - t0
            assert s["title"] and "Can’t open" not in s["title"], (url, s, snap["text"][:400])
            assert snap["refs"] > 0, snap
            log = client_call(page, "b.networkLog()")
            log, seen = log[seen:], len(log)
            failed = [e for e in log if e.get("error")]
            print(f"{url}: “{s['title']}” in {took:.1f} s, {snap['refs']} refs, {len(log)} requests ({len(failed)} failed)")
        assert not errors, errors
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="web")
    ap.add_argument("--url", action="append")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--route", default="tor:", help="tor: (default) or a relay URL, for comparison")
    a = ap.parse_args()
    check(a.runtime, a.url or ["https://example.com/", "https://news.ycombinator.com/"], a.timeout, a.route)
