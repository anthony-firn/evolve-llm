"""Opt-in UA/TLS comparison against live services, with no real account or credentials.

python3 -B tests/browser_identity_live_test.py --runtime web --output /tmp/browser-identities.json
Results omit IPs, cookies, session URLs and Google response bodies. Synthetic identifier
validation is not a test of authenticated sign-in or Google's acceptance of a real account.
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from browser_test import serve
from browser_view_browser_test import client_call, find_ref

CASES = [(p, "default") for p in ["legacy", "firefox-esr", "firefox", "chrome", "evolve"]] + [("firefox-esr", "firefox-order")]
TLS_URL = "https://tls.peet.ws/api/all"
GOOGLE = "https://accounts.google.com/v3/signin/identifier?flowName={}&flowEntry=ServiceLogin&hl=en"


def fingerprint(data):
    tls = data.get("tls", {})
    return {"http_version": data.get("http_version"), "user_agent": data.get("user_agent"),
            "ja3": tls.get("ja3"), "ja3_hash": tls.get("ja3_hash"), "ja4": tls.get("ja4"),
            "ciphers": tls.get("ciphers"), "extensions": tls.get("extensions")}


def poll(page, callback, timeout=90):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = callback()
        if last:
            return last
        page.wait_for_timeout(500)
    raise TimeoutError("The live page did not become ready")


def body(page):
    return json.loads(client_call(page, "b.evaluate('document.body?.textContent || \"\"')"))


def google_check(page, flow):
    client_call(page, f"b.navigate({json.dumps(GOOGLE.format(flow))})")
    snap = poll(page, lambda: (lambda s: s if 'textbox "Email or phone"' in s.get("text", "") and not s.get("loading") else None)(client_call(page, "b.snapshot(7000)")))
    ref = find_ref(snap["text"], 'textbox "Email or phone"')
    assert client_call(page, f"b.type({ref}, 'evolve-browser-compat@example.invalid')")["ok"]
    snap = client_call(page, "b.snapshot(7000)")
    ref = find_ref(snap["text"], 'button "Next"')
    assert client_call(page, f"b.click({ref})")["ok"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        text = client_call(page, "b.snapshot(7000)")["text"].lower()
        if "couldn’t find" in text or "couldn't find" in text:
            return "identifier-not-found"
        if "may not be secure" in text:
            return "browser-rejected"
        if "enter your password" in text:
            return "password-prompt-unexpected"
        page.wait_for_timeout(500)
    return "stalled-after-next"


def check(runtime, output, google=True, only=None, tls_only_profile=None):
    results = {"cases": []}
    server = serve(runtime)
    base = f"http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            try:
                # A real Chrome baseline; its network route differs from the engine's Tor route.
                context = browser.new_context()
                try:
                    page = context.new_page()
                    response = page.goto(TLS_URL, timeout=60000)
                    results["native_chrome"] = fingerprint(response.json())
                except Exception as e:
                    results["native_chrome"] = {"error": type(e).__name__}
                finally:
                    context.close()
                for identity, tls in CASES:
                    if only and identity != only:
                        continue
                    if tls_only_profile and tls != tls_only_profile:
                        continue
                    result = {"identity": identity, "tls": tls}
                    context = browser.new_context(viewport={"width": 1360, "height": 900})
                    try:
                        context.add_init_script(f"localStorage.setItem('evolve-browser-identity', {json.dumps(identity)});localStorage.setItem('evolve-browser-tls', {json.dumps(tls)});localStorage.setItem('evolve-browser-relay','tor:')")
                        page = context.new_page()
                        page.goto(base)
                        page.locator("#open-browser").dispatch_event("click")
                        page.wait_for_selector("#browser-panel:not([hidden]) .browser-toolbar")
                        client_call(page, "b.ready")
                        client_call(page, f"b.navigate({json.dumps(TLS_URL)})")
                        def tls_data():
                            if not client_call(page, "b.state")["url"]:
                                return None
                            try:
                                data = json.loads(body(page))
                                return data if "tls" in data else None
                            except (ValueError, TypeError):
                                return None
                        result["fingerprint"] = fingerprint(poll(page, tls_data))
                        result["navigator"] = json.loads(client_call(page, "b.evaluate('JSON.stringify({ua:navigator.userAgent,vendor:navigator.vendor,clientHints:typeof navigator.userAgentData})')"))
                        if isinstance(result["navigator"], str):
                            result["navigator"] = json.loads(result["navigator"])
                        assert result["navigator"]["ua"] == result["fingerprint"]["user_agent"]
                        if tls == "firefox-order":
                            client_call(page, "b.navigate('https://expired.badssl.com/')")
                            def certificate_error():
                                logs = client_call(page, "b.networkLog()")
                                return next((e["error"] for e in logs if e.get("url") == "https://expired.badssl.com/" and e.get("error")), None)
                            error = poll(page, certificate_error)
                            result["certificate_error"] = error
                            result["certificate_check"] = "invalid-certificate-rejected" if "cert" in error.lower() else "inconclusive-network"
                        if google:
                            result["google"] = {}
                            for flow in ["GlifWebSignIn", "WebLiteSignIn"]:
                                try:
                                    result["google"][flow] = google_check(page, flow)
                                except Exception as e:
                                    result["google"][flow] = {"error": type(e).__name__}
                    except Exception as e:
                        result["error"] = type(e).__name__
                    finally:
                        context.close()
                    results["cases"].append(result)
                    output.write_text(json.dumps(results, indent=2))
                    print(json.dumps({k: v for k, v in result.items() if k not in ["fingerprint", "navigator"]}), flush=True)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", type=Path, default=Path("web"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tls-only", action="store_true")
    ap.add_argument("--tls-profile", choices=["default", "firefox-order"])
    ap.add_argument("--identity", choices=["legacy", "firefox-esr", "firefox", "chrome", "evolve"])
    args = ap.parse_args()
    check(args.runtime, args.output, not args.tls_only, args.identity, args.tls_profile)
