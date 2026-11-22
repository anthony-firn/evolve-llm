"""Production-origin phone pairing with public relays and a scripted laptop model.

Use --candidate to test the checkout's devices.mjs before deployment. No model
weights are downloaded and no local-machine connector operations are performed.
"""
import argparse
from pathlib import Path
import json
import secrets
from playwright.sync_api import sync_playwright

ap=argparse.ArgumentParser();ap.add_argument('--candidate',action='store_true');args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
base='https://www.evolvetech.dev/play/evolve-llm/'
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    host=browser.new_context();phone=browser.new_context(**p.devices['Pixel 7'])
    code=secrets.token_hex(10).upper()
    host.add_init_script('localStorage.setItem("evolve-llm-hosting","1");localStorage.setItem("evolve-llm-host-code",'+json.dumps(code)+')')
    hp=host.new_page();pp=phone.new_page();errors=[]
    for page in (hp,pp):
        page.on('pageerror',lambda e:errors.append(str(e)))
        if args.candidate:
            page.route('**/devices.mjs',lambda r:r.fulfill(path=root/'web/devices.mjs',content_type='text/javascript'))
    hp.route('**/worker.js',lambda r:r.fulfill(path=root/'tests/chat-model-worker.mjs',content_type='text/javascript'))
    hp.goto(base,wait_until='domcontentloaded')
    hp.wait_for_selector('#gpu-chip[data-state=ok]')
    hp.set_input_files('#file-input',{'name':'fixture.gguf','mimeType':'application/octet-stream','buffer':b'fixture'})
    hp.wait_for_selector('#ready:not([hidden])')
    hp.wait_for_function('() => !document.querySelector("#host-info").hidden')
    pp.goto(base+'#remote='+code,wait_until='domcontentloaded')
    for attempt in range(3):
        pp.wait_for_selector('#gpu-chip[data-state=ok]',state='attached',timeout=45000)
        pp.wait_for_selector('#ready:not([hidden])',timeout=15000)
        assert pp.locator('#gpu-name').inner_text().startswith('Your computer')
        pp.locator('#prompt').fill('Name three planets');pp.locator('#send').click()
        pp.wait_for_selector('#app[data-phase=ready]',timeout=15000)
        assert 'A scripted assistant answer.' in pp.locator('.msg.assistant').last.inner_text()
        print(f'PASS: public-relay Android pairing and laptop-worker reply, connection {attempt+1}',flush=True)
        if attempt<2:pp.reload(wait_until='domcontentloaded')
    assert not errors,errors
    browser.close()
