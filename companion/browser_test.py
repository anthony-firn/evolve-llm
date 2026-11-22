"""Browser-to-connector test using the production origin and a temporary project."""
from pathlib import Path
import importlib.util
import json
import mimetypes
import tempfile
import threading
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright

spec = importlib.util.spec_from_file_location('connector', Path(__file__).with_name('local.py'))
local = importlib.util.module_from_spec(spec); spec.loader.exec_module(local)
web = Path(__file__).resolve().parents[1] / 'web'
with tempfile.TemporaryDirectory() as tmp:
    c = local.Connector(tmp, approve=lambda _: True)
    server = ThreadingHTTPServer(('127.0.0.1', local.PORT), local.handler(c))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            try: context.grant_permissions(['local-network-access'], origin=local.ORIGIN)
            except Exception: pass
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            def route(r):
                name = r.request.url.split('/play/evolve-llm/', 1)[-1].split('#')[0]
                if name in ('', 'index.html'):
                    html = '''<!doctype html><button id="open-settings">Settings</button><script type="module">
                    import {mountLocalComputer} from './local-computer.mjs';
                    let turns=0;
                    mountLocalComputer({ready:()=>true,lease:()=>()=>{},modelName:()=> 'fixture',
                    turn:async()=>++turns===1?{text:'<tool_call>{"name":"bash","arguments":{"command":"printf agent-ok > agent.txt"}}</tool_call>',reason:'stop'}:{text:'Created agent.txt',reason:'stop'}});
                    </script>'''
                    r.fulfill(status=200,content_type='text/html',body=html,headers={'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; connect-src http://127.0.0.1:8791"})
                else:
                    path = web / name
                    r.fulfill(status=200 if path.is_file() else 404,content_type='text/javascript',body=path.read_bytes() if path.is_file() else b'')
            page.route(local.ORIGIN + '/**', route)
            page.goto(local.ORIGIN + '/play/evolve-llm/#local=' + c.token)
            page.locator('#local-status').filter(has_text='Connected').wait_for(timeout=15000)
            assert '#local=' not in page.url
            page.locator('#local-task').fill('printf browser-ok > browser.txt')
            page.get_by_role('button',name='Run command',exact=True).click()
            page.locator('#local-status').filter(has_text='Finished').wait_for()
            assert Path(tmp, 'browser.txt').read_text() == 'browser-ok'
            page.locator('#local-task').fill('Create agent.txt')
            page.get_by_role('button',name='Ask Pi',exact=True).click()
            page.locator('#local-log').filter(has_text='Created agent.txt').wait_for()
            assert Path(tmp, 'agent.txt').read_text() == 'agent-ok'
            page.get_by_role('button',name='Disconnect',exact=True).click()
            page.locator('#local-status').filter(has_text='Disconnected').wait_for()
            assert not errors, errors
            browser.close()
            print('PASS: production-origin pairing, real command, Pi tool call, credential removal, disconnect.')
    finally:
        server.shutdown();server.server_close()
