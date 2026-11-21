"""Exercise the real main chat and Pi against a temporary local project with a scripted model."""
from pathlib import Path
import importlib.util
import mimetypes
import tempfile
import threading
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright

spec = importlib.util.spec_from_file_location('connector', Path(__file__).with_name('local.py'))
local = importlib.util.module_from_spec(spec); spec.loader.exec_module(local)
root = Path(__file__).resolve().parents[1]
worker = (root / 'tests/chat-model-worker.mjs').read_text() + '''
const original = self.onmessage;
self.onmessage = event => {
 const m=event.data;
 if(m.type!=='agent_turn')return original(event);
 const first=m.messages.find(x=>x.role==='user')?.content||'';
 const task=first.split('Current task: ').at(-1);
 if(task.includes('long task')){self.pending=m.requestId;return;}
 const results=m.messages.filter(x=>x.role==='tool');
 let text;
 if(!results.length)text=tool('bash',{command:'printf main-chat-ok > chat.txt'});
 else if(results.at(-1).content.includes('denied'))text='The operation was denied.';
 else if(task.includes('previous'))text=first.includes('Created chat.txt')?'Previous task context received.':'Context missing.';
 else text='Created chat.txt on the connected computer.';
 emit('agent_done',{requestId:m.requestId,text,reason:'end',n:20});
};
'''
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
            page = context.new_page(); errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            def route(r):
                name=r.request.url.split('/play/evolve-llm/',1)[-1].split('#')[0].split('?')[0] or 'index.html'
                if name=='worker.js':
                    return r.fulfill(status=200,content_type='text/javascript',body=worker)
                path=root/'web'/name
                r.fulfill(status=200 if path.is_file() else 404,content_type='text/javascript' if path.suffix in ('.js','.mjs') else mimetypes.guess_type(str(path))[0] or 'application/octet-stream',body=path.read_bytes() if path.is_file() else b'',headers={'Content-Security-Policy':"default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self'; connect-src 'self' http://127.0.0.1:8791; img-src 'self' data:"})
            page.route(local.ORIGIN+'/**',route)
            page.goto(local.ORIGIN+'/play/evolve-llm/#local='+c.token)
            page.locator('#local-status').filter(has_text='Connected').wait_for()
            page.locator('#local-computer').get_by_role('button',name='Close',exact=True).click()
            page.wait_for_selector('#gpu-chip[data-state=ok]')
            page.set_input_files('#file-input',{'name':'fixture.gguf','mimeType':'application/octet-stream','buffer':b'fixture'})
            page.wait_for_selector('#ready:not([hidden])')
            def send(text):
                page.locator('#prompt').fill(text);page.locator('#send').click()
                page.wait_for_selector('#app[data-phase=ready]',timeout=20000)
                return page.locator('.msg.assistant').last.inner_text()
            result=send('Create chat.txt on this computer')
            assert 'Created chat.txt' in result,result
            assert 'Local computer activity' in result,result
            assert Path(tmp,'chat.txt').read_text()=='main-chat-ok'
            assert 'Previous task context received.' in send('Describe the previous task')
            page.locator('#chat-mode').select_option('chat')
            assert 'A scripted chat answer.' in send('Create another file')
            page.locator('#chat-mode').select_option('build')
            c.approve=lambda _:False
            result=send('Create a file again')
            assert 'denied' in result.lower() and '1 tool errors' in result,result
            page.locator('#prompt').fill('long task');page.locator('#send').click()
            page.wait_for_selector('#app[data-phase=agent]')
            page.locator('#stop').click()
            page.wait_for_selector('#app[data-phase=ready]',timeout=15000)
            assert 'cancel' in page.locator('.msg.assistant').last.inner_text().lower()
            page.locator('#open-local-computer').click()
            page.get_by_role('button',name='Disconnect',exact=True).click()
            page.locator('#local-status').filter(has_text='Disconnected').wait_for()
            page.locator('#local-computer').get_by_role('button',name='Close',exact=True).click()
            page.locator('#chat-mode').select_option('auto')
            assert 'A scripted chat answer.' in send('Hello')
            assert not errors,errors
            browser.close()
            print('PASS: main-chat native tools, activity, follow-up context, Chat exclusion, denial, Stop and disconnect.')
    finally:
        server.shutdown();server.server_close()
