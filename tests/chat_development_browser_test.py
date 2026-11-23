"""The main chat with a scripted model: one assistant, no modes. Real app, agent core, files and previews.

A question gets a plain answer with no steps shown; a request to make something runs tools, shows
one collapsed line of steps and a live preview; follow-ups see the conversation; Stop cancels; the
Terminal shows the same files; a model whose template takes no tools chats plainly. Chrome works in
the Linux VM, Firefox (no WebAssembly JSPI) in the browser sandbox.
"""
import argparse
import json
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve, Handler

def check(runtime):
 trace=Path(tempfile.mkstemp(suffix='.jsonl')[1]);Handler.trace_file=str(trace)
 server=serve(runtime)
 try:
  with sync_playwright() as p:
   for kind in [p.chromium,p.firefox]:
    browser=kind.launch();page=browser.new_page(reduced_motion='reduce');errors=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:requests.append(r.url) if not r.url.startswith('http://127.0.0.1:') else None)
    page.route('**/worker.js',lambda route:route.fulfill(path=Path(__file__).with_name('chat-model-worker.mjs'),content_type='text/javascript'))
    page.goto(f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/')
    page.wait_for_selector('#gpu-chip[data-state=ok]')
    def load(name):
     page.set_input_files('#file-input',{'name':name,'mimeType':'application/octet-stream','buffer':b'fixture'})
     page.wait_for_selector('#ready:not([hidden])')
    def send(text):
     page.locator('#prompt').fill(text);page.locator('#send').click()
     page.wait_for_selector('#app[data-phase=ready]',timeout=45000)
     return page.locator('.msg.assistant').last
    load('scripted-model.gguf')
    assert page.locator('#chat-mode').count()==0
    linux=page.evaluate("typeof WebAssembly.Suspending==='function'")
    # A question: just the answer.
    message=send('What is WebGPU?')
    assert 'WebGPU is a scripted answer.' in message.inner_text(),message.inner_text()
    assert not message.locator('.work-steps').is_visible() and not message.locator('.chat-app').is_visible()
    # Making something: one collapsed line of steps, and the app.
    message=send('Make me a Snake game I can play')
    assert 'ready' in message.inner_text(),message.inner_text()
    frame=message.frame_locator('iframe');assert frame.locator('canvas').is_visible()
    frame.locator('#restart').click();assert frame.locator('#score').inner_text()=='0'
    steps=message.locator('.work-steps')
    assert steps.locator('summary').first.inner_text()=='Worked · 1 step',steps.inner_text()
    assert not page.locator('#workbench').is_visible()
    message=send('Make the snake game background blue')
    frame=message.frame_locator('iframe')
    assert frame.locator('body').evaluate('(e)=>getComputedStyle(e).backgroundColor')=='rgb(0, 0, 255)'
    steps=message.locator('.work-steps');steps.locator('summary').first.click()
    assert 'Read snake.html' in steps.inner_text() and 'Edited snake.html' in steps.inner_text() and 'Opened snake.html' in steps.inner_text(),steps.inner_text()
    assert 'Pi' not in message.inner_text(),message.inner_text()
    # Follow-ups see the conversation.
    assert 'You sent 4 messages.' in send('How many messages have I sent?').inner_text()
    message=page.locator('.msg.assistant').nth(2)
    for width in [390,1440]:
     page.set_viewport_size({'width':width,'height':900})
     assert page.locator('#prompt').is_visible()
     assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
     message.scroll_into_view_if_needed()
     message.get_by_role('button',name='Full size',exact=True).click()
     assert message.locator('.chat-app').evaluate('(e)=>e.classList.contains("expanded")')
     message.get_by_role('button',name='Exit full size',exact=True).click()
     page.screenshot(animations='disabled',path=f'/tmp/llm-chat-assistant-{kind.name}-{width}.png')
    message.get_by_role('button',name='Stop preview',exact=True).click()
    assert not message.locator('.chat-app').is_visible()
    message=send('Build a calculator')
    assert message.frame_locator('iframe').locator('h1').inner_text()=='Calculator'
    # Stop.
    page.locator('#prompt').fill('write a long task');page.locator('#send').click()
    page.wait_for_selector('#app[data-phase=agent]')
    assert page.locator('#file-input').is_disabled() and page.locator('#stop').is_visible()
    page.locator('#stop').click()
    page.wait_for_selector('#app[data-phase=ready]',timeout=15000)
    assert 'Stopped.' in page.locator('.msg.assistant').last.inner_text(),page.locator('.msg.assistant').last.inner_text()
    message=send('Delete calculator.html')
    message.locator('.work-steps summary').first.click()
    assert 'Deleted calculator.html.' in message.inner_text() and 'Ran rm calculator.html' in message.locator('.work-steps').inner_text(),message.inner_text()
    # The Terminal shows the assistant's files.
    page.locator('#open-terminal').dispatch_event("click")
    if linux:
     page.wait_for_selector('.vm-console .vt-term:not([hidden])')
     listing=page.evaluate("async()=>(await (await import('./vm/client.mjs')).sharedVm().exec('ls')).stdout")
     assert 'snake.html' in listing and 'calculator.html' not in listing,listing
    else:
     files=page.locator('#workspace-files');files.get_by_role('button',name='snake.html',exact=True).wait_for()
     assert files.get_by_role('button',name='calculator.html',exact=True).count()==0
    page.locator('#workbench').get_by_role('button',name='Close',exact=True).click()
    page.locator('#new-chat').click();assert page.locator('.chat-preview-host iframe').count()==0
    # Saved chats: the conversation comes back with its steps, and its app reopens from the files.
    # The conversation list sits beside the chat.
    saved=page.locator('#chat-list li');saved.first.wait_for()
    assert saved.count()==1 and 'What is WebGPU?' in saved.first.inner_text(),page.locator('#sidebar').inner_text()
    saved.first.locator('.chat-open').click()
    page.get_by_text('You sent 4 messages.').wait_for()
    assert page.locator('.msg.user').count()==7,page.locator('.msg.user').count()
    blue=page.locator('.msg.assistant').nth(2);blue.locator('.work-steps summary').first.click()
    assert 'Edited snake.html' in blue.locator('.work-steps').inner_text()
    blue.get_by_role('button',name='Open snake.html',exact=True).click()
    assert blue.frame_locator('iframe').locator('body').evaluate('(e)=>getComputedStyle(e).backgroundColor')=='rgb(0, 0, 255)'
    assert 'You sent 5 messages.' in send('How many messages have I sent?').inner_text()
    page.locator('#chat-list li').first.wait_for();assert page.locator('#chat-list li').count()==1
    page.locator('#chat-list li').first.get_by_role('button',name='Delete “What is WebGPU?”').click()
    page.locator('#chats-empty').wait_for()
    page.locator('#new-chat').click()
    # The local trace saw every run, model turn, tool call and result.
    events=[json.loads(l) for l in trace.read_text().splitlines()]
    kinds={e['type'] for e in events}
    assert {'run','turn','tool','result','done'}<=kinds,kinds
    assert any(e['type']=='tool' and e['name']=='edit' for e in events) and any(e['type']=='turn' and '<tool_call>' in e['text'] for e in events)
    assert not any('githubToken' in json.dumps(e) for e in events)
    trace.write_text('')
    # A model without tool support chats plainly.
    page.reload();page.wait_for_selector('#gpu-chip[data-state=ok]');load('notools-model.gguf')
    message=send('Make me a Snake game')
    assert 'A scripted chat answer.' in message.inner_text() and message.locator('.chat-work').count()==0
    assert not errors,errors
    assert not requests,requests
    print(kind.name+f': one main chat ({"Linux VM" if linux else "browser sandbox"}): answers, steps, previews, follow-ups, stop, shared files, saved chats, the local trace and plain chat passed.',flush=True)
    browser.close()
 finally:server.shutdown();server.server_close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);check(ap.parse_args().runtime)
