"""Exercise main-chat routing with a scripted model and real Pi, WASM files and previews."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve

def check(runtime):
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
    page.set_input_files('#file-input',{'name':'scripted-model.gguf','mimeType':'application/octet-stream','buffer':b'fixture'})
    page.wait_for_selector('#ready:not([hidden])')
    def send(text):
     page.locator('#prompt').fill(text);page.locator('#send').click()
     page.wait_for_selector('#app[data-phase=ready]',timeout=45000)
     return page.locator('.msg.assistant').last
    message=send('Make me a Snake game I can play')
    assert 'included Snake starter' in message.inner_text(),message.inner_text()
    assert '1 file changed' in message.inner_text(),message.inner_text()
    assert not page.locator('#workbench').is_visible()
    frame=message.frame_locator('iframe');assert frame.locator('canvas').is_visible()
    frame.locator('#restart').click();assert frame.locator('#score').inner_text()=='0'
    message=send('Make the snake game background blue')
    assert 'snake.html' in message.inner_text(),message.inner_text()
    assert 'Snake starter' not in message.inner_text(),message.inner_text()
    frame=message.frame_locator('iframe')
    assert frame.locator('body').evaluate('(e)=>getComputedStyle(e).backgroundColor')=='rgb(0, 0, 255)'
    assert frame.locator('canvas').is_visible()
    message.get_by_role('button',name='Source',exact=True).click()
    assert 'background:blue' in message.locator('.development-source').inner_text()
    message.get_by_role('button',name='Source',exact=True).click()
    for width in [390,1440]:
     page.set_viewport_size({'width':width,'height':900})
     assert page.locator('#prompt').is_visible()
     assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
     page.locator('#scroll').evaluate('(e)=>e.scrollTop=e.scrollHeight')
     message.get_by_role('button',name='Full size',exact=True).click()
     assert message.locator('.chat-app').evaluate('(e)=>e.classList.contains("expanded")')
     message.get_by_role('button',name='Exit full size',exact=True).click()
     page.screenshot(animations='disabled',path=f'/tmp/llm-chat-build-{kind.name}-{width}.png')
    message.get_by_role('button',name='Full size',exact=True).click()
    message.get_by_role('button',name='Stop preview',exact=True).click()
    assert not message.locator('.chat-app').is_visible()
    message.get_by_role('button',name='Play',exact=True).click()
    message.frame_locator('iframe').locator('canvas').wait_for()
    message.get_by_role('button',name='Exit full size',exact=True).click()
    message=send('Read README.txt and summarize it')
    assert 'Scripted summary of the saved workspace README.' in message.inner_text(),message.inner_text()
    assert 'did not produce an artifact' not in message.inner_text()
    message=send('Explain how the code works')
    assert 'scripted chat answer' in message.inner_text()
    assert message.locator('.chat-development').count()==0
    page.locator('#chat-mode').select_option('chat')
    assert 'scripted chat answer' in send('Make a snake game').inner_text()
    page.locator('#new-chat').click();assert page.locator('.chat-preview-host iframe').count()==0
    page.locator('#chat-mode').select_option('build')
    message=send('Build a calculator')
    assert 'calculator.html' in message.inner_text() and 'Snake starter' not in message.inner_text()
    assert message.frame_locator('iframe').locator('h1').inner_text()=='Calculator'
    page.locator('#prompt').fill('write a long task');page.locator('#send').click()
    page.wait_for_selector('#app[data-phase=agent]')
    assert page.locator('#file-input').is_disabled()
    assert page.locator('#chat-mode').is_disabled()
    page.locator('.msg.assistant').last.get_by_role('button',name='Open workspace',exact=True).click()
    page.locator('#workbench').get_by_role('button',name='Close',exact=True).click()
    assert page.locator('#app').get_attribute('data-phase')=='agent'
    page.locator('#stop').click()
    page.wait_for_selector('#app[data-phase=ready]',timeout=15000)
    assert 'cancel' in page.locator('.msg.assistant').last.inner_text().lower()
    page.locator('#chat-mode').select_option('auto')
    message=send('Make a Snake game and break the starter')
    assert 'starter was restored' in message.inner_text(),message.inner_text()
    assert message.frame_locator('iframe').locator('canvas').is_visible()
    # Terminal and chat see the same saved workspace; importing/running is never required.
    message.get_by_role('button',name='Open workspace',exact=True).click()
    assert page.locator('#workspace-files').get_by_role('button',name='snake.html',exact=True).is_visible()
    assert page.locator('#workspace-files').get_by_role('button',name='snake-2.html',exact=True).is_visible()
    assert page.locator('#workspace-files').get_by_role('button',name='calculator.html',exact=True).is_visible()
    page.locator('#workbench').get_by_role('button',name='Close',exact=True).click()
    message=send('Delete calculator.html')
    assert 'Deleted calculator.html' in message.inner_text(),message.inner_text()
    assert '1 file changed' in message.inner_text(),message.inner_text()
    assert message.get_by_role('button',name='Download',exact=True).count()==0
    assert errors==['broken fixture'],errors
    assert not requests,requests
    print(kind.name+': main-chat Auto/Chat/Build, playable starter, real Pi edit/test/preview, follow-up context, files, cancellation, recovery and mobile layout passed.',flush=True)
    browser.close()
 finally:server.shutdown();server.server_close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);check(ap.parse_args().runtime)
