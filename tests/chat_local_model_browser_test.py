"""Open a real local GGUF and ask the main chat for a playable Snake game."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve, FLAGS

def check(runtime,model):
 server=serve(runtime)
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=FLAGS)
   context=browser.new_context(viewport={'width':1280,'height':900})
   context.add_init_script("if(window===window.top)localStorage.setItem('evolve-llm-settings',JSON.stringify({temperature:0,maxTokens:1024,ctx:8192,thinking:false,gpu:'high-performance',gpuMemory:'auto'}))")
   context.add_init_script("""if(window===window.top){
    window.__piTokens=0;const NativeWorker=window.Worker;
    window.Worker=class extends NativeWorker{constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='agent_event'&&data.event?.type==='token')window.__piTokens++;});}};
   }""")
   page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/')
   page.wait_for_selector('#gpu-chip[data-state=ok]',timeout=30000)
   page.set_input_files('#file-input',str(model))
   page.wait_for_selector('#ready:not([hidden])',timeout=180000)
   page.locator('#prompt').fill('Make me a Snake game I can play')
   page.locator('#send').click()
   page.frame_locator('.chat-preview-host iframe').locator('canvas').wait_for(timeout=20000)
   frame=page.frame_locator('.chat-preview-host iframe')
   frame.locator('#restart').click();assert frame.locator('#score').inner_text()=='0'
   assert 'Snake starter' in page.locator('.chat-development').inner_text()
   assert not page.locator('#workbench').is_visible()
   page.wait_for_function('() => window.__piTokens>0',timeout=90000)
   # The starter is already playable while the actual model runs Pi. Stop is
   # intentionally exercised rather than claiming a tiny model customized it.
   if page.locator('#stop').is_visible():page.locator('#stop').click()
   page.wait_for_selector('#app[data-phase=ready]',timeout=20000)
   assert 'snake.html' in page.locator('.development-files').inner_text(),page.locator('.msg.assistant').last.inner_text()
   page.locator('.development-files').get_by_role('button',name='Play',exact=True).click()
   page.frame_locator('.chat-preview-host iframe').locator('canvas').wait_for(timeout=10000)
   page.screenshot(path='/tmp/llm-main-chat-real-model.png')
   assert not errors,errors
   print('Real GGUF/WebGPU main chat: Auto → shared WASM workspace → labeled playable Snake starter → actual Pi model tokens → cancellation and saved file passed.')
   browser.close()
 finally:server.shutdown();server.server_close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);a=ap.parse_args();check(a.runtime,a.model)
