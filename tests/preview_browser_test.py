"""Run a real canvas app and verify preview isolation under production CSP in both browsers."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve

def check(runtime):
 server=serve(runtime)
 try:
  with sync_playwright() as p:
   for kind in [p.chromium,p.firefox]:
    browser=kind.launch();page=browser.new_page();remote=[]
    page.on('request',lambda r:remote.append(r.url) if not r.url.startswith('http://127.0.0.1:') else None)
    page.goto(f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/')
    page.locator('#open-terminal').click();page.wait_for_function("() => document.querySelector('#terminal-status')?.textContent.startsWith('Ready')")
    page.locator('#terminal-input').fill('demo snake');page.locator('.terminal-form button[type=submit]').click()
    page.wait_for_function("() => !document.querySelector('.terminal-form button[type=submit]').disabled",timeout=20000)
    log=page.locator('.terminal-log').inner_text();assert '"ok": true' in log,(kind.name,log)
    frame=page.frame_locator('.preview-host iframe');assert frame.locator('canvas').is_visible()
    first=frame.locator('canvas').evaluate('(c)=>c.toDataURL()');page.wait_for_timeout(200);second=frame.locator('canvas').evaluate('(c)=>c.toDataURL()');assert first!=second
    frame.locator('#restart').click();assert frame.locator('#score').inner_text()=='0'
    page.screenshot(path=f'/tmp/llm-snake-preview-{kind.name}.png')
    page.set_viewport_size({'width':390,'height':850});assert page.locator('#terminal-input').is_visible();assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path=f'/tmp/llm-snake-preview-mobile-{kind.name}.png')
    page.set_viewport_size({'width':1280,'height':720})
    # A faulty app must return an error; a hostile app gets neither parent storage nor network.
    result=page.evaluate(r'''async()=>{
     const {AppPreview}=await import('./preview-client.mjs');const host=document.createElement('div');document.body.append(host);const preview=new AppPreview({container:host});
     try{
      const broken=await preview.open('<script>throw new Error("broken-game")<\/script>',{smoke:true});
      const isolated=await preview.open('<script>try{parent.document.body;console.error("PARENT-ACCESS")}catch{}try{localStorage.getItem("private");console.error("STORAGE-ACCESS")}catch{}fetch("https://example.com/private").catch(()=>{});<\/script>',{smoke:true});
      return {broken,isolated};
     }finally{preview.dispose();host.remove()}
    }''')
    assert result['broken']['ok'] is False and any('broken-game' in s for s in result['broken']['errors']),result
    assert not any('ACCESS' in s for s in result['isolated']['errors']),result
    assert 'connect-src' in result['isolated']['violations'],result
    assert not remote,remote
    print(kind.name+': playable Snake, animation, restart, runtime errors, parent/storage isolation and denied network passed.',flush=True)
    # Actual Pi core + WASM + UI auto-preview with a deterministic model fixture.
    page.reload()
    page.evaluate(r'''async()=>{
      const {mountWorkbench}=await import('./workbench.mjs');const html=await(await fetch('./examples/snake.html')).text();let round=0;
      const workbench=await mountWorkbench({ready:()=>true,lease:()=>()=>{},getNetwork:()=>({enabled:false}),setNetwork:()=>{},modelName:()=> 'Scripted transport fixture',turn:async()=>{
        round++;return {text:round===1?'<tool_call>'+JSON.stringify({name:'write',arguments:{path:'/workspace/fixture-game.html',text:html}})+'</tool_call>':round===2?'<tool_call>'+JSON.stringify({name:'browser_test',arguments:{path:'/workspace/fixture-game.html'}})+'</tool_call>':'The runtime check passed.',reason:'end'};
      }});await workbench.open();
    }''')
    page.locator('#terminal-input').fill('pi Create and test a game');page.locator('.terminal-form button[type=submit]').click()
    page.wait_for_function("() => !document.querySelector('.terminal-form button[type=submit]').disabled",timeout=30000)
    assert '2 tool calls' in page.locator('.terminal-log').text_content(),page.locator('.terminal-log').text_content()
    assert page.frame_locator('.preview-host iframe').locator('canvas').is_visible()
    print(kind.name+': Pi write → browser_test → auto-play integration passed.',flush=True)
    browser.close()
 finally:server.shutdown();server.server_close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);check(ap.parse_args().runtime)
