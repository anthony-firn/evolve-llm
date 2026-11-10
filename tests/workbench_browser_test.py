"""Exercise the shipped WASM worker, Pi core, permissions and persistence under production CSP."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve


def check(runtime):
    server=serve(runtime)
    base=f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/'
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch()
            page=browser.new_page(); errors=[]; remote=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:remote.append(r.url) if not r.url.startswith(base.split('/play/')[0]+'/') else None)
            page.goto(base);page.locator('#open-terminal').click()
            page.wait_for_function("() => document.querySelector('#terminal-status').textContent.startsWith('Ready')")
            def command(text):
                page.locator('#terminal-input').fill(text)
                page.locator('.terminal-form button[type=submit]').click()
                page.wait_for_function("() => !document.querySelector('.terminal-form button[type=submit]').disabled")
                return page.locator('.terminal-log').inner_text()
            assert 'apple\npear' in command('printf "pear\\napple\\npear\\n" | sort | uniq > fruit.txt && cat fruit.txt')
            assert 'Only /workspace paths' in command('cat /etc/passwd')
            assert 'off' in command('curl https://example.com').lower()
            assert not remote,remote
            page.reload();page.locator('#open-terminal').click()
            assert 'apple\npear' in command('cat fruit.txt')
            # Real Pi agent core + actual sandbox WASM, using a deterministic model transport.
            result=page.evaluate("""async()=>{
              const {Workspace}=await import('./shell-client.mjs');const {createPiTask}=await import('./pi-agent.bundle.mjs');
              const workspace=new Workspace();let turn=0;const events=[];
              try {
                const task=createPiTask({workspace,network:{options:()=>({enabled:false})},onEvent:e=>events.push(e.type),turn:async(messages)=>{
                  turn++;return turn===1?{text:'<tool_call>{"name":"write","arguments":{"path":"/workspace/pi.txt","text":"verified"}}</tool_call>',reason:'end'}:turn===2?{text:'<tool_call>{"name":"read","arguments":{"path":"/workspace/pi.txt"}}</tool_call>',reason:'end'}:{text:'Checked.',reason:'end'};
                }});
                const taskResult=await task.run('Write and read');const file=await workspace.request('read',{path:'/workspace/pi.txt'});
                const snapshot=await workspace.request('snapshot');
                let rejected=false;try{await workspace.request('restore',{snapshot:JSON.stringify({cwd:'/workspace',dirs:['/workspace'],files:{'/etc/passwd':'bad'}})})}catch{rejected=true}
                const after=await workspace.request('read',{path:'/workspace/pi.txt'});
                return {calls:taskResult.calls,file:file.text,after:after.text,rejected,events};
              }finally{workspace.dispose()}
            }""")
            assert result['calls']==2 and result['file']=='verified' and result['after']=='verified' and result['rejected'],result
            for width in [390,1440]:
                page.set_viewport_size({'width':width,'height':850})
                assert page.locator('#terminal-input').is_visible()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width
            assert not errors,errors
            print('WASM pipelines, confinement, Web off, IndexedDB persistence, real Pi tools, atomic restore, mobile layout and CSP passed.')
            browser.close()
    finally:
        server.shutdown();server.server_close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True)
    check(parser.parse_args().runtime)
