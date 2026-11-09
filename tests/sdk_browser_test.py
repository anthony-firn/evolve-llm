"""Exercise the actual SDK Worker path; optional local GGUF checks real GPU inference."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve, FLAGS


def check(runtime, model=None, chromium=False):
    server=serve(runtime)
    base=f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/'
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(**({} if chromium else {'executable_path':'/usr/bin/google-chrome','args':FLAGS}))
            page=browser.new_page();errors=[];remote=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:remote.append(r.url) if not r.url.startswith(base) else None)
            page.goto(base+'examples/',wait_until='networkidle')
            page.locator('#fixture').click()
            page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('Fixture complete')")
            assert page.locator('#output').inner_text()=='This is a scripted SDK fixture.'
            assert not remote,remote
            page.locator('#fixture').click();page.locator('#cancel').click()
            page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('aborted:')")
            assert page.locator('#fixture').is_enabled()
            print('Actual module Worker, streamed fixture, cancellation and zero external requests passed.',flush=True)
            if model:
                page.evaluate("() => {const i=document.createElement('input');i.type='file';i.id='model-file';document.body.append(i)}")
                page.locator('#model-file').set_input_files(str(model))
                result=page.evaluate("""async () => {
                  const {createEngine}=await import('../sdk.mjs');
                  const engine=createEngine({timeoutMs:120000});
                  try {
                    const adapter=await engine.initialize();
                    await engine.load({files:[document.querySelector('#model-file').files[0]]},{contextSize:512});
                    let text='', count=0;
                    for await(const e of engine.generate([{role:'user',content:'Name three planets in our solar system.'}],{maxTokens:16,temperature:0})) {
                      if(e.type==='token')text+=e.text;
                      if(e.type==='done')count=e.tokens;
                    }
                    const abort=new AbortController();
                    let cancelled=false;
                    try {for await(const e of engine.generate([{role:'user',content:'Count from 1 to 1000.'}],{signal:abort.signal,maxTokens:1000})) {if(e.type==='token')abort.abort()}}
                    catch(e){cancelled=e.code==='aborted'}
                    return {adapter:adapter.description,text,count,cancelled};
                  } finally {engine.dispose()}
                }""")
                assert result['text'].strip() and result['count']>0,result
                assert result['cancelled'],result
                print('Actual GPU SDK run:',result,flush=True)
            assert not errors,errors
            browser.close()
    finally:
        server.shutdown();server.server_close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--chromium',action='store_true',help='Use Playwright Chromium for the no-GPU fixture in CI')
    args=parser.parse_args();check(args.runtime,args.model,args.chromium)
