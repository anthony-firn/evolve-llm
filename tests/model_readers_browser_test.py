"""Real OPFS lock contention: one worker owns a write lock, another reads saved weights."""
import argparse
import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_test import serve
ROOT=Path(__file__).resolve().parents[1]

def check():
    with tempfile.TemporaryDirectory(prefix='model-reader-test-') as name:
        runtime=Path(name)
        for filename in ['model-readers.mjs']:
            shutil.copy2(ROOT/'web'/filename,runtime/filename)
        shutil.copy2(ROOT/'tests/model-reader-fixture.mjs',runtime/'fixture.mjs')
        (runtime/'index.html').write_text('<!doctype html><title>Saved model reader regression</title>')
        server=serve(runtime)
        try:
            with sync_playwright() as p:
                for browser_type in [p.chromium,p.firefox]:
                    browser=browser_type.launch();page=browser.new_page()
                    page.goto(f'http://127.0.0.1:{server.server_address[1]}/play/evolve-llm/')
                    result=page.evaluate('''async()=>{
                     const a=new Worker('./fixture.mjs',{type:'module'}),b=new Worker('./fixture.mjs',{type:'module'});
                     const request=(worker,type)=>new Promise((resolve,reject)=>{worker.onmessage=e=>resolve(e.data);worker.onerror=reject;worker.postMessage({type})});
                     try {
                      const lock=await request(a,'lock');
                      const locked=await request(b,'read');
                      const unlock=await request(a,'unlock');
                      const unlocked=await request(b,'read');return {lock,locked,unlock,unlocked};
                     }finally{a.terminate();b.terminate()}
                    }''')
                    assert result['lock']=={'ok':True},result
                    for state in ['locked','unlocked']:
                        assert result[state]=={'bytes':[3,2],'size':4},(browser_type.name,result)
                    print(browser_type.name+': reads succeed during and after an exclusive OPFS lock.',flush=True)
                    browser.close()
        finally:
            server.shutdown();server.server_close()
if __name__=='__main__':check()
