import { AppPreview } from './preview-client.mjs';
import { Workspace } from './shell-client.mjs';
import { createWebTools, stripToolCalls } from './tools.js';
const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
const button=(label,fn)=>{const e=node('button',label);e.type='button';e.addEventListener('click',fn);return e};
async function database(mode, value) {
  return new Promise((resolve,reject)=>{
    const open=indexedDB.open('evolve-llm-workspace-v1',1);
    open.onupgradeneeded=()=>open.result.createObjectStore('workspace');
    open.onerror=()=>reject(new Error('Browser storage is unavailable.'));
    open.onsuccess=()=>{const db=open.result,tx=db.transaction('workspace',mode==='get'?'readonly':'readwrite'),store=tx.objectStore('workspace');
      const op=mode==='get'?store.get('current'):store.put(value,'current');let result;
      op.onsuccess=()=>result=op.result;tx.oncomplete=()=>{db.close();resolve(result)};tx.onerror=()=>{db.close();reject(new Error('Workspace could not be saved.'))};};
  });
}
export async function mountWorkbench({ ready, turn, lease, getNetwork, setNetwork, modelName, taskTimeLimit = () => 180000 }) {
  const root=node('section',undefined,'workbench');root.id='workbench';root.hidden=true;root.setAttribute('aria-label','WASM terminal and Pi agent');
  const head=node('header'),title=node('div');title.append(node('strong','Terminal'),node('small','The assistant\u2019s files · runs in this page'));
  const tokenLabel=node('label'),tokenLimit=node('select');tokenLimit.id='pi-output-tokens';for(const value of [1024,2048,4096]){const option=node('option',String(value));option.value=value;tokenLimit.append(option)}tokenLimit.value='2048';tokenLabel.append(node('span','Pi tokens'),tokenLimit);
  const webLabel=node('label'),webToggle=node('input');webToggle.type='checkbox';webToggle.id='terminal-web';webLabel.append(webToggle,node('span','Web access'));
  const close=button('Close',()=>{if(!externalTask)cancel();preview.stop();previewPanel.hidden=true;console.classList.remove('preview-expanded');root.hidden=true});
  // The Linux terminal: the Evolve VM (web/vm/) in a GNOME-style window, in place of the console.
  const linuxToggle=button('Browser sandbox',()=>toggleLinux());linuxToggle.id='linux-terminal-toggle';linuxToggle.title='Switch between the Linux terminal (the assistant\u2019s files, git and gh) and the older browser sandbox';
  head.append(title,tokenLabel,webLabel,linuxToggle,close);
  const notice=node('p','Commands and files stay in a browser-only /workspace. Web searches and page URLs go to external services when enabled. This supports a shell subset, not host Bash.','workbench-note');
  const layout=node('div',undefined,'workbench-layout'),aside=node('aside'),files=node('div',undefined,'workbench-files');files.id='workspace-files';
  const actions=node('div',undefined,'workspace-actions');
  const filePicker=node('input');filePicker.type='file';filePicker.multiple=true;filePicker.hidden=true;
  const restorePicker=node('input');restorePicker.type='file';restorePicker.accept='.json,application/json';restorePicker.hidden=true;
  const status=node('p','Loading the WASM shell…','workbench-status');status.id='terminal-status';status.setAttribute('role','status');
  const log=node('div',undefined,'terminal-log');log.id='terminal-log';log.setAttribute('role','log');log.setAttribute('aria-label','Command output');
  const form=node('form',undefined,'terminal-form'),prompt=node('label','/workspace $'),input=node('input');input.id='terminal-input';input.autocomplete='off';input.spellcheck=false;input.maxLength=16384;input.placeholder='help · echo "hello" > note.txt · pi <task>';prompt.htmlFor=input.id;
  const submit=node('button','Run');submit.type='submit';const stop=button('Stop',()=>cancel());stop.disabled=true;
  form.append(prompt,input,submit,stop);
  const previewPanel=node('section',undefined,'app-preview');previewPanel.hidden=true;
  const previewTitle=node('strong','App preview'),previewActions=node('div',undefined,'preview-actions'),previewHost=node('div',undefined,'preview-host');
  previewPanel.append(previewTitle,previewActions,previewHost);
  const preview=new AppPreview({container:previewHost,onResult:result=>{if(result.late)record('Preview runtime error',result.errors.join('\n'),'error')}});let previewPath=null;
  const console=node('div',undefined,'terminal-console');console.append(log,previewPanel,form,status);
  const vmHost=node('div',undefined,'vm-console');vmHost.hidden=true;
  aside.append(node('h2','Workspace files'),actions,files);layout.append(aside,console,vmHost);root.append(head,notice,layout,filePicker,restorePicker);document.body.append(root);
  let vmWindow=null;
  // The Linux terminal needs WebAssembly JSPI (Chrome, Edge); elsewhere the sandbox is the terminal.
  let linuxAvailable=typeof WebAssembly.Suspending==='function';linuxToggle.hidden=!linuxAvailable;
  function showLinux(on){vmHost.hidden=!on;aside.hidden=on;console.hidden=on;notice.hidden=on;layout.classList.toggle('vm-mode',on);linuxToggle.textContent=on?'Browser sandbox':'Linux terminal';}
  async function toggleLinux(){
    const on=vmHost.hidden;showLinux(on);
    if(on&&!vmWindow){
      if(!document.querySelector('link[href$="vm/terminal.css"]')){const l=document.createElement('link');l.rel='stylesheet';l.href=new URL('./vm/terminal.css',import.meta.url).href;document.head.append(l)}
      const [{sharedVm},{TerminalWindow}]=await Promise.all([import('./vm/client.mjs'),import('./vm/terminal.mjs')]);
      const vm=sharedVm();
      // A build without the VM (or one that fails to start): the sandbox is the terminal.
      try{await vm.ready}catch{linuxAvailable=false;linuxToggle.hidden=true;showLinux(false);await boot();input.focus();return}
      vm.onAgent=runLinuxTask;
      vmWindow=new TerminalWindow({vm,container:vmHost});await vmWindow.open();
    }
    if(on)vmWindow?.active?.term.focus();
    else{await boot();input.focus()}
  }
  let initialized=false, initialization=null, task=null, busy=false, externalTask=false, saving=Promise.resolve(), agent=null, abortWeb=new AbortController(), history=[],cursor=0;
  function record(label,text='',className='') {
    const row=node('section',undefined,'terminal-entry '+className),labelNode=node('strong',label),output=node('pre',String(text).slice(0,65536));row.append(labelNode,output);log.append(row);
    while(log.children.length>32)log.firstElementChild.remove();log.scrollTop=log.scrollHeight;return output;
  }
  function checkpoint(snapshot) {
    const data=JSON.parse(snapshot);prompt.textContent=data.cwd+' $';files.replaceChildren();
    for(const path of Object.keys(data.files).sort())files.append(button(path.replace('/workspace/',''),()=>run(async()=>{
      const result=await workspace.request('read',{path},{signal:task.signal});record(path,result.text);
    })));
    if(!Object.keys(data.files).length)files.append(node('p','No files yet.'));
    saving=saving.then(()=>database('put',snapshot)).catch(()=>{status.textContent='Files are available this session, but browser storage could not save them. Export a copy.'});
  }
  const workspace=new Workspace({onCheckpoint:checkpoint});
  const web=createWebTools();
  const network={options:()=>({...getNetwork()}),run:async(name,args,signal)=>web(name,args,{...getNetwork(),signal:signal?AbortSignal.any([signal,abortWeb.signal]):abortWeb.signal})};
  function cancel(){agent?.abort();task?.abort();}
  async function openPreview(path,{signal,smoke=false}={}){
    if(!/\.html?$/i.test(path))throw new Error('Preview supports .html files.');
    const file=await workspace.request('read',{path},{signal});previewPanel.hidden=false;console.classList.add('preview-expanded');previewPath=path;previewTitle.textContent=path+' · '+(smoke?'Smoke test':'Play');
    const result=await preview.open(file.text,{signal,smoke});record('Browser check · '+path,JSON.stringify(result,null,2),result.ok?'':'error');return result;
  }
  previewActions.append(button('Show/hide log',()=>console.classList.toggle('preview-expanded')),button('Full size',()=>root.classList.toggle('preview-wide')),button('Reload',()=>run(()=>openPreview(previewPath,{signal:task.signal}))),button('Test',()=>run(()=>openPreview(previewPath,{signal:task.signal,smoke:true}))),button('Stop preview',()=>{preview.stop();previewPanel.hidden=true;console.classList.remove('preview-expanded')}));
  function setBusy(value){busy=value;submit.disabled=value;stop.disabled=!value;for(const control of actions.querySelectorAll('button'))control.disabled=value;for(const control of files.querySelectorAll('button'))control.disabled=value;}
  async function run(fn,{rethrow=false}={}) {
    if(busy){if(rethrow)throw new Error('Workspace is busy.');return;}externalTask=rethrow;setBusy(true);task=new AbortController();const limit=taskTimeLimit();const deadline=limit?setTimeout(()=>task?.abort(),limit):null;
    try{await boot();const result=await fn();await saving;status.textContent='Ready · files saved in this browser';return result;}
    catch(error){status.textContent=String(error.message||error);record('Error',String(error.message||error),'error');if(rethrow)throw error;}
    finally{clearTimeout(deadline);agent=null;task=null;externalTask=false;setBusy(false);if(!root.hidden)input.focus()}
  }
  async function boot(){
    if(initialized)return;
    if(initialization)return initialization;
    initialization=(async()=>{
      let saved;try{saved=await database('get')}catch{status.textContent='Browser storage is unavailable; this session can still run.'}
      if(saved){await workspace.request('restore',{snapshot:saved});checkpoint(saved)}
      else{await workspace.request('write',{path:'/workspace/README.txt',text:'Browser-only workspace. Run help for commands.\nUse pi followed by a task to ask the loaded model to act through Pi.\nNo host files, processes or credentials are available.\n'});}
      initialized=true;status.textContent='Ready · WASM shell · Pi agent core 1.5.1';
      record('Welcome','Try: printf "pear\\napple\\npear\\n" | sort | uniq\nTry: echo "hello" > note.txt && cat note.txt\nTry: pi Create a compact Snake game in snake.html, browser_test it, then preview it\nTry a bundled game: demo snake\nRun HTML: preview snake.html · Check HTML: test snake.html\nNetwork: search "WebGPU" or curl https://example.com (Web must be on).');
    })().catch(error=>{initialization=null;throw error});return initialization;
  }
  async function command(line){
    console.classList.remove('preview-expanded');
    record(prompt.textContent+' '+line);
    if(line==='demo snake'){const response=await fetch(new URL('./examples/snake.html',import.meta.url));if(!response.ok)throw new Error('Bundled example is unavailable.');await workspace.request('write',{path:'/workspace/snake-example.html',text:await response.text()},{signal:task.signal});record('Bundled example','Loaded the included Snake example, ready for Pi to edit.');await openPreview('/workspace/snake-example.html',{signal:task.signal,smoke:true});return}
    if(/^(preview|test)\s+/.test(line)){const [operation,...parts]=line.split(/\s+/);await openPreview(parts.join(' '),{signal:task.signal,smoke:operation==='test'});return}
    if(/^pi(?:\s|$)/.test(line)){
      const question=line.slice(2).trim();if(!question){record('Pi','Use pi followed by a task. For coding, choose Qwen2.5 Coder 1.5B or larger; tiny chat models often fail tool tasks. Example: pi Create a compact Snake game in snake.html, run browser_test, fix any errors, then preview it. Use a coding model; each task allows 10 model turns and 12 tool calls.');return}
      return performTask(question);
    }else{
      const result=await workspace.request('exec',{command:line,network:network.options()},{signal:task.signal});
      record('exit '+result.code,(result.stdout||'')+(result.stderr||''),result.code?'error':'');
    }
  }
  // `pi TASK` typed in the Linux terminal: Pi works in that directory with the VM's shell, files,
  // git and gh, and writes its progress to the terminal.
  async function runLinuxTask(question,cwd,write,signal){
    if(!ready()){write('\x1b[33mpi: load a tool-capable model first (Models), then run pi again.\x1b[0m\n');return 1}
    if(busy||agent){write('pi: Pi is already working on a task.\n');return 1}
    busy=true;const release=lease();const controller=new AbortController();
    const cancel=()=>{controller.abort();agent?.abort()};signal?.addEventListener('abort',cancel,{once:true});
    const limit=taskTimeLimit();const timer=limit?setTimeout(cancel,Math.max(limit*5,15*60000)):null;
    const dim=s=>`\x1b[2m${s}\x1b[0m`;
    let calls=0,errors=0,turns=0;
    const show=r=>{const t=(r?.content||[]).map(c=>c.text||'').join('\n')||JSON.stringify(r);const lines=t.split('\n');return lines.slice(0,12).join('\n')+(lines.length>12?`\n… ${lines.length-12} more lines`:'')};
    const onEvent=e=>{
      if(e.type==='token')write(dim(e.text));
      else if(e.type==='tool'){calls++;const a=e.args||{};write(`\n\x1b[1;36m● ${e.name}\x1b[0m ${e.name==='bash'?'$ '+a.command:a.path||a.query||a.url||''}\n`)}
      else if(e.type==='tool-result'){if(e.isError)errors++;write((e.isError?'\x1b[31m':'\x1b[2m')+show(e.result)+'\x1b[0m\n')}
      else if(e.type==='retry')write('\x1b[33m(the model\'s tool call was invalid; asking again)\x1b[0m\n');
      else if(e.type==='answer')turns++;
    };
    try{
      const [{createPiTask},{VmWorkspace},{sharedVm}]=await Promise.all([import('./pi-agent.bundle.mjs'),import('./vm/workspace.mjs'),import('./vm/client.mjs')]);
      write(dim(`Pi · ${modelName()} · ${cwd}\n`));
      agent=createPiTask({turn,workspace:new VmWorkspace(sharedVm(),{cwd}),network,maxTokens:Number(tokenLimit.value),modelName:modelName(),onEvent,linux:{cwd}});
      if(controller.signal.aborted)throw new Error('cancelled');
      const result=await agent.run(question);
      const last=result?.messages?.filter(m=>m.role==='assistant').at(-1);
      const answer=stripToolCalls((last?.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n')).trim();
      if(answer)write(`\n${answer}\n`);
      write(dim(`\n${turns} model turns · ${calls} tool calls · ${errors} tool errors\n`));
      return errors&&!answer?1:0;
    }catch(error){
      write(`\n\x1b[31mpi: ${controller.signal.aborted?'stopped':error.message||error}\x1b[0m\n`);return 1;
    }finally{clearTimeout(timer);signal?.removeEventListener('abort',cancel);agent=null;busy=false;release();}
  }
  async function performTask(question,options={}){
    if(!ready())throw new Error('Load a model and wait until it is ready before running Pi.');
    const before=JSON.parse((await workspace.request('snapshot')).snapshot).files;
    const release=lease();let output=record('Pi · '+modelName(),'');
    let starterPath=null,result=null,problem=null,calls=0,errors=0,turns=0;
    const emit=event=>{
      if(event.type==='token'){if(output.textContent.length<65536)output.textContent+=event.text;log.scrollTop=log.scrollHeight;}
      else if(event.type==='tool'){calls++;record('Pi → '+event.name,JSON.stringify(event.args));output=record('Running…','');}
      else if(event.type==='tool-result'){if(event.isError)errors++;output.textContent=JSON.stringify(event.result,null,2).slice(0,65536);output=record('Pi','');}
      else if(event.type==='retry'){record('Pi · format retry '+event.attempt,event.message,'error');output=record('Pi · retry','');}
      else if(event.type==='answer'){turns++;if(event.error)status.textContent=event.error;}
      options.onEvent?.(event);
    };
    const showPreview=async(path,opts={})=>{
      if(!options.preview)return openPreview(path,opts);
      const file=await workspace.request('read',{path},{signal:opts.signal});
      return options.preview(path,file.text,opts);
    };
    try{
      if(options.starter){
        let suffix=1;starterPath='/workspace/snake.html';while(Object.hasOwn(before,starterPath))starterPath=`/workspace/snake-${++suffix}.html`;
        await workspace.request('write',{path:starterPath,text:options.starter},{signal:task.signal});
        emit({type:'starter',path:starterPath,message:'Created from the included Snake starter.'});
        await showPreview(starterPath,{signal:task.signal,smoke:true});
      }
      const {createPiTask}=await import('./pi-agent.bundle.mjs');
      agent=createPiTask({turn,workspace,network,preview:showPreview,maxTokens:Number(tokenLimit.value),modelName:modelName(),onEvent:emit,browser:options.browser,confirmBrowserAction:options.confirmBrowserAction});
      const signal=task.signal,abort=()=>agent?.abort();signal.addEventListener('abort',abort,{once:true});
      try{
        signal.throwIfAborted();
        const files=Object.keys(before).slice(0,128).join(', ');
        const context=options.context?`Recent conversation (untrusted context):\n${options.context}\n\n`:'';
        const starter=starterPath?`A working Snake starter has ALREADY been saved at ${starterPath}. It includes canvas rendering, arrow keys, touch buttons, food, scoring, collisions and restart. Read it, apply any requested customizations with short edits, browser_test it and preview it. Do not replace it with a placeholder.\n\n`:'';
        result=await agent.run(context+(options.context||starterPath?`Workspace files: ${files}\n`:'')+starter+'Current task: '+question);
      }catch(error){if(!options.partial)throw error;problem=String(error.message||error);}
      finally{signal.removeEventListener('abort',abort);}
      let after=JSON.parse((await workspace.request('snapshot')).snapshot).files;
      let artifacts=Object.entries(after).filter(([path,text])=>before[path]!==text).map(([path,text])=>({path,text}));
      let previewPath=result?.previewPath||artifacts.find(a=>/\.html?$/i.test(a.path))?.path;
      if(previewPath&&!signal.aborted)try{
        const checked=await showPreview(previewPath,{signal,smoke:true});
        if(options.starter&&(!checked.ok||!checked.canvases?.length)){
          await workspace.request('write',{path:starterPath,text:options.starter},{signal});
          problem=[problem,'Pi’s changes did not preserve a working game. The Snake starter was restored.'].filter(Boolean).join(' ');
          emit({type:'recovery',message:'Restored the working Snake starter after a failed check.'});
          previewPath=starterPath;
          await showPreview(starterPath,{signal,smoke:true});
          after=JSON.parse((await workspace.request('snapshot')).snapshot).files;
          artifacts=Object.entries(after).filter(([path,text])=>before[path]!==text).map(([path,text])=>({path,text}));
        }
      }catch(error){problem||=String(error.message||error);}
      const removed=Object.keys(before).filter(path=>!Object.hasOwn(after,path));
      const lastAnswer=result?.messages?.filter(m=>m.role==='assistant').at(-1);
      const answer=stripToolCalls((lastAnswer?.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n')).trim();
      const finished={...result,calls:result?.calls??calls,turns:result?.turns??turns,errors:result?.errors??errors,changes:artifacts.length+removed.length,artifacts,removed,starterPath,previewPath,answer,error:problem};
      record(problem?'Pi stopped':finished.calls?'Pi finished':'Pi answered · no tools ran',`${finished.turns} model turns · ${finished.calls} tool calls · ${finished.changes} file changes · ${finished.errors} tool errors`);
      return finished;
    }catch(error){
      if(!options.partial)throw error;
      let after;
      try{after=JSON.parse((await workspace.request('snapshot')).snapshot).files;}
      catch{after=JSON.parse(workspace.checkpoint||'{"files":{}}').files;}
      const artifacts=Object.entries(after).filter(([path,text])=>before[path]!==text).map(([path,text])=>({path,text}));
      const removed=Object.keys(before).filter(path=>!Object.hasOwn(after,path));
      return {calls,turns,errors,changes:artifacts.length+removed.length,artifacts,removed,starterPath,error:String(error.message||error)};
    }finally{release();}
  }
  actions.append(button('Import text',()=>filePicker.click()),button('Export',()=>run(async()=>{
    const {snapshot}=await workspace.request('snapshot');const url=URL.createObjectURL(new Blob([snapshot],{type:'application/json'}));const a=node('a');a.href=url;a.download='evolve-workspace.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  })),button('Restore',()=>restorePicker.click()));
  filePicker.addEventListener('change',()=>run(async()=>{
    for(const file of filePicker.files){if(file.size>4*1024*1024)throw new Error('Each imported text file must fit the 4 MiB workspace.');const text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());await workspace.request('write',{path:'/workspace/'+file.name.replace(/[\\/]/g,'_'),text},{signal:task.signal});record('Imported',file.name)}filePicker.value='';
  }));
  restorePicker.addEventListener('change',()=>run(async()=>{
    const file=restorePicker.files[0];if(!file)return;if(file.size>26*1024*1024)throw new Error('Workspace snapshot is too large.');
    if(!confirm('Replace the current browser workspace with this snapshot?'))return;
    const snapshot=await file.text();await workspace.request('restore',{snapshot},{signal:task.signal});checkpoint(snapshot);restorePicker.value='';record('Restored','Workspace snapshot loaded.');
  }));
  form.addEventListener('submit',e=>{e.preventDefault();const line=input.value.trim();if(!line||busy)return;input.value='';history.push(line);history=history.slice(-50);cursor=history.length;run(()=>command(line))});
  input.addEventListener('keydown',e=>{if(e.key==='ArrowUp'||e.key==='ArrowDown'){e.preventDefault();cursor=Math.max(0,Math.min(history.length,cursor+(e.key==='ArrowUp'?-1:1)));input.value=history[cursor]||''}if(e.ctrlKey&&e.key.toLowerCase()==='c'){e.preventDefault();cancel()}});
  webToggle.addEventListener('change',()=>setNetwork(webToggle.checked));
  return {runTask:(question,options)=>run(()=>performTask(question,options),{rethrow:true}),cancel,
    /** The sandbox's /workspace, saved in this browser: the assistant's files where the Linux VM cannot run. */
    async sandbox(){await boot();return workspace},async open(){root.hidden=false;webToggle.checked=getNetwork().enabled;if(linuxAvailable&&!vmWindow&&vmHost.hidden){await toggleLinux();return}if(vmHost.hidden){await boot();input.focus()}else vmWindow?.active?.term.focus()},setNetwork(){if(busy)cancel();abortWeb.abort();abortWeb=new AbortController();webToggle.checked=getNetwork().enabled},dispose(){cancel();workspace.dispose();preview.dispose();root.remove()}};
}
