import { AppPreview } from './preview-client.mjs';
import { Workspace } from './shell-client.mjs';
import { createWebTools } from './tools.js';
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
export async function mountWorkbench({ ready, turn, lease, getNetwork, setNetwork, modelName }) {
  const root=node('section',undefined,'workbench');root.id='workbench';root.hidden=true;root.setAttribute('aria-label','WASM terminal and Pi agent');
  const head=node('header'),title=node('div');title.append(node('strong','Terminal & Pi'),node('small','WASM workspace · local model'));
  const tokenLabel=node('label'),tokenLimit=node('select');tokenLimit.id='pi-output-tokens';for(const value of [1024,2048,4096]){const option=node('option',String(value));option.value=value;tokenLimit.append(option)}tokenLimit.value='2048';tokenLabel.append(node('span','Pi tokens'),tokenLimit);
  const webLabel=node('label'),webToggle=node('input');webToggle.type='checkbox';webToggle.id='terminal-web';webLabel.append(webToggle,node('span','Web access'));
  const close=button('Close',()=>{cancel();preview.stop();previewPanel.hidden=true;console.classList.remove('preview-expanded');root.hidden=true});head.append(title,tokenLabel,webLabel,close);
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
  aside.append(node('h2','Workspace files'),actions,files);layout.append(aside,console);root.append(head,notice,layout,filePicker,restorePicker);document.body.append(root);
  let initialized=false, initialization=null, task=null, busy=false, saving=Promise.resolve(), agent=null, abortWeb=new AbortController(), history=[],cursor=0;
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
  async function run(fn) {
    if(busy)return;setBusy(true);task=new AbortController();const deadline=setTimeout(()=>task?.abort(),180000);
    try{await boot();await fn();await saving;status.textContent='Ready · files saved in this browser';}
    catch(error){status.textContent=String(error.message||error);record('Error',String(error.message||error),'error')}
    finally{clearTimeout(deadline);agent=null;task=null;setBusy(false);input.focus()}
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
    record(prompt.textContent+' '+line);
    if(line==='demo snake'){const response=await fetch(new URL('./examples/snake.html',import.meta.url));if(!response.ok)throw new Error('Bundled example is unavailable.');await workspace.request('write',{path:'/workspace/snake-example.html',text:await response.text()},{signal:task.signal});record('Bundled example','Loaded the included Snake example, ready for Pi to edit.');await openPreview('/workspace/snake-example.html',{signal:task.signal,smoke:true});return}
    if(/^(preview|test)\s+/.test(line)){const [operation,...parts]=line.split(/\s+/);await openPreview(parts.join(' '),{signal:task.signal,smoke:operation==='test'});return}
    if(/^pi(?:\s|$)/.test(line)){
      const question=line.slice(2).trim();if(!question){record('Pi','Use pi followed by a task. Choose a tool-capable model such as Qwen. Example: pi Create a compact Snake game in snake.html, run browser_test, fix any errors, then preview it. Use a coding model; each task allows 6 model turns and 12 tool calls.');return}
      if(!ready())throw new Error('Load a model and wait until it is ready before running Pi. Shell commands work without a model.');
      const release=lease();let output=record('Pi · '+modelName(),'');
      try{
        const {createPiTask}=await import('./pi-agent.bundle.mjs');
        agent=createPiTask({turn,workspace,network,preview:openPreview,maxTokens:Number(tokenLimit.value),modelName:modelName(),onEvent:event=>{
          if(event.type==='token'){if(output.textContent.length<65536)output.textContent+=event.text;log.scrollTop=log.scrollHeight}
          else if(event.type==='tool'){record('Pi → '+event.name,JSON.stringify(event.args));output=record('Running…','')}
          else if(event.type==='tool-result'){output.textContent=JSON.stringify(event.result,null,2).slice(0,65536);output=record('Pi','')}
          else if(event.type==='answer'&&event.error)output.textContent=event.error;
        }});
        const signal=task.signal,abort=()=>agent?.abort();signal.addEventListener('abort',abort,{once:true});
        try{const result=await agent.run(question);record(result.calls?'Pi finished':'Pi answered · no tools ran',`${result.turns} model turns · ${result.calls} tool calls`);if(result.previewPath)await openPreview(result.previewPath,{signal,smoke:true});status.textContent=`Pi finished · ${result.turns} model turns · ${result.calls} tool calls`;}
        finally{signal.removeEventListener('abort',abort)}
      }finally{release()}
    }else{
      const result=await workspace.request('exec',{command:line,network:network.options()},{signal:task.signal});
      record('exit '+result.code,(result.stdout||'')+(result.stderr||''),result.code?'error':'');
    }
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
  return {async open(){root.hidden=false;webToggle.checked=getNetwork().enabled;await boot();input.focus()},setNetwork(){if(busy)cancel();abortWeb.abort();abortWeb=new AbortController();webToggle.checked=getNetwork().enabled},dispose(){cancel();workspace.dispose();preview.dispose();root.remove()}};
}
