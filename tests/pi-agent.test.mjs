import test from 'node:test';
import assert from 'node:assert/strict';
import { createPiTask, askedSteps, namedRepos, wantsSources, wantsPlacement, sedWithoutInPlace, wantsMore, sedInPlaceFile, changedLines, unreachable } from '../web/pi-agent.mjs';

test('real Pi agent loop invokes sandbox tools and feeds their results back to the local transport',async()=>{
 const files=new Map(),events=[];let round=0;
 const workspace={async request(name,args){if(name==='write'){files.set(args.path,args.text);return {ok:true}}if(name==='read')return {text:files.get(args.path)};throw Error('Unexpected operation')}};
 const task=createPiTask({workspace,network:{options:()=>({enabled:false}),run:()=>{throw Error('No network permission')}},onEvent:e=>events.push(e),turn:async(messages,schemas)=>{
  assert(schemas.some(t=>t.function.name==='bash'));
  assert.equal(messages[0].role,'system');assert.match(messages[0].content,/untrusted data/);
  round++;
  if(round===1)return {text:'<tool_call>{"name":"write","arguments":{"path":"/workspace/note.txt","text":"verified"}}</tool_call>',reason:'end',n:10};
  if(round===2)return {text:'<tool_call>{"name":"read","arguments":{"path":"/workspace/note.txt"}}</tool_call>',reason:'end',n:10};
  assert(messages.some(m=>m.role==='tool'&&m.content.includes('verified')));
  return {text:'The file was written and read back.',reason:'end',n:10};
 }});
 const result=await task.run('Create a note and check it');
 assert.equal(result.calls,2);assert.equal(result.turns,3);assert.equal(files.get('/workspace/note.txt'),'verified');assert.equal(events.filter(e=>e.type==='tool').length,2);
});

test('Pi task turn budget terminates a model that repeatedly requests tools',async()=>{
 let executed=0;
 const task=createPiTask({workspace:{request:async()=>{executed++;return {stdout:'x',code:0}}},network:{options:()=>({enabled:false})},turn:async()=>({text:'<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',reason:'end',n:10})});
 await assert.rejects(task.run('Loop'),/10-turn limit/);assert.equal(executed,10);
});

test('truncated tool calls are not executed and malformed arguments do not cross the tool boundary',async()=>{
 let n=0,executed=0;
 const task=createPiTask({workspace:{request:async()=>{executed++;return{ok:true}}},network:{options:()=>({enabled:false})},turn:async()=>{
  n++;return n===1?{text:'<tool_call>{"name":"write","arguments":{"path":"/workspace/a","text":"x"}}</tool_call>',reason:'length',n:1}:{text:'Stopped at the output limit.',reason:'end',n:1};
 }});
 // The cut-off call never runs; the model is asked once more (see the cut-off test below).
 await task.run('Write');assert.equal(executed,0);assert.equal(n,2);
});

test('invalid tool JSON cannot be reported as successful execution',async()=>{
 const task=createPiTask({workspace:{request:async()=>{throw Error('Must not execute')}},network:{options:()=>({enabled:false})},turn:async()=>({text:'<tool_call>{{"name":"write","arguments":{}}</tool_call>',reason:'end'})});
 await assert.rejects(task.run('Create a file'),/invalid or incomplete/);
});
test('Pi can smoke-test and preview HTML and passes the selected output budget',async()=>{
 let n=0;const calls=[];
 const task=createPiTask({maxTokens:4096,workspace:{request:async()=>({ok:true})},network:{options:()=>({enabled:false})},preview:async(path,opts)=>{calls.push([path,opts.smoke]);return{ok:true,errors:[]}},turn:async(_,schemas,opts)=>{
 assert.equal(opts.maxTokens,4096);assert(schemas.some(x=>x.function.name==='browser_test'));n++;
 return {text:n===1?'<tool_call>{"name":"browser_test","arguments":{"path":"/workspace/game.html"}}</tool_call>':n===2?'<tool_call>{"name":"preview","arguments":{"path":"/workspace/game.html"}}</tool_call>':'Opened.',reason:'end'};
 }});const completed=await task.run('Test and play');assert.equal(completed.messages.find(m=>m.role==='toolResult').isError,false);assert.deepEqual(calls,[['/workspace/game.html',true],['/workspace/game.html',false]]);
});

test('malformed output gets bounded model feedback and can recover without executing invalid replies',async()=>{
 let n=0,executed=0;const events=[];
 const task=createPiTask({onEvent:e=>events.push(e),workspace:{request:async(name)=>{executed++;assert.equal(name,'read');return{text:'actual file'}}},network:{options:()=>({enabled:false})},turn:async(messages)=>{
  n++;if(n===1)return {text:'<tool_call>{{"name":"edit","arguments":{}}</tool_call>',reason:'end'};
  if(n===2){assert.match(messages.at(-1).content,/SINGLE opening brace/);return{text:'<tool_call>{"name":"read","arguments":{"path":"/workspace/game.html"}}</tool_call>',reason:'end'}}
  return {text:'Read the file.',reason:'end'};
 }});const result=await task.run('Read game.html');assert.equal(result.repairs,1);assert.equal(result.turns,3);assert.equal(executed,1);assert.equal(events.filter(e=>e.type==='retry').length,1);
});
test('edits require a prior read and an exact nonempty string; a failed edit shows the file for correction',async()=>{
 let n=0;const operations=[];
 const task=createPiTask({workspace:{request:async(name,args)=>{operations.push(name);return name==='read'?{text:'background:red;'}:{ok:true}}},network:{options:()=>({enabled:false})},turn:async(messages)=>{
 n++;const call=n===1?{name:'edit',arguments:{path:'/workspace/game.html',oldText:'',newText:'blue'}}:n===2?{name:'read',arguments:{path:'/workspace/game.html'}}:n===3?{name:'edit',arguments:{path:'/workspace/game.html',oldText:'background:red;',newText:'background:blue;'}}:null;
 if(n===2)assert(messages.some(m=>m.role==='tool'&&m.content.includes('oldText is empty')&&m.content.includes('background:red;')));
 return {text:call?'<tool_call>'+JSON.stringify(call)+'</tool_call>':'Changed.',reason:'end'};
 }});const result=await task.run('Make the file blue');assert.deepEqual(operations,['read','read','edit']);assert.equal(result.changes,1);assert.equal(result.errors,1);assert.equal(result.lastToolFailed,false);
});

test('a failed tool followed by a confident answer still reports zero changes and a tool error',async()=>{
 let n=0;
 const task=createPiTask({workspace:{request:async()=>{throw Error('File not found')}},network:{options:()=>({enabled:false})},turn:async(_,schemas)=>{
  assert(!schemas.some(t=>['open_url','web_search'].includes(t.function.name)));
  return {text:++n===1?'<tool_call>{"name":"read","arguments":{"path":"/workspace/missing.html"}}</tool_call>':'Done; the game is blue.',reason:'end'};
 }});
 const result=await task.run('Change the game');
 assert.equal(result.changes,0);assert.equal(result.errors,1);assert.equal(result.lastToolFailed,true);assert.equal(result.previewPath,null);
});

test('a complete fenced JSON call executes without a format retry',async()=>{
 let n=0;
 const task=createPiTask({workspace:{request:async()=>({text:'Actual file'})},network:{options:()=>({enabled:false})},turn:async()=>({
  text:++n===1?'```json\n{"name":"read","arguments":{"path":"/workspace/a.txt"}}\n```':'Read the file.',reason:'end'
 })});
 const result=await task.run('Read a.txt');assert.equal(result.calls,1);assert.equal(result.repairs,0);
});

test('a JSON-only batch reads, edits, and checks the actual file sequentially',async()=>{
 let n=0,source='background:red';const operations=[];
 const task=createPiTask({workspace:{request:async(name,args)=>{
  operations.push(name);
  if(name==='read')return {text:source};
  assert.equal(name,'edit');assert.equal(args.oldText,source);source=args.newText;return {ok:true};
 }},network:{options:()=>({enabled:false})},preview:async()=>{operations.push('browser_test');assert.equal(source,'background:blue');return{ok:true}},turn:async()=>({text:++n===1?[
  {name:'read',arguments:{path:'/workspace/game.html'}},
  {name:'edit',arguments:{path:'/workspace/game.html',oldText:'background:red',newText:'background:blue'}},
  {name:'browser_test',arguments:{path:'/workspace/game.html'}},
 ].map(call=>'```json\n'+JSON.stringify(call)+'\n```').join('\n'):'Checked.',reason:'end'})});
 const result=await task.run('Make the background blue');
 assert.deepEqual(operations,['read','edit','browser_test']);assert.equal(result.changes,1);assert.equal(result.errors,0);
});

test('a successful final preview stops inference and calls after preview are not executed',async()=>{
 let generated=0,opened=0;
 const task=createPiTask({workspace:{request:async()=>{throw Error('Must not write after preview')}},network:{options:()=>({enabled:false})},preview:async()=>{opened++;return{ok:true}},turn:async()=>{
  generated++;assert.equal(generated,1,'Must not ask the model for another turn after preview');
  return {text:[{name:'preview',arguments:{path:'/workspace/game.html'}},{name:'write',arguments:{path:'/workspace/game.html',text:'damaged'}}].map(call=>'<tool_call>'+JSON.stringify(call)+'</tool_call>').join('\n'),reason:'end'};
 }});
 const result=await task.run('Preview the game');assert.equal(opened,1);assert.equal(result.turns,1);assert.equal(result.calls,1);
});

test('a failed preview allows the model to continue repairing the app',async()=>{
 let n=0,opened=0;
 const task=createPiTask({workspace:{},network:{options:()=>({enabled:false})},preview:async()=>({ok:++opened===2}),turn:async()=>{
  n++;return{text:'<tool_call>{"name":"preview","arguments":{"path":"/workspace/game.html"}}</tool_call>',reason:'end'};
 }});
 const result=await task.run('Preview the game');assert.equal(n,2);assert.equal(result.errors,1);assert.equal(result.lastToolFailed,false);
});

test('a failed edit must be corrected before final preview can finish the task',async()=>{
 let n=0,opened=0,edits=0;
 // The first edit misses (as when oldText is not in the file); the preview after it must refuse.
 const task=createPiTask({workspace:{request:async(name)=>{if(name==='read')return{text:'red'};if(name==='edit'&&++edits===1)throw new Error('oldText is not in /workspace/game.html');return{ok:true}}},network:{options:()=>({enabled:false})},preview:async()=>{opened++;return{ok:true}},turn:async()=>{
  n++;
  const calls=n===1?[{name:'edit',arguments:{path:'/workspace/game.html',oldText:'red',newText:'blue'}},{name:'preview',arguments:{path:'/workspace/game.html'}}]:[
   {name:'read',arguments:{path:'/workspace/game.html'}},{name:'edit',arguments:{path:'/workspace/game.html',oldText:'red',newText:'blue'}},{name:'preview',arguments:{path:'/workspace/game.html'}}
  ];
  assert(n<=2,'Model should stop after corrected final preview');
  return{text:calls.map(call=>'<tool_call>'+JSON.stringify(call)+'</tool_call>').join('\n'),reason:'end'};
 }});
 const result=await task.run('Edit and preview');assert.equal(n,2);assert.equal(opened,1);assert.equal(result.changes,1);assert.equal(result.errors,2);
});

// ---- Browser tools (shared live page, browser/API.md section 4) ----
import { normalizeBrowserUrl } from '../web/pi-agent.mjs';
const LOGIN = '# Login\n[1] textbox "acct" value=""  [2] password "pw" (value hidden)\n[3] button "login"';
function fakeBrowser(pages = { 'https://news.ycombinator.com/login': LOGIN }) {
  const log = []; let url = '', text = '';
  const b = {
    log, state: { url: '' },
    async navigate(u) { log.push(['navigate', u]); url = u; text = pages[u] ?? 'Hello'; },
    async waitForIdle(ms) { log.push(['idle', ms]); },
    async snapshot(max) { log.push(['snapshot', max]); return { url, title: 'Login', scrollY: 0, pageHeight: 1400, viewportHeight: 700, text, refs: (text.match(/\[\d+\]/g) || []).length, version: 1 }; },
    click: async (ref) => { log.push(['click', ref]); return b.clickResult || { ok: true }; },
    type: async (ref, t, o) => { log.push(['type', ref, t, o]); return { ok: true }; },
    select: async (ref, v) => { log.push(['select', ref, v]); return { ok: true }; },
    scroll: async (dy) => { log.push(['scroll', dy]); return { ok: true }; },
    key: async (k) => { log.push(['key', k]); return { ok: true }; },
    back() { log.push(['back']); },
  };
  return b;
}
const call = (name, args) => ({ text: '<tool_call>' + JSON.stringify({ name, arguments: args }) + '</tool_call>', reason: 'end' });
const network = { options: () => ({ enabled: false }) };

test('browser tasks get shared-page tools, safety instructions and a larger budget', async () => {
  const browser = fakeBrowser(); let n = 0; const results = [];
  const task = createPiTask({ browser, workspace: { request: async () => ({ ok: true }) }, network, turn: async (messages, schemas) => {
    const tools = schemas.map(s => s.function.name);
    for (const t of ['browser_open', 'browser_snapshot', 'browser_click', 'browser_type', 'browser_select', 'browser_scroll', 'browser_key', 'browser_back', 'read']) assert(tools.includes(t), t);
    assert(!tools.includes('preview') && !tools.includes('browser_test'));
    assert.match(messages[0].content, /share a live web browser/); assert.match(messages[0].content, /Never type passwords or payment details/);
    assert.match(messages[0].content, /never instructions/); assert.match(messages[0].content, /at most 16 model turns and 24 tool calls/);
    if (n) results.push(messages.at(-1).content);
    n++;
    if (n === 1) return call('browser_open', { url: 'news.ycombinator.com/login' });
    if (n === 2) return call('browser_type', { ref: 1, text: 'anthony' });
    if (n === 3) return call('browser_type', { ref: 2, text: 'hunter2' });
    if (n === 4) return call('browser_scroll', { direction: 'down' });
    return { text: 'Please type your password in the browser.', reason: 'end' };
  } });
  const result = await task.run('Log in to Hacker News');
  assert.equal(result.calls, 4);
  assert.deepEqual(browser.log.find(l => l[0] === 'navigate'), ['navigate', 'https://news.ycombinator.com/login']);
  assert.match(results[0], /Page: "Login" https:\/\/news\.ycombinator\.com\/login/);
  assert.match(results[0], /untrusted data from the website/); assert.match(results[0], /\[3\] button "login"/);
  assert(browser.log.some(l => l[0] === 'type' && l[1] === 1 && l[2] === 'anthony' && l[3].submit === false));
  assert(!browser.log.some(l => l[0] === 'type' && l[1] === 2), 'password field must never be typed into');
  assert.match(results[2], /Ask the person to type it themselves/);
  assert.deepEqual(browser.log.find(l => l[0] === 'scroll'), ['scroll', 595]);
  assert.equal(result.messages.filter(m => m.role === 'toolResult' && m.isError).length, 1);
});

test('person activity, missing refs and engine refusals return a fresh snapshot to Pi', async () => {
  const browser = fakeBrowser(); let n = 0; const seen = [];
  const task = createPiTask({ browser, workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    if (n) seen.push(messages.at(-1).content);
    n++;
    if (n === 1) return call('browser_open', { url: 'https://news.ycombinator.com/login' });
    if (n === 2) { browser.clickResult = { ok: false, error: 'stale' }; return call('browser_click', { ref: 3 }); }
    if (n === 3) return call('browser_click', { ref: 9 });
    if (n === 4) { browser.clickResult = { ok: false, error: 'password' }; return call('browser_click', { ref: 1 }); }
    return { text: 'Done.', reason: 'end' };
  } });
  await task.run('Click login');
  assert.match(seen[1], /did not run: the person is using the page right now/); assert.match(seen[1], /\[3\] button "login"/);
  assert.match(seen[2], /no element \[9\]/);
  assert.match(seen[3], /password, one-time code or payment details/);
  assert.equal(browser.log.filter(l => l[0] === 'click').length, 2);
});

test('risky clicks need the person: refused without approval, done with it', async () => {
  const pages = { 'https://shop.example/cart': '[1] button "Place order"  [2] link "Continue shopping" -> /\n[3] link "Remove item" -> /cart/rm' };
  for (const approve of [undefined, false, true]) {
    const browser = fakeBrowser(pages); let n = 0; const asked = []; const seen = [];
    const task = createPiTask({ browser, workspace: { request: async () => ({}) }, network,
      confirmBrowserAction: approve === undefined ? undefined : async (text) => { asked.push(text); return approve; },
      turn: async (messages) => {
        if (n) seen.push(messages.at(-1).content);
        n++;
        if (n === 1) return call('browser_open', { url: 'https://shop.example/cart' });
        if (n === 2) return call('browser_click', { ref: 1 });
        if (n === 3) return call('browser_click', { ref: 2 });
        if (n === 4) return call('browser_click', { ref: 3 });
        return { text: 'Asked the person.', reason: 'end' };
      } });
    await task.run('Buy it');
    const clicks = browser.log.filter(l => l[0] === 'click').map(l => l[1]);
    if (approve) { assert.deepEqual(clicks, [1, 2, 3]); assert.equal(asked.length, 2); assert.match(asked[0], /Place order/); }
    else { assert.deepEqual(clicks, [2]); assert.match(seen[1], /needs the person's approval/); assert.match(seen[3], /needs the person's approval/); }
  }
});

test('the browser accessor is resolved lazily, once, and older snapshots are compacted', async () => {
  let made = 0; const browser = fakeBrowser(); let n = 0; let last;
  const task = createPiTask({ browser: async () => { made++; return browser; }, workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    n++;
    if (n === 1) { assert.equal(made, 0); return call('browser_open', { url: 'https://news.ycombinator.com/login' }); }
    if (n === 2) return call('browser_snapshot', {});
    if (n === 3) return call('browser_key', { key: 'Tab' });
    last = messages; return { text: 'ok', reason: 'end' };
  } });
  await task.run('Look');
  assert.equal(made, 1);
  const tools = last.filter(m => m.role === 'tool');
  assert.equal(tools.length, 3);
  assert.match(tools[0].content, /older page snapshot omitted/); assert.doesNotMatch(tools[0].content, /button "login"/);
  assert.match(tools[2].content, /Press Tab: done/); assert.match(tools[2].content, /button "login"/);
});

test('browser tasks stop at their own turn budget; other tasks keep theirs', async () => {
  const task = createPiTask({ browser: fakeBrowser(), workspace: { request: async () => ({}) }, network, turn: async () => call('browser_snapshot', {}) });
  await assert.rejects(task.run('Loop'), /16-turn limit/);
});

test('only http and https pages can be opened, without embedded credentials', () => {
  assert.equal(normalizeBrowserUrl('news.ycombinator.com'), 'https://news.ycombinator.com/');
  assert.equal(normalizeBrowserUrl('http://localhost:8080/x'), 'http://localhost:8080/x');
  assert.equal(normalizeBrowserUrl('localhost:3000'), 'https://localhost:3000/');
  for (const bad of ['javascript:alert(1)', 'file:///etc/passwd', 'data:text/html,hi', 'https://user:pw@example.com/', ''])
    assert.throws(() => normalizeBrowserUrl(bad), bad);
});

test('Pi runs tools a model calls in its own syntax (Kimi K3.1) and returns structured history',async()=>{
 const files=new Map();let round=0;let history=null;
 const workspace={async request(name,args){if(name==='write'){files.set(args.path,args.text);return {ok:true}}throw Error('Unexpected operation')}};
 const k3=(name,args)=>`<|open|>response<|sep|>Writing it.<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="${name}" index="1"<|sep|>`+Object.entries(args).map(([k,v])=>`<|open|>argument key="${k}" type="string"<|sep|>${v}<|close|>argument<|sep|>`).join('')+'<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>';
 const task=createPiTask({workspace,network:{options:()=>({enabled:false})},turn:async(messages)=>{
  round++;
  if(round===1)return {text:k3('write',{path:'/workspace/a.txt',text:'<b>kimi</b>'}),reason:'end',n:40};
  history=messages;
  return {text:'<|open|>response<|sep|>Done.<|close|>response<|sep|><|close|>message<|sep|>',reason:'end',n:5};
 }});
 const result=await task.run('Write a file');
 assert.equal(files.get('/workspace/a.txt'),'<b>kimi</b>');assert.equal(result.calls,1);
 const call=history.find(m=>m.role==='assistant'&&m.tool_calls);
 assert.deepEqual(call.tool_calls[0].function,{name:'write',arguments:{path:'/workspace/a.txt',text:'<b>kimi</b>'}});
 assert.equal(call.content,'Writing it.');
 const res=history.find(m=>m.role==='tool');
 assert.equal(res.name,'write');assert.equal(res.tool_call_id,call.tool_calls[0].id);assert.match(call.tool_calls[0].id,/^functions\.write:\d+$/);
});

test('an edit needs no read when oldText matches once; a miss in a long file shows the closest lines',async()=>{
 const long=Array.from({length:300},(_,i)=>i===150?'python3 -m http.server 8080 --directory dist':`line ${i} of a long README with enough words to pass three thousand characters`).join('\n');
 let n=0,edited=null;const results=[];
 const task=createPiTask({workspace:{request:async(name,args)=>{if(name==='read')return{text:long};if(name==='edit'){if(!long.includes(args.oldText))throw new Error(`oldText is not in ${args.path}; read the file again and copy the exact text`);edited=args;return{ok:true}}return{}}},network:{options:()=>({enabled:false})},
  onEvent:e=>{if(e.type==='tool-result')results.push(e.result.content[0].text)},turn:async()=>{
  n++;const call=n===1?{name:'edit',arguments:{path:'README.md',oldText:'python3 -m http.server 8000',newText:'x'}}:n===2?{name:'edit',arguments:{path:'README.md',oldText:'python3 -m http.server 8080 --directory dist',newText:'y'}}:null;
  return{text:call?'<tool_call>'+JSON.stringify(call)+'</tool_call>':'Added the line.',reason:'end'};
 }});
 const result=await task.run('Add a line to README.md');
 assert.match(results[0],/oldText is not in README\.md, so nothing was changed\. The closest lines of README\.md \(lines 147-155\)/);
 assert.match(results[0],/151: python3 -m http\.server 8080 --directory dist/);
 assert.equal(edited.newText,'y');assert.equal(result.changes,1);
});

test('an empty reply after tool results is asked to continue, and a long read says it was cut',async()=>{
 let n=0;const seen=[];
 const task=createPiTask({linux:{cwd:'/home/user'},workspace:{request:async(name)=>name==='read'?{text:'x'.repeat(20000)}:{}},network:{options:()=>({enabled:false})},turn:async(messages)=>{
  n++;seen.push(messages.at(-1));
  if(n===1)return{text:'<tool_call>'+JSON.stringify({name:'read',arguments:{path:'README.md'}})+'</tool_call>',reason:'end'};
  if(n===2)return{text:'',reason:'end'};
  return{text:'Read it; nothing else to do.',reason:'end'};
 }});
 const result=await task.run('Read README.md');
 assert.match(seen[1].content,/more characters not shown\. To find a line, run: grep -n/);
 assert.match(seen[2].content,/Your reply was empty\. Continue the task/);
 assert.equal(result.repairs,1);assert.equal(n,3);
});

test('write does not replace an existing file the task has not read; it may after a read',async()=>{
 const files=new Map([['/workspace/calc.py','def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n']]);let n=0;const results=[];
 const task=createPiTask({workspace:{request:async(name,args)=>{if(name==='read'){if(!files.has(args.path))throw Error('no such file');return{text:files.get(args.path)}}if(name==='write'){files.set(args.path,args.text);return{ok:true}}return{}}},network:{options:()=>({enabled:false})},
  onEvent:e=>{if(e.type==='tool-result')results.push([e.isError,e.result.content[0].text])},turn:async()=>{
  n++;const call=n===1?{name:'write',arguments:{path:'/workspace/calc.py',text:'def add(a, b):\n    return a + b\n'}}:n===2?{name:'write',arguments:{path:'/workspace/new.txt',text:'fresh'}}:n===3?{name:'read',arguments:{path:'/workspace/calc.py'}}:n===4?{name:'write',arguments:{path:'/workspace/calc.py',text:'whole new file'}}:null;
  return{text:call?'<tool_call>'+JSON.stringify(call)+'</tool_call>':'Done.',reason:'end'};
 }});
 await task.run('Fix calc.py');
 assert.equal(results[0][0],true);assert.match(results[0][1],/calc\.py already exists, so nothing was written\. To change part of it, use edit/);
 assert.equal(results[1][0],false);assert.equal(files.get('/workspace/new.txt'),'fresh');
 assert.equal(results[3][0],false);assert.equal(files.get('/workspace/calc.py'),'whole new file');
});

test('the same failing command a third time stops the task with the reason',async()=>{
 let n=0;
 const task=createPiTask({linux:{cwd:'/home/user'},workspace:{cwd:'/home/user',request:async(name)=>name==='exec'?{code:1,stdout:'',stderr:'boom',cwd:'/home/user'}:{}},network:{options:()=>({enabled:false})},turn:async()=>{
  n++;return{text:'<tool_call>'+JSON.stringify({name:'bash',arguments:{command:'make it work'}})+'</tool_call>',reason:'end'};
 }});
 await assert.rejects(task.run('Do it'),/The same command failed 3 times, so the task stopped: make it work/);
 assert.equal(n,3);
});

test('a request about failing CI says where the reason is',async()=>{
 const seen=[];const vm={read:async()=>({ok:false}),exec:async()=>({code:0,stdout:'',stderr:''})};
 const task=createPiTask({linux:{cwd:'/home/user'},workspace:{cwd:'/home/user',vm,path:p=>p,request:async()=>({})},network:{options:()=>({enabled:false})},turn:async messages=>{seen.push(messages.at(-1).content);return{text:'Done.',reason:'end'}}});
 await task.run('CI is failing for the GitHub repository demo/app. Fix it.');
 assert.match(seen[0],/To find out why CI fails, run gh run list, then gh run view ID/);
 await task.run('Add a line to README.md in demo/app.');
 assert.doesNotMatch(seen.at(-1),/find out why CI fails/);
 assert.doesNotMatch(seen.at(-1),/To put a line after another line/);
 await task.run('In README.md of demo/app, right after the line "npm test", add the line "npm run lint".');
 assert.match(seen.at(-1),/To put a line after another line, use edit with oldText "that line" and newText "that line\\nnew line"/);
});

test('askedSteps reads commit, push and pull request from a request, and "do not"',()=>{
 assert.deepEqual(askedSteps('Fix it, commit it on a new branch, push it and open a pull request.'),{commit:true,push:true,pr:true});
 assert.deepEqual(askedSteps('Fix it on a new branch named fix-ci, test the fix, push it and open a pull request.'),{commit:true,push:true,pr:true});
 assert.deepEqual(askedSteps('Commit it on a new branch named docs. Do not push it.'),{commit:true,push:false,pr:false});
 assert.deepEqual(askedSteps('Merge pull request #1 in demo/app.'),{commit:false,push:false,pr:false});
 assert.deepEqual(askedSteps('Find the commit that broke the build.'),{commit:false,push:false,pr:false});
 assert.deepEqual(askedSteps('Fix it on a branch named fix-add and send me a pull request.'),{commit:true,push:true,pr:true});
 assert.deepEqual(askedSteps('Use a branch called docs-serve and open a PR for it.'),{commit:true,push:true,pr:true});
});

test('a final reply before the asked commit, push and pull request gets at most two nudges naming the next command',async()=>{
 const seen=[];let status='On branch fix\nChanges not committed (git commit -m "MESSAGE" saves them all):\n\tmodified:   calc.py\n';
 const workspace={cwd:'/home/user/app',vm:{read:async()=>({ok:false})},path:p=>p,request:async(name,args)=>{
  if(name!=='exec')return{};
  if(args.command==='git status')return{code:0,stdout:status,stderr:''};
  if(args.command==='gh pr view')return{code:1,stdout:'',stderr:'gh: no open pull request'};
  return{code:0,stdout:'',stderr:''};
 }};
 const task=createPiTask({linux:{cwd:'/home/user/app'},workspace,network:{options:()=>({enabled:false})},turn:async messages=>{
  seen.push(messages.at(-1).content);
  return{text:'I fixed it and committed it. Next steps: push and open a pull request.',reason:'end'};
 }});
 const r=await task.run('Fix calc.py, commit it, push it and open a pull request.');
 assert.equal(seen.length,3);
 assert.match(seen[1],/^Not finished yet: Your changes are not committed yet\. Make that tool call now: git commit -m "What changed"\./);
 status='On branch fix\nYour branch is ahead of \'origin/main\' by 1 commit (not pushed; git push uploads them).\nnothing to commit, working tree clean\n';
 seen.length=0;await task.run('Now push it and open a pull request.');
 assert.match(seen[1],/Your commits are not pushed yet\. Make that tool call now: git push\./);
 status='On branch fix\nnothing to commit, working tree clean\n';
 seen.length=0;await task.run('Open a pull request for it.');
 assert.equal(seen.length,1,'no pull request nudge before anything was pushed');
 seen.length=0;await task.run('What does calc.py do?');
 assert.equal(seen.length,1);
});

test('an edit aimed at the wrong file is told where the text is, and the same failing edit a third time stops',async()=>{
 const files=new Map([['/home/user/app/calc.py','def add(a, b):\n    return a - b\n'],['/home/user/app/test_calc.py','from calc import add\n']]);const results=[];
 const workspace={cwd:'/home/user/app',vm:{read:async p=>({ok:files.has(p)})},path:p=>p.startsWith('/')?p:`/home/user/app/${p.replace(/^\.\//,'')}`,request:async(name,args)=>{
  if(name==='read'){const t=files.get(workspace.path(args.path));if(t==null)throw Error('no such file');return{text:t}}
  if(name==='edit'){const t=files.get(workspace.path(args.path));if(!t.includes(args.oldText))throw Error(`oldText is not in ${args.path}; read the file again`);return{}}
  if(name==='exec'&&args.command.startsWith('grep -rlF'))return{code:0,stdout:'calc.py\n',stderr:''};
  return{code:0,stdout:'',stderr:''};
 }};
 const call={name:'edit',arguments:{path:'test_calc.py',oldText:'def add(a, b):\n    return a - b',newText:'def add(a, b):\n    return a + b'}};
 const task=createPiTask({linux:{cwd:'/home/user/app'},workspace,network:{options:()=>({enabled:false})},onEvent:e=>{if(e.type==='tool-result')results.push(e.result.content[0].text)},turn:async()=>({text:'<tool_call>'+JSON.stringify(call)+'</tool_call>',reason:'end'})});
 await assert.rejects(task.run('Fix add in calc.py'),/The same edit call failed 3 times, so the task stopped/);
 assert.match(results[0],/oldText is not in test_calc\.py, so nothing was changed\. That text is in calc\.py: call edit with that path\./);
});

test('namedRepos finds OWNER/NAME however a request names the repository, and not file paths',()=>{
 const names=p=>namedRepos(p).map(([o,n])=>`${o}/${n}`);
 assert.deepEqual(names('Clone the GitHub repository demo/app. In tools/serve_local.py add x.'),['demo/app']);
 assert.deepEqual(names('In demo/app on GitHub, README.md has the line'),['demo/app']);
 assert.deepEqual(names('calc.py in demo/app (GitHub) is broken'),['demo/app']);
 assert.deepEqual(names('Please merge PR #1 on demo/app (it is on GitHub).'),['demo/app']);
 assert.deepEqual(names("The checks on demo/app's main branch are red."),['demo/app']);
 assert.deepEqual(names('See https://github.com/anthony-firn/evolve-llm/pull/4'),['anthony-firn/evolve-llm']);
 assert.deepEqual(names('Run python3 tests/vm_browser_test.py and fix web/app.js'),[]);
});

test('a request about the repository the paired computer builds says to build and test there',async()=>{
 const seen=[];
 const workspace={cwd:'/home/user',vm:{read:async()=>({ok:false})},path:p=>p,request:async()=>({code:0,stdout:'',stderr:''})};
 let building='demo/app';
 const task=createPiTask({linux:{cwd:'/home/user',building:()=>building},workspace,network:{options:()=>({enabled:false})},turn:async m=>{seen.push(m.at(-1).content);return{text:'Done.',reason:'end'}}});
 await task.run('Fix the tests of the GitHub repository demo/app.');
 assert.match(seen[0],/Your computer is paired for building demo\/app: in its clone, cargo, npm, npx and node run on the computer/);
 building=null;await task.run('Fix the tests of the GitHub repository demo/app.');
 assert.doesNotMatch(seen.at(-1),/paired for building/);
});

test('a request for sources answered from memory is asked to search first, once, only with web access',async()=>{
 assert.ok(wantsSources('Cite the code section and link to it.'));assert.ok(wantsSources('Look up when it was first lit'));assert.ok(!wantsSources('Fix calc.py'));
 for(const enabled of [true,false]){
  const seen=[];let n=0;
  const task=createPiTask({workspace:{request:async()=>({})},network:{options:()=>({enabled}),run:async()=>({text:'[1] Sunvale City Code 12-4',sources:[{url:'https://x'}]})},turn:async messages=>{
   n++;seen.push(messages.at(-1).content);
   return{text:n===2&&enabled?'<tool_call>'+JSON.stringify({name:'web_search',arguments:{query:'Sunvale deposit'}})+'</tool_call>':'Landlords have 30 days.',reason:'end'};
  }});
  await task.run('How many days does a Sunvale landlord have to return a deposit? Cite the section.');
  // Then, its answer having no link, it is given the search's results to cite.
  if(enabled){assert.match(seen[1],/Not finished yet: This question asks for sources, and you have not searched yet\. Make that tool call now: web_search/);assert.match(seen[3],/Pages your search found: https:\/\/x/);assert.equal(n,4);}
  else assert.equal(n,1);
 }
});

test('a line asked for after another line but appended with >> gets a note on how to place it',async()=>{
 assert.ok(wantsPlacement('In README.md, right after the line "npm test", add the line "x".'));assert.ok(!wantsPlacement('Add a line to README.md'));
 const results=[];let n=0;
 const workspace={cwd:'/home/user/app',vm:{read:async()=>({ok:false})},path:p=>p,request:async(name)=>name==='exec'?{code:0,stdout:'',stderr:'',cwd:'/home/user/app'}:{}};
 const task=createPiTask({linux:{cwd:'/home/user/app'},workspace,network:{options:()=>({enabled:false})},onEvent:e=>{if(e.type==='tool-result')results.push(e.result.content[0].text)},turn:async()=>{
  n++;return{text:n===1?'<tool_call>'+JSON.stringify({name:'bash',arguments:{command:"echo 'npm run lint' >> README.md"}})+'</tool_call>':'Done.',reason:'end'};
 }});
 await task.run('In README.md of demo/app, right after the line "npm test", add the line "npm run lint".');
 assert.match(results[0],/>> added the text at the end of README\.md, but the request asks for it next to a particular line\. Undo it with: git restore README\.md/);
});

test('sedWithoutInPlace finds a substitution that leaves its file unchanged',()=>{
 assert.equal(sedWithoutInPlace("cd app && sed -n '1s/def add(a, b):/x/' calc.py"),'calc.py');
 assert.equal(sedWithoutInPlace("sed 's/a - b/a + b/' calc.py"),'calc.py');
 // As models send it: the replacement's \n arrives as a real newline.
 assert.equal(sedWithoutInPlace("cd app && sed -n '1s/def add(a, b):/def add(a, b):\n    return a + b/' calc.py"),'calc.py');
 assert.equal(sedWithoutInPlace("sed -i 's/a - b/a + b/' calc.py"),null);
 assert.equal(sedWithoutInPlace("sed 's/a/b/' calc.py > new.py"),null);
 assert.equal(sedWithoutInPlace("sed -n '1,20p' calc.py"),null);
 assert.equal(sedWithoutInPlace("cat calc.py | sed 's/a/b/'"),null);
});

test('grep finding nothing is said plainly, with the file that has the text',async()=>{
 const results=[];let n=0;
 const workspace={cwd:'/home/user/app',vm:{read:async()=>({ok:false})},path:p=>p,request:async(name,args)=>{
  if(name!=='exec')return{};
  if(args.command.startsWith('grep -rlF'))return{code:0,stdout:'calc.py\n',stderr:''};
  return{code:1,stdout:'',stderr:'',cwd:'/home/user/app'};
 }};
 const task=createPiTask({linux:{cwd:'/home/user/app'},workspace,network:{options:()=>({enabled:false})},onEvent:e=>{if(e.type==='tool-result')results.push(e.result.content[0].text)},turn:async()=>{
  n++;return{text:n===1?'<tool_call>'+JSON.stringify({name:'bash',arguments:{command:"cd app && grep -n 'def add' test_calc.py"}})+'</tool_call>':'Done.',reason:'end'};
 }});
 await task.run('Find add');
 assert.match(results[0],/grep found no "def add" in test_calc\.py\. It is in calc\.py\./);
 assert.doesNotMatch(results[0],/stopped at the first error/);
});

test('an answer without a link, when sources were asked for, is sent back with the pages opened',async()=>{
 const prompts=[];let n=0;
 const network={options:()=>({enabled:true}),run:async(name)=>name==='open_url'?{text:'Lit in 1887.',sources:[{title:'Lighthouse',url:'https://en.wikipedia.org/wiki/Lighthouse'}]}:{text:'',sources:[]}};
 const task=createPiTask({workspace:{request:async()=>({})},network,turn:async(messages)=>{
  n++;prompts.push(JSON.stringify(messages.at(-1)));
  if(n===1)return{text:'<tool_call>'+JSON.stringify({name:'open_url',arguments:{url:'https://en.wikipedia.org/wiki/Lighthouse'}})+'</tool_call>',reason:'end'};
  if(n===2)return{text:'It was first lit in 1887. Source: Lighthouse (Wikipedia)',reason:'end'};
  return{text:'It was first lit in 1887. Source: https://en.wikipedia.org/wiki/Lighthouse',reason:'end'};
 }});
 await task.run('When was the lighthouse first lit? Cite a link.');
 assert.equal(n,3);
 assert.match(prompts[2],/your answer has no link\. Give your answer again with the link of the page it came from\. Pages you opened: https:\/\/en\.wikipedia\.org\/wiki\/Lighthouse/);
});

test('a button clicked again on the same page is held once and explained, then allowed', async () => {
  const pages = { 'https://shop.example/': '# Shop\nCart: 1 item(s)\n[1] button "Add Red Kettle to cart"  [2] button "Add Blue Kettle to cart"' };
  const browser = fakeBrowser(pages); let n = 0; const seen = [];
  const task = createPiTask({ browser, workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    if (n) seen.push(messages.at(-1).content);
    n++;
    if (n === 1) return call('browser_open', { url: 'https://shop.example/' });
    if (n <= 4) return call('browser_click', { ref: 2 });
    if (n === 5) return call('browser_click', { ref: 1 });
    return { text: 'Done.', reason: 'end' };
  } });
  await task.run('Add two blue kettles to the cart');
  assert.deepEqual(browser.log.filter(l => l[0] === 'click').map(l => l[1]), [2, 2, 1]);
  assert.match(JSON.stringify(seen[2]), /was not done: you already clicked button \\"Add Blue Kettle to cart\\" on this page, and the page then said: \\"Cart: 1 item\(s\)\\"/);
});

test('a model that reads images gets the newest page screenshot with the browser result', async () => {
  const browser = fakeBrowser({ 'https://shop.example/': '# Shop\n[1] button "Buy"' });
  let shots = 0;
  browser.screenshot = async ({ unit, maxSide }) => { shots++; return { rgb: new Uint8Array(unit * unit * 3), width: unit, height: unit, maxSide, shot: shots }; };
  let n = 0, seen = null, system = '';
  const task = createPiTask({ browser, vision: { unit: 32 }, workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    n++;
    system = messages[0].content;
    if (n === 1) return call('browser_open', { url: 'https://shop.example/' });
    if (n === 2) return call('browser_snapshot', {});
    seen = messages; return { text: 'A shop.', reason: 'end' };
  } });
  await task.run('What is on the page?');
  const tools = seen.filter((m) => m.role === 'tool');
  assert.equal(tools.length, 2);
  assert.equal(tools[0].images, undefined);
  assert.deepEqual([tools[1].images.length, tools[1].images[0].width, tools[1].images[0].shot], [1, 32, 2]);
  assert.match(system, /screenshot of the page/);
  // Without vision: no screenshots, no prompt about them.
  const plain = fakeBrowser({ 'https://shop.example/': '# Shop' });
  plain.screenshot = async () => { throw new Error('not asked for'); };
  let m = 0, prompt = '';
  await createPiTask({ browser: plain, workspace: { request: async () => ({}) }, network, turn: async (messages) => { m++; prompt = messages[0].content; return m === 1 ? call('browser_open', { url: 'https://shop.example/' }) : { text: 'ok', reason: 'end' }; } }).run('Look');
  assert.doesNotMatch(prompt, /screenshot of the page/);
});

test('vision models can click a point of the screenshot: it becomes the element there, with the same checks', async () => {
  const pages = { 'https://shop.example/': '# Shop\n[1] button "Add Kettle to cart"  [2] button "Place order"' };
  const browser = fakeBrowser(pages);
  browser.screenshot = async ({ unit }) => ({ rgb: new Uint8Array(unit * unit * 3), width: unit, height: unit });
  const asked = [];
  browser.refAt = async (x, y) => { asked.push([x, y]); return y < 400 ? { ok: true, ref: 1 } : { ok: true, ref: 2 }; };
  const snapshot = browser.snapshot;
  browser.snapshot = async (max) => ({ ...(await snapshot(max)), viewportWidth: 1000, viewportHeight: 800 });
  let n = 0; const seen = []; let tools = [];
  const task = createPiTask({ browser, vision: { unit: 32 }, workspace: { request: async () => ({}) }, network, turn: async (messages, schemas) => {
    if (n) seen.push(JSON.stringify(messages.at(-1).content)); else tools = schemas.map((t) => t.function?.name || t.name);
    n++;
    if (n === 1) return call('browser_open', { url: 'https://shop.example/' });
    if (n === 2) return call('browser_click_at', { x: 500, y: 250 });
    if (n === 3) return call('browser_click_at', { x: 500, y: 900 });
    if (n === 4) return call('browser_click_at', { x: 1500, y: 10 });
    return { text: 'Done.', reason: 'end' };
  } });
  await task.run('Add the kettle');
  assert(tools.includes('browser_click_at'));
  assert.deepEqual(asked, [[500, 200], [500, 720]]);
  assert.deepEqual(browser.log.filter((l) => l[0] === 'click').map((l) => l[1]), [1]);
  assert.match(seen[2], /needs the person's approval/);
  assert.match(seen[3], /x and y go from 0 to 1000/);
});

test('typing without an element number goes to the element clicked last on the same page', async () => {
  const pages = { 'https://shop.example/': '# Shop\n[1] textbox "Search"  [2] button "Go"' };
  const browser = fakeBrowser(pages); let n = 0; const seen = [];
  const task = createPiTask({ browser, workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    if (n) seen.push(JSON.stringify(messages.at(-1).content));
    n++;
    if (n === 1) return call('browser_type', { text: 'too early' });
    if (n === 2) return call('browser_open', { url: 'https://shop.example/' });
    if (n === 3) return call('browser_click', { ref: 1 });
    if (n === 4) return call('browser_type', { text: 'kettle' });
    return { text: 'Done.', reason: 'end' };
  } });
  await task.run('Search for kettles');
  assert.deepEqual(browser.log.filter((l) => l[0] === 'type').map((l) => [l[1], l[2]]), [[1, 'kettle']]);
  assert.match(seen[0], /ref must be an element number/);
});

test('a reply cut off at the output limit is asked again once, by how it was cut', async () => {
  for (const [cut, want] of [
    ['<think>\nLet me weigh every option ' + 'again '.repeat(50), /You thought past the reply limit/],
    ['<think>ok</think>\n\n \n\n \n\n', /only blank lines/],
    ['<tool_call>\n{"name": "write", "arguments": {"path": "counter.html", "text": "<html>' + 'x'.repeat(200), /cut off at the output limit.*small and simple/],
  ]) {
    let n = 0; const asks = [];
    const task = createPiTask({ workspace: { request: async () => ({}) }, network: { options: () => ({ enabled: false }) }, turn: async (messages) => {
      n++; if (n > 1) asks.push(messages.at(-1).content);
      return n === 1 ? { text: cut, reason: 'length' } : { text: 'Done.', reason: 'end' };
    } });
    await task.run('Make counter.html');
    assert.equal(n, 2); assert.match(asks[0], want);
  }
  // A second cut-off in the same task stops it, as before.
  let m = 0;
  const twice = createPiTask({ workspace: { request: async () => ({}) }, network: { options: () => ({ enabled: false }) }, turn: async () => { m++; return { text: '<think>' + 'more '.repeat(20), reason: 'length' }; } });
  await assert.rejects(twice.run('Merge it'), /output\/context limit/);
  assert.equal(m, 2);
});

test('an answer from search results alone, without a link, is sent back with the results\' addresses', async () => {
  const prompts = []; let n = 0;
  const network = { options: () => ({ enabled: true }), run: async () => ({ text: '1. Lighthouse: lit in 1887', sources: [{ title: 'Lighthouse', url: 'https://en.wikipedia.org/wiki/Lighthouse' }, { title: 'Other', url: 'https://example.org/b' }] }) };
  const task = createPiTask({ workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    n++; prompts.push(JSON.stringify(messages.at(-1)));
    if (n === 1) return { text: '<tool_call>' + JSON.stringify({ name: 'web_search', arguments: { query: 'lighthouse' } }) + '</tool_call>', reason: 'end' };
    return { text: n === 2 ? 'It was lit in 1887. Source: Lighthouse' : 'It was lit in 1887. https://en.wikipedia.org/wiki/Lighthouse', reason: 'end' };
  } });
  await task.run('When was the lighthouse first lit? Include the link you used.');
  assert.equal(n, 3);
  assert.match(prompts[2], /Pages your search found: https:\/\/en\.wikipedia\.org\/wiki\/Lighthouse https:\/\/example\.org\/b/);
});

test('a button clicked again stays held when the request asks for it once', async () => {
  assert.ok(wantsMore('Add two kettles')); assert.ok(wantsMore('click it 3 times')); assert.ok(!wantsMore('Go to http://127.0.0.1:37187/ and add the Blue Kettle'));
  const pages = { 'https://shop.example/': '# Shop\nCart: 1 item(s)\n[1] button "Add Red Kettle to cart"  [2] button "Add Blue Kettle to cart"' };
  const browser = fakeBrowser(pages); let n = 0; const seen = [];
  const task = createPiTask({ browser, workspace: { request: async () => ({}) }, network, turn: async (messages) => {
    if (n) seen.push(JSON.stringify(messages.at(-1).content));
    n++;
    if (n === 1) return call('browser_open', { url: 'https://shop.example/' });
    if (n <= 4) return call('browser_click', { ref: 2 });
    return { text: 'Done.', reason: 'end' };
  } });
  await task.run('Go to https://shop.example/ and add the Blue Kettle to the cart.');
  assert.deepEqual(browser.log.filter(l => l[0] === 'click').map(l => l[1]), [2]);
  assert.match(seen[3], /was not done again: the request asks for it once/);
});

test('a file changed in place by sed is shown as it now reads', async () => {
  assert.equal(sedInPlaceFile("cd app && sed -i 's/a/b/' calc.py && git add calc.py"), 'calc.py');
  assert.equal(sedInPlaceFile("sed -n '1p' calc.py"), null);
  assert.equal(changedLines('a\nb\nc', 'a\nb\nc'), null);
  let file = 'def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n';
  const results = []; let n = 0;
  const workspace = { cwd: '/home/user/app', path: (p) => '/home/user/app/' + p, vm: { read: async () => ({ ok: false }) }, request: async (name, args) => {
    if (name === 'read') return { text: file };
    if (name === 'exec') { if (/sed -i/.test(args.command)) file = file.replace('def add(a, b):', 'def add(a, b):\n    return a + b'); return { code: 0, stdout: '[fix-add af6d630] Fix add', stderr: '', cwd: '/home/user/app' }; }
    return {};
  } };
  const task = createPiTask({ linux: { cwd: '/home/user/app' }, workspace, network: { options: () => ({ enabled: false }) }, onEvent: (e) => { if (e.type === 'tool-result') results.push(e.result.content[0].text); }, turn: async () => {
    n++; return { text: n === 1 ? '<tool_call>' + JSON.stringify({ name: 'bash', arguments: { command: "sed -i 's/def add(a, b):/def add(a, b):\\n    return a + b/' calc.py && git commit -m 'Fix add'" } }) + '</tool_call>' : 'Done.', reason: 'end' };
  } });
  await task.run('Fix add');
  assert.match(results[0], /calc\.py now reads:\n1: def add\(a, b\):\n2:     return a \+ b\n3:     return a - b\n4: /);
});

test('a line left under a return at the same indent is named as unreachable', () => {
  assert.equal(unreachable('1: def add(a, b):\n2:     return a + b\n3:     return a - b\n4: '), 'line 3 can never run: it comes right after a return at the same indent. If the new line replaces it, remove line 3.');
  assert.equal(unreachable('1: def add(a, b):\n2:     return a + b\n3: \n4: def sub(a, b):'), null);
  assert.equal(unreachable('1:   if (x) {\n2:     return 1;\n3:   }\n4:   return 2;'), null);
  assert.equal(unreachable('1:     if x:\n2:         return 1\n3:     else:'), null);
});
