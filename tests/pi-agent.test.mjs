import test from 'node:test';
import assert from 'node:assert/strict';
import { createPiTask } from '../web/pi-agent.mjs';

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
 await assert.rejects(task.run('Write'),/output\/context limit/);assert.equal(executed,0);
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
test('edits require a prior read and an exact nonempty string; failures return to Pi for correction',async()=>{
 let n=0;const operations=[];
 const task=createPiTask({workspace:{request:async(name,args)=>{operations.push(name);return name==='read'?{text:'background:red;'}:{ok:true}}},network:{options:()=>({enabled:false})},turn:async(messages)=>{
 n++;const call=n===1?{name:'edit',arguments:{path:'/workspace/game.html',oldText:'',newText:'blue'}}:n===2?{name:'read',arguments:{path:'/workspace/game.html'}}:n===3?{name:'edit',arguments:{path:'/workspace/game.html',oldText:'background:red;',newText:'background:blue;'}}:null;
 if(n===2)assert(messages.some(m=>m.role==='tool'&&m.content.includes('Read this file first')));
 return {text:call?'<tool_call>'+JSON.stringify(call)+'</tool_call>':'Changed.',reason:'end'};
 }});const result=await task.run('Make the file blue');assert.deepEqual(operations,['read','edit']);assert.equal(result.changes,1);assert.equal(result.errors,1);assert.equal(result.lastToolFailed,false);
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
 let n=0,opened=0;
 const task=createPiTask({workspace:{request:async(name)=>name==='read'?{text:'red'}:{ok:true}},network:{options:()=>({enabled:false})},preview:async()=>{opened++;return{ok:true}},turn:async()=>{
  n++;
  const calls=n===1?[{name:'edit',arguments:{path:'/workspace/game.html',oldText:'red',newText:'blue'}},{name:'preview',arguments:{path:'/workspace/game.html'}}]:[
   {name:'read',arguments:{path:'/workspace/game.html'}},{name:'edit',arguments:{path:'/workspace/game.html',oldText:'red',newText:'blue'}},{name:'preview',arguments:{path:'/workspace/game.html'}}
  ];
  assert(n<=2,'Model should stop after corrected final preview');
  return{text:calls.map(call=>'<tool_call>'+JSON.stringify(call)+'</tool_call>').join('\n'),reason:'end'};
 }});
 const result=await task.run('Edit and preview');assert.equal(n,2);assert.equal(opened,1);assert.equal(result.changes,1);assert.equal(result.errors,2);
});
