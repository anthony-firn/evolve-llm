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
 await assert.rejects(task.run('Loop'),/6-turn limit/);assert.equal(executed,6);
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
