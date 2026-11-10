import test from 'node:test';
import assert from 'node:assert/strict';
import { Workspace } from '../web/shell-client.mjs';

test('abort terminates only the owned worker; restart restores the last checkpoint and ignores late events',async()=>{
 const workers=[];
 const workspace=new Workspace({workerFactory:()=>{const worker={messages:[],postMessage(message){this.messages.push(message)},terminate(){this.terminated=true}};workers.push(worker);return worker}});
 const abort=new AbortController();const first=workspace.request('write',{path:'a',text:'saved'},{signal:abort.signal});
 const worker=workers[0];worker.onmessage({data:{id:1,result:{ok:true},snapshot:'checkpoint'}});await first;
 const pending=workspace.request('read',{path:'a'},{signal:abort.signal});abort.abort();await assert.rejects(pending,/cancelled/);assert(worker.terminated);
 const resumed=workspace.request('read',{path:'a'});const replacement=workers[1];assert.equal(replacement.messages[0].type,'restore');assert.equal(replacement.messages[0].snapshot,'checkpoint');
 worker.onmessage({data:{id:replacement.messages[0].id,result:{text:'late'}}});
 replacement.onmessage({data:{id:replacement.messages[0].id,result:{ok:true}}});await Promise.resolve();
 replacement.onmessage({data:{id:replacement.messages[1].id,result:{text:'saved'}}});assert.deepEqual(await resumed,{text:'saved'});workspace.dispose();
});
test('hard timeout preserves a checkpoint and releases a stuck worker',async()=>{
 let terminated=false;const workspace=new Workspace({timeoutMs:5,workerFactory:()=>({postMessage(){},terminate(){terminated=true}})});
 await assert.rejects(workspace.request('snapshot'),/time limit/);assert(terminated);assert.equal(workspace.active,false);workspace.dispose();
});
