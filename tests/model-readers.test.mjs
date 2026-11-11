import test from 'node:test';
import assert from 'node:assert/strict';
import { storedReader, openModelReaders, storageError } from '../web/model-readers.mjs';
const locked = () => new DOMException('No modification allowed', 'NoModificationAllowedError');
class SyncReader {
  readAsArrayBuffer(blob) { return new Uint8Array(blob.bytes).buffer; }
}
const file = {size: 4, slice: (start,end) => ({bytes: [4,3,2,1].slice(start,end)})};

test('a locked saved model falls back to bounded file reads without requesting write access',async()=>{
  const modes=[];
  const reader=await storedReader({createSyncAccessHandle:async opts=>{modes.push(opts.mode);throw locked()},getFile:async()=>file},SyncReader);
  const target=new Uint8Array(2);reader.read(1,target);
  assert.deepEqual([...target],[3,2]);assert.equal(reader.size,4);assert.deepEqual(modes,['read-only']);reader.close();
  assert.throws(()=>reader.read(3,target),/shorter/);
});
test('unavailable sync access uses FileReaderSync; permission errors remain errors',async()=>{
  const reader=await storedReader({getFile:async()=>file},SyncReader);
  assert.equal(reader.size,4);
  await assert.rejects(storedReader({createSyncAccessHandle:async()=>{throw new DOMException('Denied','NotAllowedError')}}),/Denied/);
});
test('direct reads remain zero-copy and a failed split-file open closes earlier handles',async()=>{
  let closed=0;
  const access={getSize:()=>4,read:target=>{target.set([4,3]);return 2},close:()=>closed++};
  const handle={createSyncAccessHandle:async()=>access};
  const reader=await storedReader(handle);const target=new Uint8Array(2);reader.read(0,target);assert.deepEqual([...target],[4,3]);reader.close();
  await assert.rejects(openModelReaders({names:['first','missing']},async()=>({getFileHandle:async name=>{if(name==='missing')throw Error('Missing part');return handle}})),/Missing part/);
  assert.equal(closed,2);
});
test('a size inspection failure closes its acquired handle and lock errors explain recovery',async()=>{
  let closed=false;
  await assert.rejects(storedReader({createSyncAccessHandle:async()=>({getSize(){throw Error('Invalid')},close(){closed=true}})}),/Invalid/);
  assert(closed);assert.match(storageError(locked()),/Close other Evolve LLM tabs/);assert.equal(storageError(Error('Other')),'Other');
});
