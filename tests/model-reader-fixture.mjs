import { storedReader } from './model-readers.mjs';
let lock;
self.onmessage=async({data})=>{
 try {
  const root=await navigator.storage.getDirectory();
  const dir=await root.getDirectoryHandle('reader-regression',{create:true});
  const handle=await dir.getFileHandle('weights.bin',{create:true});
  if(data.type==='lock'){
   lock=await handle.createSyncAccessHandle();lock.truncate(0);lock.write(new Uint8Array([4,3,2,1]));lock.flush();
   self.postMessage({ok:true});return;
  }
  if(data.type==='unlock'){lock.close();lock=null;self.postMessage({ok:true});return}
  if(data.type==='read'){
   const reader=await storedReader(handle),target=new Uint8Array(2);
   try{reader.read(1,target);self.postMessage({bytes:[...target],size:reader.size})}finally{reader.close()}
  }
 }catch(error){self.postMessage({error:String(error),name:error.name})}
};
