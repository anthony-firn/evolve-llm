// Scripted model protocol fixture. The application, the assistant (Pi's agent core), the Linux VM
// or browser sandbox and previews stay real. A file named *notools* loads as a model whose chat
// template takes no tools, so the main chat answers it plainly.
const emit = (type, data={}) => postMessage({type,...data});
const tool = (name,args) => `<tool_call>${JSON.stringify({name,arguments:args})}</tool_call>`;
self.onmessage=({data:m})=>{
 if(m.type==='hello'){
  emit('adapter',{adapter:{ok:true,description:'Scripted model fixture',maxBufferSize:1073741824}});
  emit('cache',{models:[],supported:true});
 }else if(m.type==='load'){
  const names=(m.source?.files||[m.source?.file]).map(f=>f?.name||'');
  emit('loaded',{
   source:{key:null,id:'fixture'},model:{parameters:1000000,active_parameters:1000000,config:{n_layer:2},weight_types:{Q4_K:1}},
   chat:{name:'Scripted model fixture',context:8192,weight_bytes:100,state_bytes:100,placement:{streaming:false}},
   tools:!names.some(n=>n.includes('notools')),
  });
 }else if(m.type==='generate'){
  emit('token',{text:'A scripted chat answer.'});emit('done',{tokens:5,ms:100,used:30,context:8192,reason:'end'});
 }else if(m.type==='stop'){
  if(self.pending){emit('agent_done',{requestId:self.pending,text:'',reason:'stopped',n:0});self.pending=null;}
 }else if(m.type==='agent_turn'){
  const last=m.messages.findLastIndex(x=>x.role==='user');
  const request=String(m.messages[last]?.content||'');
  const results=m.messages.slice(last+1).filter(x=>x.role==='tool');
  const users=m.messages.filter(x=>x.role==='user').length;
  let text;
  if(request.includes('long task')){self.pending=m.requestId;return;}
  if(/what is webgpu/i.test(request))text='WebGPU is a scripted answer.';
  else if(/how many messages/i.test(request))text=`You sent ${users} messages.`;
  else if(/already saved at/.test(request))text='Your Snake game is ready below.';
  else if(/background blue/i.test(request)){
   if(!results.length)text=tool('read',{path:'snake.html'});
   else if(results.length===1)text=tool('edit',{path:'snake.html',oldText:'background:#101827',newText:'background:blue'});
   else text=tool('browser_test',{path:'snake.html'})+'\n'+tool('preview',{path:'snake.html'});
  }else if(/calculator/i.test(request)&&/delete/i.test(request)){
   text=results.length?'Deleted calculator.html.':tool('bash',{command:'rm calculator.html'});
  }else if(/calculator/i.test(request)){
   text=!results.length?tool('write',{path:'calculator.html',text:'<!doctype html><title>Calculator</title><h1>Calculator</h1><button>1</button>'}):tool('browser_test',{path:'calculator.html'})+'\n'+tool('preview',{path:'calculator.html'});
  }else text='A scripted assistant answer.';
  emit('agent_done',{requestId:m.requestId,text,reason:'end',n:20});
 }
};
