// Scripted model protocol fixture. The application, Pi, shell WASM and previews stay real.
const emit = (type, data={}) => postMessage({type,...data});
const tool = (name,args) => `<tool_call>${JSON.stringify({name,arguments:args})}</tool_call>`;
self.onmessage=({data:m})=>{
 if(m.type==='hello'){
  emit('adapter',{adapter:{ok:true,description:'Scripted model fixture',maxBufferSize:1073741824}});
  emit('cache',{models:[],supported:true});
 }else if(m.type==='load')emit('loaded',{
  source:{key:null,id:'fixture'},model:{parameters:1000000,active_parameters:1000000,config:{n_layer:2},weight_types:{Q4_K:1}},
  chat:{name:'Scripted model fixture',context:8192,weight_bytes:100,state_bytes:100,placement:{streaming:false}},
 });
 else if(m.type==='generate'){
  emit('token',{text:'A scripted chat answer.'});emit('done',{tokens:5,ms:100,used:30,context:8192,reason:'end'});
 }else if(m.type==='stop'){
  if(self.pending){emit('agent_done',{requestId:self.pending,text:'',reason:'stopped',n:0});self.pending=null;}
 }else if(m.type==='agent_turn'){
  const initial=m.messages.find(x=>x.role==='user')?.content||'';
  const request=initial.split('Current task: ').at(-1);
  if(request.includes('long task')){self.pending=m.requestId;return;}
  const path=initial.match(/ALREADY been saved at (\/workspace\/[^. ]+\.html)/)?.[1]||'/workspace/snake.html';
  const results=m.messages.filter(x=>x.role==='tool');
  let text;
  if(request.includes('break the starter')){
   text=results.length?'unfinished':tool('write',{path,text:'<script>throw new Error("broken fixture")</script>'});
   emit('agent_done',{requestId:m.requestId,text,reason:results.length?'length':'end',n:20});return;
  }
  if(/read readme/i.test(request)){
   text=results.length?'Scripted summary of the saved workspace README.':tool('read',{path:'/workspace/README.txt'});
  }else if(/delete calculator/i.test(request)){
   text=results.length?'Deleted the file.':tool('bash',{command:'rm /workspace/calculator.html'});
  }else if(/blue background|background blue/i.test(request)){
   if(!results.length)text=tool('read',{path:'/workspace/snake.html'});
   else if(results.length===1)text=tool('edit',{path:'/workspace/snake.html',oldText:'background:#101827',newText:'background:blue'});
   else text=tool('browser_test',{path:'/workspace/snake.html'})+'\n'+tool('preview',{path:'/workspace/snake.html'});
  }else if(request.includes('calculator')){
   text=!results.length?tool('write',{path:'/workspace/calculator.html',text:'<!doctype html><title>Calculator</title><h1>Calculator</h1><button>1</button>'}):tool('browser_test',{path:'/workspace/calculator.html'})+'\n'+tool('preview',{path:'/workspace/calculator.html'});
  }else text=tool('browser_test',{path})+'\n'+tool('preview',{path});
  emit('agent_done',{requestId:m.requestId,text,reason:'end',n:20});
 }
};
