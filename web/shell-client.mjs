/** Dedicated WASM worker; hard time limit restores the last committed checkpoint. */
export class Workspace {
  constructor({ workerFactory = () => new Worker(new URL('./shell-worker.mjs', import.meta.url), { type: 'module' }), onCheckpoint = () => {}, timeoutMs = 20000 } = {}) {
    this.workerFactory=workerFactory; this.onCheckpoint=onCheckpoint; this.timeoutMs=timeoutMs; this.pending=null; this.worker=null; this.sequence=0; this.checkpoint=null; this.disposed=false;
  }
  attach() {
    if (this.disposed) throw new Error('Workspace is closed.');
    if (this.worker) return;
    const worker=this.worker=this.workerFactory();
    worker.onmessage=({data})=>{
      if(this.worker!==worker||!this.pending||data.id!==this.pending.id)return;
      if(data.snapshot){this.checkpoint=data.snapshot;try{this.onCheckpoint(data.snapshot)}catch{}}
      data.error?this.finish(new Error(data.error)):this.finish(null,data.result);
    };
    worker.onerror=event=>{event.preventDefault?.();if(this.worker===worker){this.finish(new Error('Workspace worker failed. The last checkpoint was retained.'));this.resetWorker()}};
  }
  resetWorker(){this.worker?.terminate();this.worker=null;}
  finish(error,value){const pending=this.pending;if(!pending)return;clearTimeout(pending.timer);pending.signal?.removeEventListener('abort',pending.abort);this.pending=null;error?pending.reject(error):pending.resolve(value);}
  async request(type,data={}, {signal}={}) {
    if(this.active)throw new Error('Workspace is busy.');signal?.throwIfAborted();this.active=true;
    try {
    const fresh=!this.worker;this.attach();
    if(fresh&&this.checkpoint)await this.send('restore',{snapshot:this.checkpoint},signal);
    const result=await this.send(type,data,signal);
    if(type==='restore')this.checkpoint=data.snapshot;
    return result;
    } finally {this.active=false}
  }
  send(type,data,signal){
    return new Promise((resolve,reject)=>{
      const id=++this.sequence;
      const abort=()=>{this.finish(new Error('Workspace operation cancelled; last checkpoint retained.'));this.resetWorker()};
      const timer=setTimeout(()=>{this.finish(new Error('Workspace time limit exceeded; last checkpoint retained.'));this.resetWorker()},this.timeoutMs);
      this.pending={id,resolve,reject,timer,signal,abort};signal?.addEventListener('abort',abort,{once:true});
      if(signal?.aborted)abort();else{try{this.worker.postMessage({id,type,...data})}catch{abort()}}
    });
  }
  dispose(){this.disposed=true;this.finish(new Error('Workspace closed.'));this.resetWorker();}
}
