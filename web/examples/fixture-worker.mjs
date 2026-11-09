// Protocol fixture only. Does not initialize WASM, download weights or perform inference.
let stopped = false;
self.onmessage = ({ data }) => {
  if (data.type === 'hello') self.postMessage({ type: 'adapter', adapter: { ok: true, description: 'Protocol fixture; no GPU used' } });
  else if (data.type === 'load') { stopped = false; self.postMessage({ type: 'loaded', fixture: true }); }
  else if (data.type === 'generate') {
    const tokens = ['This ', 'is a ', 'scripted ', 'SDK fixture.']; let i = 0;
    const tick = () => { if (stopped) return; if (i < tokens.length) { self.postMessage({ type: 'token', text: tokens[i++] }); setTimeout(tick, 30); }
      else self.postMessage({ type: 'done', reason: 'fixture', tokens: tokens.length, ms: 120 }); };
    tick();
  } else if (data.type === 'stop' || data.type === 'cancel') stopped = true;
};
