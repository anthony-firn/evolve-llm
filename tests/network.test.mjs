import test from 'node:test';
import assert from 'node:assert/strict';
import { publicUrl, boundedText } from '../web/network.js';
import { createWebTools, openUrl } from '../web/tools.js';

test('private hosts and URL credentials never reach an external reader',async()=>{
 const blocked=['http://127.0.0.1/','http://2130706433/','http://10.1.2.3/','http://169.254.169.254/','http://192.168.1.2/','http://172.31.2.1/','http://100.64.1.1/','https://localhost/','https://x.local/','http://[::1]/','https://u:p@example.com/'];
 let fetched=0;for(const url of blocked){assert.throws(()=>publicUrl(url));const r=await openUrl(url,{fetchFn:()=>{fetched++;throw Error('no')}});assert.equal(r.sources.length,0)}assert.equal(fetched,0);
 assert.equal(publicUrl('https://example.com/path#fragment'),'https://example.com/path');
});
test('web capability is explicit, responses are bounded, cache is short-lived and auth is not ambient',async()=>{
 let requests=0;
 const run=createWebTools(async(url,options)=>{requests++;assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');assert.equal(options.referrerPolicy,'no-referrer');return new Response('Title: Example\n\n'+ 'a'.repeat(90000))});
 await assert.rejects(run('open_url',{url:'https://example.com'},{enabled:false}),/off/);assert.equal(requests,0);
 const a=await run('open_url',{url:'https://example.com'},{enabled:true});assert.equal(a.text.length,6000);
 const b=await run('open_url',{url:'https://example.com'},{enabled:true});assert.equal(b.cached,true);assert.equal(requests,1);
 const controller=new AbortController();controller.abort();await assert.rejects(run('open_url',{url:'https://example.com'},{enabled:true,signal:controller.signal}));
 await assert.rejects(boundedText(new Response('x'.repeat(100)),10),/size/);
});
test('aborting a provider read stops the request and cannot become model evidence',async()=>{
 const controller=new AbortController();
 const run=createWebTools(async(_,opts)=>new Promise((resolve,reject)=>opts.signal.addEventListener('abort',()=>reject(opts.signal.reason),{once:true})));
 const pending=run('open_url',{url:'https://example.com'},{enabled:true,signal:controller.signal});controller.abort();await assert.rejects(pending);
});
