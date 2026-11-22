import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeToolFormat, toolTransport, toolResult } from '../web/tool-transport.mjs';

test('GGUF double-braced tool example disables the malformed native tool template',()=>{
 assert.equal(nativeToolFormat('<tools>web_search</tools><tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'),false);
 assert.equal(nativeToolFormat('<tools>web_search</tools><tool_call>{"name": <name>, "arguments": {}}</tool_call>'),true);
 const messages=[{role:'system',content:'Local agent policy'},{role:'user',content:'Read game.html'}],tools=[{type:'function',function:{name:'read'}}];
 const fallback=toolTransport(messages,tools,false);
 assert.equal(fallback.options.tools,undefined);assert.equal(fallback.messages.filter(m=>m.role==='system').length,1);
 assert.match(fallback.messages[0].content,/Local agent policy/);assert.match(fallback.messages[0].content,/<tool_call>\s*\{"name"/);assert.doesNotMatch(fallback.messages[0].content,/\{\{"name"/);
 assert.equal(fallback.messages.at(-1).content,'Read game.html');
 assert.deepEqual(toolTransport(messages,tools,true),{messages,options:{enable_thinking:false,tools}});
});

test('fallback preserves policy and tool feedback for templates without a tool role',()=>{
 const messages=[{role:'system',content:'Policy'},{role:'user',content:'Edit a file'},{role:'assistant',content:'Call read'},{role:'tool',content:'File contents'}];
 const result=toolTransport(messages,[],false);
 assert.equal(result.messages.at(-1).role,'user');
 assert.equal(result.messages.at(-1).content,'<tool_response>\nFile contents\n</tool_response>');
 assert.deepEqual(messages.at(-1),{role:'tool',content:'File contents'});
});

test('images on tool results survive the text tool format', () => {
  const image = { rgb: new Uint8Array(3), width: 1, height: 1 };
  const { messages } = toolTransport([{ role: 'user', content: 'Look' }, toolResult('call_0', 'browser_open', 'Page'), ], [], false);
  assert.equal(messages[2].images, undefined);
  const withImage = toolTransport([{ role: 'user', content: 'Look' }, { ...toolResult('call_0', 'browser_open', 'Page'), images: [image] }], [], false);
  assert.equal(withImage.messages[2].images[0], image);
  const native = toolTransport([{ ...toolResult('call_0', 'browser_open', 'Page'), images: [image] }], [], true);
  assert.equal(native.messages[0].images[0], image);
});
