import test from 'node:test';
import assert from 'node:assert/strict';
import { createEngine, EngineError } from '../web/sdk.mjs';
class WorkerFixture {
  constructor() { this.messages = []; this.terminated = false; }
  postMessage(message) { this.messages.push(message); }
  emit(data) { this.onmessage?.({ data }); }
  terminate() { this.terminated = true; }
}
function setup(timeoutMs = 1000) {
  const workers = [];
  const engine = createEngine({ workerFactory: () => { const w = new WorkerFixture(); workers.push(w); return w; }, timeoutMs });
  return { engine, workers };
}
async function loaded(engine, workers) {
  const loading = engine.load('smollm2-135m'); workers.at(-1).emit({ type: 'loaded', info: 'fixture' }); await loading;
}
test('SDK uses the real worker protocol and streams incremental text with tools disabled', async () => {
  const { engine, workers } = setup();
  const ready = engine.initialize(); workers[0].emit({ type: 'adapter', adapter: { ok: true } }); await ready;
  await loaded(engine, workers);
  const stream = engine.generate([{ role: 'user', content: 'Hi' }], { temperature: 0, seed: 42 });
  assert.equal(workers[0].messages.at(-1).internet.enabled, false);
  assert.equal(workers[0].messages.at(-1).sampler.seed, 42);
  workers[0].emit({ type: 'token', text: 'Hello ' }); workers[0].emit({ type: 'token', text: 'there' }); workers[0].emit({ type: 'done', tokens: 2 });
  const events = []; for await (const event of stream) events.push(event);
  assert.equal(events.map(e => e.text || '').join(''), 'Hello there'); assert.equal(events.at(-1).type, 'done');
  engine.dispose(); assert(workers[0].terminated);
});
test('busy calls cannot change loaded state or mix a second operation into the first', async () => {
  const { engine, workers } = setup(); await loaded(engine, workers);
  const stream = engine.generate([{ role: 'user', content: 'Hi' }]);
  await assert.rejects(engine.load('smollm2-135m'), e => e.code === 'busy');
  assert.throws(() => engine.generate([{ role: 'user', content: 'Again' }]), e => e.code === 'busy');
  workers[0].emit({ type: 'done' }); for await (const event of stream) {}
  const second = engine.generate([{ role: 'user', content: 'Again' }]); await second.return(); engine.dispose();
});
test('aborting kills the old worker, rejects pending reads, and ignores its late messages', async () => {
  const { engine, workers } = setup(); await loaded(engine, workers);
  const controller = new AbortController(); const stream = engine.generate([{ role: 'user', content: 'Hi' }], { signal: controller.signal });
  const waiting = stream.next(); controller.abort(); await assert.rejects(waiting, e => e.code === 'aborted'); assert(workers[0].terminated);
  assert.throws(() => engine.generate([{ role: 'user', content: 'Hi' }]), e => e.code === 'not_loaded');
  const load = engine.load('smollm2-135m'); workers[0].emit({ type: 'loaded', stale: true });
  assert.equal(workers.length, 2); workers[1].emit({ type: 'loaded', fresh: true }); assert.equal((await load).fresh, true); engine.dispose();
});
test('timeout, worker failure and consumer early exit release resources', async () => {
  const one = setup(10); await assert.rejects(one.engine.load('smollm2-135m'), e => e.code === 'timeout'); assert(one.workers[0].terminated);
  const two = setup(); const load = two.engine.load('smollm2-135m'); two.workers[0].onerror({ preventDefault() {} }); await assert.rejects(load, e => e.code === 'worker');
  const three = setup(); await loaded(three.engine, three.workers); const stream = three.engine.generate([{ role: 'user', content: 'Hi' }]);
  three.workers[0].emit({ type: 'token', text: 'Partial' }); for await (const e of stream) break; assert(three.workers[0].terminated); three.engine.dispose();
});
test('errors do not reflect worker contents, queued tokens are bounded, and disposed calls fail', async () => {
  const { engine, workers } = setup(); const load = engine.load('smollm2-135m'); workers[0].emit({ type: 'error', message: 'PRIVATE PROMPT' });
  await assert.rejects(load, e => e instanceof EngineError && e.code === 'engine' && !e.message.includes('PRIVATE'));
  await loaded(engine, workers); const stream = engine.generate([{ role: 'user', content: 'Hi' }]);
  for (let i = 0; i < 1025; i++) workers.at(-1).emit({ type: 'token', text: 'x' });
  await assert.rejects(stream.next(), e => e.code === 'backpressure'); engine.dispose();
  assert.throws(() => engine.generate([{ role: 'user', content: 'Hi' }]), e => e.code === 'disposed');
});
test('unsupported GPU, pre-aborted calls and invalid models fail without a download', async () => {
  const { engine, workers } = setup(); const ready = engine.initialize(); workers[0].emit({ type: 'adapter', adapter: { ok: false } });
  await assert.rejects(ready, e => e.code === 'unsupported_gpu');
  await assert.rejects(engine.load('not-a-model'), TypeError);
  const abort = new AbortController(); abort.abort(); await assert.rejects(engine.load('smollm2-135m', { signal: abort.signal }), e => e.code === 'aborted');
  assert.equal(workers[0].messages.length, 1); engine.dispose();
});

test('GPU budget has explicit MB units and is converted to the runtime GB setting', async () => {
  const { engine, workers } = setup(); const load = engine.load('smollm2-135m', { gpuMemoryMB: 512 });
  assert.equal(workers[0].messages[0].gpuMemory, 0.5); workers[0].emit({ type: 'loaded' }); await load; engine.dispose();
});
