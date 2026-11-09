import { createEngine, EngineError, type EngineEvent } from '../web/sdk.mjs';
const engine = createEngine({ timeoutMs: 60000 });
const controller = new AbortController();
await engine.initialize({ signal: controller.signal });
await engine.load('smollm2-135m', { gpuMemoryMB: 512, onProgress: e => console.log(e.type) });
try {
  for await (const event of engine.generate([{ role: 'user', content: 'Hi' }], { signal: controller.signal })) {
    const typed: EngineEvent = event;
    if (typed.type === 'token') console.log(typed.text.toUpperCase());
    if (typed.type === 'done') console.log(typed.tokens.toFixed(0));
  }
} catch (error) { if (error instanceof EngineError) console.log(error.code); else throw error; }
engine.dispose();
// @ts-expect-error roles are deliberately restricted to textual conversation roles
engine.generate([{ role: 'developer', content: 'Hi' }]);
// @ts-expect-error the budget is in MB and uses a number, not a unit-bearing string
engine.load('smollm2-135m', { gpuMemoryMB: '512 MB' });
