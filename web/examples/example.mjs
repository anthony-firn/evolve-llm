import { createEngine } from '../sdk.mjs';
import { MODELS } from '../models.js';
const $ = id => document.getElementById(id);
let controller = null;
async function run(fixture) {
  controller = new AbortController();
  for (const id of ['fixture', 'real']) $(id).disabled = true;
  $('cancel').disabled = false; $('output').textContent = ''; $('events').textContent = '';
  const engine = createEngine(fixture ? { workerURL: new URL('./fixture-worker.mjs', import.meta.url) } : {});
  try {
    $('status').textContent = fixture ? 'Protocol fixture running; no inference or download.' : 'Checking WebGPU…';
    await engine.initialize({ signal: controller.signal });
    const model = MODELS.find(m => m.id.includes('smollm2'));
    await engine.load(model.id, { contextSize: 512, signal: controller.signal,
      onProgress: event => { $('status').textContent = `Real model: ${event.phase || event.type}`; } });
    $('status').textContent = fixture ? 'Scripted fixture output' : 'Real GPU inference';
    for await (const event of engine.generate([{ role: 'user', content: $('prompt').value }], { maxTokens: 64, signal: controller.signal })) {
      if (event.type === 'token') $('output').textContent += event.text;
      else $('events').textContent += JSON.stringify(event) + '\n';
    }
    $('status').textContent = fixture ? 'Fixture complete. This was not model-generated text.' : 'GPU inference complete. Model unloaded.';
  } catch (error) { $('status').textContent = `${error.code || 'error'}: ${error.message}`; }
  finally { engine.dispose(); controller = null; for (const id of ['fixture', 'real']) $(id).disabled = false; $('cancel').disabled = true; }
}
$('fixture').addEventListener('click', () => run(true));
$('real').addEventListener('click', () => run(false));
$('cancel').addEventListener('click', () => controller?.abort());
