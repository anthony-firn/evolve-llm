// A live trace of the assistant's work for whoever runs Evolve LLM on their own computer
// (tools/serve_local.py): each run, model turn, tool call and result, and approval goes to the local
// server, which appends it to a file to follow while it runs. It is on only when the page comes from
// this computer AND that server answers …/__trace; production and plain static hosting never send.
// Events never carry settings secrets (the GitHub token and API keys stay out of them).

const LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(globalThis.location?.hostname);
let enabled = null;
let queue = Promise.resolve();
let seq = 0;

function on() {
  enabled ??= LOCAL ? fetch(new URL('./__trace', import.meta.url), { cache: 'no-store' }).then((r) => r.status === 204, () => false) : Promise.resolve(false);
  return enabled;
}

const clip = (v, n) => (typeof v === 'string' && v.length > n ? `${v.slice(0, n)}… (${v.length - n} more characters)` : v);

/** Record one event, in order. Strings longer than 8,000 characters are cut. */
export function trace(event) {
  const body = { seq: ++seq, time: new Date().toISOString(), ...Object.fromEntries(Object.entries(event).map(([k, v]) => [k, clip(v, 8000)])) };
  queue = queue.then(on).then((yes) => yes && fetch(new URL('./__trace', import.meta.url), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).catch(() => {});
}
