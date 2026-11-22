// The main chat's assistant: Pi's agent core with the loaded model, working in the Evolve VM (the
// same files, git and gh as the Linux terminal), with web tools when Web access is on, the shared
// browser for requests about a website, and app previews in the conversation. The person sees the
// reply and a short list of steps; the model sees one prompt and one set of tools. Browsers without
// WebAssembly JSPI (Firefox, Safari) cannot run the VM: there it works in the browser sandbox's
// /workspace instead, with the same files and previews but no git.
import { createWebTools, stripToolCalls } from './tools.js';

const HOME = '/home/user';

/** `turn(messages, tools, opts)`: one model turn (app.js agentTurn). `sandbox()`: the browser
 *  sandbox's workspace (web/workbench.mjs), used when the VM cannot run. */
export function createAssistant({ turn, getNetwork, sandbox }) {
  let linux = typeof WebAssembly.Suspending === 'function';
  let workspace = null; // one for the conversation, so a `cd` carries over between messages
  let agent = null;
  let abortWeb = new AbortController();
  const web = createWebTools();
  const network = {
    options: () => ({ ...getNetwork() }),
    run: (name, args, signal) => web(name, args, { ...getNetwork(), signal: signal ? AbortSignal.any([signal, abortWeb.signal]) : abortWeb.signal }),
  };
  async function getWorkspace() {
    if (linux && !workspace) {
      try {
        const [{ sharedVm }, { VmWorkspace }] = await Promise.all([import('./vm/client.mjs'), import('./vm/workspace.mjs')]);
        const vm = sharedVm();
        await vm.ready;
        workspace = new VmWorkspace(vm, { cwd: HOME });
      } catch {
        linux = false; // the VM is not served here (a build without it): the sandbox still works
      }
    }
    return linux ? workspace : sandbox();
  }
  /** The included Snake game, saved under an unused name: tiny models then only customize it. */
  async function saveStarter(ws, view) {
    const response = await fetch(new URL('./examples/snake.html', import.meta.url));
    if (!response.ok) throw new Error('The Snake starter could not be loaded.');
    const dir = linux ? ws.cwd : '/workspace';
    const exists = (path) => ws.request('read', { path }).then(() => true, () => false);
    let path = `${dir}/snake.html`;
    for (let n = 2; await exists(path); n++) path = `${dir}/snake-${n}.html`;
    await ws.request('write', { path, text: await response.text() });
    view.event({ type: 'starter', path });
    return path;
  }

  /** Answer one message: {answer, calls, error}. `history`: earlier [{role, content}] turns. */
  async function run(question, { history = [], view, modelName, maxTokens = 2048, browser, confirmBrowserAction, starter = false, onText = () => {}, onEvent: observe = () => {} }) {
    const ws = await getWorkspace();
    const preview = async (path, opts = {}) => {
      const file = await ws.request('read', { path }, { signal: opts.signal });
      return view.preview(file.path || path, file.text, opts);
    };
    let prompt = question;
    if (starter) {
      const path = await saveStarter(ws, view);
      await preview(path, { smoke: true }).catch(() => {});
      prompt = `A working Snake game is already saved at ${path} and open for the person. It has arrow keys, touch buttons, food, scoring, collisions and Restart. If they asked for changes, read it, make short edits, browser_test it and preview it; otherwise just tell them it is ready.\n\n${question}`;
    }
    const { createPiTask } = await import('./pi-agent.bundle.mjs');
    let turnText = '', shown = '';
    const onEvent = (e) => {
      if (e.type === 'token') {
        turnText += e.text;
        const t = stripToolCalls(turnText);
        if (t.trim()) { shown = t; onText(t); }
      } else if (e.type === 'answer') turnText = '';
      view.event(e);
      if (e.type !== 'token') observe(e);
    };
    agent = createPiTask({ turn, workspace: ws, network, onEvent, modelName, maxTokens, preview, browser, confirmBrowserAction, linux: linux ? { cwd: ws.cwd } : null, history });
    let result = null, error = null;
    try { result = await agent.run(prompt); }
    catch (e) { error = String(e.message || e); }
    finally { agent = null; }
    const last = result?.messages?.filter((m) => m.role === 'assistant').at(-1);
    const answer = stripToolCalls((last?.content || []).filter((c) => c.type === 'text').map((c) => c.text).join('\n')).trim();
    return { answer: answer || shown.trim(), calls: result?.calls ?? view.steps, error };
  }

  return {
    run,
    /** A file of the assistant's workspace (to reopen an app from a saved chat): {path, text}. */
    async read(path) { return (await getWorkspace()).request('read', { path }); },
    cancel() { agent?.abort(); abortWeb.abort(); abortWeb = new AbortController(); },
    /** A new conversation starts in the home folder again; its files stay. */
    reset() { if (workspace) workspace.cwd = HOME; },
    /** Whether files, git and gh live in the Linux VM here (else the browser sandbox). */
    get linux() { return linux; },
  };
}
