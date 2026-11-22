import { AppPreview } from './preview-client.mjs';

// A web address: https://… or a bare domain such as news.ycombinator.com (not file names like app.js).
const WEB_ADDRESS = /\bhttps?:\/\/[^\s<>"']+|\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|org|net|io|dev|ai|app|edu|gov|co|uk|us|ca|de|fr|eu|news|info|me|tv|xyz|site|tech|blog|so|gg|sh|ly)\b(?:\/[^\s<>"']*)?/i;
const BROWSER_ACTION = /\b(?:go to|goto|open|visit|browse|navigate|log ?in|sign ?in|click|press|tap|fill (?:in|out)|type|search|look up|check|find|scroll|read|show|order|book|subscribe|download|upvote|vote|reply|post|comment)\b/i;
const BROWSER_PLACE = /\b(?:website|web ?site|site|web ?page|webpage|browser|tab|link|button|form|account|inbox|dashboard|front page|homepage|home page)\b/i;

/** Whether a request is about using a live web page (the assistant then gets the shared browser). */
export function wantsBrowser(text) {
  const t = String(text);
  if (/^\s*(?:explain|teach|can you explain|tell me (?:how|why|about)|how (?:do|does|can|should) (?:i|you|we|one)\b)/i.test(t) && !/\bhttps?:\/\//i.test(t)) return false;
  if (/\b(?:in|with|using|on) (?:the|my|your) (?:web )?browser\b/i.test(t)) return true;
  if (WEB_ADDRESS.test(t)) return !/^\s*(?:what is|what's|define)\s+(?:https?:\/\/)?[\w.-]+\s*\??\s*$/i.test(t);
  return BROWSER_ACTION.test(t) && BROWSER_PLACE.test(t) && !/\b[\w-]+\.(?:html?|css|[cm]?js|json|txt|md|csv)\b|\/workspace\//i.test(t);
}

const LAW = /§|\b(?:laws?|legal(?:ly)?|statutes?|ordinances?|regulations?|(?:city|civil|penal|tax|municipal) code|court|lawsuit|sue|liab(?:le|ility)|contract|lease|landlord|tenants?|eviction|attorney|lawyer|copyright|trademark|patent|GDPR|CCPA|compliance)\b/i;
const CODING = /\b(?:code|coding|function|bug|fix|tests?|compile|build|refactor|repo(?:sitory)?|clone|commit|branch|pull request|PR|merge|git|gh|CI|npm|cargo|python3?|javascript|typescript|rust|html|css|script|app|game|api|sql)\b|\.(?:py|[cm]?js|ts|rs|html?|css|json|md|toml|ya?ml)\b/i;
const RESEARCH = /\b(?:look (?:up|into)|research|search (?:for|the web)|find out|sources?|cite|citation|according to|latest|news|who (?:is|was|were|designed|invented|founded|wrote)|when (?:was|did|is|were)|what year|how many)\b/i;

/** What kind of work a request is: browser (a live web page), law, coding, research or chat.
 *  Used to pick the model that does that kind of work best on this device. */
export function kindOf(text) {
  const t = String(text);
  if (wantsBrowser(t)) return 'browser';
  if (LAW.test(t)) return 'law';
  if (CODING.test(t)) return 'coding';
  if (RESEARCH.test(t)) return 'research';
  return 'chat';
}

export function wantsSnakeStarter(text) {
  const creation=/\b(make|build|create|develop|implement|code)\b/i.test(text);
  const newGame=/\b(?:a|an|new|another)\s+(?:[\w-]+\s+){0,2}snake\s+(?:game|clone)\b/i.test(text);
  const shortRequest=/\b(?:make|build|create|develop|implement|code)\s+(?:(?:me|us)\s+)?snake(?:\s+game)?[.!?]?\s*$/i.test(text);
  return creation&&(newGame||shortRequest);
}

const node = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const button = (text, fn) => { const b = node('button', text); b.type = 'button'; b.addEventListener('click', fn); return b; };
const fileName = (p) => String(p || '').replace(/\/+$/, '').split('/').pop();
const site = (u) => String(u || '').replace(/^https?:\/\//, '').slice(0, 80);

/** A step of the assistant's work, as the person reads it. */
export function stepLabel(name, args = {}) {
  switch (name) {
    case 'bash': return `Ran ${String(args.command || '').split('\n').filter(Boolean).join('; ').slice(0, 140)}`;
    case 'read': return `Read ${fileName(args.path)}`;
    case 'write': return `Wrote ${fileName(args.path)}`;
    case 'edit': return `Edited ${fileName(args.path)}`;
    case 'web_search': return `Searched the web for “${String(args.query || '').slice(0, 100)}”`;
    case 'open_url': return `Read ${site(args.url)}`;
    case 'preview': return `Opened ${fileName(args.path)}`;
    case 'browser_test': return `Tested ${fileName(args.path)}`;
    case 'browser_open': return `Opened ${site(args.url)}`;
    default: return name.startsWith('browser_') ? 'Used the browser' : name;
  }
}

const outputText = (result) => (result?.content || []).map((c) => c.text || '').join('\n') || (result === undefined ? '' : JSON.stringify(result, null, 2));

// The assistant's work under its reply: one collapsed line of steps (each with its output) and, when
// it opens an app, a live preview. DOM nodes belong to the message, so streaming never reloads it.
// `record` is what a saved chat keeps; `restore` rebuilds the view from it, with a button that
// reopens the app through `openFile(path)` -> {path, text}.
export function createWorkView({ restore = null, openFile = null } = {}) {
  const root = node('section', undefined, 'chat-work');
  const steps = node('details', undefined, 'work-steps'); steps.hidden = true;
  const summary = node('summary', 'Working…'), list = node('ol');
  steps.append(summary, list);
  const panel = node('section', undefined, 'chat-app'); panel.hidden = true;
  const title = node('strong'), actions = node('div', undefined, 'preview-actions'), host = node('div', undefined, 'chat-preview-host');
  panel.append(title, actions, host);
  const preview = new AppPreview({ container: host, onResult: (result) => { if (result.late) add('The app reported an error', result.errors?.join('\n'), true); } });
  root.append(steps, panel);
  let current = null, lastCheck = null, stopped = false, count = 0, failed = 0, live = null;
  const saved = [];
  function add(label, detail = '', error = false) {
    saved.push({ label, output: String(detail).slice(0, 2000), error });
    const item = node('li', undefined, error ? 'step-error' : '');
    const d = node('details'), out = node('pre', String(detail).slice(0, 6000));
    d.append(node('summary', label), out); item.append(d); list.append(item);
    while (list.children.length > 60) list.firstElementChild.remove();
    steps.hidden = false;
    summary.textContent = `Working · ${label}`;
    return { item, out };
  }
  async function show(path, html, options = {}) {
    if (stopped) throw new Error('This preview has been closed.');
    current = { path, html }; panel.hidden = false; title.textContent = fileName(path);
    lastCheck = null;
    try { lastCheck = await preview.open(html, options); }
    catch (error) { lastCheck = { ok: false, errors: [String(error.message || error)] }; throw error; }
    return lastCheck;
  }
  const fullSize = button('Full size', () => { panel.classList.toggle('expanded'); fullSize.textContent = panel.classList.contains('expanded') ? 'Exit full size' : 'Full size'; });
  actions.append(fullSize, button('Reload', () => show(current.path, current.html).catch(() => {})), button('Stop preview', () => { preview.stop(); panel.hidden = true; }));
  function event(e) {
    if (e.type === 'starter') { count++; add(`Started from the included Snake game (${fileName(e.path)})`); }
    else if (e.type === 'tool') { count++; live = add(stepLabel(e.name, e.args), ''); }
    else if (e.type === 'tool-result' && live) {
      live.out.textContent = outputText(e.result).slice(0, 6000);
      Object.assign(saved.at(-1), { output: live.out.textContent.slice(0, 2000), error: Boolean(e.isError) });
      if (e.isError) { failed++; live.item.classList.add('step-error'); }
      live = null;
    }
  }
  function finish() {
    summary.textContent = `Worked · ${count} step${count === 1 ? '' : 's'}${failed ? ` · ${failed} failed` : ''}`;
    steps.open = false;
  }
  if (restore) {
    for (const step of restore.steps || []) { count++; if (step.error) failed++; add(step.label, step.output, step.error); }
    finish();
    if (restore.preview && openFile) {
      const reopen = button(`Open ${fileName(restore.preview)}`, () => openFile(restore.preview).then((f) => { reopen.remove(); return show(f.path || restore.preview, f.text); }).catch((e) => { reopen.textContent = String(e.message || e); }));
      reopen.className = 'reopen'; root.append(reopen);
    }
  }
  return { root, event, preview: show, finish, get check() { return lastCheck; }, get steps() { return count; },
    get record() { return { steps: saved.slice(), preview: current?.path || restore?.preview || null }; },
    dispose() { stopped = true; preview.dispose(); } };
}
