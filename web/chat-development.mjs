import { AppPreview } from './preview-client.mjs';

// A web address: https://… or a bare domain such as news.ycombinator.com (not file names like app.js).
const WEB_ADDRESS = /\bhttps?:\/\/[^\s<>"']+|\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|org|net|io|dev|ai|app|edu|gov|co|uk|us|ca|de|fr|eu|news|info|me|tv|xyz|site|tech|blog|so|gg|sh|ly)\b(?:\/[^\s<>"']*)?/i;
const BROWSER_ACTION = /\b(?:go to|goto|open|visit|browse|navigate|log ?in|sign ?in|click|press|tap|fill (?:in|out)|type|search|look up|check|find|scroll|read|show|order|book|subscribe|download|upvote|vote|reply|post|comment)\b/i;
const BROWSER_PLACE = /\b(?:website|web ?site|site|web ?page|webpage|browser|tab|link|button|form|account|inbox|dashboard|front page|homepage|home page)\b/i;

/** Whether a request is about using a live web page (Pi with the shared browser). */
export function wantsBrowser(text) {
  const t = String(text);
  if (/^\s*(?:explain|teach|can you explain|tell me (?:how|why|about)|how (?:do|does|can|should) (?:i|you|we|one)\b)/i.test(t) && !/\bhttps?:\/\//i.test(t)) return false;
  if (/\b(?:in|with|using|on) (?:the|my|your) (?:web )?browser\b/i.test(t)) return true;
  if (WEB_ADDRESS.test(t)) return !/^\s*(?:what is|what's|define)\s+(?:https?:\/\/)?[\w.-]+\s*\??\s*$/i.test(t);
  return BROWSER_ACTION.test(t) && BROWSER_PLACE.test(t) && !/\b[\w-]+\.(?:html?|css|[cm]?js|json|txt|md|csv)\b|\/workspace\//i.test(t);
}

export function wantsDevelopment(text, mode = 'auto', active = false) {
  if (mode === 'chat') return false;
  if (mode === 'build') return true;
  if (/^\s*pi\s+/i.test(text)) return true;
  if (wantsBrowser(text)) return true;
  if (/^\s*(?:explain|teach|what\b|why\b|how\b|can you explain|tell me (?:how|why|about))/i.test(text)) return false;
  const action = /^\s*(?:(?:can|could|would) you\s+)?code\b/i.test(text) || /\b(make|build|create|develop|implement|write|fix|edit|modify|update|change|add|remove|run|test|preview|open|read|summarize|save|rename|delete)\b/i.test(text);
  const object = /\b[\w-]+\.(?:html?|css|[cm]?js|json|txt|md|csv)\b|\/workspace\//i.test(text) || /\b(app|application|game|snake|website|page|calculator|script|file|code|html|css|javascript|bug|project|workspace)\b/i.test(text);
  // While Pi works on a task, short replies (approvals, "I logged in") continue it.
  return action && (object || active) || active && /^\s*(?:try again|retry|do it|continue|go ahead|yes\b|yep\b|ok(?:ay)?\b|sure\b|confirm(?:ed)?\b|approved?\b|done\b|i(?:'m| am| have|'ve)? (?:logged|signed) in\b|logged in\b|signed in\b)/i.test(text);
}

export function wantsSnakeStarter(text) {
  const creation=/\b(make|build|create|develop|implement|code)\b/i.test(text);
  const newGame=/\b(?:a|an|new|another)\s+(?:[\w-]+\s+){0,2}snake\s+(?:game|clone)\b/i.test(text);
  const shortRequest=/\b(?:make|build|create|develop|implement|code)\s+(?:(?:me|us)\s+)?snake(?:\s+game)?[.!?]?\s*$/i.test(text);
  return creation&&(newGame||shortRequest);
}

export function developmentSummary(result, check) {
  const files = result.artifacts?.map(a => a.path.replace('/workspace/', '')) || [];
  const lines = [];
  if (files.length) lines.push(`Saved ${files.join(', ')}${result.starterPath ? ' using the included Snake starter' : ''}.`);
  else if(!result.removed?.length)lines.push('No workspace files changed.');
  if(result.removed?.length)lines.push('Deleted '+result.removed.map(path=>path.replace('/workspace/','')).join(', ')+'.');
  if (check?.ok) lines.push('The app opens in the preview below. Its runtime check passed; this does not verify every gameplay rule.');
  else if (check) lines.push('The app did not pass its runtime check. See the activity details.');
  if (result.error) lines.push(`Pi could not finish all of the work: ${result.error}`);
  else if (result.errors) lines.push(`${result.errors} tool error${result.errors === 1 ? '' : 's'} occurred. Review the saved result.`);
  else if (!files.length && !result.removed?.length && !check){
    if(result.answer&&result.calls)lines.push('Pi’s response:\n\n'+result.answer);
    else lines.push('Pi did not produce an artifact. Try a smaller task or a stronger coding model.');
  }
  return lines.join('\n\n');
}

const node = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const button = (text, fn) => { const b = node('button', text); b.type = 'button'; b.addEventListener('click', fn); return b; };

// DOM nodes belong to the message, so streaming updates never reload its iframe.
export function createDevelopmentView({ openWorkspace }) {
  const root = node('section', undefined, 'chat-development');
  const activity = node('details', undefined, 'development-activity'); activity.open = true;
  const status = node('summary', 'Preparing workspace…'), log = node('ol');
  activity.append(status, log);
  const files = node('div', undefined, 'development-files');
  const panel = node('section', undefined, 'chat-app'); panel.hidden = true;
  const title = node('strong'), actions = node('div', undefined, 'preview-actions'), host = node('div', undefined, 'chat-preview-host');
  panel.append(title, actions, host);
  const source = node('pre', undefined, 'development-source'); source.hidden = true;
  const preview = new AppPreview({container:host,onResult:result=>{if(result.late)add('Runtime error',result.errors?.join('\n'),true);}});
  root.append(activity, files, panel, source, button('Open workspace', openWorkspace));
  let current = null, lastCheck = null, stopped = false;
  function add(label, detail, error = false) {
    const item = node('li', undefined, error ? 'activity-error' : '');
    if (detail) {const d = node('details');d.append(node('summary',label),node('pre',String(detail).slice(0,6000)));item.append(d);}
    else item.textContent = label;
    log.append(item); while(log.children.length>40)log.firstElementChild.remove();
    status.textContent = label;
  }
  async function show(path, html, options={}) {
    if(stopped)throw new Error('This preview has been closed.');
    current = {path,html}; panel.hidden = false; title.textContent = path.replace('/workspace/', '');
    lastCheck=null;
    try{lastCheck=await preview.open(html,options);}
    catch(error){lastCheck={ok:false,errors:[String(error.message||error)]};add('Preview stopped',error.message,true);throw error;}
    add(lastCheck.ok?'Runtime check passed':'Runtime check failed', JSON.stringify(lastCheck,null,2),!lastCheck.ok);
    return lastCheck;
  }
  const fullSize=button('Full size',()=>{panel.classList.toggle('expanded');fullSize.textContent=panel.classList.contains('expanded')?'Exit full size':'Full size';});
  actions.append(fullSize,button('Reload',()=>show(current.path,current.html).catch(e=>add(e.message,null,true))),button('Test',()=>show(current.path,current.html,{smoke:true}).catch(e=>add(e.message,null,true))),button('Stop preview',()=>{preview.stop();panel.hidden=true;}));
  function event(e) {
    if(e.type==='starter')add('Created '+e.path.replace('/workspace/','')+' from the Snake starter');
    else if(e.type==='tool')add('Pi · '+e.name+(e.args?.path?' · '+e.args.path.replace('/workspace/',''):''),JSON.stringify(e.args,null,2));
    else if(e.type==='tool-result')add(e.name+(e.isError?' failed':' completed'),JSON.stringify(e.result,null,2),Boolean(e.isError));
    else if(e.type==='retry')add('Retrying the tool call',e.message);
    else if(e.type==='recovery')add(e.message,null,true);
  }
  function finish(result) {
    files.replaceChildren();
    for(const file of result.artifacts||[]){
      const row=node('div',undefined,'development-file');row.append(node('span',file.path.replace('/workspace/','')),
        button('Source',()=>{const same=source.dataset.path===file.path;source.textContent=file.text;source.hidden=same&&!source.hidden;source.dataset.path=file.path;}),
        button('Download',()=>{const url=URL.createObjectURL(new Blob([file.text],{type:'text/plain'}));const a=node('a');a.href=url;a.download=file.path.split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}));
      if(/\.html?$/i.test(file.path))row.append(button('Play',()=>show(file.path,file.text).catch(e=>add(e.message,null,true))));
      files.append(row);
    }
    for(const path of result.removed||[])files.append(node('div','Deleted '+path.replace('/workspace/',''),'development-file'));
    status.textContent=result.error?'Work stopped · review results':result.errors?'Work finished with tool errors':'Work finished';
    activity.open=Boolean(result.error||result.errors);
  }
  return {root,event,preview:show,finish,get check(){return lastCheck;},dispose(){stopped=true;preview.dispose();}};
}
