/**
 * The workbench's single shared browser: the panel the person sees and the page Pi uses.
 *
 * For Pi tools: `const browser = await ensureBrowser({ show: true })` opens the panel (so the
 * person watches) and returns the BrowserClient; then use browser.navigate(), waitForIdle(),
 * snapshot(), click(), type(), select(), scroll(), key() (browser/API.md section 4).
 *
 * pi-agent.mjs is bundled by esbuild into pi-agent.bundle.mjs. Importing this module from it
 * would bundle a second copy with its own (empty) singleton, so pass the accessor in instead,
 * e.g. from workbench.mjs: `createPiTask({ ..., browser: () => import('./browser-session.mjs').then(m => m.ensureBrowser({ show: true })) })`.
 */

let client = null;
let pending = null;
let panel = null;

function loadStyles() {
  if (document.querySelector('link[data-browser-styles]')) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('./browser.css', import.meta.url).href;
  link.dataset.browserStyles = '';
  document.head.append(link);
}

/**
 * The relay this deployment provides, from browser-config.json next to this module
 * (`{"relayUrl": "wss://…/"}`), or undefined for a relay on this computer.
 */
async function siteRelay() {
  try {
    const response = await fetch(new URL('./browser-config.json', import.meta.url), { cache: 'no-cache' });
    if (!response.ok) return undefined;
    const config = await response.json();
    const url = typeof config?.relayUrl === 'string' ? config.relayUrl.trim() : '';
    return /^wss?:\/\//.test(url) ? url : undefined;
  } catch {
    return undefined;
  }
}

/** The browser if it has been created, else null. */
export function getBrowser() {
  return client;
}

/** Create the browser (once) and return it. `show` opens the panel. */
export async function ensureBrowser({ show = false } = {}) {
  if (!client) {
    pending ||= (async () => {
      loadStyles();
      const { BrowserClient } = await import('./browser-client.mjs');
      const relayUrl = await siteRelay();
      panel = document.createElement('section');
      panel.id = 'browser-panel';
      panel.className = 'browser-window';
      panel.hidden = !show;
      panel.setAttribute('aria-label', 'Web browser');
      document.body.append(panel);
      const c = new BrowserClient({ relayUrl });
      c.mount(panel, { onClose: closeBrowser });
      client = c;
      return c;
    })().catch((error) => { pending = null; panel?.remove(); panel = null; throw error; });
    await pending;
  }
  if (show) panel.hidden = false;
  return client;
}

export async function openBrowser() {
  const c = await ensureBrowser({ show: true });
  panel.hidden = false;
  if (!c.state.url) panel.querySelector('.browser-address-input')?.focus();
  return c;
}

/** Hide the panel. The page keeps running so Pi can continue and the person can return. */
export function closeBrowser() {
  if (panel) panel.hidden = true;
}

export function isBrowserOpen() {
  return Boolean(panel && !panel.hidden);
}
