// Dedicated worker for the WASM browser engine: Blitz layout, Boa JavaScript, rustls networking.
import { createBrowserHost } from './browser-host.mjs';

async function loadFonts() {
  const base = new URL('./browser-fonts/', import.meta.url);
  const response = await fetch(new URL('fonts.json', base));
  if (!response.ok) throw new Error('The browser fonts are missing. Build them with `node tools/build_workbench.mjs`.');
  const manifest = await response.json();
  return Promise.all(manifest.files.map(async (name) => {
    const r = await fetch(new URL(name, base));
    if (!r.ok) throw new Error(`Font ${name} could not be loaded (HTTP ${r.status}).`);
    return [name, new Uint8Array(await r.arrayBuffer())];
  }));
}

async function load() {
  let pkg;
  try {
    pkg = await import('./browser-pkg/browser.js');
  } catch (e) {
    throw new Error('The browser engine is not built yet. Run `node tools/build_workbench.mjs` (it builds browser/ to web/browser-pkg/).');
  }
  const [, fonts] = await Promise.all([pkg.default(), loadFonts()]);
  return { Browser: pkg.Browser, fonts };
}

const host = createBrowserHost({ load, post: (message) => self.postMessage(message) });
self.onmessage = ({ data }) => { host.handle(data); };
self.addEventListener('unhandledrejection', (e) => { if (/unreachable|RuntimeError/.test(String(e.reason))) e.preventDefault(); });
