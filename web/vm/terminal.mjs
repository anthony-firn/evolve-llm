// A terminal window for the Evolve VM, laid out and driven like GNOME Terminal: tabs, a menu,
// Find, zoom, profiles, and the same shortcuts. Each tab is a shell on its own VM terminal; the
// emulator is ghostty-web (Ghostty's terminal core compiled to WebAssembly).

import { FitAddon, Ghostty, Terminal } from '../vendor/ghostty-web/ghostty-web.mjs';

const THEMES = {
  Dark: { background: '#1e1e1e', foreground: '#d0cfcc', cursor: '#d0cfcc', selectionBackground: '#4a6c9b', black: '#171421', red: '#c01c28', green: '#26a269', yellow: '#a2734c', blue: '#12488b', magenta: '#a347ba', cyan: '#2aa1b3', white: '#d0cfcc', brightBlack: '#5e5c64', brightRed: '#f66151', brightGreen: '#33d17a', brightYellow: '#e9ad0c', brightBlue: '#2a7bde', brightMagenta: '#c061cb', brightCyan: '#33c7de', brightWhite: '#ffffff' },
  Light: { background: '#ffffff', foreground: '#171421', cursor: '#171421', selectionBackground: '#b5d5ff', black: '#171421', red: '#c01c28', green: '#26a269', yellow: '#a2734c', blue: '#12488b', magenta: '#a347ba', cyan: '#2aa1b3', white: '#d0cfcc', brightBlack: '#5e5c64', brightRed: '#f66151', brightGreen: '#33d17a', brightYellow: '#e9ad0c', brightBlue: '#2a7bde', brightMagenta: '#c061cb', brightCyan: '#33c7de', brightWhite: '#ffffff' },
  'Solarized Dark': { background: '#002b36', foreground: '#839496', cursor: '#93a1a1', selectionBackground: '#073642', black: '#073642', red: '#dc322f', green: '#859900', yellow: '#b58900', blue: '#268bd2', magenta: '#d33682', cyan: '#2aa198', white: '#eee8d5', brightBlack: '#586e75', brightRed: '#cb4b16', brightGreen: '#586e75', brightYellow: '#657b83', brightBlue: '#839496', brightMagenta: '#6c71c4', brightCyan: '#93a1a1', brightWhite: '#fdf6e3' },
  Ubuntu: { background: '#300a24', foreground: '#eeeeec', cursor: '#eeeeec', selectionBackground: '#6a4a62', black: '#2e3436', red: '#cc0000', green: '#4e9a06', yellow: '#c4a000', blue: '#3465a4', magenta: '#75507b', cyan: '#06989a', white: '#d3d7cf', brightBlack: '#555753', brightRed: '#ef2929', brightGreen: '#8ae234', brightYellow: '#fce94f', brightBlue: '#729fcf', brightMagenta: '#ad7fa8', brightCyan: '#34e2e2', brightWhite: '#eeeeec' },
};
const PREFS_KEY = 'evolve-terminal-prefs';
const FONT = '"DejaVu Sans Mono", "Ubuntu Mono", "Cascadia Mono", Menlo, Consolas, monospace';

let engine = null;
function loadEngine() {
  engine ??= Ghostty.load(new URL('../vendor/ghostty-web/ghostty-vt.wasm', import.meta.url).href);
  return engine;
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function loadPrefs() {
  try {
    return { theme: 'Dark', fontSize: 14, ...JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') };
  } catch {
    return { theme: 'Dark', fontSize: 14 };
  }
}

export class TerminalWindow {
  /** `vm`: a VmClient. `container`: where the window goes. */
  constructor({ vm, container }) {
    this.vm = vm;
    this.prefs = loadPrefs();
    this.tabs = [];
    this.active = null;
    this.root = el('section', 'vt-window');
    this.root.setAttribute('aria-label', 'Terminal');
    const header = el('header', 'vt-header');
    this.tabStrip = el('div', 'vt-tabs');
    this.tabStrip.setAttribute('role', 'tablist');
    const add = el('button', 'vt-icon vt-new', '+');
    add.title = 'New Tab (Ctrl+Shift+T)';
    add.setAttribute('aria-label', 'New tab');
    add.onclick = () => this.newTab();
    const findBtn = el('button', 'vt-icon', '⌕');
    findBtn.title = 'Find (Ctrl+Shift+F)';
    findBtn.setAttribute('aria-label', 'Find');
    findBtn.onclick = () => this.toggleFind();
    const menuBtn = el('button', 'vt-icon', '☰');
    menuBtn.title = 'Menu';
    menuBtn.setAttribute('aria-label', 'Terminal menu');
    menuBtn.setAttribute('aria-haspopup', 'menu');
    this.menu = this.buildMenu();
    menuBtn.onclick = (e) => {
      e.stopPropagation();
      this.menu.hidden = !this.menu.hidden;
    };
    document.addEventListener('click', () => { this.menu.hidden = true; });
    header.append(this.tabStrip, add, el('span', 'vt-spacer'), findBtn, menuBtn, this.menu);
    this.findBar = this.buildFind();
    this.body = el('div', 'vt-body');
    this.root.append(header, this.findBar, this.body);
    container.append(this.root);
    this.applyTheme();
    new ResizeObserver(() => this.fit()).observe(this.body);
  }

  buildMenu() {
    const menu = el('div', 'vt-menu');
    menu.hidden = true;
    menu.setAttribute('role', 'menu');
    const item = (label, keys, fn) => {
      const b = el('button', 'vt-menu-item');
      b.setAttribute('role', 'menuitem');
      b.append(el('span', '', label), el('kbd', '', keys || ''));
      b.onclick = () => {
        menu.hidden = true;
        fn();
        this.active?.term.focus();
      };
      menu.append(b);
    };
    item('New Tab', 'Ctrl+Shift+T', () => this.newTab());
    item('Close Tab', 'Ctrl+Shift+W', () => this.closeTab(this.active));
    menu.append(el('hr'));
    item('Copy', 'Ctrl+Shift+C', () => this.copy());
    item('Paste', 'Ctrl+Shift+V', () => this.paste());
    item('Select All', 'Ctrl+Shift+A', () => this.active?.term.selectAll());
    item('Find…', 'Ctrl+Shift+F', () => this.toggleFind(true));
    menu.append(el('hr'));
    item('Zoom In', 'Ctrl++', () => this.zoom(1));
    item('Zoom Out', 'Ctrl+−', () => this.zoom(-1));
    item('Normal Size', 'Ctrl+0', () => this.zoom(0));
    menu.append(el('hr'));
    for (const name of Object.keys(THEMES)) item(`${name === this.prefs.theme ? '● ' : ''}${name} theme`, '', () => this.setTheme(name));
    menu.append(el('hr'));
    item('Reset and Clear', '', () => this.active?.term.reset());
    return menu;
  }

  buildFind() {
    const bar = el('div', 'vt-find');
    bar.hidden = true;
    this.findInput = el('input');
    this.findInput.type = 'search';
    this.findInput.placeholder = 'Find';
    this.findInput.setAttribute('aria-label', 'Find in terminal');
    this.findCount = el('span', 'vt-find-count');
    const prev = el('button', 'vt-icon', '↑');
    prev.setAttribute('aria-label', 'Previous match');
    const next = el('button', 'vt-icon', '↓');
    next.setAttribute('aria-label', 'Next match');
    const close = el('button', 'vt-icon', '✕');
    close.setAttribute('aria-label', 'Close find');
    prev.onclick = () => this.find(-1);
    next.onclick = () => this.find(1);
    close.onclick = () => this.toggleFind(false);
    this.findInput.oninput = () => this.find(0);
    this.findInput.onkeydown = (e) => {
      if (e.key === 'Enter') this.find(e.shiftKey ? -1 : 1);
      if (e.key === 'Escape') this.toggleFind(false);
    };
    bar.append(this.findInput, this.findCount, prev, next, close);
    return bar;
  }

  /** Open the first tab. */
  async open() {
    await loadEngine();
    await this.newTab();
  }

  async newTab() {
    const ghostty = await loadEngine();
    const host = el('div', 'vt-term');
    this.body.append(host);
    const term = new Terminal({ ghostty, fontSize: this.prefs.fontSize, fontFamily: FONT, theme: THEMES[this.prefs.theme], scrollback: 10000, cursorBlink: true });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(host);
    const tabBtn = el('button', 'vt-tab');
    tabBtn.setAttribute('role', 'tab');
    const label = el('span', 'vt-tab-title', 'user@evolve: ~');
    const x = el('span', 'vt-tab-close', '✕');
    x.setAttribute('aria-label', 'Close tab');
    tabBtn.append(label, x);
    const tab = { term, fit, host, tabBtn, label, tty: null };
    tabBtn.onclick = (e) => {
      if (e.target === x) this.closeTab(tab);
      else this.focusTab(tab);
    };
    this.tabStrip.append(tabBtn);
    this.tabs.push(tab);
    term.attachCustomKeyEventHandler((e) => this.shortcut(e));
    term.onTitleChange((t) => { label.textContent = t || 'Terminal'; });
    tab.tty = await this.vm.session({
      onOutput: (data) => term.write(data),
      onExit: () => this.closeTab(tab, true),
    });
    term.onData((d) => this.vm.input(tab.tty, d));
    term.onResize(({ cols, rows }) => this.vm.resize(tab.tty, cols, rows));
    this.focusTab(tab);
    return tab;
  }

  focusTab(tab) {
    if (!tab) return;
    this.active = tab;
    for (const t of this.tabs) {
      t.host.hidden = t !== tab;
      t.tabBtn.setAttribute('aria-selected', String(t === tab));
      t.tabBtn.classList.toggle('active', t === tab);
    }
    this.fit();
    tab.term.focus();
  }

  closeTab(tab, exited = false) {
    if (!tab || !this.tabs.includes(tab)) return;
    if (!exited && tab.tty !== null) this.vm.close(tab.tty);
    tab.term.dispose();
    tab.host.remove();
    tab.tabBtn.remove();
    const i = this.tabs.indexOf(tab);
    this.tabs.splice(i, 1);
    if (this.active === tab) this.focusTab(this.tabs[Math.min(i, this.tabs.length - 1)]);
    if (!this.tabs.length) this.newTab();
  }

  fit() {
    if (!this.active || this.active.host.hidden) return;
    try {
      this.active.fit.fit();
    } catch { /* not laid out yet */ }
    const { cols, rows } = this.active.term;
    if (this.active.tty !== null) this.vm.resize(this.active.tty, cols, rows);
  }

  /**
   * GNOME Terminal's shortcuts. Returns true when it handled the key (ghostty-web then keeps it
   * from the program; note this is the reverse of xterm.js).
   */
  shortcut(e) {
    if (e.type !== 'keydown') return false;
    const k = e.key.toLowerCase();
    const run = (fn) => {
      e.preventDefault();
      fn();
      return true;
    };
    if (e.ctrlKey && e.shiftKey) {
      if (k === 't') return run(() => this.newTab());
      if (k === 'w') return run(() => this.closeTab(this.active));
      if (k === 'c') return run(() => this.copy());
      if (k === 'v') return run(() => this.paste());
      if (k === 'f') return run(() => this.toggleFind(true));
      if (k === 'a') return run(() => this.active.term.selectAll());
    }
    if (e.ctrlKey && !e.shiftKey && !e.altKey) {
      if (k === '=' || k === '+') return run(() => this.zoom(1));
      if (k === '-') return run(() => this.zoom(-1));
      if (k === '0') return run(() => this.zoom(0));
      if (k === 'pageup' || k === 'pagedown') {
        return run(() => {
          const i = this.tabs.indexOf(this.active) + (k === 'pageup' ? -1 : 1);
          this.focusTab(this.tabs[(i + this.tabs.length) % this.tabs.length]);
        });
      }
    }
    if (e.altKey && !e.ctrlKey && /^[1-9]$/.test(e.key) && this.tabs[Number(e.key) - 1]) return run(() => this.focusTab(this.tabs[Number(e.key) - 1]));
    return false;
  }

  async copy() {
    const text = this.active?.term.getSelection();
    if (text) await navigator.clipboard?.writeText(text).catch(() => {});
  }

  async paste() {
    const text = await navigator.clipboard?.readText().catch(() => '');
    if (text) this.active?.term.paste(text);
  }

  zoom(step) {
    this.prefs.fontSize = step === 0 ? 14 : Math.min(32, Math.max(8, this.prefs.fontSize + step));
    for (const t of this.tabs) t.term.options.fontSize = this.prefs.fontSize;
    this.savePrefs();
    this.fit();
  }

  setTheme(name) {
    this.prefs.theme = name;
    for (const t of this.tabs) t.term.options.theme = THEMES[name];
    this.savePrefs();
    this.applyTheme();
    this.menu.replaceWith((this.menu = this.buildMenu()));
  }

  applyTheme() {
    const t = THEMES[this.prefs.theme];
    this.root.style.setProperty('--vt-bg', t.background);
    this.root.style.setProperty('--vt-fg', t.foreground);
  }

  savePrefs() {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(this.prefs)); } catch { /* storage unavailable */ }
  }

  toggleFind(show = this.findBar.hidden) {
    this.findBar.hidden = !show;
    if (show) {
      this.findInput.focus();
      this.findInput.select();
      this.find(0);
    } else {
      this.active?.term.clearSelection();
      this.active?.term.focus();
    }
  }

  /** Every match of the Find text in the active tab's scrollback and screen. */
  matches() {
    const q = this.findInput.value.toLowerCase();
    const out = [];
    const term = this.active?.term;
    if (!q || !term) return out;
    const buf = term.buffer.active;
    for (let y = 0; y < buf.length; y++) {
      const line = buf.getLine(y)?.translateToString(true).toLowerCase() || '';
      for (let x = line.indexOf(q); x !== -1; x = line.indexOf(q, x + 1)) out.push([y, x]);
    }
    return out;
  }

  /** Select the next (1), previous (-1) or first-from-the-end (0) match. */
  find(dir) {
    const all = this.matches();
    if (!all.length) {
      this.findCount.textContent = this.findInput.value ? 'No matches' : '';
      this.active?.term.clearSelection();
      return 0;
    }
    if (dir === 0 || this.findAt === undefined) this.findAt = all.length - 1;
    else this.findAt = (this.findAt + dir + all.length) % all.length;
    const [y, x] = all[this.findAt];
    const term = this.active.term;
    term.select(x, y, this.findInput.value.length);
    term.scrollToLine(Math.max(0, y - Math.floor(term.rows / 2)));
    this.findCount.textContent = `${this.findAt + 1} of ${all.length}`;
    return all.length;
  }

  /** The active tab's visible text (tests and Pi's view of the terminal). */
  text() {
    const buf = this.active?.term.buffer.active;
    if (!buf) return '';
    const lines = [];
    for (let y = 0; y < buf.length; y++) lines.push(buf.getLine(y)?.translateToString(true) ?? '');
    return lines.join('\n').replace(/\n+$/, '');
  }
}
