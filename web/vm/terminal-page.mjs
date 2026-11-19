// A full-page terminal on the Evolve VM (terminal.html), also used by tests.
import { VmClient } from './client.mjs';
import { TerminalWindow } from './terminal.mjs';

const vm = new VmClient();
const win = new TerminalWindow({ vm, container: document.getElementById('term') });
window.vmTerminal = win;
window.vmClient = vm;
win.open().then(() => { window.vmTerminalReady = true; });
