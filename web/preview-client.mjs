/** Execute selected HTML only in the separate, CSP-restricted opaque-origin frame. */
export class AppPreview {
  constructor({ container, onResult = () => {} }) {
    this.container = container; this.onResult = onResult; this.pending = null; this.frame = null;
    this.listen = event => {
      if (event.source !== this.frame?.contentWindow) return;
      if(event.data?.type==='evolve-preview-error'&&event.data.id===this.id&&typeof event.data.message==='string'){this.onResult({ok:false,late:true,errors:[event.data.message.slice(0,1000)]});return}
      if(!this.pending)return;
      const data = event.data || {}, pending = this.pending;
      if (data.type === 'evolve-preview-ready' && !pending.sent) {
        pending.sent = true;
        this.frame.contentWindow.postMessage({ type: 'evolve-preview-load', id: pending.id, html: pending.html, smoke: pending.smoke }, '*');
      } else if (data.type === 'evolve-preview-result' && data.id === pending.id && data.result && typeof data.result === 'object') {
        // Preview output is untrusted. Bound and copy it before showing it or sending it to Pi.
        let result;
        try { result = JSON.parse(JSON.stringify(data.result).slice(0, 12000)); } catch { return; }
        this.finish(null, result); this.onResult(result);
      }
    };
    addEventListener('message', this.listen);
  }
  finish(error, result) {
    const pending = this.pending; if (!pending) return;
    this.pending = null; clearTimeout(pending.timer); pending.signal?.removeEventListener('abort', pending.abort);
    error ? pending.reject(error) : pending.resolve(result);
  }
  async open(html, { signal, smoke = false } = {}) {
    if (typeof html !== 'string' || new TextEncoder().encode(html).byteLength > 524288) throw new Error('Preview supports HTML files up to 512 KiB.');
    signal?.throwIfAborted(); this.stop();
    const fenced = /^\s*```(?:html)?\s*([\s\S]*?)\s*```\s*$/.exec(html); if (fenced) html = fenced[1];
    this.frame = document.createElement('iframe'); this.frame.title = 'Workspace app preview';
    this.frame.setAttribute('sandbox', 'allow-scripts'); this.frame.referrerPolicy = 'no-referrer';
    const frame = this.frame;
    const result = new Promise((resolve, reject) => {
      const abort = () => { this.finish(new Error('Preview cancelled.')); frame.remove(); };
      this.pending = { id: crypto.randomUUID(), html, smoke, resolve, reject, signal, abort,
        timer: setTimeout(() => { this.finish(new Error('Preview did not report within 10 seconds. Check for an infinite loop or blocked script.')); frame.remove(); }, 10000) };
      signal?.addEventListener('abort', abort, { once: true });
    });
    this.id=this.pending.id;
    frame.src = new URL('./preview/', import.meta.url).href; this.container.replaceChildren(frame);
    return result;
  }
  stop() { this.finish(new Error('Preview stopped.')); this.frame?.remove(); this.frame = null; }
  dispose() { this.stop(); removeEventListener('message', this.listen); }
}
