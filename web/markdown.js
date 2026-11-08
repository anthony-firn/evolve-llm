// A small Markdown subset for model replies. It parses into a plain tree and builds DOM
// nodes with textContent only, so model output can never inject HTML or scripts.

/** Split a reply into reasoning (<think>…</think>) and answer parts. */
export function splitThinking(text) {
  const parts = [];
  let rest = text;
  while (rest.length) {
    const open = rest.indexOf('<think>');
    if (open === -1) { parts.push({ kind: 'answer', text: rest }); break; }
    if (open > 0) parts.push({ kind: 'answer', text: rest.slice(0, open) });
    const close = rest.indexOf('</think>', open + 7);
    if (close === -1) { parts.push({ kind: 'think', text: rest.slice(open + 7), done: false }); break; }
    parts.push({ kind: 'think', text: rest.slice(open + 7, close), done: true });
    rest = rest.slice(close + 8);
  }
  return parts.filter((p) => p.kind === 'think' || p.text.trim() !== '');
}

/** Block structure: paragraphs, headings, lists, quotes, rules and fenced code. */
export function parseBlocks(src) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*(```+|~~~+)\s*([\w+#.-]*)\s*$/);
    if (fence) {
      const marker = fence[1];
      const body = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        if (lines[i].trim().startsWith(marker[0].repeat(marker.length)) && lines[i].trim().replace(/[`~]/g, '') === '') { closed = true; i++; break; }
        body.push(lines[i]);
        i++;
      }
      const text = body.join('\n');
      blocks.push({ type: 'code', lang: fence[2] || '', text: closed ? text : text.replace(/\n$/, ''), closed });
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) { blocks.push({ type: 'heading', level: heading[1].length, text: heading[2] }); i++; continue; }
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) { blocks.push({ type: 'rule' }); i++; continue; }
    if (/^\s{0,3}>/.test(line)) {
      const body = [];
      while (i < lines.length && /^\s{0,3}>/.test(lines[i])) { body.push(lines[i].replace(/^\s{0,3}>\s?/, '')); i++; }
      blocks.push({ type: 'quote', children: parseBlocks(body.join('\n')) });
      continue;
    }
    const item = line.match(/^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/);
    if (item) {
      const ordered = /\d/.test(item[2]);
      const start = ordered ? parseInt(item[2], 10) : 1;
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/);
        if (m && /\d/.test(m[2]) === ordered && m[1].length <= item[1].length + 1) {
          items.push([m[3]]);
          i++;
        } else if (lines[i].trim() && /^\s+/.test(lines[i]) && items.length) {
          items[items.length - 1].push(lines[i].trim());
          i++;
        } else break;
      }
      blocks.push({ type: 'list', ordered, start, items: items.map((t) => t.join('\n')) });
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(```|~~~)/.test(lines[i]) && !/^\s{0,3}(#{1,6}\s|>)/.test(lines[i]) && !/^(\s*)([-*+]|\d{1,9}[.)])\s+/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    blocks.push({ type: 'paragraph', text: para.join('\n') });
  }
  return blocks;
}

/** Inline spans: code, bold, italic, links (http/https only) and line breaks. */
export function parseInline(text) {
  const out = [];
  const re = /(`+)([\s\S]*?[^`])\1(?!`)|\*\*([^*\n][\s\S]*?)\*\*|__([^_\n][\s\S]*?)__|\*([^*\s][^*\n]*?)\*|(?<![\w])_([^_\s][^_\n]*?)_(?![\w])|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])|\n/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ type: 'text', text: text.slice(last, m.index) });
    if (m[1]) out.push({ type: 'code', text: m[2] });
    else if (m[3] !== undefined || m[4] !== undefined) out.push({ type: 'strong', children: parseInline(m[3] ?? m[4]) });
    else if (m[5] !== undefined || m[6] !== undefined) out.push({ type: 'em', children: parseInline(m[5] ?? m[6]) });
    else if (m[7] !== undefined) out.push({ type: 'link', href: m[8], children: parseInline(m[7]) });
    else if (m[9] !== undefined) out.push({ type: 'link', href: m[9], children: [{ type: 'text', text: m[9] }] });
    else out.push({ type: 'break' });
    last = re.lastIndex;
  }
  if (last < text.length) out.push({ type: 'text', text: text.slice(last) });
  return out;
}

function inlineNodes(doc, spans) {
  const frag = doc.createDocumentFragment();
  for (const s of spans) {
    let node;
    switch (s.type) {
      case 'text': node = doc.createTextNode(s.text); break;
      case 'break': node = doc.createElement('br'); break;
      case 'code': node = doc.createElement('code'); node.textContent = s.text; break;
      case 'strong': node = doc.createElement('strong'); node.append(inlineNodes(doc, s.children)); break;
      case 'em': node = doc.createElement('em'); node.append(inlineNodes(doc, s.children)); break;
      case 'link':
        node = doc.createElement('a');
        node.href = s.href;
        node.rel = 'noopener noreferrer nofollow';
        node.target = '_blank';
        node.append(inlineNodes(doc, s.children));
        break;
      default: continue;
    }
    frag.append(node);
  }
  return frag;
}

/** Build DOM for Markdown text. `onCopy(text)` is called by code-block copy buttons. */
export function renderMarkdown(doc, text, onCopy) {
  const frag = doc.createDocumentFragment();
  const build = (blocks, parent) => {
    for (const b of blocks) {
      let el;
      switch (b.type) {
        case 'heading':
          el = doc.createElement(`h${Math.min(6, b.level + 2)}`);
          el.append(inlineNodes(doc, parseInline(b.text)));
          break;
        case 'rule': el = doc.createElement('hr'); break;
        case 'quote': el = doc.createElement('blockquote'); build(b.children, el); break;
        case 'list': {
          el = doc.createElement(b.ordered ? 'ol' : 'ul');
          if (b.ordered && b.start !== 1) el.start = b.start;
          for (const item of b.items) {
            const li = doc.createElement('li');
            li.append(inlineNodes(doc, parseInline(item)));
            el.append(li);
          }
          break;
        }
        case 'code': {
          el = doc.createElement('figure');
          el.className = 'code';
          const bar = doc.createElement('figcaption');
          const lang = doc.createElement('span');
          lang.textContent = b.lang || 'text';
          bar.append(lang);
          if (onCopy) {
            const btn = doc.createElement('button');
            btn.type = 'button';
            btn.className = 'copy-code';
            btn.textContent = 'Copy';
            btn.addEventListener('click', () => onCopy(b.text, btn));
            bar.append(btn);
          }
          const pre = doc.createElement('pre');
          const code = doc.createElement('code');
          code.textContent = b.text;
          pre.append(code);
          el.append(bar, pre);
          break;
        }
        default:
          el = doc.createElement('p');
          el.append(inlineNodes(doc, parseInline(b.text)));
      }
      parent.append(el);
    }
  };
  build(parseBlocks(text), frag);
  return frag;
}
