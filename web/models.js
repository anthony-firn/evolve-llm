// Models offered in the picker. Files download directly from Hugging Face into this browser.
// Any other GGUF file on Hugging Face, or one already on the device, can be opened too.

const LLAMA = { license: 'Llama 3.2 Community License', licenseUrl: 'https://www.llama.com/llama3_2/license/' };
const APACHE = { license: 'Apache 2.0', licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0' };

export const MODELS = [
  { id: 'smollm2-135m', name: 'SmolLM2 135M', maker: 'Hugging Face', repo: 'bartowski/SmolLM2-135M-Instruct-GGUF',
    file: 'SmolLM2-135M-Instruct-Q8_0.gguf', bytes: 144811360, quant: 'Q8_0', note: 'Tiny. A quick first test.', ...APACHE },
  { id: 'smollm2-360m', name: 'SmolLM2 360M', maker: 'Hugging Face', repo: 'HuggingFaceTB/SmolLM2-360M-Instruct-GGUF',
    file: 'smollm2-360m-instruct-q8_0.gguf', bytes: 386404992, quant: 'Q8_0', note: 'Small English assistant.', ...APACHE },
  { id: 'qwen2.5-0.5b', name: 'Qwen2.5 0.5B', maker: 'Qwen', repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf', bytes: 491400032, quant: 'Q4_K_M', note: 'Fast and multilingual. Recommended.', recommended: true, ...APACHE },
  { id: 'qwen3-0.6b', name: 'Qwen3 0.6B', maker: 'Qwen', repo: 'Qwen/Qwen3-0.6B-GGUF',
    file: 'Qwen3-0.6B-Q8_0.gguf', bytes: 639446688, quant: 'Q8_0', note: 'Can think step by step before answering.', thinking: true, ...APACHE },
  { id: 'llama3.2-1b', name: 'Llama 3.2 1B', maker: 'Meta', repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
    file: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', bytes: 807694464, quant: 'Q4_K_M', note: 'General assistant with a long context.', ...LLAMA },
  { id: 'qwen2.5-coder-1.5b', name: 'Qwen2.5 Coder 1.5B', maker: 'Qwen', repo: 'bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF',
    file: 'Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf', bytes: 986048800, quant: 'Q4_K_M', note: 'Writes and explains code.', ...APACHE },
  { id: 'smollm2-1.7b', name: 'SmolLM2 1.7B', maker: 'Hugging Face', repo: 'HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF',
    file: 'smollm2-1.7b-instruct-q4_k_m.gguf', bytes: 1055609536, quant: 'Q4_K_M', note: 'Stronger English assistant.', ...APACHE },
  { id: 'qwen2.5-1.5b', name: 'Qwen2.5 1.5B', maker: 'Qwen', repo: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf', bytes: 1117320736, quant: 'Q4_K_M', note: 'Better answers, still quick on most GPUs.', ...APACHE },
  { id: 'qwen3-1.7b', name: 'Qwen3 1.7B', maker: 'Qwen', repo: 'Qwen/Qwen3-1.7B-GGUF',
    file: 'Qwen3-1.7B-Q8_0.gguf', bytes: 1834426016, quant: 'Q8_0', note: 'Thinking mode with more knowledge.', thinking: true, ...APACHE },
  { id: 'llama3.2-3b', name: 'Llama 3.2 3B', maker: 'Meta', repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
    file: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', bytes: 2019377696, quant: 'Q4_K_M', note: 'Needs about 3 GB of GPU memory.', ...LLAMA },
  { id: 'qwen3-4b', name: 'Qwen3 4B', maker: 'Qwen', repo: 'Qwen/Qwen3-4B-GGUF',
    file: 'Qwen3-4B-Q4_K_M.gguf', bytes: 2497280256, quant: 'Q4_K_M', note: 'Largest here. Needs about 3.5 GB of GPU memory.', thinking: true, ...APACHE },
];

export function hfUrl(repo, file, revision = 'main') {
  const path = file.split('/').map(encodeURIComponent).join('/');
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path}`;
}

/**
 * Accepts "owner/repo/file.gguf", "owner/repo/sub/dir/file.gguf" or a huggingface.co URL
 * (resolve or blob form). Returns { repo, file, revision } or throws with a readable message.
 */
export function parseHfReference(input) {
  let text = String(input || '').trim();
  if (!text) throw new Error('Enter a Hugging Face file, for example Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf');
  let revision = 'main';
  if (/^https?:\/\//i.test(text)) {
    let url;
    try { url = new URL(text); } catch { throw new Error('That address is not a valid URL.'); }
    if (url.hostname !== 'huggingface.co' && url.hostname !== 'www.huggingface.co') {
      throw new Error('Only huggingface.co files can be downloaded here. Use “Open a file” for other sources.');
    }
    const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
    const at = parts.findIndex((p) => p === 'resolve' || p === 'blob');
    if (at !== 2 || parts.length < 5) throw new Error('Use a link to a single .gguf file on huggingface.co.');
    revision = parts[3];
    text = [parts[0], parts[1], ...parts.slice(4)].join('/');
  }
  const parts = text.split('/').filter(Boolean);
  if (parts.length < 3) throw new Error('Use owner/repository/file.gguf.');
  const file = parts.slice(2).join('/');
  if (!/\.gguf$/i.test(file)) throw new Error('The file must be a .gguf model.');
  if (/-\d{5}-of-\d{5}\.gguf$/i.test(file)) throw new Error('Split GGUF files are not supported yet. Choose a single-file model.');
  const valid = /^[\w.-]+$/;
  if (!valid.test(parts[0]) || !valid.test(parts[1]) || parts.slice(2).some((p) => !valid.test(p) || p === '..')) {
    throw new Error('That name has characters a Hugging Face path cannot contain.');
  }
  return { repo: `${parts[0]}/${parts[1]}`, file, revision };
}

/** A stable local key for a downloaded file. */
export function cacheKey(repo, file, revision = 'main') {
  return `${repo}/${revision}/${file}`.replace(/[^\w.-]+/g, '__');
}

export function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1000 && i < units.length - 1) { n /= 1000; i++; }
  return `${n >= 100 || i === 0 ? Math.round(n) : n.toFixed(n >= 10 ? 1 : 2)} ${units[i]}`;
}
