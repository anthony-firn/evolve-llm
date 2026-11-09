// Models offered in the picker. Files download directly from Hugging Face into this browser.
// Any other GGUF file on Hugging Face, or one already on the device, can be opened too.

const LLAMA = { license: 'Llama 3.2 Community License', licenseUrl: 'https://www.llama.com/llama3_2/license/' };
const APACHE = { license: 'Apache 2.0', licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0' };
const MIT = { license: 'MIT', licenseUrl: 'https://opensource.org/license/mit' };
const KIMI = { license: 'Modified MIT', licenseUrl: 'https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/main/LICENSE' };

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

// Mixture-of-experts models. The ones larger than GPU memory keep their attention weights on the
// GPU and read the experts each token chooses from browser storage, so they run slowly but run.
// `bytes` is the download size; split models list their parts when they are chosen.
export const LARGE = [
  { id: 'olmoe-1b-7b', name: 'OLMoE 1B-7B', maker: 'Ai2', repo: 'allenai/OLMoE-1B-7B-0125-Instruct-GGUF',
    file: 'OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf', bytes: 4210000000, quant: 'Q4_K_M', params: '7B, 1.3B active',
    note: '64 experts, 8 per token. Fits on GPUs with about 5 GB.', ...APACHE },
  { id: 'qwen3-30b-a3b', name: 'Qwen3 30B-A3B', maker: 'Qwen', repo: 'unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF',
    file: 'Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf', bytes: 11790000000, quant: 'UD-Q2_K_XL', params: '30B, 3.3B active',
    note: 'Strong general model. Streams experts from storage when GPU memory is short.', ...APACHE },
  { id: 'qwen3-235b-a22b', name: 'Qwen3 235B-A22B', maker: 'Qwen', repo: 'unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF',
    file: 'UD-Q2_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf', bytes: 88770000000, quant: 'UD-Q2_K_XL', params: '235B, 22B active',
    note: 'Two files. Needs about 90 GB of free browser storage; each token reads gigabytes, so replies are slow.', ...APACHE },
  { id: 'qwen3-coder-480b', name: 'Qwen3 Coder 480B-A35B', maker: 'Qwen', repo: 'unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF',
    file: 'Qwen3-Coder-480B-A35B-Instruct-UD-IQ1_M.gguf', bytes: 149720000000, quant: 'UD-IQ1_M', params: '480B, 35B active',
    note: 'Needs about 150 GB of free browser storage. Slow: expect seconds per token.', ...APACHE },
  { id: 'deepseek-v3.1', name: 'DeepSeek V3.1', maker: 'DeepSeek', repo: 'unsloth/DeepSeek-V3.1-GGUF',
    file: 'UD-IQ1_S/DeepSeek-V3.1-UD-IQ1_S-00001-of-00004.gguf', bytes: 192310000000, quant: 'UD-IQ1_S', params: '671B, 37B active',
    note: 'Four files, about 190 GB of storage. Latent attention keeps the context small. Slow.', ...MIT },
  { id: 'kimi-k2', name: 'Kimi K2 Instruct', maker: 'Moonshot AI', repo: 'unsloth/Kimi-K2-Instruct-0905-GGUF',
    file: 'UD-TQ1_0/Kimi-K2-Instruct-0905-UD-TQ1_0-00001-of-00006.gguf', bytes: 248610000000, quant: 'UD-TQ1_0', params: '1T, 32B active',
    note: 'One trillion parameters in six files, about 250 GB of storage. Slow.', ...KIMI },
];

/** For `name-00001-of-00004.gguf`, every part's path; null for a single file. */
export function splitParts(file) {
  const m = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i.exec(file);
  if (!m) return null;
  const n = Number(m[3]);
  return Array.from({ length: n }, (_, i) => `${m[1]}-${String(i + 1).padStart(5, '0')}-of-${m[3]}.gguf`);
}

/** GGUF files of a Hugging Face repository with their sizes, split parts grouped. */
export async function listRepo(repo, revision = 'main', fetchFn = fetch) {
  const res = await fetchFn(`https://huggingface.co/api/models/${repo}/tree/${encodeURIComponent(revision)}?recursive=1`);
  if (res.status === 404 || res.status === 401) throw new Error('Hugging Face has no public repository with that name.');
  if (!res.ok) throw new Error(`Hugging Face answered with HTTP ${res.status}.`);
  const files = (await res.json()).filter((f) => f.type === 'file' && /\.gguf$/i.test(f.path) && !/mmproj/i.test(f.path));
  const groups = new Map();
  for (const f of files) {
    const parts = splitParts(f.path);
    const key = parts ? parts[0] : f.path;
    const g = groups.get(key) || { file: key, parts: [], bytes: 0 };
    g.parts.push({ file: f.path, bytes: f.lfs?.size ?? f.size ?? 0 });
    g.bytes += f.lfs?.size ?? f.size ?? 0;
    groups.set(key, g);
  }
  for (const g of groups.values()) g.parts.sort((a, b) => a.file.localeCompare(b.file));
  return [...groups.values()].sort((a, b) => a.bytes - b.bytes);
}

/** Every part of a file (or split model) with sizes, from the repository listing. */
export async function resolveParts(repo, file, revision = 'main', fetchFn = fetch) {
  const groups = await listRepo(repo, revision, fetchFn);
  const g = groups.find((x) => x.file === file || x.parts.some((p) => p.file === file));
  if (!g) throw new Error(`${file} is not in ${repo}.`);
  const want = splitParts(g.file)?.length || 1;
  if (g.parts.length !== want) throw new Error(`Only ${g.parts.length} of the ${want} parts of this model are in the repository.`);
  return { file: g.file, bytes: g.bytes, parts: g.parts.map((p) => ({ ...p, url: hfUrl(repo, p.file, revision) })) };
}

export function hfUrl(repo, file, revision = 'main') {
  const path = file.split('/').map(encodeURIComponent).join('/');
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path}`;
}

/**
 * Accepts "owner/repo", "owner/repo/file.gguf", "owner/repo/sub/dir/file.gguf" or a huggingface.co
 * URL (resolve or blob form). Returns { repo, file, revision }, with file null for a repository,
 * or throws with a readable message. Any part of a split model stands for the whole model.
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
  if (parts.length === 2) {
    // A whole repository: its files are listed to choose from.
    if (!parts.every((p) => /^[\w.-]+$/.test(p))) throw new Error('That name has characters a Hugging Face path cannot contain.');
    return { repo: `${parts[0]}/${parts[1]}`, file: null, revision };
  }
  if (parts.length < 3) throw new Error('Use owner/repository, or owner/repository/file.gguf.');
  const file = parts.slice(2).join('/');
  if (!/\.gguf$/i.test(file)) throw new Error('The file must be a .gguf model.');
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
