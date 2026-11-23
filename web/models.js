// The models the app runs, chosen by the app itself (web/model-manager.mjs): only models on the
// frontier of coding quality against time per answer, so no model here is both slower and worse
// than another that fits the same device. Files download from Hugging Face (and mirrors, see
// web/mirrors.mjs) into this browser. Developers can still open other GGUF files.
//
// `modes`: Artificial Analysis's Coding Index (Terminal-Bench Hard and SciCode, run by them) and
// the output tokens the model spent on their Intelligence Index, without and with thinking, from
// their table of 2026-06-01 (Intelligence Index v4.0, the last with a Coding Index):
// https://web.archive.org/web/20260601053007/https://artificialanalysis.ai/models
// `speed`: tokens per second on the reference laptop (RTX 3050 6 GB, evolve-llm bench,
// 2026-11-22; Qwen3 Coder 30B-A3B estimated from its reads per token, see docs/frontier.json). Time per
// answer is tokens / speed, so a model that writes three times as much is three times slower.
// `experts`: a mixture-of-experts model that keeps its attention on the GPU and reads the experts
// each token picks from browser storage; it needs `minGpu` bytes of GPU memory and room to save it.

const APACHE = { license: 'Apache 2.0', licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0' };

export const MODELS = [
  { id: 'qwen3-0.6b', name: 'Qwen3 0.6B', maker: 'Qwen', repo: 'Qwen/Qwen3-0.6B-GGUF',
    file: 'Qwen3-0.6B-Q8_0.gguf', bytes: 639446688, sha256: '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031', quant: 'Q8_0',
    note: 'For phones and small GPUs: quick answers to simple requests.', thinking: true, speed: 96.4,
    modes: [{ thinking: false, coding: 1.35, tokens: 2.9e6 }, { thinking: true, coding: 0.93, tokens: 19.5e6 }], ...APACHE },
  { id: 'qwen3.5-2b', name: 'Qwen3.5 2B', maker: 'Qwen', repo: 'unsloth/Qwen3.5-2B-GGUF',
    file: 'Qwen3.5-2B-Q4_K_M.gguf', bytes: 1280835840, sha256: 'aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223', quant: 'Q4_K_M',
    note: 'The strongest model for phones and GPUs under 3 GB.', thinking: true, speed: 47.3,
    modes: [{ thinking: false, coding: 4.92, tokens: 100.2e6 }, { thinking: true, coding: 3.45, tokens: 389.5e6 }], ...APACHE },
  { id: 'qwen3-4b-2507', name: 'Qwen3 4B Instruct 2507', maker: 'Qwen', repo: 'unsloth/Qwen3-4B-Instruct-2507-GGUF',
    file: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', bytes: 2497281120, sha256: '3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597', quant: 'Q4_K_M',
    note: 'Fast, short answers: the first choice for everyday requests on most computers.', speed: 22.5,
    modes: [{ thinking: false, coding: 9.05, tokens: 21.8e6 }], ...APACHE },
  { id: 'qwen3.5-4b', name: 'Qwen3.5 4B', maker: 'Qwen', repo: 'unsloth/Qwen3.5-4B-GGUF',
    file: 'Qwen3.5-4B-Q4_K_M.gguf', bytes: 2740937888, sha256: '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4', quant: 'Q4_K_M',
    note: 'For coding and research that need more care. A small cache keeps long chats fast.', thinking: true, speed: 20.3,
    modes: [{ thinking: false, coding: 13.67, tokens: 75.1e6 }, { thinking: true, coding: 17.49, tokens: 236.7e6 }], ...APACHE },
  { id: 'qwen3-coder-30b-a3b', name: 'Qwen3 Coder 30B-A3B', maker: 'Qwen', repo: 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF',
    file: 'Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf', bytes: 12848766112, sha256: 'd6b85d2b6633c55be3751848acc74fab40d28d93b27dabacd96ceabe10962057', quant: 'UD-IQ3_XXS',
    params: '30B, 3.3B active', note: 'For hard coding work: trained for coding agents. Reads its experts from storage.', speed: 2.0,
    experts: { minGpu: 2.5e9 }, modes: [{ thinking: false, coding: 19.36, tokens: 13.2e6 }], ...APACHE },
  { id: 'qwen3.6-35b-a3b', name: 'Qwen3.6 35B-A3B', maker: 'Qwen', repo: 'unsloth/Qwen3.6-35B-A3B-GGUF',
    file: 'Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf', bytes: 16845511648, sha256: 'a832b9689925f1bd335bbe985cdfb06c36bf2cf268f4f8f6eceafa3ceb515617', quant: 'UD-Q3_K_XL',
    params: '35B, 3B active', note: 'For the hardest requests: thinks before it answers, slowly. Reads its experts from storage.', thinking: true, speed: 1.5,
    experts: { minGpu: 2.5e9 }, modes: [{ thinking: false, coding: 17.6, tokens: 24.3e6 }, { thinking: true, coding: 35.15, tokens: 143.2e6 }], ...APACHE },
  // Images: the one catalog model that reads them. Used for requests with pictures or screenshots.
  { id: 'qwen3-vl-2b', name: 'Qwen3-VL 2B', maker: 'Qwen', repo: 'Qwen/Qwen3-VL-2B-Instruct-GGUF',
    file: 'Qwen3VL-2B-Instruct-Q4_K_M.gguf', bytes: 1107409952, sha256: '089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae', quant: 'Q4_K_M',
    note: 'Sees images: reads screenshots and pictures. Downloads its image encoder too.', speed: 53.1,
    vision: { file: 'mmproj-Qwen3VL-2B-Instruct-F16.gguf', bytes: 819394848 }, ...APACHE },
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
    // The SHA-256 Hugging Face lists for LFS files names the file's swarm and manifest.
    g.parts.push({ file: f.path, bytes: f.lfs?.size ?? f.size ?? 0, sha256: f.lfs?.oid ?? null });
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
