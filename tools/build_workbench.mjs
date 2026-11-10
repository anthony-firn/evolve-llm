import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { mkdir,writeFile,readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
execFileSync('cargo',['build','--locked','--manifest-path','sandbox/Cargo.toml','--release','--target','wasm32-unknown-unknown'],{cwd:root,stdio:'inherit'});
await mkdir(new URL('../web/shell-pkg/',import.meta.url),{recursive:true});
execFileSync('wasm-bindgen',['sandbox/target/wasm32-unknown-unknown/release/evolve_shell.wasm','--target','web','--out-dir','web/shell-pkg','--out-name','shell'],{cwd:root,stdio:'inherit'});
await build({entryPoints:[new URL('../web/pi-agent.mjs',import.meta.url).pathname],outfile:new URL('../web/pi-agent.bundle.mjs',import.meta.url).pathname,bundle:true,format:'esm',platform:'browser',minify:true,legalComments:'external',metafile:true}).then(result=>writeFile(new URL('../web/shell-pkg/pi-build.json',import.meta.url),JSON.stringify({packages:['@earendil-works/pi-agent-core@1.5.1','@earendil-works/pi-ai@1.5.1','@earendil-works/pi-schema@1.5.1'],bytes:Object.entries(result.metafile.outputs).find(([name])=>name.endsWith('pi-agent.bundle.mjs'))[1].bytes},null,2)+'\n'));
await writeFile(new URL('../web/shell-pkg/PI-LICENSE.txt',import.meta.url),await readFile(new URL('../vendor/PI-LICENSE.txt',import.meta.url)));

const hash=async path=>createHash('sha256').update(await readFile(new URL(path,import.meta.url))).digest('hex');
const metaPath=new URL('../web/shell-pkg/pi-build.json',import.meta.url);
const meta=JSON.parse(await readFile(metaPath,'utf8'));
meta.shell_wasm_sha256=await hash('../web/shell-pkg/shell_bg.wasm');meta.bundle_sha256=await hash('../web/pi-agent.bundle.mjs');
await writeFile(metaPath,JSON.stringify(meta,null,2)+'\n');
