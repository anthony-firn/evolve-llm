import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { mkdir,writeFile,readFile,readdir,copyFile,rm,stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync,spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const hash=async path=>createHash('sha256').update(await readFile(new URL(path,import.meta.url))).digest('hex');
const browserOnly=process.argv.includes('--browser-only');

if(!browserOnly){
execFileSync('cargo',['build','--locked','--manifest-path','sandbox/Cargo.toml','--release','--target','wasm32-unknown-unknown'],{cwd:root,stdio:'inherit'});
await mkdir(new URL('../web/shell-pkg/',import.meta.url),{recursive:true});
execFileSync('wasm-bindgen',['sandbox/target/wasm32-unknown-unknown/release/evolve_shell.wasm','--target','web','--out-dir','web/shell-pkg','--out-name','shell'],{cwd:root,stdio:'inherit'});
await build({entryPoints:[new URL('../web/pi-agent.mjs',import.meta.url).pathname],outfile:new URL('../web/pi-agent.bundle.mjs',import.meta.url).pathname,bundle:true,format:'esm',platform:'browser',minify:true,legalComments:'external',metafile:true}).then(result=>writeFile(new URL('../web/shell-pkg/pi-build.json',import.meta.url),JSON.stringify({packages:['@earendil-works/pi-agent-core@1.5.1','@earendil-works/pi-ai@1.5.1','@earendil-works/pi-schema@1.5.1'],bytes:Object.entries(result.metafile.outputs).find(([name])=>name.endsWith('pi-agent.bundle.mjs'))[1].bytes},null,2)+'\n'));
await copyFile(new URL('../web/pi-agent.bundle.mjs',import.meta.url),new URL('../web/local-pi-agent.bundle.mjs',import.meta.url));
await writeFile(new URL('../web/shell-pkg/PI-LICENSE.txt',import.meta.url),await readFile(new URL('../vendor/PI-LICENSE.txt',import.meta.url)));

const metaPath=new URL('../web/shell-pkg/pi-build.json',import.meta.url);
const meta=JSON.parse(await readFile(metaPath,'utf8'));
meta.shell_wasm_sha256=await hash('../web/shell-pkg/shell_bg.wasm');meta.bundle_sha256=await hash('../web/pi-agent.bundle.mjs');
await writeFile(metaPath,JSON.stringify(meta,null,2)+'\n');
}

// ---- The web browser engine (browser/): Blitz, Stylo, Boa and rustls compiled to WebAssembly ----
// Slow (a release build with LTO); skip with EVOLVE_SKIP_BROWSER=1, or build only it with --browser-only.
if(process.env.EVOLVE_SKIP_BROWSER!=='1'){
  // Same WebAssembly features as the LLM engine (tools/build_web.py); every browser with WebGPU has them.
  const features='+simd128,+bulk-memory,+nontrapping-fptoint,+sign-ext,+mutable-globals';
  const rustflags=process.env.RUSTFLAGS??`-C target-feature=${features}`;
  const locked=existsSync(new URL('../browser/Cargo.lock',import.meta.url));
  execFileSync('cargo',['build',...(locked?['--locked']:[]),'--release','--target','wasm32-unknown-unknown','--manifest-path','browser/Cargo.toml'],{cwd:root,stdio:'inherit',env:{...process.env,RUSTFLAGS:rustflags}});
  const pkg=new URL('../web/browser-pkg/',import.meta.url);
  await rm(pkg,{recursive:true,force:true});await mkdir(pkg,{recursive:true});
  execFileSync('wasm-bindgen',['browser/target/wasm32-unknown-unknown/release/evolve_browser.wasm','--target','web','--out-dir','web/browser-pkg','--out-name','browser'],{cwd:root,stdio:'inherit'});
  const wasm=fileURLToPath(new URL('browser_bg.wasm',pkg));
  const wasmOpt=spawnSync('wasm-opt',['--version'],{stdio:'ignore'}).status===0;
  if(wasmOpt)execFileSync('wasm-opt',['-O2','--enable-bulk-memory','--enable-nontrapping-float-to-int','--enable-simd','--enable-sign-ext','--enable-mutable-globals','--enable-reference-types','--enable-multivalue','-o',wasm,wasm],{cwd:root,stdio:'inherit'});

  // Tor's cached consensus (relay list), bundled as the fallback for when webtor's daily snapshot
  // is unreachable. The engine skips copies older than three days while a fresher one exists.
  const torOut=new URL('../web/browser-tor/',import.meta.url);await mkdir(torOut,{recursive:true});
  for(const n of ['consensus.txt.br','microdescriptors.txt.br']){
    try{const r=await fetch(`https://raw.githubusercontent.com/ethereum/webtor-rs/main/webtor/src/cached/${n}`);if(!r.ok)throw new Error(`HTTP ${r.status}`);await writeFile(new URL(n,torOut),Buffer.from(await r.arrayBuffer()));}
    catch(e){console.warn(`Tor consensus ${n} not refreshed (${e.message}); keeping the previous copy if there is one`);}
  }

  // Fonts: WebAssembly has no system fonts, so the engine gets these files at start-up.
  const fontSrc=new URL('../browser/fonts/',import.meta.url),fontOut=new URL('../web/browser-fonts/',import.meta.url);
  await rm(fontOut,{recursive:true,force:true});await mkdir(fontOut,{recursive:true});
  const names=(await readdir(fontSrc)).sort();
  const files=names.filter(n=>/\.(ttf|otf|woff2?)$/i.test(n)),licenses=names.filter(n=>/\.(txt|md)$/i.test(n));
  for(const n of [...files,...licenses])await copyFile(new URL(n,fontSrc),new URL(n,fontOut));
  await writeFile(new URL('fonts.json',fontOut),JSON.stringify({files,licenses},null,2)+'\n');

  const fontHashes={};for(const n of files)fontHashes[n]=await hash(`../web/browser-fonts/${n}`);
  await writeFile(new URL('BUILD.json',pkg),JSON.stringify({
    wasm_bytes:(await stat(wasm)).size,wasm_sha256:await hash('../web/browser-pkg/browser_bg.wasm'),
    wasm_opt:wasmOpt,rustflags,fonts:fontHashes,
  },null,2)+'\n');
  console.log(`built web/browser-pkg: engine ${((await stat(wasm)).size/1e6).toFixed(2)} MB${wasmOpt?' (wasm-opt -O2)':''}, ${files.length} fonts`);
}
