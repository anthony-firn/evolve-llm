"""Package the browser SDK with an identified, unchanged engine runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def package(runtime, out):
    metadata = json.loads((runtime.parent / 'BUILD.json').read_text())
    engine_commit = metadata.get('engine_commit', metadata['commit'])
    if not re.fullmatch(r'[a-f0-9]{40}', engine_commit) or metadata.get('modified'):
        raise ValueError('An identified clean runtime release is required')
    wasm = runtime / 'evolve_llm_bg.wasm'
    digest = hashlib.sha256(wasm.read_bytes()).hexdigest()
    if digest != metadata['wasm_sha256'] or digest != json.loads((runtime / 'BUILD.json').read_text())['wasm_sha256']:
        raise ValueError('Runtime checksum mismatch')
    subprocess.run(['git', 'diff', '--exit-code', engine_commit, 'HEAD', '--', 'src', 'Cargo.toml', 'Cargo.lock'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'Cargo.toml', 'Cargo.lock'], cwd=ROOT):
        raise ValueError('Engine sources have local changes')
    names = ['icon.svg', 'markdown.js', 'index.html', 'app.js', 'app.css', 'sdk.mjs', 'sdk.d.mts', 'worker.js', 'models.js', 'tools.js', 'network.js',
        'workbench.mjs', 'shell-client.mjs', 'shell-worker.mjs', 'pi-agent.bundle.mjs', 'pi-agent.bundle.mjs.LEGAL.txt',
        'shell-pkg/shell.js', 'shell-pkg/shell_bg.wasm', 'shell-pkg/pi-build.json', 'shell-pkg/PI-LICENSE.txt',
        'examples/index.html', 'examples/example.mjs', 'examples/example.css', 'examples/fixture-worker.mjs']
    files = {name: (ROOT / 'web' / name).read_bytes() for name in names if (ROOT / 'web' / name).is_file()}
    for required in ['shell-pkg/shell_bg.wasm', 'pi-agent.bundle.mjs', 'network.js']:
        if required not in files: raise ValueError('Build the workbench first: node tools/build_workbench.mjs')
    files.update({'pkg/' + name: (runtime / name).read_bytes() for name in ['evolve_llm.js', 'evolve_llm_bg.wasm', 'BUILD.json']})
    files['WORKBENCH.md'] = (ROOT / 'docs/workbench.md').read_bytes()
    files['README.md'] = (ROOT / 'docs/embedding.md').read_bytes()
    for name in ['LICENSE', 'NOTICE']: files[name] = (ROOT / name).read_bytes()
    provenance = {'source': 'https://github.com/anthony-firn/evolve-llm',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'modified': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        'engine_commit': engine_commit, 'wasm_sha256': digest,
        'files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'evolve-llm-browser-0.3.0.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in {**files, 'BUILD.json': (json.dumps(provenance, indent=2) + '\n').encode()}.items():
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)); entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    print(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True, help='pkg/ directory inside a prior identified release')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    package(args.runtime.resolve(), args.out.resolve())
