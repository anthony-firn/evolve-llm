"""Build the browser app into dist/: the engine as WebAssembly plus the static web/ files.

Needs Rust with the wasm32-unknown-unknown target and wasm-bindgen-cli matching Cargo.lock.
Uses wasm-opt when it is installed.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def locked_version(name):
    lock = (ROOT / "Cargo.lock").read_text()
    m = re.search(rf'name = "{name}"\nversion = "([^"]+)"', lock)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    a = ap.parse_args()
    out = a.out.resolve()
    run("cargo", "build", "--release", "--lib", "--target", "wasm32-unknown-unknown", "--no-default-features")
    want = locked_version("wasm-bindgen")
    have = subprocess.run(["wasm-bindgen", "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    if want and have != want:
        raise SystemExit(f"wasm-bindgen-cli {have} does not match the locked crate {want}: cargo install wasm-bindgen-cli --version {want}")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "web", out)
    wasm = ROOT / "target/wasm32-unknown-unknown/release/evolve_llm.wasm"
    run("wasm-bindgen", str(wasm), "--target", "web", "--no-typescript", "--out-dir", str(out / "pkg"))
    bg = out / "pkg/evolve_llm_bg.wasm"
    if shutil.which("wasm-opt"):
        run("wasm-opt", "-O3", "--enable-bulk-memory", "--enable-nontrapping-float-to-int", "--enable-sign-ext", "-o", str(bg), str(bg))
    (out / "pkg/BUILD.json").write_text(json.dumps({
        "wasm_bytes": bg.stat().st_size,
        "wasm_sha256": hashlib.sha256(bg.read_bytes()).hexdigest(),
        "wasm_bindgen": have,
    }, indent=2) + "\n")
    print(f"built {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: engine {bg.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
