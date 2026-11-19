#!/usr/bin/env python3
"""Build the Evolve VM into web/vm/: the kernel (Rust, wasm-bindgen) in kernel/, and the WASI
programs (shell, core utilities) in bin/, with programs.json listing what /bin holds.

Needs Rust with the wasm32-unknown-unknown and wasm32-wasip1 targets, and wasm-bindgen-cli.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "vm"


def run(*args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    vm = ROOT / "vm"
    run("cargo", "build", "--release", "--target", "wasm32-unknown-unknown", "--manifest-path", str(vm / "Cargo.toml"))
    run("wasm-bindgen", str(vm / "target/wasm32-unknown-unknown/release/evolve_vm.wasm"), "--target", "web", "--no-typescript", "--out-dir", str(OUT / "kernel"))
    progs = vm / "programs"
    run("cargo", "build", "--release", "--target", "wasm32-wasip1", "--manifest-path", str(progs / "Cargo.toml"))
    (OUT / "bin").mkdir(parents=True, exist_ok=True)
    for name in ("sh", "utils"):
        shutil.copy(progs / f"target/wasm32-wasip1/release/{name}.wasm", OUT / "bin" / f"{name}.wasm")
    # The utility names, read from the source so the list cannot drift.
    src = (progs / "src/bin/utils.rs").read_text()
    names = re.findall(r'"([^"]+)"', src.split("pub const NAMES")[1].split("];")[0])
    (OUT / "programs.json").write_text(json.dumps({"utils": names}, indent=1) + "\n")
    print(f"built web/vm: kernel {(OUT / 'kernel/evolve_vm_bg.wasm').stat().st_size // 1024} KB, sh {(OUT / 'bin/sh.wasm').stat().st_size // 1024} KB, utils {(OUT / 'bin/utils.wasm').stat().st_size // 1024} KB ({len(names)} utilities)")


if __name__ == "__main__":
    main()
