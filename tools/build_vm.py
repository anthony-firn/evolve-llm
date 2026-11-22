#!/usr/bin/env python3
"""Build the Evolve VM into web/vm/: the kernel (Rust, wasm-bindgen) in kernel/, and the WASI
programs (shell, core utilities, git/gh, pi, python3) in bin/, with programs.json listing what
/bin holds.

Needs Rust with the wasm32-unknown-unknown and wasm32-wasip1 targets, and wasm-bindgen-cli.
python3 is RustPython, a Python 3 interpreter in Rust, built from its crates.io release with that
release's own Cargo.lock (--locked) and the Rust it needs (rustup installs it for this step only);
--no-python skips it.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "vm"
# RustPython: the release, the Rust it builds with, and the features for WASI (its standard
# library frozen in; no threads, sockets or native allocator).
PYTHON = ("rustpython", "0.6.0", "1.97.0", "freeze-stdlib,stdlib,stdio,importlib,host_env")


def run(*args, cwd=ROOT, env=None):
    subprocess.run(args, cwd=cwd, check=True, env=env)


def main():
    vm = ROOT / "vm"
    run("cargo", "build", "--release", "--target", "wasm32-unknown-unknown", "--manifest-path", str(vm / "Cargo.toml"))
    run("wasm-bindgen", str(vm / "target/wasm32-unknown-unknown/release/evolve_vm.wasm"), "--target", "web", "--no-typescript", "--out-dir", str(OUT / "kernel"))
    progs = vm / "programs"
    run("cargo", "build", "--release", "--target", "wasm32-wasip1", "--manifest-path", str(progs / "Cargo.toml"))
    (OUT / "bin").mkdir(parents=True, exist_ok=True)
    for name in ("sh", "utils", "gh", "pi"):
        shutil.copy(progs / f"target/wasm32-wasip1/release/{name}.wasm", OUT / "bin" / f"{name}.wasm")
    if "--no-python" not in sys.argv:
        crate, version, rust, features = PYTHON
        run("rustup", "toolchain", "install", rust, "--profile", "minimal", "--target", "wasm32-wasip1")
        root = vm / "target" / "python"
        env = {**os.environ, "RUSTUP_TOOLCHAIN": rust}
        run("cargo", "install", f"{crate}@{version}", "--locked", "--root", str(root), "--target", "wasm32-wasip1",
            "--no-default-features", "--features", features, env=env)
        shutil.copy(root / "bin" / "rustpython.wasm", OUT / "bin" / "python.wasm")
    # The utility names, read from the source so the list cannot drift.
    src = (progs / "src/bin/utils.rs").read_text()
    names = re.findall(r'"([^"]+)"', src.split("pub const NAMES")[1].split("];")[0])
    (OUT / "programs.json").write_text(json.dumps({"utils": names}, indent=1) + "\n")
    print(f"built web/vm: kernel {(OUT / 'kernel/evolve_vm_bg.wasm').stat().st_size // 1024} KB, sh {(OUT / 'bin/sh.wasm').stat().st_size // 1024} KB, utils {(OUT / 'bin/utils.wasm').stat().st_size // 1024} KB ({len(names)} utilities), gh {(OUT / 'bin/gh.wasm').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
