#!/usr/bin/env python3
"""Compile docs/DOCS.tex (IEEEtran) into docs/DOCS.pdf with Tectonic.

Tectonic is a self-contained TeX engine. It is taken from PATH, or downloaded
once into <repo>/.tools/; the LaTeX packages it needs are fetched on first use.

Run:  python docs/build_pdf.py [--tex docs/DOCS.tex] [--out docs/DOCS.pdf] [--keep]
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
TECTONIC_VERSION = "0.17.0"


def ensure_tectonic() -> str:
    """Path to a Tectonic binary: on PATH, or downloaded once into <repo>/.tools/."""
    found = shutil.which("tectonic")
    if found:
        return found
    exe = "tectonic.exe" if os.name == "nt" else "tectonic"
    local = REPO_DIR / ".tools" / "tectonic" / exe
    if local.exists():
        return str(local)
    arch = "aarch64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
    if os.name == "nt":
        asset = f"tectonic-{TECTONIC_VERSION}-{arch}-pc-windows-msvc.zip"
    elif sys.platform == "darwin":
        asset = f"tectonic-{TECTONIC_VERSION}-{arch}-apple-darwin.tar.gz"
    else:
        asset = f"tectonic-{TECTONIC_VERSION}-{arch}-unknown-linux-musl.tar.gz"
    url = ("https://github.com/tectonic-typesetting/tectonic/releases/download/"
           f"tectonic%40{TECTONIC_VERSION}/{asset}")
    print(f"[*] Downloading Tectonic {TECTONIC_VERSION} into .tools/ ...")
    data = urllib.request.urlopen(url, timeout=120).read()
    local.parent.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(local.parent)
    else:
        tarfile.open(fileobj=io.BytesIO(data)).extractall(local.parent)
    os.chmod(local, 0o755)
    return str(local)


def compile_tex(tex: Path, out: Path, keep: bool = False) -> None:
    """Compile tex (relative paths resolve against its folder) and copy the PDF to out."""
    tex = tex.resolve()
    build = tex.parent / ".tex-build"
    build.mkdir(exist_ok=True)
    result = subprocess.run([ensure_tectonic(), "-X", "compile", "--keep-logs", "--outdir", str(build), tex.name],
                            cwd=tex.parent, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = build / (tex.stem + ".log")
    if result.returncode != 0:
        print(result.stdout[-4000:], result.stderr[-4000:])
        raise SystemExit(f"Tectonic failed; see {log}")
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    for marker in ("Missing character", "undefined", "Overfull"):
        hits = [line for line in text.splitlines() if marker in line and "Font shape" not in line]
        if hits:
            print(f"[WARN] {len(hits)} x {marker}: {hits[0].strip()}")
    shutil.copyfile(build / (tex.stem + ".pdf"), out)
    if not keep:
        shutil.rmtree(build, ignore_errors=True)
    print(f"[SUCCESS] PDF created: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex", type=Path, default=DOCS_DIR / "DOCS.tex")
    ap.add_argument("--out", type=Path, default=DOCS_DIR / "DOCS.pdf")
    ap.add_argument("--keep", action="store_true", help="keep the build folder (log, aux files)")
    args = ap.parse_args()
    compile_tex(args.tex, args.out, args.keep)


if __name__ == "__main__":
    main()
