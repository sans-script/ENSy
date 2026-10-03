#!/usr/bin/env python3
"""Start ENSy: the FastAPI backend (port 3000) and the Vite dev server (port 5173).

    python start.py                  # both servers; open http://localhost:5173
    python start.py --backend-only   # backend only; it also serves the built
                                     # frontend (npm run build) at http://localhost:3000

HF_TOKEN (needed for the gated google/gemma-2-2b) is read from the environment
or from a .env file in the repository root (git-ignored), e.g.

    HF_TOKEN=hf_...

Ctrl+C stops both servers, including their child processes.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_PORT, FRONTEND_PORT = 3000, 5173
WINDOWS = os.name == "nt"


def load_dotenv(path: Path) -> list[str]:
    """KEY=VALUE lines from .env; variables already set in the environment win."""
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def python_exe() -> str:
    venv = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
    return str(venv) if venv.exists() else sys.executable


def vite_cmd() -> list[str]:
    local = ROOT / "node_modules" / ".bin" / ("vite.cmd" if WINDOWS else "vite")
    if not local.exists():
        sys.exit("node_modules missing: run `npm install` first")
    return [str(local), "--host", "0.0.0.0", "--port", str(FRONTEND_PORT), "--strictPort"]


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def spawn(name: str, cmd: list[str]) -> subprocess.Popen:
    kwargs = {"cwd": ROOT, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
              "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1}
    if WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # own process group, stopped as a whole
    proc = subprocess.Popen(cmd, **kwargs)

    def pump():
        for line in proc.stdout:
            print(f"[{name}] {line}", end="", flush=True)

    threading.Thread(target=pump, daemon=True).start()
    return proc


def stop(proc: subprocess.Popen) -> None:
    """Stop a process and its children (the .venv launcher starts a child Python)."""
    if proc.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def wait_healthy(proc: subprocess.Popen, timeout: float = 120) -> bool:
    url = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend-only", action="store_true", help="start only the backend")
    args = ap.parse_args()

    # SIGTERM (and Ctrl+Break on Windows) stop both servers just like Ctrl+C
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    if WINDOWS:
        signal.signal(signal.SIGBREAK, interrupt)

    loaded = load_dotenv(ROOT / ".env")
    if loaded:
        print(f"[start] loaded from .env: {', '.join(loaded)}")
    if not os.environ.get("HF_TOKEN"):
        print("[start] HF_TOKEN not set: Gemma 2 (gated) will be unavailable; "
              "add HF_TOKEN=hf_... to .env")

    ports = [BACKEND_PORT] + ([] if args.backend_only else [FRONTEND_PORT])
    busy = [p for p in ports if port_busy(p)]
    if busy:
        sys.exit(f"[start] port(s) {busy} already in use: stop the running server first")

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        backend = spawn("backend", [python_exe(), "app.py"])
        procs.append(("backend", backend))
        if not wait_healthy(backend):
            sys.exit("[start] the backend did not become healthy; see the log above")
        print(f"[start] backend ready: http://localhost:{BACKEND_PORT}")
        if not args.backend_only:
            procs.append(("frontend", spawn("frontend", vite_cmd())))
            print(f"[start] frontend: http://localhost:{FRONTEND_PORT}  (Ctrl+C stops both)")
        while all(p.poll() is None for _, p in procs):
            time.sleep(0.5)
        for name, p in procs:
            if p.poll() is not None:
                print(f"[start] {name} exited with code {p.returncode}; stopping the rest")
    except KeyboardInterrupt:
        print("\n[start] stopping...")
    finally:
        for _, p in reversed(procs):
            stop(p)
        print("[start] all servers stopped")


if __name__ == "__main__":
    main()
