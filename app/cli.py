"""CLI entry point for `uvx splitscore`.

Detects GPU vendor, installs the correct torch + onnxruntime backends,
then starts the server.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import webbrowser


def _detect_vendor() -> str:
    """Return gpu vendor: nvidia | amd | intel | apple | none."""
    system = platform.system()

    # NVIDIA
    if shutil.which("nvidia-smi"):
        return "nvidia"

    # Apple Silicon
    if system == "Darwin" and platform.machine() == "arm64":
        return "apple"

    # AMD (Linux only — ROCm)
    if system == "Linux":
        try:
            lspci = subprocess.check_output(["lspci"], text=True, timeout=5)
            if "amd" in lspci.lower():
                return "amd"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Intel (Windows — DirectML)
    if system == "Windows":
        try:
            lspci = subprocess.check_output(["lspci"], text=True, timeout=5)
            if "intel" in lspci.lower():
                return "intel"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return "none"


def _pip_install(args: list[str]) -> None:
    """Install packages into the running interpreter, preferring the uv CLI.

    uvx ephemeral environments ship no pip (``python -m pip`` fails there), so
    install via ``uv pip install --python sys.executable``, which resolves into
    the exact environment the CLI is running from.
    """
    if shutil.which("uv"):
        subprocess.check_call(
            ["uv", "pip", "install", "--system-certs", "--python", sys.executable, *args])
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def _install_onnxruntime(vendor: str) -> None:
    """Install the correct onnxruntime variant for the GPU vendor."""
    pkgs = {
        "nvidia": "onnxruntime-gpu>=1.21,<1.27",
        "amd": "onnxruntime-rocm",
        "intel": "onnxruntime-directml",
        "apple": "onnxruntime",
        "none": "onnxruntime",
    }
    pkg = pkgs[vendor]
    print(f"Installing {pkg} ...")
    _pip_install([pkg])


_CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"  # CUDA 12.8, matches onnxruntime-gpu <1.27
# torch cu128 wheels bundle the CUDA 12.8 runtime + cuDNN 9 DLLs in torch/lib
# and declare no nvidia-* pip deps — only this standard set.
_CUDA_TORCH_DEPS = [
    "filelock",
    "typing-extensions>=4.10.0",
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2",
    "fsspec>=0.8.5",
    "setuptools",
]


def _latest_cu128_torch_version() -> str:
    """Latest stable torch version on the cu128 wheel index, e.g. '2.9.1+cu128'."""
    url = f"{_CUDA_TORCH_INDEX}/torch/"
    page = urllib.request.urlopen(url, timeout=30).read().decode()
    versions = [m.group(1) for m in re.finditer(r"torch-(\d+\.\d+\.\d+)\+cu128-", page)]
    if not versions:
        raise RuntimeError("no torch+cu128 wheel found on download.pytorch.org")
    latest = max(versions, key=lambda s: tuple(int(x) for x in s.split(".")))
    return f"{latest}+cu128"


def _install_torch(vendor: str, force: bool = False) -> None:
    """Install the correct torch variant for the GPU vendor."""
    if not force:
        try:
            import torch
            if vendor in ("nvidia", "amd") and torch.cuda.is_available():
                return
            if vendor in ("apple", "none"):
                return  # default PyPI torch is fine
        except ImportError:
            pass

    if vendor == "nvidia":
        # Install from the cu128 index ALONE with --no-deps. PyPI's CPU torch is
        # versioned higher, so a bare `torch>=2.7` (or adding PyPI as a fallback
        # index) would resolve to the CPU wheel instead of the CUDA build whose
        # bundled cublas64_12/cudart64_12/cudnn64_9.dll are what onnxruntime's
        # CUDAExecutionProvider needs on Windows — no system CUDA/cuDNN install.
        version = _latest_cu128_torch_version()
        print(f"Installing torch {version} (CUDA 12.8 build) ...")
        _pip_install([
            "--index-url", _CUDA_TORCH_INDEX,
            "--no-deps",
            *(["--force-reinstall"] if force else []),
            f"torch=={version}",
        ])
        # Runtime deps come from PyPI (satisfied ones are skipped).
        _pip_install(_CUDA_TORCH_DEPS)
    elif vendor == "amd":
        print("Installing torch (amd/ROCm) ...")
        _pip_install([
            "--index-url", "https://download.pytorch.org/whl/rocm6.2",
            "--extra-index-url", "https://pypi.org/simple",
            *(["--force-reinstall"] if force else []),
            "torch>=2.7",
        ])
    else:
        print("Installing torch (default) ...")
        _pip_install(["torch>=2.7"])


def _ensure_backends() -> None:
    """Make sure torch and onnxruntime are importable with correct backends."""
    # Check torch
    try:
        import torch
    except ImportError:
        vendor = _detect_vendor()
        print(f"GPU detected: {vendor}")
        _install_torch(vendor)
    else:
        vendor = _detect_vendor()
        if vendor == "nvidia" and not torch.cuda.is_available():
            # torch resolved to the CPU wheel (e.g. `uvx splitscore` pulls
            # torch from PyPI, ignoring the project's cu130 index). Upgrade it
            # to the CUDA build, then re-exec so the fresh process imports the
            # CUDA torch — its bundled cuDNN DLLs are what let onnxruntime's
            # CUDAExecutionProvider load.
            if os.environ.get("SPLITSCORE_TORCH_UPGRADED") == "1":
                print("torch still has no CUDA support after upgrade; running on CPU.")
            else:
                print("torch is CPU-only; upgrading to the CUDA build ...")
                _install_torch("nvidia", force=True)
                os.environ["SPLITSCORE_TORCH_UPGRADED"] = "1"
                print("torch upgraded; restarting to load the CUDA build ...")
                os.execv(sys.executable, [sys.executable, *sys.argv])

    # Check onnxruntime GPU provider
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except ImportError:
        vendor = _detect_vendor()
        _install_onnxruntime(vendor)
        return

    has_gpu = any(p != "CPUExecutionProvider" for p in providers)
    if not has_gpu and vendor not in ("apple", "none"):
        print(f"onnxruntime has no GPU provider ({providers}); reinstalling ...")
        _install_onnxruntime(vendor)


def _find_free_port(start: int) -> int:
    """Return the first available port starting from *start*."""
    import socket
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found near {start}")


def main() -> None:
    import argparse
    _ensure_backends()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    port = _find_free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    uvicorn.run("app.main:app", host="127.0.0.1", port=port)
