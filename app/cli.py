"""CLI entry point for `uvx splitscore`.

Detects GPU vendor, installs the correct torch + onnxruntime backends,
then starts the server.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def _install_torch(vendor: str) -> None:
    """Install the correct torch variant for the GPU vendor."""
    try:
        import torch
        # Check if the correct backend is actually available
        if vendor == "nvidia" and torch.cuda.is_available():
            return
        if vendor == "amd" and torch.cuda.is_available():  # ROCm uses cuda device
            return
        if vendor in ("apple", "none"):
            return  # default torch is fine
    except ImportError:
        pass

    index_urls = {
        "nvidia": "https://download.pytorch.org/whl/cu130",
        "amd": "https://download.pytorch.org/whl/rocm6.2",
    }
    if vendor in index_urls:
        print(f"Installing torch ({vendor}) ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--index-url", index_urls[vendor],
            "--trusted-host", "download.pytorch.org",
            "torch>=2.7",
        ])
    else:
        print("Installing torch (default) ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.7"])


def _ensure_backends() -> None:
    """Make sure torch and onnxruntime are importable with correct backends."""
    # Check torch
    try:
        import torch  # noqa: F401
    except ImportError:
        vendor = _detect_vendor()
        print(f"GPU detected: {vendor}")
        _install_torch(vendor)
    else:
        vendor = _detect_vendor()

    # Check onnxruntime GPU provider
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_gpu = any(p not in ("CPUExecutionProvider",) for p in providers)
        if not has_gpu and vendor not in ("apple", "none"):
            _install_onnxruntime(vendor)
    except ImportError:
        _install_onnxruntime(vendor)


def main() -> None:
    _ensure_backends()
    import uvicorn
    url = "http://127.0.0.1:8000"
    webbrowser.open(url)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
