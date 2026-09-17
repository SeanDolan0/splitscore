"""GPU detection and backend resolution for onnxruntime and torch."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    vendor: str          # nvidia | amd | intel | apple | none
    name: str            # human-readable name, e.g. "NVIDIA GeForce RTX 4060"
    preferred_device: str  # cuda | mps | xpu | cpu


def _run(cmd: list[str], timeout: float = 5) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _detect_nvidia() -> GpuInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not out:
        return None
    name = out.splitlines()[0].strip()
    return GpuInfo(vendor="nvidia", name=name, preferred_device="cuda")


def _detect_amd() -> GpuInfo | None:
    if platform.system() != "Linux":
        return None
    lspci = _run(["lspci"])
    if not lspci or "amd" not in lspci.lower():
        return None
    # Check for ROCm runtime
    if not shutil.which("rocminfo"):
        return GpuInfo(vendor="amd", name="AMD GPU (no ROCm)", preferred_device="cpu")
    return GpuInfo(vendor="amd", name="AMD GPU (ROCm)", preferred_device="cuda")


def _detect_intel() -> GpuInfo | None:
    if platform.system() != "Windows":
        return None  # DirectML is Windows-only
    lspci = _run(["lspci"])  # may not exist on Windows
    if not lspci:
        return None
    if "intel" not in lspci.lower():
        return None
    return GpuInfo(vendor="intel", name="Intel GPU (DirectML)", preferred_device="cpu")


def _detect_apple_silicon() -> GpuInfo | None:
    if platform.system() != "Darwin":
        return None
    if platform.machine() != "arm64":
        return None
    return GpuInfo(vendor="apple", name="Apple Silicon (MPS)", preferred_device="mps")


def detect_gpu() -> GpuInfo:
    """Detect the best available GPU and return device info.

    Detection order: NVIDIA → AMD → Intel → Apple Silicon → CPU fallback.
    """
    for detector in (_detect_nvidia, _detect_amd, _detect_intel, _detect_apple_silicon):
        result = detector()
        if result:
            return result
    return GpuInfo(vendor="none", name="No GPU detected", preferred_device="cpu")


# --- onnxruntime provider resolution ---

# Maps device string → ordered list of onnxruntime execution providers to try.
_PROVIDER_MAP: dict[str, list[str]] = {
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "mps": ["CPUExecutionProvider"],   # ONNX has no Apple GPU provider
    "xpu": ["CPUExecutionProvider"],   # ONNX has no Intel XPU provider
    "cpu": ["CPUExecutionProvider"],
}


def _dll_search_dirs() -> list[str]:
    """Directory roots where onnxruntime provider DLLs may be found."""
    dirs: list[str] = []
    seen: set[str] = set()
    roots = list(os.environ.get("PATH", "").split(os.pathsep))
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        roots.append(cuda_path)
    for root in roots:
        for candidate in (root, os.path.join(root, "bin")):
            if candidate and os.path.isdir(candidate):
                key = os.path.abspath(candidate)
                if key not in seen:
                    seen.add(key)
                    dirs.append(key)
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            key = str(lib)
            if key not in seen:
                dirs.append(key)
    except Exception:
        pass
    return dirs


def _has_cudnn_dll() -> bool:
    """True if the onnxruntime CUDA provider's cuDNN dependency is present.

    onnxruntime-gpu <1.27 on Windows loads CUDAExecutionProvider at session
    creation, which hard-fails with a noisy native error when cudnn64_9.dll is
    missing (e.g. a CPU-only torch wheel provides none). Detect that up front
    and drop CUDA entirely so we fall back to CPU cleanly instead.
    """
    if sys.platform != "win32":
        return True  # other platforms: let session creation validate
    for name in ("cudnn64_9.dll", "cudnn64_8.dll"):
        for d in _dll_search_dirs():
            if os.path.exists(os.path.join(d, name)):
                return True
    return False


def resolve_onnx_provider(device: str) -> list[str]:
    """Return ordered list of onnxruntime providers for the given device."""
    if device == "cuda" and not _has_cudnn_dll():
        return ["CPUExecutionProvider"]
    return _PROVIDER_MAP.get(device, ["CPUExecutionProvider"])


# --- torch device resolution ---

def resolve_torch_device(device: str) -> str:
    """Resolve a device string ('auto', 'cuda', 'mps', 'xpu', 'cpu').

    Returns a torch-compatible device string. Falls back to 'cpu' if the
    requested backend is not available.
    """
    import torch

    if device == "auto":
        gpu = detect_gpu()
        device = gpu.preferred_device

    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if device in ("cuda", "mps", "xpu"):
        # Requested but unavailable — fall back to CPU with a warning
        import warnings
        warnings.warn(f"{device} requested but not available, falling back to CPU")
    return "cpu"
