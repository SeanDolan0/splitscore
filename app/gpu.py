"""GPU detection and backend resolution for onnxruntime and torch."""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass


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


def resolve_onnx_provider(device: str) -> list[str]:
    """Return ordered list of onnxruntime providers for the given device."""
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
