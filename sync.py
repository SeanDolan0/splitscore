"""Detect GPU and install the correct torch + onnxruntime backends.

Usage:
    uv run python sync.py          # auto-detect GPU
    uv run python sync.py --cpu    # force CPU-only
    uv run python sync.py --cuda 12.6  # force specific CUDA version

How it works:
    torch is NOT a static dependency in pyproject.toml because the correct
    wheel index (cu118, cu130, rocm, etc.) depends on the user's GPU.
    This script rewrites the [tool.uv.sources] section of pyproject.toml to
    point torch at the right index, then runs `uv lock` + `uv sync` so that
    every future `uv run` resolves the correct variant automatically.
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
PYPROJECT = PROJECT_DIR / "pyproject.toml"

CUDA_TAG_MAP: dict[str, str] = {
    "11.8": "cu118", "12.1": "cu121", "12.4": "cu124",
    "12.6": "cu126", "12.8": "cu128", "13.0": "cu130",
}


def _detect_vendor() -> str:
    """Return gpu vendor: nvidia | amd | intel | apple | none."""
    system = platform.system()

    if shutil.which("nvidia-smi"):
        return "nvidia"
    if system == "Darwin" and platform.machine() == "arm64":
        return "apple"
    if system == "Linux":
        try:
            lspci = subprocess.check_output(["lspci"], text=True, timeout=5)
            if "amd" in lspci.lower():
                return "amd"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    if system == "Windows":
        try:
            lspci = subprocess.check_output(["lspci"], text=True, timeout=5)
            if "intel" in lspci.lower():
                return "intel"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return "none"


def _detect_cuda_version() -> str | None:
    """Detect CUDA version from nvidia-smi driver version."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        driver = out.strip().splitlines()[0].strip()
        major = int(driver.split(".")[0])
        if major >= 600: return "13.0"
        if major >= 570: return "12.8"
        if major >= 560: return "12.6"
        if major >= 550: return "12.4"
        if major >= 535: return "12.2"
    except (subprocess.SubprocessError, IndexError, ValueError):
        pass
    return None


def _find_best_tag(version: str) -> str:
    if version in CUDA_TAG_MAP:
        return CUDA_TAG_MAP[version]
    parts = [float(v) for v in CUDA_TAG_MAP]
    detected = float(version)
    candidates = sorted([v for v in parts if v <= detected], reverse=True)
    return CUDA_TAG_MAP[str(candidates[0])] if candidates else CUDA_TAG_MAP[min(parts, key=float)]


def _torch_index_url(cuda_tag: str | None) -> str | None:
    """Return the PyTorch wheel index URL for the given CUDA tag, or None for CPU."""
    if cuda_tag:
        return f"https://download.pytorch.org/whl/{cuda_tag}"

    return None


def _configure_pyproject(vendor: str, cuda_tag: str | None) -> None:
    """Rewrite pyproject.toml's [tool.uv] section with the correct torch index.

    For nvidia/amd: adds a [[tool.uv.index]] entry and [tool.uv.sources]
    mapping so uv resolves torch from the vendor-specific wheel index.
    For cpu/apple/none: removes the index and source entries.
    """
    text = PYPROJECT.read_text(encoding="utf-8")

    # Remove existing uv.index and uv.sources blocks (idempotent).
    # Match [[tool.uv.index]] ... up to next [[ or end of [tool.uv] section.
    text = re.sub(
        r'\n?\[\[tool\.uv\.index\]\].*?(?=\n\[(?!tool\.uv\.index\])|\Z)',
        '', text, flags=re.DOTALL,
    )
    text = re.sub(
        r'\n?\[tool\.uv\.sources\].*?(?=\n\[(?!tool\.uv\.sources\])|\Z)',
        '', text, flags=re.DOTALL,
    )

    # Ensure torch>=2.7 is a direct dependency (required for [tool.uv.sources]).
    if 'torch' not in text.split('dependencies')[1].split(']')[0] if 'dependencies' in text else True:
        text = text.replace(
            'dependencies = [',
            'dependencies = [\n    "torch>=2.7",',
            1,
        )

    index_url = _torch_index_url(_find_best_tag(cuda_tag) if cuda_tag else None)
    if index_url:
        # Add the vendor-specific index and source mapping.
        uv_sources_block = f"""
[[tool.uv.index]]
name = "pytorch"
url = "{index_url}"

[tool.uv.sources]
torch = {{ index = "pytorch" }}
"""
        # Insert before [tool.setuptools] or at end of file.
        marker = "\n[build-system]"
        if marker in text:
            text = text.replace(marker, uv_sources_block + marker)
        else:
            text = text.rstrip() + "\n" + uv_sources_block + "\n"
    else:
        # CPU-only: make sure torch is still a dependency but from default PyPI.
        pass

    PYPROJECT.write_text(text, encoding="utf-8")


def _uv_lock_and_sync() -> int:
    """Run `uv lock` then `uv sync` to resolve and install all deps.

    UV_SYSTEM_CERTS=1: some corporate/firewall setups intercept TLS and uv's
    bundled certs fail.  Passing --system-certs (and the env var for child
    processes) lets uv use the OS certificate store instead.
    """
    env = {**os.environ, "UV_SYSTEM_CERTS": "1"}
    for cmd in (
        ["uv", "lock", "--system-certs"],
        ["uv", "sync", "--system-certs"],
    ):
        print(f"  $ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=PROJECT_DIR, env=env)
        if rc != 0:
            return rc
    return 0


def _install_onnxruntime(vendor: str) -> None:
    """Install the correct onnxruntime variant for the detected GPU."""
    pkgs = {
        "nvidia": "onnxruntime-gpu>=1.21,<1.27",
        "amd": "onnxruntime-rocm",
        "intel": "onnxruntime-directml",
        "apple": "onnxruntime",
        "none": "onnxruntime",
    }
    pkg = pkgs[vendor]
    cmd = ["uv", "pip", "install", "--system-certs", "--force-reinstall", pkg]
    print(f"  $ {' '.join(cmd)}")
    env = {**os.environ, "UV_SYSTEM_CERTS": "1"}
    subprocess.check_call(cmd, cwd=PROJECT_DIR, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-detect GPU and install correct backends")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="Force CPU-only")
    group.add_argument("--cuda", metavar="VERSION", help="Force CUDA version (e.g. 12.6)")
    args = parser.parse_args()

    print("=== GPU detection ===")

    if args.cpu:
        vendor = "none"
        cuda_tag = None
        print("  forced CPU-only")
    elif args.cuda:
        vendor = "nvidia"
        cuda_tag = _find_best_tag(args.cuda)
        print(f"  forced CUDA {args.cuda} -> torch tag: {cuda_tag}")
    else:
        vendor = _detect_vendor()
        cuda_tag = _detect_cuda_version() if vendor == "nvidia" else None
        label = f"{vendor}" + (f" (CUDA {cuda_tag})" if cuda_tag else "")
        print(f"  detected: {label}")

    print("\n=== Configure torch index ===")
    _configure_pyproject(vendor, cuda_tag)

    print("\n=== uv lock + sync (resolve & install all deps) ===")
    rc = _uv_lock_and_sync()
    if rc != 0:
        return rc

    print("\n=== onnxruntime install ===")
    _install_onnxruntime(vendor)

    # Verify CUDA provider works (warn if toolkit DLLs are missing)
    if vendor == "nvidia":
        print("\n=== CUDA provider check ===")
        try:
            import importlib
            importlib.invalidate_caches()
            ort = importlib.import_module("onnxruntime")
            if "CUDAExecutionProvider" in ort.get_available_providers():
                print("  CUDA provider available")
            else:
                print("  WARNING: CUDA provider listed but failed to load")
                print("  Install the CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
                print("  App will still work on CPU.")
        except Exception as exc:
            print(f"  WARNING: onnxruntime check failed: {exc}")

    print("\nDone. Run `uv run python -m app` to start the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
