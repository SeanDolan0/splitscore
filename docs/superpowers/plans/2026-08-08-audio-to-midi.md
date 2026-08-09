# AudioToMIDI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app that separates an uploaded audio file into 6 stems with BS-RoFormer-SW, then transcribes user-selected stems to per-stem MIDI files with MuScriptor.

**Architecture:** FastAPI backend serves a vanilla single-page frontend. Separation runs the BS-RoFormer-SW ONNX model through onnxruntime-gpu inside a torch STFT/ISTFT wrapper with chunked crossfade overlap-add. Transcription wraps MuScriptor's `TranscriptionModel`. A `Pipeline` coordinates the two stages and pushes progress events over SSE. All model-heavy code is structured for dependency injection so tests never touch the GPU, the HF token, or the 300 MB+ weights.

**Tech Stack:** Python 3.13, uv, FastAPI, uvicorn, onnxruntime-gpu, torch (cu128), soundfile, huggingface-hub, muscriptor 0.3.0, pytest/pytest-asyncio/httpx.

**Spec:** `docs/superpowers/specs/2026-08-08-audio-to-midi-design.md`

## Global Constraints

- Python >= 3.13, managed by uv (already installed: uv 0.11.26).
- **torch must use the CUDA 12.8 backend.** `torch-backend` is only a `uv pip` CLI flag, NOT a `pyproject.toml` setting. In `pyproject.toml` you must pin torch to the cu128 wheel index via `[[tool.uv.index]]` + `[tool.uv.sources]` (see Task 1). On this machine torch-cuda must report `torch.cuda.is_available() == True`.
- MuScriptor weights are **gated** (HF login + CC BY-NC 4.0 non-commercial license). The app must never crash on a missing token — it must surface a setup hint. Tests must not require the token.
- MuScriptor stem order (separation output): `["bass", "drums", "other", "vocals", "guitar", "piano"]`. Do not reorder.
- Separation ONNX I/O is fixed by the model card: inputs `spec_real`/`spec_imag` `float32 [1,2,1025,T]`; outputs `out_spec_real`/`out_spec_imag` `float32 [1,6,2,1025,T]`; T is fixed at 345 frames (4 s @ 44.1 kHz). STFT params are fixed: `torch.stft(audio, n_fft=2048, hop_length=512, win_length=2048, window=hann, center=True, normalized=False)`. Sample rate is fixed at 44100 Hz, stereo.
- Every adjustable setting from the spec's Settings tables is exposed in the GUI and persisted in `app/settings.json` (separation device/precision; model size; per-stem instrument; temperature; beam size; batch size; transcription device; output folder; keep-stems; remember-selection).
- One job at a time — guard both pipeline stages with a single lock on the `Pipeline` instance. The running app has exactly one `Pipeline` (module-level in `app/main.py`), so the instance lock IS the global lock. It is per-instance rather than module-level so pytest-asyncio's per-test event loops don't trip `asyncio.Lock` cross-loop binding errors.
- Model runs never block the SSE event stream: `Pipeline.separate`/`Pipeline.transcribe` run blocking work inside `asyncio.to_thread`, and events pushed from a worker thread go through `loop.call_soon_threadsafe` (asyncio.Queue is not thread-safe).
- SSE terminal events are `done`, `failed`, and `cancelled`. `error` is non-terminal (a single stem failing must not cut the stream). The SSE route ends the stream only on a terminal event.
- All errors surface in the UI; no silent failures.
- A failing task leaves tests red; a passing task ends with `git commit` per task. Never skip the commit step.
- Developer rules that always apply: immutable data patterns, early returns over deep nesting, functions < 50 lines, files < 800 lines, no `console.log`/`print` debug leftovers, no hardcoded secrets. Tests target the named assert(s) — the plan's test code is authoritative; do not "improve" assertions beyond what each step specifies.

## File Structure

```
audioToMidi/
├── pyproject.toml            # deps + cu128 index/sources pin (Task 1)
├── .gitignore                # output/, caches, __pycache__ (Task 1)
├── README.md                 # setup + HF token + license (Task 9)
├── app/
│   ├── __init__.py           # (Task 1)
│   ├── __main__.py           # python -m app entrypoint (Task 7)
│   ├── settings.py           # Settings dataclass + JSON persistence (Task 2)
│   ├── separator.py          # STFT/ISTFT, chunking, ONNX session, separate() (Tasks 3-4)
│   ├── transcribe.py         # MuScriptor wrapper, instruments list (Task 5)
│   ├── pipeline.py           # Job + Pipeline orchestration, SSE events (Task 6)
│   ├── main.py               # FastAPI routes, SSE, static (Task 7)
│   └── static/
│       ├── index.html        # (Task 8)
│       ├── app.js            # (Task 8)
│       └── style.css         # (Task 8)
├── tests/
│   ├── test_settings.py      # (Task 2)
│   ├── test_separator.py     # (Tasks 3-4)
│   ├── test_transcribe.py    # (Task 5)
│   ├── test_pipeline.py      # (Task 6)
│   └── test_main.py          # (Task 7)
└── output/                   # <job_id>/stems/*.wav, <job_id>/midi/*.mid (created at runtime)
```

---

## Task 1: Project scaffold — pyproject.toml, package layout, uv sync

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Modify: `README.md` (stub — filled in Task 9)

**Interfaces:**
- Consumes: nothing.
- Produces: a `uv sync`-able project; `app` importable as a package; torch CUDA-enabled. This is the foundation every later task builds on.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "audio-to-midi"
version = "0.1.0"
description = "Separate audio into stems with BS-RoFormer-SW, then transcribe to MIDI with MuScriptor"
requires-python = ">=3.13,<3.14"   # verified on 3.13.14; <3.14 keeps uv on the tested interpreter
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "onnxruntime-gpu>=1.20",
    "torch>=2.7",
    "soundfile>=0.12",
    "huggingface-hub>=0.24",
    "muscriptor==0.3.0",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["."]   # bare `uv run pytest` (console script) does NOT add the project root to sys.path

# Resolve only for the platforms the cu128 torch source targets. Without this,
# uv also resolves for macOS where muscriptor 0.3.0 pins torch<2.3 and the whole
# resolution fails ("This project requires torch>=2.7" on macOS x86_64).
[tool.uv]
environments = [
    "sys_platform == 'win32'",
    "sys_platform == 'linux'",
]

# Pin torch to the CUDA 12.8 wheel index on Linux/Windows (macOS has no CUDA wheels).
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
output/
.hf_cache/
```

- [ ] **Step 3: Create `app/__init__.py`**

```python
"""AudioToMIDI — stem separation + MIDI transcription."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Run `uv sync` and verify the environment**

Run:
```bash
uv sync
```
Expected: resolves and installs all deps including the cu128 torch build (this downloads ~2.5 GB the first time — allow several minutes).

- [ ] **Step 5: Verify torch CUDA + muscriptor import**

Run:
```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available()); import muscriptor; import fastapi; import onnxruntime; print('imports ok')"
```
Expected: `cuda: True` (the RTX 5070 Ti with cu128 torch) and `imports ok`.
If `cuda: False`, STOP — the cu128 index/source pin is wrong; fix `pyproject.toml` before continuing (do not proceed on CPU).

- [ ] **Step 6: Verify pytest runs (empty collection)**

Run:
```bash
uv run pytest
```
Expected: `no tests ran` (exit code 5) or `collected 0 items`. If it errors on config, fix `[tool.pytest.ini_options]`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore app/__init__.py
git commit -m "chore: scaffold uv project with cu128 torch + core deps"
```

> Note: `uv.lock` is committed — it pins the resolved cu128 torch build and all transitive deps (uv-idiomatic reproducibility). On this machine all uv commands need `--system-certs` (TLS-interception proxy) — e.g. `uv sync --system-certs`.

**Step 5 verification re: `--system-certs`:** the TLS-interception proxy on this machine breaks uv's default cert bundle. Run uv commands as `uv --system-certs <subcommand>` (or `uv sync --system-certs`) throughout this project.

---

## Task 2: Settings module

**Files:**
- Create: `app/settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: nothing (standalone).
- Produces (used by Tasks 6, 7, 8):
  - `@dataclass class Settings` with fields (all lowercase, exactly these names):
    `separation_device: str = "auto"`, `separation_precision: str = "fp16"`,
    `model_size: str = "large"`, `instrument_by_stem: dict = field(default_factory=dict)`,
    `temperature: float = 0.0`, `beam_size: int = 4`, `batch_size: int = 4`,
    `transcription_device: str = "auto"`, `output_folder: str = "./output"`,
    `keep_stems: bool = True`, `remember_selection: bool = True`
  - `STEMS: list[str] = ["bass", "drums", "other", "vocals", "guitar", "piano"]` (module constant)
  - `DEFAULT_SETTINGS: Settings` (module singleton)
  - `SETTINGS_FILE: Path = Path(__file__).parent / "settings.json"` (module constant, so tests can monkeypatch it)
  - `load_settings() -> Settings`
  - `save_settings(s: Settings) -> None`
  - `resolve_device(requested: str) -> str` — `"auto"` → `"cuda"` if `torch.cuda.is_available()` else `"cpu"`; `"cuda"`/`"cpu"` returned as-is.

- [ ] **Step 1: Write the failing test**

```python
from app.settings import Settings, STEMS, DEFAULT_SETTINGS, load_settings, save_settings, resolve_device

def test_stems_order():
    assert STEMS == ["bass", "drums", "other", "vocals", "guitar", "piano"]

def test_defaults_match_spec():
    s = DEFAULT_SETTINGS
    assert s.separation_device == "auto"
    assert s.separation_precision == "fp16"
    assert s.model_size == "large"
    assert s.temperature == 0.0
    assert s.beam_size == 4
    assert s.batch_size == 4
    assert s.output_folder == "./output"
    assert s.keep_stems is True
    assert s.remember_selection is True

def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "settings.json")
    s = Settings(model_size="small", temperature=0.5)
    save_settings(s)
    loaded = load_settings()
    assert loaded.model_size == "small"
    assert loaded.temperature == 0.5
    assert loaded.output_folder == "./output"  # untouched fields keep defaults

def test_load_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "nope.json")
    assert load_settings() == DEFAULT_SETTINGS

def test_resolve_device_auto_and_explicit(monkeypatch):
    monkeypatch.setattr("app.settings.torch", _FakeTorch(cuda=True))
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr("app.settings.torch", _FakeTorch(cuda=False))
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"

class _FakeTorch:
    """Mirror real torch: `torch.cuda` is an attribute whose .is_available() is called."""
    def __init__(self, cuda):
        self.cuda = _FakeCuda(cuda)

class _FakeCuda:
    def __init__(self, available):
        self._available = available
    def is_available(self):
        return self._available
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.settings'`.

- [ ] **Step 3: Write minimal implementation — `app/settings.py`**

```python
"""Persistent app settings."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# Order must match the BS-RoFormer-SW ONNX output channels (0-5).
STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"]

SETTINGS_FILE = Path(__file__).parent / "settings.json"


@dataclass
class Settings:
    separation_device: str = "auto"      # auto | cuda | cpu
    separation_precision: str = "fp16"   # fp16 | fp32
    model_size: str = "large"            # small | medium | large
    instrument_by_stem: dict = field(default_factory=dict)  # stem -> instrument name or "" (auto)
    temperature: float = 0.0             # 0 = deterministic
    beam_size: int = 4                   # 1 = greedy
    batch_size: int = 4
    transcription_device: str = "auto"   # auto | cuda | cpu
    output_folder: str = "./output"
    keep_stems: bool = True
    remember_selection: bool = True


DEFAULT_SETTINGS = Settings()


def load_settings() -> Settings:
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS
    data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    merged = asdict(DEFAULT_SETTINGS)
    merged.update({k: v for k, v in data.items() if k in merged})
    return Settings(**merged)


def save_settings(settings: Settings) -> None:
    SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def resolve_device(requested: str) -> str:
    if requested in ("cuda", "cpu"):
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_settings.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app/settings.py tests/test_settings.py
git commit -m "feat: add persistent settings module"
```

---

## Task 3: Separator DSP primitives — STFT/ISTFT, chunking, overlap-add

**Files:**
- Create: `app/separator.py` (only the DSP primitives — session/`separate()` come in Task 4)
- Test: `tests/test_separator.py`

**Interfaces:**
- Consumes: nothing (pure torch).
- Produces (used by Task 4 and the Task 4/6 tests):
  - `SR = 44100`, `N_FFT = 2048`, `HOP_LENGTH = 512`, `WIN_LENGTH = 2048`, `CHUNK_FRAMES = 345`, `CHUNK_HOP_FRAMES = 220`, `OVERLAP_FRAMES = 125` (module constants)
  - `def stft(audio: torch.Tensor) -> torch.Tensor` — input `[2, N]` float32 at 44.1 kHz; returns complex spectrogram `[2, 1025, F]` via `torch.stft(..., n_fft=2048, hop_length=512, win_length=2048, window=hann, center=True, normalized=False, return_complex=True)`.
  - `def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor` — inverse of `stft`; **always pass `length`** when you know the original signal length (torch's default istft output is truncated for `center=True`).
  - `def split_into_chunks(spec: torch.Tensor) -> list[torch.Tensor]` — splits the `[2, 1025, F]` spectrogram into a list of `[2, 1025, 345]` tensors, stride `CHUNK_HOP_FRAMES`. One chunk if `F <= CHUNK_FRAMES`; else `(F - CHUNK_FRAMES + CHUNK_HOP_FRAMES - 1) // CHUNK_HOP_FRAMES + 1` chunks, last chunk zero-padded to exactly 345 frames.
  - `def overlap_add(chunks: list[torch.Tensor], total_frames: int) -> torch.Tensor` — inverse of `split_into_chunks`: overlap-adds the chunk spectrograms back to `total_frames` frames, crossfading the 125-frame overlap regions with a linear ramp, returning `[2, 1025, total_frames]`.
  - `def mask_and_synthesize(input_spec: torch.Tensor, mask: torch.Tensor, length: int | None = None) -> torch.Tensor` — applies one mask `[2, 1025, F]` to `input_spec`, returns the ISTFT audio for that stem. (Task 4 uses the equivalent inline form so it can accumulate chunks before the single overlap-add; keep this helper as part of the public surface.)

- [ ] **Step 1: Write the failing test**

```python
import torch
from app.separator import stft, istft, split_into_chunks, overlap_add, CHUNK_FRAMES, CHUNK_HOP_FRAMES

def _tone(seconds=0.5, freq=440.0, sr=44100):
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    return 0.5 * torch.sin(2 * torch.pi * freq * t)

def _stereo_tone(seconds=0.5):
    mono = _tone(seconds)
    return torch.stack([mono, mono])

def test_stft_istft_roundtrip():
    audio = _stereo_tone(0.4)
    spec = stft(audio)
    assert spec.shape[0] == 2 and spec.shape[1] == 1025
    recon = istft(spec, length=audio.shape[1])
    assert recon.shape == audio.shape
    err = (recon - audio).abs().mean().item()
    assert err < 1e-3, f"roundtrip error too high: {err}"

def test_split_into_chunks_counts():
    spec = stft(_stereo_tone(10.0))  # ~862 frames -> 4 chunks
    chunks = split_into_chunks(spec)
    expected = (spec.shape[-1] - CHUNK_FRAMES + CHUNK_HOP_FRAMES - 1) // CHUNK_HOP_FRAMES + 1
    assert len(chunks) == expected == 4
    for c in chunks:
        assert c.shape[-1] == CHUNK_FRAMES

def test_split_short_audio_single_padded_chunk():
    spec = stft(_stereo_tone(0.5))  # ~44 frames, well under 345
    chunks = split_into_chunks(spec)
    assert len(chunks) == 1
    assert chunks[0].shape[-1] == CHUNK_FRAMES

def test_overlap_add_reconstructs_input_within_tolerance():
    spec = stft(_stereo_tone(10.0))  # 4 chunks, exercises every crossfade branch
    chunks = split_into_chunks(spec)
    assert len(chunks) == 4
    back = overlap_add(chunks, spec.shape[-1])
    assert back.shape == spec.shape
    rel = (back - spec).abs().mean().item() / spec.abs().mean().item()
    assert rel < 0.04, f"reconstruction rel error too high: {rel}"

def test_overlap_add_total_frames_respected():
    spec = stft(_stereo_tone(10.0))
    chunks = split_into_chunks(spec)
    back = overlap_add(chunks, spec.shape[-1])
    assert back.shape[-1] == spec.shape[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_separator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.separator'`.

- [ ] **Step 3: Write minimal implementation — `app/separator.py` (DSP part)**

```python
"""BS-RoFormer-SW separation: torch DSP around the ONNX model."""

from __future__ import annotations

import torch

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
CHUNK_FRAMES = 345        # model traced at 4 s @ 44.1 kHz (176400 samples)
CHUNK_HOP_FRAMES = 220    # 125-frame overlap (~1.45 s) for crossfading
OVERLAP_FRAMES = CHUNK_FRAMES - CHUNK_HOP_FRAMES  # 125

# Order must match the ONNX output channels.
STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"]


def _window():
    return torch.hann_window(WIN_LENGTH)


def stft(audio: torch.Tensor) -> torch.Tensor:
    """[2, N] float32 @44.1kHz -> [2, 1025, F] complex spectrogram."""
    return torch.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=_window(), center=True, return_complex=True,
    )


def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """[2, 1025, F] -> [2, N] audio. Pass length to reconstruct the exact N."""
    return torch.istft(
        spec, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=_window(), center=True, length=length,
    )


def split_into_chunks(spec: torch.Tensor) -> list[torch.Tensor]:
    """Split [2,1025,F] into [2,1025,345] chunks, stride CHUNK_HOP_FRAMES, pad last."""
    total = spec.shape[-1]
    if total <= CHUNK_FRAMES:
        return [torch.nn.functional.pad(spec, (0, CHUNK_FRAMES - total))]
    n_chunks = (total - CHUNK_FRAMES + CHUNK_HOP_FRAMES - 1) // CHUNK_HOP_FRAMES + 1
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_HOP_FRAMES
        piece = spec[..., start:start + CHUNK_FRAMES]
        if piece.shape[-1] < CHUNK_FRAMES:
            piece = torch.nn.functional.pad(piece, (0, CHUNK_FRAMES - piece.shape[-1]))
        chunks.append(piece)
    return chunks


def overlap_add(chunks: list[torch.Tensor], total_frames: int) -> torch.Tensor:
    """Reverse of split_into_chunks with a linear crossfade over the overlap.

    Chunk i starts at frame i*CHUNK_HOP_FRAMES. The overlap region gets a
    linear ramp 0->1 on the incoming chunk and 1->0 on existing content, which
    sums to exactly 1 for identical overlap content (exact reconstruction).
    """
    if not chunks:
        raise ValueError("no chunks to overlap-add")
    out = chunks[0][..., :total_frames].clone()
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        start = i * CHUNK_HOP_FRAMES
        if start >= total_frames:
            break
        end = min(start + CHUNK_FRAMES, total_frames)
        over = min(OVERLAP_FRAMES, end - start)
        ramp = torch.linspace(0.0, 1.0, over).reshape(1, 1, over)  # float32
        out[..., start:start + over] = (
            out[..., start:start + over] * (1 - ramp) + chunk[..., :over] * ramp)
        if end > out.shape[-1]:
            tail = end - out.shape[-1]
            out = torch.cat([out, chunk[..., over:over + tail]], dim=-1)
    return out[..., :total_frames]


def mask_and_synthesize(input_spec: torch.Tensor, mask: torch.Tensor,
                        length: int | None = None) -> torch.Tensor:
    """Apply a per-stem mask [2,1025,F] to input_spec, ISTFT back to audio [2,N]."""
    return istft(input_spec * mask, length=length)
```

> Note: `mask_and_synthesize` takes a single mask; Task 4 applies masks per stem while accumulating chunks, then does one `overlap_add` + `istft(length=...)` per stem.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_separator.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app/separator.py tests/test_separator.py
git commit -m "feat: add STFT/ISTFT chunking and overlap-add primitives"
```

---

## Task 4: Separator ONNX session + `separate()`

**Files:**
- Modify: `app/separator.py`
- Modify: `tests/test_separator.py` (append tests)

**Interfaces:**
- Consumes: Task 3 primitives (`stft`, `istft`, `split_into_chunks`, `overlap_add`, constants).
- Produces (used by Task 6 pipeline and Task 7 backend):
  - `class Separator:` with:
    - `__init__(self, precision: str = "fp16", device: str = "auto", session_factory=None)` — `session_factory` is an injectable callable `(precision, device) -> session`; default builds the onnxruntime session. Tests pass a fake. On any session-build exception, falls back to a CPU session.
    - `def separate(self, in_path: str | Path, out_dir: str | Path, on_progress: Callable[[float], None] | None = None) -> list[Path]` — decodes audio, separates, writes 6 WAVs named `<out_dir>/<stem>.wav` in `STEMS` order, calls `on_progress(pct)` with 0→100, returns the written paths. Runs synchronously (the pipeline wraps it in a thread).
  - Session contract (for fakes): `.run(None, {"spec_real": r, "spec_imag": i}) -> (out_real, out_imag)` where `r`/`i` are `[1,2,1025,345]` and outputs are `[1,6,2,1025,345]`.
  - `def load_audio(path: str | Path) -> tuple[torch.Tensor, int]` — decode to stereo float32 at 44.1 kHz (soundfile); mono→stereo dup, >2ch trimmed, non-44.1 kHz resampled with linear interpolation. Returns `(audio [2,N], sr)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import soundfile as sf
import torch
from app.separator import Separator, load_audio, STEMS

class FakeSession:
    """Identity separation: masks are all ones, so each stem = the original."""
    def __init__(self, precision, device):
        self.precision = precision
        self.device = device
        self.calls = 0
    def run(self, output_names, feeds):
        self.calls += 1
        r = feeds["spec_real"]  # [1,2,1025,345]
        i = feeds["spec_imag"]
        ones = np.ones_like(r)
        zeros = np.zeros_like(i)
        out_r = np.stack([ones] * 6, axis=1)   # [1,6,2,1025,345]
        out_i = np.stack([zeros] * 6, axis=1)
        return out_r, out_i


def _make_wav(path, seconds=1.0, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.4 * np.sin(2 * np.pi * 330.0 * t)
    stereo = np.stack([sig, sig], axis=1).astype(np.float32)
    sf.write(str(path), stereo, sr)
    return path


def test_load_audio_returns_stereo_44k(tmp_path):
    path = _make_wav(tmp_path / "a.wav", seconds=0.5)
    audio, sr = load_audio(path)
    assert sr == 44100
    assert audio.shape[0] == 2
    assert audio.shape[1] > 0
    assert audio.dtype == torch.float32


def test_separate_writes_six_stems_in_order(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    out = tmp_path / "out"
    sep = Separator(session_factory=FakeSession)
    results = sep.separate(src, out)
    assert [p.name for p in results] == [f"{s}.wav" for s in STEMS]
    for p in results:
        data, sr = sf.read(str(p))
        assert sr == 44100
        assert data.shape[0] > 0


def test_separate_reports_progress(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    sep = Separator(session_factory=FakeSession)
    seen = []
    sep.separate(src, tmp_path / "out", on_progress=lambda pct: seen.append(pct))
    assert seen
    assert seen[-1] == 100.0


def test_identity_separation_reconstructs_original(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    sep = Separator(session_factory=FakeSession)
    results = sep.separate(src, tmp_path / "out")
    orig, _ = sf.read(str(src))
    stem0, _ = sf.read(str(results[0]))
    # Ones mask -> stem 0 should equal the original (same length, close values).
    assert stem0.shape[0] == orig.shape[0]
    rel = np.abs(stem0 - orig).mean() / (np.abs(orig).mean() + 1e-9)
    assert rel < 0.05, f"identity separation drifted: {rel:.3f}"


def test_separate_uses_fake_session_precision_and_device(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=1.0)
    seen = {}

    def factory(precision, device):
        seen["precision"] = precision
        seen["device"] = device
        return FakeSession(precision, device)

    sep = Separator(precision="fp32", device="cpu", session_factory=factory)
    sep.separate(src, tmp_path / "out")
    assert seen == {"precision": "fp32", "device": "cpu"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_separator.py -v`
Expected: FAIL — `ImportError: cannot import name 'Separator'` (and `load_audio`).

- [ ] **Step 3: Append to `app/separator.py`**

```python
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


MODEL_ID = "elicwhite/bs-roformer-sw-6stem-onnx"
MODEL_FILES = {"fp16": "bs_roformer_sw_6stem_fp16.onnx", "fp32": "bs_roformer_sw_6stem_fp32.onnx"}
MODEL_CACHE = Path.home() / ".cache" / "audio-to-midi" / "separator"


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode any soundfile-readable file to stereo float32 @ 44.1 kHz."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")  # [N, ch]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    audio = torch.from_numpy(data.T).float()  # [ch, N]
    if sr != SR:
        target = int(audio.shape[1] * SR / sr)
        audio = torch.stack([
            torch.nn.functional.interpolate(
                audio[c:c+1, None, :], size=target, mode="linear", align_corners=False)[0, 0]
            for c in range(2)])
    return audio, SR


def _download_model(precision: str) -> Path:
    """Download the ONNX file to MODEL_CACHE if missing; return its path."""
    from huggingface_hub import hf_hub_download
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=MODEL_ID, filename=MODEL_FILES[precision], cache_dir=MODEL_CACHE))


def _default_session_factory(precision: str, device: str) -> ort.InferenceSession:
    model_path = _download_model(precision)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(model_path), providers=providers)


class Separator:
    def __init__(self, precision: str = "fp16", device: str = "auto", session_factory=None):
        self.precision = precision if precision in MODEL_FILES else "fp16"
        from app.settings import resolve_device
        self.device = resolve_device(device)
        factory = session_factory or _default_session_factory
        try:
            self.session = factory(self.precision, self.device)
        except Exception:
            # GPU runtime failed (missing provider, download hiccup) -> CPU.
            self.session = _default_session_factory(self.precision, "cpu")
            self.device = "cpu"

    def _run_chunk(self, spec_chunk: torch.Tensor):
        """[2,1025,345] complex -> ([6,2,1025,345] real, [6,2,1025,345] imag) as tensors."""
        r = spec_chunk[None, ...].real.float().numpy()
        i = spec_chunk[None, ...].imag.float().numpy()
        out_r, out_i = self.session.run(None, {"spec_real": r, "spec_imag": i})
        return torch.from_numpy(out_r), torch.from_numpy(out_i)

    def separate(self, in_path, out_dir, on_progress=None) -> list[Path]:
        audio, _ = load_audio(in_path)
        spec = stft(audio.to("cpu"))
        chunks = split_into_chunks(spec)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        acc = [None] * len(STEMS)  # per-stem masked chunk lists
        for idx, chunk in enumerate(chunks):
            masks_r, masks_i = self._run_chunk(chunk)
            for s in range(len(STEMS)):
                mask = masks_r[0, s] + 1j * masks_i[0, s]  # [2,1025,345]
                masked = chunk * mask
                acc[s] = acc[s] or []
                acc[s].append(masked)
            if on_progress:
                on_progress(100.0 * (idx + 1) / len(chunks))
        results = []
        for s, name in enumerate(STEMS):
            recon = overlap_add(acc[s], spec.shape[-1])
            audio_s = istft(recon, length=audio.shape[1]).clamp(-1.0, 1.0)
            path = out_dir / f"{name}.wav"
            sf.write(str(path), audio_s.T.numpy(), SR)
            results.append(path)
        return results
```

> Note: `mask_and_synthesize` (Task 3) is kept as a small public helper; `separate` uses the equivalent inline `chunk * mask` so it can accumulate masked chunks before the single overlap-add per stem. Do not remove `mask_and_synthesize` — it is part of the module's public surface.
>
> Note: if the exact ONNX filename is wrong, `hf_hub_download` raises a 404 at first run. List the real files to fix `MODEL_FILES`:
> `uv run python -c "from huggingface_hub import list_repo_files; print(list_repo_files('elicwhite/bs-roformer-sw-6stem-onnx'))"`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_separator.py -v`
Expected: PASS (10 passed). The `FakeSession` prevents any real model download; the identity test confirms the whole chunk/mask/overlap path.

- [ ] **Step 5: Commit**

```bash
git add app/separator.py tests/test_separator.py
git commit -m "feat: add ONNX separation with device fallback and stem WAV output"
```

---

## Task 5: Transcriber module (MuScriptor wrapper)

**Files:**
- Create: `app/transcribe.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: nothing (wraps muscriptor).
- Produces (used by Task 6 pipeline and Task 7 backend):
  - `def list_instruments() -> list[str]` — returns MuScriptor's instrument vocabulary; caches the result in a module-level `_INSTRUMENTS`; on any failure returns `[]`.
  - `class Transcriber:` with:
    - `__init__(self, model_size: str = "large", device: str = "auto")` — raises `RuntimeError` with a setup hint if `device == "cuda"` but torch has no CUDA.
    - `def transcribe(self, stem_path: str | Path, stem: str, instruments: str | None = None, temperature: float = 0.0, beam_size: int = 4, batch_size: int = 4) -> bytes` — returns MIDI bytes via `model.transcribe_to_midi(...)`.
  - `transcribe_to_midi` call contract (verified against muscriptor 0.3.0 source):
    `transcribe_to_midi(audio, use_sampling, temperature, cfg_coef, instruments, batch_size, no_eos_is_ok, beam_size, prelude_forcing, detect_tempo) -> bytes`
    — pass `use_sampling=temperature > 0`, `temperature=temperature if temperature > 0 else 1.0`, `instruments=[instruments] if instruments else None`, `beam_size=beam_size`, `batch_size=batch_size` (the raw int), `prelude_forcing=(batch_size == 1)`. (`prelude_forcing=True` with `batch_size > 1` raises ValueError in muscriptor, so it must be `False` for batched runs.)

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock, patch
import app.transcribe as tr
from app.transcribe import Transcriber, list_instruments

def test_list_instruments_caches_and_returns():
    tr._INSTRUMENTS = None
    with patch("app.transcribe._fetch_instruments", return_value=["acoustic_piano", "drums"]) as f:
        assert list_instruments() == ["acoustic_piano", "drums"]
        assert list_instruments() == ["acoustic_piano", "drums"]
        assert f.call_count == 1  # cached after first call

def test_list_instruments_handles_failure():
    tr._INSTRUMENTS = None
    with patch("app.transcribe._fetch_instruments", side_effect=RuntimeError("no cli")):
        assert list_instruments() == []

def test_transcribe_passes_instruments_and_temperature():
    fake_model = MagicMock()
    fake_model.transcribe_to_midi.return_value = b"\x00MIDI"
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber(model_size="large", device="cpu")
        out = t.transcribe("s.wav", "vocals", instruments="vocals", temperature=0.8, beam_size=4, batch_size=4)
        assert out == b"\x00MIDI"
        kwargs = fake_model.transcribe_to_midi.call_args[1]
        assert kwargs["instruments"] == ["vocals"]
        assert kwargs["use_sampling"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["beam_size"] == 4
        assert kwargs["batch_size"] == 4
        assert kwargs["prelude_forcing"] is False  # batch_size>1 forces this

def test_transcribe_none_instruments_and_deterministic():
    fake_model = MagicMock()
    fake_model.transcribe_to_midi.return_value = b"midi"
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber()
        t.transcribe("s.wav", "piano", instruments=None, temperature=0.0, batch_size=1)
        kwargs = fake_model.transcribe_to_midi.call_args[1]
        assert kwargs["instruments"] is None
        assert kwargs["use_sampling"] is False
        assert kwargs["temperature"] == 1.0
        assert kwargs["batch_size"] == 1
        assert kwargs["prelude_forcing"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.transcribe'`.

- [ ] **Step 3: Write minimal implementation — `app/transcribe.py`**

```python
"""MuScriptor wrapper: transcribe a stem audio file to MIDI bytes."""

from __future__ import annotations

import shutil
import subprocess

from muscriptor import TranscriptionModel

_INSTRUMENTS: list[str] | None = None


def _fetch_instruments() -> list[str]:
    """Run `muscriptor list-instruments` and parse names (lines, stripped, non-empty)."""
    if shutil.which("muscriptor") is None:
        raise RuntimeError("muscriptor CLI not found")
    out = subprocess.check_output(["muscriptor", "list-instruments"], text=True, timeout=60)
    return [line.strip() for line in out.splitlines() if line.strip()]


def list_instruments() -> list[str]:
    """MuScriptor instrument vocabulary, cached; [] on failure."""
    global _INSTRUMENTS
    if _INSTRUMENTS is None:
        try:
            _INSTRUMENTS = _fetch_instruments()
        except Exception:
            _INSTRUMENTS = []
    return _INSTRUMENTS


class Transcriber:
    def __init__(self, model_size: str = "large", device: str = "auto"):
        self.model_size = model_size
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "cuda requested for transcription but torch has no CUDA. "
                    "Reinstall with the cu128 torch backend (see README).")
        self.model = TranscriptionModel.load_model(model_size)

    def transcribe(self, stem_path, stem, instruments=None, temperature=0.0,
                   beam_size=4, batch_size=4) -> bytes:
        return self.model.transcribe_to_midi(
            str(stem_path),
            instruments=[instruments] if instruments else None,
            use_sampling=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            beam_size=beam_size,
            batch_size=batch_size,
            prelude_forcing=(batch_size == 1),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS (4 passed). Tests patch `app.transcribe.TranscriptionModel`, so no weights download and no HF token needed.

- [ ] **Step 5: Commit**

```bash
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: add MuScriptor transcription wrapper"
```

---

## Task 6: Pipeline — Job, orchestration, SSE event stream

**Files:**
- Create: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 2 `Settings`, `STEMS`; Task 4 `Separator`; Task 5 `Transcriber`.
- Produces (used by Task 7 backend):
  - `@dataclass class Job:` fields `id: str`, `status: str = "created"`, `song_name: str = ""`, `input_path: Path | None = None`, `output_dir: Path | None = None`, `error: str | None = None`, `cancel: asyncio.Event = field(default_factory=asyncio.Event)`, `events: asyncio.Queue = field(default_factory=asyncio.Queue)`. Status values: `created, separating, ready, transcribing, done, failed, cancelled` (constants `STATUS_*`).
  - `class Pipeline:` with:
    - `__init__(self, settings: Settings, separator_factory=None, transcriber_factory=None)` — factories injectable for tests. Creates `self._lock = asyncio.Lock()` (the one-at-a-time guard) and `self.jobs: dict[str, Job] = {}` (job registry the backend reads).
    - `def create_job(self, song_name: str, input_path: str | Path) -> Job` — creates `output/<job_id>/stems|midi/` dirs and registers `self.jobs[job.id] = job`.
    - `async def separate(self, job: Job) -> None` — acquires `self._lock`; sets `status="separating"`; builds the separator in a thread; runs `sep.separate` in a thread with an `on_progress` that pushes `{"type":"progress","phase":"separating","pct":...}`; sets `status="ready"`; pushes `{"type":"stems","stems":STEMS}`. On `CancelledError`/cancel flag → `status="cancelled"` + `{"type":"cancelled"}` (terminal). On exception → `status="failed"`, `error=str`, push `{"type":"failed","message":...}` (terminal).
    - `async def transcribe(self, job: Job, stems: list[str], instrument_by_stem: dict[str, str], temperature: float, beam_size: int, batch_size: int) -> None` — acquires `self._lock`; builds the transcriber in a thread; per stem: push `{"type":"progress","phase":"transcribing","stem":stem,"pct":0}`, transcribe in a thread, write `<output_dir>/midi/<song>_<stem>.mid`, push `{"type":"midi","stem":stem,"file":...}`. Per-stem failure → push `{"type":"error",...}` (NON-terminal), continue. Sets `status="done"` + `{"type":"done"}` (terminal) at the end; job-level exception → `{"type":"failed"}` (terminal).
  - **Thread-safety rule:** `asyncio.Queue` is not thread-safe. Events pushed from a worker thread (separator progress) must go through `loop.call_soon_threadsafe(job.events.put_nowait, event)`. Events pushed from the coroutine (everything else) use plain `put_nowait`.

- [ ] **Step 1: Write the failing test**

```python
import asyncio
from pathlib import Path
from app.pipeline import Pipeline, Job
from app.settings import Settings

class FakeSeparator:
    def __init__(self, precision="fp16", device="auto", session_factory=None):
        pass
    def separate(self, in_path, out_dir, on_progress=None):
        for pct in (10, 50, 100):
            on_progress(pct)

class FakeTranscriber:
    def __init__(self, model_size="large", device="auto"):
        pass
    def transcribe(self, stem_path, stem, instruments=None, temperature=0.0,
                   beam_size=4, batch_size=4) -> bytes:
        return f"midi:{stem}".encode()


def _drain(q: asyncio.Queue) -> list[dict]:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


async def test_separate_flow_and_events(tmp_path):
    job = Job(id="j1", song_name="song", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)
    pipe = Pipeline(Settings(), separator_factory=FakeSeparator)
    await pipe.separate(job)
    await asyncio.sleep(0)
    assert job.status == "ready"
    events = _drain(job.events)
    assert [e["pct"] for e in events if e["type"] == "progress"] == [10, 50, 100]
    assert events[-1] == {"type": "stems",
                          "stems": ["bass", "drums", "other", "vocals", "guitar", "piano"]}


async def test_transcribe_writes_midi_per_stem(tmp_path):
    job = Job(id="j1", song_name="my song", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out" / "midi").mkdir(parents=True)
    pipe = Pipeline(Settings(), transcriber_factory=FakeTranscriber)
    await pipe.transcribe(job, ["vocals", "piano"], {"vocals": "", "piano": ""},
                          temperature=0.0, beam_size=4, batch_size=4)
    assert job.status == "done"
    files = sorted(p.name for p in (tmp_path / "out" / "midi").glob("*.mid"))
    assert files == ["my song_piano.mid", "my song_vocals.mid"]
    midi_events = [e for e in _drain(job.events) if e["type"] == "midi"]
    assert {e["stem"] for e in midi_events} == {"vocals", "piano"}


async def test_per_stem_failure_does_not_stop_pipeline(tmp_path):
    class Flaky(FakeTranscriber):
        def transcribe(self, *a, **k):
            if a[1] == "drums":
                raise RuntimeError("boom")
            return b"ok"
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out" / "midi").mkdir(parents=True)
    pipe = Pipeline(Settings(), transcriber_factory=Flaky)
    await pipe.transcribe(job, ["vocals", "drums"], {}, 0.0, 4, 4)
    assert job.status == "done"  # drums failed but vocals succeeded
    assert (tmp_path / "out" / "midi" / "s_vocals.mid").exists()


async def test_cancelled_separation_sets_cancelled(tmp_path):
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    job.cancel.set()
    pipe = Pipeline(Settings(), separator_factory=FakeSeparator)
    await pipe.separate(job)
    assert job.status == "cancelled"
    assert _drain(job.events)[-1] == {"type": "cancelled"}


async def test_separation_failure_sets_failed_with_error(tmp_path):
    class Boom:
        def __init__(self, precision="fp16", device="auto", session_factory=None):
            pass
        def separate(self, in_path, out_dir, on_progress=None):
            raise RuntimeError("no model weights")
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    pipe = Pipeline(Settings(), separator_factory=Boom)
    await pipe.separate(job)
    assert job.status == "failed"
    assert "model weights" in job.error
    assert _drain(job.events)[-1]["type"] == "failed"


async def test_create_job_registers_and_makes_dirs(tmp_path):
    pipe = Pipeline(Settings())
    job = pipe.create_job("song", tmp_path / "in.wav")
    assert job.id in pipe.jobs
    assert job.output_dir is not None
    assert (job.output_dir / "stems").is_dir()
    assert (job.output_dir / "midi").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline'`.

- [ ] **Step 3: Write minimal implementation — `app/pipeline.py`**

```python
"""Job orchestration: separate then transcribe, with SSE event stream + cancel."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.separator import STEMS, Separator
from app.settings import Settings
from app.transcribe import Transcriber

STATUS_CREATED = "created"
STATUS_SEPARATING = "separating"
STATUS_READY = "ready"
STATUS_TRANSCRIBING = "transcribing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    song_name: str = ""
    status: str = STATUS_CREATED
    input_path: Path | None = None
    output_dir: Path | None = None
    error: str | None = None
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    events: asyncio.Queue = field(default_factory=asyncio.Queue)


class Pipeline:
    def __init__(self, settings: Settings, separator_factory=None, transcriber_factory=None):
        self.settings = settings
        self._sep_factory = separator_factory or Separator
        self._tr_factory = transcriber_factory or Transcriber
        self._lock = asyncio.Lock()
        self.jobs: dict[str, Job] = {}

    def create_job(self, song_name: str, input_path: str | Path) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], song_name=song_name, input_path=Path(input_path))
        out = Path(self.settings.output_folder) / job.id
        job.output_dir = out
        (out / "stems").mkdir(parents=True, exist_ok=True)
        (out / "midi").mkdir(parents=True, exist_ok=True)
        self.jobs[job.id] = job
        return job

    @staticmethod
    def _emit(job: Job, event: dict) -> None:
        """Push an event from the event loop thread (plain put_nowait)."""
        job.events.put_nowait(event)

    async def separate(self, job: Job) -> None:
        if job.cancel.is_set():
            job.status = STATUS_CANCELLED
            self._emit(job, {"type": "cancelled"})
            return
        job.status = STATUS_SEPARATING
        async with self._lock:
            try:
                if job.cancel.is_set():
                    job.status = STATUS_CANCELLED
                    self._emit(job, {"type": "cancelled"})
                    return
                loop = asyncio.get_running_loop()

                def emit(event: dict) -> None:
                    # Called from the separator's worker thread -> thread-safe push.
                    loop.call_soon_threadsafe(job.events.put_nowait, event)

                sep = await asyncio.to_thread(
                    self._sep_factory, precision=self.settings.separation_precision,
                    device=self.settings.separation_device)
                await asyncio.to_thread(
                    sep.separate, str(job.input_path), job.output_dir / "stems",
                    lambda pct: emit({"type": "progress", "phase": "separating", "pct": pct}))
                job.status = STATUS_READY
                emit({"type": "stems", "stems": STEMS})
            except asyncio.CancelledError:
                job.status = STATUS_CANCELLED
                self._emit(job, {"type": "cancelled"})
            except Exception as exc:
                job.status = STATUS_FAILED
                job.error = str(exc)
                self._emit(job, {"type": "failed", "message": str(exc)})

    async def transcribe(self, job: Job, stems: list[str], instrument_by_stem: dict[str, str],
                         temperature: float, beam_size: int, batch_size: int) -> None:
        job.status = STATUS_TRANSCRIBING
        async with self._lock:
            try:
                tr = await asyncio.to_thread(
                    self._tr_factory, model_size=self.settings.model_size,
                    device=self.settings.transcription_device)
                for stem in stems:
                    if job.cancel.is_set():
                        job.status = STATUS_CANCELLED
                        self._emit(job, {"type": "cancelled"})
                        return
                    self._emit(job, {"type": "progress", "phase": "transcribing",
                                     "stem": stem, "pct": 0})
                    try:
                        midi_bytes = await asyncio.to_thread(
                            tr.transcribe, job.output_dir / "stems" / f"{stem}.wav", stem,
                            instrument_by_stem.get(stem) or None, temperature, beam_size, batch_size)
                        out = job.output_dir / "midi" / f"{job.song_name}_{stem}.mid"
                        out.write_bytes(midi_bytes)
                        self._emit(job, {"type": "midi", "stem": stem, "file": out.name})
                    except Exception as exc:
                        # One stem failing must not stop the others (non-terminal).
                        self._emit(job, {"type": "error", "message": f"{stem}: {exc}"})
            except Exception as exc:
                job.status = STATUS_FAILED
                job.error = str(exc)
                self._emit(job, {"type": "failed", "message": str(exc)})
                return
        if job.cancel.is_set():
            job.status = STATUS_CANCELLED
            self._emit(job, {"type": "cancelled"})
        else:
            job.status = STATUS_DONE
            self._emit(job, {"type": "done"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (6 passed). Uses per-instance lock (no module-global lock, so pytest-asyncio's per-test loops don't trip cross-loop binding).

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: add job pipeline with SSE events and cancellation"
```

---

## Task 7: FastAPI backend — routes, SSE, static serving, upload validation

**Files:**
- Create: `app/main.py`
- Create: `app/__main__.py`
- Create: `app/static/index.html` (stub — the full page replaces it in Task 8; the `StaticFiles` mount at `app/static/` raises `RuntimeError: Directory ... does not exist` at import if the dir is absent, so a stub must exist before the Task 7 tests run)
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: Task 2 `Settings`, Task 6 `Pipeline`/`Job`/status constants.
- Produces: the runnable app. `uv run python -m app` starts uvicorn on `127.0.0.1:8000` and opens the browser.
- Routes:
  - `POST /api/jobs` — multipart field `file`; validates non-empty + allowed extensions (`wav, mp3, flac, ogg, m4a, aiff`); saves to `output/<job_id>/input/<orig>`; kicks off `asyncio.create_task(pipeline.separate(job))`; returns `{"job_id": ...}`.
  - `GET /api/jobs/{job_id}` — `{"job_id", "status", "error", "song_name", "midi": [filenames]}`.
  - `GET /api/jobs/{job_id}/events` — SSE stream; yields `data: <json>\n\n` for each event in `job.events` plus a heartbeat comment every 15 s; ends after a terminal event (`done`/`failed`/`cancelled`).
  - `POST /api/jobs/{job_id}/transcribe` — body `{"stems": [...]}`; 404 unknown job, 409 if status not `ready`/`done`/`failed`, 400 if any stem ∉ `STEMS`; starts transcribe task; returns `{"job_id"}`.
  - `POST /api/jobs/{job_id}/cancel` — sets `job.cancel`; returns `{"ok": true}`.
  - `GET /api/settings` — returns current settings.
  - `PUT /api/settings` — saves settings; returns the saved settings.
  - `GET /output/{job_id}/midi/{filename}` — MIDI file as `audio/midi` download.
  - `GET /output/{job_id}/stems/{filename}` — stem WAV as `audio/wav` (browser preview).
  - Static mount at `/` (must be registered after all API routes).

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import io
import wave
from pathlib import Path
from fastapi.testclient import TestClient
import app.main as main


def _make_upload_bytes() -> bytes:
    """Minimal valid WAV (1s of silence, 16 kHz mono) built entirely in memory."""
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000)
    w.close()
    return buf.getvalue()


class FakePipeline:
    def __init__(self, settings, separator_factory=None, transcriber_factory=None):
        self.settings = settings
        self.jobs = {}
    def create_job(self, song_name, input_path):
        from app.pipeline import STATUS_READY, Job
        job = Job(id="abc", song_name=song_name, input_path=Path(input_path))
        job.output_dir = Path(self.settings.output_folder) / "abc"
        (job.output_dir / "stems").mkdir(parents=True, exist_ok=True)
        (job.output_dir / "midi").mkdir(parents=True, exist_ok=True)
        job.status = STATUS_READY
        self.jobs[job.id] = job
        return job
    async def separate(self, job):
        await asyncio.sleep(0)


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(main, "PIPELINE", FakePipeline(main.PIPELINE.settings))
    monkeypatch.setattr(main.PIPELINE.settings, "output_folder", str(tmp_path))
    return TestClient(main.app)


def test_upload_creates_job(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.wav", _make_upload_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["job_id"] == "abc"

def test_upload_rejects_bad_extension(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400
    assert "extension" in r.json()["detail"].lower()

def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.wav", b"", "audio/wav")})
    assert r.status_code == 400

def test_settings_roundtrip(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.put("/api/settings", json={"model_size": "small", "temperature": 0.5})
    assert r.json()["model_size"] == "small"
    r2 = client.get("/api/settings")
    assert r2.json()["model_size"] == "small"

def test_transcribe_requires_valid_stems(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.wav", _make_upload_bytes(), "audio/wav")})
    job_id = r.json()["job_id"]
    r = client.post(f"/api/jobs/{job_id}/transcribe", json={"stems": ["not-a-stem"]})
    assert r.status_code == 400

def test_index_html_served(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "audio-midi-app" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Create the stub `app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>AudioToMIDI</title></head>
<body><div id="audio-midi-app">AudioToMIDI</div></body></html>
```

> Note: this stub is required so `app.main` imports (its `StaticFiles` mount needs `app/static/` to exist). Task 8 overwrites it with the full page.

- [ ] **Step 4: Write minimal implementation — `app/main.py`**

```python
"""FastAPI backend: routes, SSE event stream, static frontend."""

from __future__ import annotations

import asyncio
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline import STATUS_DONE, STATUS_FAILED, STATUS_READY, Pipeline
from app.separator import STEMS
from app.settings import Settings, load_settings, save_settings

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aiff"}

app = FastAPI(title="AudioToMIDI")
PIPELINE = Pipeline(load_settings())

_HEARTBEAT = ":" + " " * 15 + "\n\n"  # SSE comment keeps the connection alive

TERMINAL_EVENTS = {"done", "failed", "cancelled"}


class TranscribeBody(BaseModel):
    stems: list[str]


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    name = file.filename or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported extension '.{ext}'; allowed: {sorted(ALLOWED_EXTENSIONS)}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file upload")
    job = PIPELINE.create_job(Path(name).stem, "unused")
    in_dir = job.output_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    job.input_path = in_dir / name
    job.input_path.write_bytes(data)
    asyncio.create_task(PIPELINE.separate(job))
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = PIPELINE.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    midi = [p.name for p in (job.output_dir / "midi").glob("*.mid")]
    return {"job_id": job.id, "status": job.status, "error": job.error,
            "song_name": job.song_name, "midi": midi}


@app.post("/api/jobs/{job_id}/transcribe")
async def transcribe(job_id: str, body: TranscribeBody):
    job = PIPELINE.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    if job.status not in (STATUS_READY, STATUS_DONE, STATUS_FAILED):
        raise HTTPException(409, f"Job not ready (status={job.status})")
    bad = [s for s in body.stems if s not in STEMS]
    if bad:
        raise HTTPException(400, f"Unknown stems: {bad}")
    s = PIPELINE.settings
    asyncio.create_task(PIPELINE.transcribe(
        job, body.stems, s.instrument_by_stem, s.temperature, s.beam_size, s.batch_size))
    return {"job_id": job.id}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: str):
    job = PIPELINE.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    job.cancel.set()
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str):
    job = PIPELINE.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")

    async def gen():
        while True:
            try:
                event = await asyncio.wait_for(job.events.get(), timeout=15.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in TERMINAL_EVENTS:
                    return
            except asyncio.TimeoutError:
                yield _HEARTBEAT
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/settings")
async def get_settings():
    return load_settings()


@app.put("/api/settings")
async def put_settings(body: dict):
    merged = asdict(load_settings())
    merged.update({k: v for k, v in body.items() if k in merged})
    save_settings(Settings(**merged))
    return merged


@app.get("/output/{job_id}/midi/{filename}")
async def download_midi(job_id: str, filename: str):
    path = Path(PIPELINE.settings.output_folder) / job_id / "midi" / filename
    if not path.is_file():
        raise HTTPException(404, "MIDI not found")
    return FileResponse(path, media_type="audio/midi", filename=filename)


@app.get("/output/{job_id}/stems/{filename}")
async def download_stem(job_id: str, filename: str):
    path = Path(PIPELINE.settings.output_folder) / job_id / "stems" / filename
    if not path.is_file():
        raise HTTPException(404, "Stem not found")
    return FileResponse(path, media_type="audio/wav")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


def main() -> None:
    import uvicorn
    url = "http://127.0.0.1:8000"
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

**Also create `app/__main__.py`** (`python -m app` requires it):

```python
from app.main import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py -v`
Expected: PASS (6 passed). Uses `FakePipeline`, so no real separation, no model downloads, no HF token.

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/__main__.py app/static/index.html tests/test_main.py
git commit -m "feat: add FastAPI backend with SSE progress and job routes"
```

---

## Task 8: Frontend — vanilla single-page UI

**Files:**
- Create: `app/static/index.html`
- Create: `app/static/app.js`
- Create: `app/static/style.css`
- Modify: `tests/test_main.py` (append one test)

**Interfaces:**
- Consumes: Task 7 routes. Talks to: `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/events` (EventSource), `POST /api/jobs/{id}/transcribe`, `POST /api/jobs/{id}/cancel`, `GET/PUT /api/settings`, `GET /output/{id}/midi/{f}` and `GET /output/{id}/stems/{f}`.
- Produces: the GUI. The 6 stem cards are **static HTML** with `data-stem` attributes (the Task 8 test asserts on them); `showStems()` fills in the audio preview + instrument value after separation.

- [ ] **Step 1: Write the failing test (appended to `tests/test_main.py`)**

```python
def test_frontend_marks_stem_checkboxes(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.get("/")
    html = r.text
    for stem in ["vocals", "piano", "guitar", "bass", "drums", "other"]:
        assert f'data-stem="{stem}"' in html
    assert "audio-midi-app" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — `index.html` does not exist yet (404 on `/`).

- [ ] **Step 3: Create `app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AudioToMIDI</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <div id="audio-midi-app" class="shell">
    <header class="topbar">
      <h1>AudioToMIDI</h1>
      <p class="tagline">separate → transcribe → MIDI</p>
    </header>

    <section id="drop-zone" class="panel dropzone" tabindex="0" role="button"
             aria-label="Upload audio file">
      <input type="file" id="file-input" accept=".wav,.mp3,.flac,.ogg,.m4a,.aiff" hidden>
      <p class="drop-hint">Drag &amp; drop an audio file</p>
      <p class="drop-sub">or click to browse (wav, mp3, flac, ogg, m4a, aiff)</p>
    </section>

    <section id="job-panel" class="panel hidden">
      <div class="job-row">
        <span id="job-status" class="job-status">created</span>
        <button id="btn-cancel" class="ghost" type="button">cancel</button>
      </div>
      <div class="progress"><div id="progress-bar" class="progress-bar"></div></div>
      <p id="progress-text" class="muted"></p>
      <p id="job-error" class="error hidden"></p>
    </section>

    <section id="stem-panel" class="panel hidden">
      <h2>Stems</h2>
      <div id="stem-grid" class="stem-grid">
        <label class="stem-card"><input type="checkbox" data-stem="bass" checked><span class="stem-name">bass</span><audio controls preload="none"></audio><input class="inst" data-instrument data-stem="bass" placeholder="instrument (auto)"></label>
        <label class="stem-card"><input type="checkbox" data-stem="drums"><span class="stem-name">drums</span><audio controls preload="none"></audio><input class="inst" data-instrument data-stem="drums" placeholder="instrument (auto)"></label>
        <label class="stem-card"><input type="checkbox" data-stem="other"><span class="stem-name">other</span><audio controls preload="none"></audio><input class="inst" data-instrument data-stem="other" placeholder="instrument (auto)"></label>
        <label class="stem-card"><input type="checkbox" data-stem="vocals" checked><span class="stem-name">vocals</span><audio controls preload="none"></audio><input class="inst" data-instrument data-stem="vocals" placeholder="instrument (auto)"></label>
        <label class="stem-card"><input type="checkbox" data-stem="guitar" checked><span class="stem-name">guitar</span><audio controls preload="none"></audio><input class="inst" data-instrument data-stem="guitar" placeholder="instrument (auto)"></label>
        <label class="stem-card"><input type="checkbox" data-stem="piano" checked><span class="stem-name">piano</span><audio controls preload="none"></audio><input class="inst" data-instrument data-stem="piano" placeholder="instrument (auto)"></label>
      </div>
      <div class="actions">
        <button id="btn-transcribe" class="primary" type="button">Transcribe selected to MIDI</button>
      </div>
      <div id="results" class="results"></div>
    </section>

    <aside class="panel settings">
      <h2>Settings</h2>
      <div id="settings-form" class="settings-form"></div>
      <button id="btn-save-settings" class="ghost" type="button">Save settings</button>
      <p id="settings-note" class="muted"></p>
    </aside>
  </div>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Create `app/static/app.js`**

```javascript
const STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"];
const DEFAULT_CHECKED = ["vocals", "piano", "guitar", "bass"];
let settings = {};
let currentJob = null;
let eventSource = null;

const $ = (id) => document.getElementById(id);

// ---------- settings ----------
async function loadSettings() {
  settings = await (await fetch("/api/settings")).json();
  renderSettingsForm();
}
function renderSettingsForm() {
  const fields = [
    ["separation_precision", "Separation precision", ["fp16", "fp32"]],
    ["model_size", "Model size", ["small", "medium", "large"]],
    ["separation_device", "Separation device", ["auto", "cuda", "cpu"]],
    ["transcription_device", "Transcription device", ["auto", "cuda", "cpu"]],
    ["temperature", "Temperature (0 = deterministic)", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5]],
    ["beam_size", "Beam size", [1, 2, 3, 4, 5, 6, 8]],
    ["batch_size", "Batch size", [1, 2, 4, 8]],
    ["output_folder", "Output folder", null],
  ];
  $("settings-form").innerHTML = fields.map(([key, label, options]) => {
    const val = settings[key];
    let control;
    if (Array.isArray(options)) {
      control = `<select id="s-${key}">${options.map((o) => `<option value="${o}" ${String(o) === String(val) ? "selected" : ""}>${o}</option>`).join("")}</select>`;
    } else {
      control = `<input id="s-${key}" type="text" value="${val}">`;
    }
    return `<label class="field"><span>${label}</span>${control}</label>`;
  }).join("");
  [["keep_stems", "Keep stem WAVs after conversion"],
   ["remember_selection", "Remember last stem selection"]].forEach(([key, label]) => {
    $("settings-form").insertAdjacentHTML("beforeend",
      `<label class="field toggle"><input type="checkbox" id="s-${key}" ${settings[key] ? "checked" : ""}> <span>${label}</span></label>`);
  });
}
async function saveSettings() {
  const body = {};
  ["separation_precision", "model_size", "separation_device", "transcription_device",
   "temperature", "beam_size", "batch_size", "output_folder"].forEach((key) => {
    body[key] = $(`s-${key}`).value;
  });
  body.temperature = parseFloat(body.temperature);
  body.beam_size = parseInt(body.beam_size, 10);
  body.batch_size = parseInt(body.batch_size, 10);
  ["keep_stems", "remember_selection"].forEach((key) => {
    body[key] = $(`s-${key}`).checked;
  });
  const resp = await fetch("/api/settings", {
    method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  settings = await resp.json();
  $("settings-note").textContent = "Settings saved.";
}

// ---------- upload ----------
function setupDropzone() {
  const dz = $("drop-zone");
  const input = $("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("over");
    if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) upload(input.files[0]); });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
}
async function upload(file) {
  $("job-error").classList.add("hidden");
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch("/api/jobs", { method: "POST", body: fd });
  if (!resp.ok) { showError((await resp.json()).detail); return; }
  const { job_id } = await resp.json();
  currentJob = job_id;
  $("job-panel").classList.remove("hidden");
  $("stem-panel").classList.add("hidden");
  $("results").innerHTML = "";
  setProgress(0, "Separating stems…");
  connectEvents(job_id);
}

// ---------- SSE ----------
function connectEvents(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  eventSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "progress") {
      if (ev.phase === "separating") setProgress(ev.pct, "Separating stems…");
      if (ev.phase === "transcribing") setProgress(ev.pct, `Transcribing ${ev.stem}…`);
    } else if (ev.type === "stems") {
      setProgress(100, "Separation done");
      showStems();
    } else if (ev.type === "midi") {
      addResult(ev.stem, ev.file);
    } else if (ev.type === "done") {
      setProgress(100, "Done");
      eventSource.close();
    } else if (ev.type === "error") {
      showError(ev.message);
    } else if (ev.type === "failed" || ev.type === "cancelled") {
      showError(ev.message || ev.type);
      eventSource.close();
    }
  };
}

// ---------- stems ----------
function showStems() {
  const remembered = settings.remember_selection
    ? (JSON.parse(localStorage.getItem("checkedStems") || "null") || DEFAULT_CHECKED)
    : DEFAULT_CHECKED;
  STEMS.forEach((stem) => {
    const card = document.querySelector(`.stem-card[data-stem="${stem}"]`);
    card.querySelector('input[type="checkbox"]').checked = remembered.includes(stem);
    card.querySelector("audio").src = `/output/${currentJob}/stems/${stem}.wav`;
    card.querySelector(".inst").value = settings.instrument_by_stem?.[stem] || "";
  });
  $("stem-panel").classList.remove("hidden");
}
function selectedStems() {
  return [...document.querySelectorAll('input[data-stem]:checked')].map((el) => el.dataset.stem);
}
async function transcribeSelected() {
  const stems = selectedStems();
  if (!stems.length) { showError("Select at least one stem."); return; }
  settings.instrument_by_stem = {};
  [...document.querySelectorAll('input[data-instrument]')].forEach((el) => {
    settings.instrument_by_stem[el.dataset.stem] = el.value.trim();
  });
  if (settings.remember_selection) localStorage.setItem("checkedStems", JSON.stringify(stems));
  await fetch(`/api/jobs/${currentJob}/transcribe`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ stems }),
  });
  setProgress(0, "Transcribing…");
}

// ---------- results ----------
function addResult(stem, file) {
  const link = document.createElement("a");
  link.href = `/output/${currentJob}/midi/${encodeURIComponent(file)}`;
  link.textContent = `download ${file}`;
  link.classList.add("result-link");
  $("results").appendChild(link);
}
function setProgress(pct, text) {
  $("progress-bar").style.width = `${pct}%`;
  $("progress-text").textContent = text;
  $("job-status").textContent = text;
}
function showError(message) {
  const el = $("job-error");
  el.textContent = message;
  el.classList.remove("hidden");
}

$("btn-save-settings").addEventListener("click", saveSettings);
$("btn-cancel").addEventListener("click", async () => {
  if (currentJob) await fetch(`/api/jobs/${currentJob}/cancel`, { method: "POST" });
});
$("btn-transcribe").addEventListener("click", transcribeSelected);
loadSettings();
setupDropzone();
```

- [ ] **Step 5: Create `app/static/style.css`**

```css
:root {
  --bg: #0d1117;
  --panel: #161b22;
  --panel-2: #1c2333;
  --line: #2a3242;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #d2a8ff;      /* violet, studio glow */
  --accent-2: #58a6ff;
  --good: #3fb950;
  --bad: #f85149;
  --radius: 10px;
  --font-mono: "Cascadia Mono", "JetBrains Mono", ui-monospace, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.shell { max-width: 880px; margin: 0 auto; padding: 24px 16px 64px; display: grid; gap: 16px; }
.topbar h1 { font-family: var(--font-mono); letter-spacing: -0.02em; margin: 0; }
.topbar .tagline { color: var(--muted); margin: 2px 0 0; }
.panel {
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
  border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px; box-shadow: 0 8px 24px rgba(0,0,0,.35);
}
.dropzone {
  border: 2px dashed var(--line); text-align: center; cursor: pointer;
  padding: 48px 16px; transition: border-color .15s, background .15s;
}
.dropzone:hover, .dropzone.over { border-color: var(--accent); background: rgba(210,168,255,.04); }
.drop-hint { font-size: 1.2rem; margin: 0; }
.drop-sub { color: var(--muted); margin: 6px 0 0; font-size: .9rem; }
.hidden { display: none !important; }
.job-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.job-status { font-family: var(--font-mono); color: var(--accent-2); }
.progress { height: 10px; border-radius: 5px; background: var(--line); overflow: hidden; }
.progress-bar { height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); transition: width .2s; }
.muted { color: var(--muted); font-size: .85rem; }
.error { color: var(--bad); font-family: var(--font-mono); font-size: .85rem; }
.stem-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.stem-card { display: grid; gap: 6px; padding: 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel-2); }
.stem-name { font-family: var(--font-mono); text-transform: capitalize; font-weight: 600; }
.stem-card audio { width: 100%; height: 34px; }
.inst { background: var(--bg); border: 1px solid var(--line); color: var(--text); border-radius: 6px; padding: 4px 8px; font-size: .8rem; }
.actions { margin-top: 16px; }
button.primary {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); color: #0d1117;
  border: 0; border-radius: var(--radius); padding: 10px 18px; font-weight: 700; cursor: pointer;
}
button.ghost { background: transparent; color: var(--muted); border: 1px solid var(--line); border-radius: var(--radius); padding: 6px 12px; cursor: pointer; }
button.primary:hover { filter: brightness(1.1); }
button.ghost:hover { color: var(--text); border-color: var(--accent); }
.settings-form { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.field { display: grid; gap: 4px; font-size: .85rem; color: var(--muted); }
.field.toggle { grid-column: 1 / -1; display: flex; align-items: center; gap: 6px; }
.field select, .field input { background: var(--bg); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
.results { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.result-link { color: var(--accent-2); font-family: var(--font-mono); }
.result-link:hover { text-decoration: underline; }
@media (max-width: 640px) { .settings-form { grid-template-columns: 1fr; } }
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — all suites (settings 5, separator 10, transcribe 4, pipeline 6, main 7).

- [ ] **Step 7: Manual smoke check**

Run: `uv run python -m app` → browser opens. Confirm the page renders, the settings panel populates from `/api/settings`, and the stem grid shows the six checkboxes. (A full separation requires the HF token + model downloads — Task 9 covers that; the page must at least render and load settings.)

- [ ] **Step 8: Commit**

```bash
git add app/static/index.html app/static/app.js app/static/style.css tests/test_main.py
git commit -m "feat: add vanilla single-page frontend"
```

---

## Task 9: README, HF token setup, first-run docs

**Files:**
- Modify: `README.md` (rewrite the stub from Task 1)
- No code.

**Interfaces:**
- Consumes: nothing.
- Produces: runnable instructions for the user; satisfies the spec's "Setup & first run" and license/token requirements.

- [ ] **Step 1: Write `README.md`**

```markdown
# AudioToMIDI

Separate an audio file into stems with **BS-RoFormer-SW**, then transcribe the stems
you choose to **MIDI** with **MuScriptor**. Local web app (FastAPI + vanilla JS).

```
audio ──▶ BS-RoFormer-SW ──▶ 6 stems ──▶ MuScriptor ──▶ per-stem .mid
```

## Requirements

- Windows 11 / Linux, Python 3.13, [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU recommended (the cu128 torch backend is pinned in `pyproject.toml`)

## Setup

```bash
uv sync                        # installs deps incl. CUDA torch (~2.5 GB first time)
```

MuScriptor weights are **gated**: you need a free Hugging Face account.

1. Open https://huggingface.co/muscriptor/muscriptor and accept the **CC BY-NC 4.0**
   (non-commercial) license.
2. Log in from the terminal:
   ```bash
   uv run hf auth login
   ```
   (or export `HF_TOKEN=hf_...` in your shell).

## Run

```bash
uv run python -m app
```

Your browser opens `http://127.0.0.1:8000`. First run downloads the separation model
(~336 MB) and the MuScriptor weights (large ~2.8 GB) into `~/.cache/`.

## Usage

1. Drop an audio file (wav / mp3 / flac / ogg / m4a / aiff).
2. Separation runs automatically — wait for the 6 stems to appear.
3. Tick the stems you want, adjust per-stem instrument (empty = auto), hit **Transcribe selected to MIDI**.
4. Download the `.mid` files. Each stem is named `<song>_<stem>.mid`.

Every setting (model size, device, precision, temperature, beam/batch size, output folder)
is adjustable in the Settings panel and persists across runs.

## Notes & limitations

- MuScriptor does not preserve velocity — output is timing, pitch, and instrument.
- Drums and "other" stems tend to transcribe poorly to readable MIDI; melodic stems
  (vocals, piano, guitar, bass) give the best results.
- MIDI + stem WAVs are written under the configured output folder
  (`./output/<job_id>/stems|midi/`).
- One job at a time by design.
```

- [ ] **Step 2: Verify README renders (no tests needed)**

Run: `git diff --stat README.md`
Expected: README rewritten (stat shows the change). Nothing to execute — this is documentation.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add setup, HF token, and usage guide"
```

---

## Self-Review (run before marking the plan complete)

Checklist, executed against the spec:

1. **Spec coverage**
   - Two-stage flow, pick stems → `POST /api/jobs` + `POST /api/jobs/{id}/transcribe` (Task 7). ✓
   - 6-stem separation, correct order → `STEMS` constant + `Separator.separate` (Tasks 3-4). ✓
   - SSE progress, cancel → `events` queue + `cancel` event (Task 6), SSE route (Task 7). ✓
   - All settings exposed + persisted → Task 2 settings, Task 7 GET/PUT routes, Task 8 settings form (incl. keep_stems + remember_selection toggles). ✓
   - Default stem selection (vocals/piano/guitar/bass checked) → static checkboxes in Task 8 `index.html`. ✓
   - Error handling: bad extension/empty upload (Task 7), per-stem failure continues + non-terminal `error` event (Task 6), GPU→CPU fallback (Task 4), HF-token setup hint in README (Task 9). ✓
   - Tests: separator roundtrip/chunk/single-chunk/reconstruct/order/precision-path (Tasks 3-4), pipeline flow/SSE/cancel/error/create_job (Task 6), settings (Task 2), transcribe mock (Task 5), backend routes (Task 7). ✓
   - Global constraint: torch cu128 verified at Task 1 Step 5. ✓
2. **Placeholder scan** — no TBD/TODO; every code step has full code; no references to undefined symbols (all interfaces defined in the Interface blocks above each task; `PIPELINE.jobs` registry is created in Task 6 and consumed by Task 7). ✓
3. **Type/name consistency** — `STEMS` in `app/separator.py` and `app/settings.py` (both defined; identical by construction). `Job`/`Pipeline` fields match between Task 6 definition and Task 7 usage. `transcribe_to_midi` kwargs verified against the real muscriptor 0.3.0 source (raw `batch_size` int, `prelude_forcing=(batch_size == 1)`). `load_settings`/`save_settings`/`SETTINGS_FILE` consistent across Tasks 2/7/8. Thread-safety: worker-thread emits use `call_soon_threadsafe` (Task 6), coroutine emits use `put_nowait`. SSE terminal events (`done`/`failed`/`cancelled`) match between Task 6 emits and Task 7 `TERMINAL_EVENTS`. Status constants imported in Task 7. ✓

If anything above fails, fix it before handing off. The plan is only complete when the self-review passes.
