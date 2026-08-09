# AudioToMIDI — Design Spec

**Date:** 2026-08-08
**Status:** Approved
**Approach:** FastAPI + vanilla frontend, ONNX separation, MuScriptor transcription (Approach 1)

## Overview

A local web app that converts an audio file into per-stem MIDI files. The pipeline is:

```
audio file ──▶ BS-RoFormer-SW (6 stems) ──▶ pick stems ──▶ MuScriptor ──▶ .mid files
```

- **Separation:** BS-RoFormer-SW splits the mix into 6 stems — bass, drums, other, vocals, guitar, piano.
- **Transcription:** MuScriptor transcribes each *selected* stem to MIDI, one `.mid` file per stem.
- **GUI:** drag-and-drop upload, stem checkboxes, live progress, download links.

The whole thing runs locally on the user's machine (localhost). It is a personal tool; the model weights are gated behind a non-commercial license.

## Environment (verified)

- Windows 11, Python 3.13.14, `uv` 0.11.26
- NVIDIA RTX 5070 Ti Laptop GPU, 12 GB VRAM (CUDA available; torch must be installed with `cu128` backend)
- No `gh` CLI, no web search in agent env — research done via WebFetch

## Constraints & requirements

1. MuScriptor model weights are **gated** on Hugging Face under a **CC BY-NC 4.0 (non-commercial)** license. The user must create a free HF account, accept the license, and provide a `HF_TOKEN`. This is a hard requirement for transcription to work.
2. Windows PyTorch defaults to CPU; GPU use requires the `cu128` torch backend, configured via uv's `torch-backend` setting in `pyproject.toml`.
3. All adjustable settings must be exposed in the web GUI (see Settings).

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Browser tab (http://localhost:8000)                      │
│  index.html + app.js + style.css   (vanilla, no build)    │
│  drag-drop · stem checkboxes · progress · download links  │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTP + SSE (progress stream)
┌──────────────────────────▼─────────────────────────────────┐
│  FastAPI backend  (app/main.py)                            │
│  routes: upload / separate / transcribe / download / SSE   │
└───────────────┬──────────────────────────┬─────────────────┘
                │                          │
    ┌───────────▼───────────┐   ┌──────────▼───────────┐
    │ app/separator.py      │   │ app/transcribe.py    │
    │ BS-RoFormer-SW (ONNX) │   │ MuScriptor wrapper   │
    │ + torch STFT wrapper  │   │ → .mid bytes         │
    │ → 6 stem WAVs         │   │                      │
    └───────────┬───────────┘   └──────────┬───────────┘
                │                          │
        onnxruntime-gpu             muscriptor (torch cu128)
        model: 336 MB fp16          model: large 1.4B / medium / small
└────────────────────────────────────────────────────────────┘
```

### Components

| File | Responsibility | Interface |
|---|---|---|
| `app/main.py` | FastAPI app: routes, static serving, job lifecycle, SSE | HTTP + SSE |
| `app/separator.py` | BS-RoFormer-SW ONNX session + STFT/ISTFT wrapper | `separate(path, cfg) -> [6 stem WAV paths]`, yields progress |
| `app/transcribe.py` | MuScriptor wrapper | `transcribe(stem_path, stem, cfg, on_progress) -> .mid bytes` |
| `app/pipeline.py` | Job state machine; wires separate→transcribe; cancellation | `Job` dataclass + coroutine |
| `app/static/` | `index.html`, `app.js`, `style.css` | browser |
| `tests/` | `test_separator.py`, `test_pipeline.py` | pytest |
| `output/` | `output/<job_id>/stems/*.wav`, `output/<job_id>/midi/*.mid` | filesystem |

### Project layout

```
audioToMidi/
├── pyproject.toml            # deps; torch-backend = cu128
├── README.md                 # setup + HF token + license note
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── separator.py
│   ├── transcribe.py
│   ├── pipeline.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── tests/
│   ├── test_separator.py
│   └── test_pipeline.py
└── output/
```

## Data flow

**Stage 1 — Upload & separate:**

```
POST /api/jobs  (multipart audio)
  → decode/validate → create job id
  → separation runs (SSE progress: "Separating… 40%")
  → 6 stem WAVs written to output/<job_id>/stems/
  → response: job id + stem metadata (name, duration)
```

**Stage 2 — Pick stems & transcribe:**

```
POST /api/jobs/<id>/transcribe  {"stems": ["vocals", "piano"]}
  → muscriptor per selected stem (SSE: "Transcribing vocals…")
  → output/<job_id>/midi/<song>_<stem>.mid
```

**Downloads / preview:**

```
GET /output/<job_id>/midi/<file>   → download MIDI
GET /output/<job_id>/stems/<file>  → <audio> preview playback in browser
```

Progress is pushed over **SSE** (`text/event-stream`) using a generator in FastAPI — no extra dependency. A **cancel** button sets a flag the pipeline checks between steps.

**Concurrency:** one job at a time, guarded by a global lock. (Per-job locks only if throughput ever matters.)

## Separation engine (app/separator.py)

- **Model:** `elicwhite/bs-roformer-sw-6stem-onnx` — `bs_roformer_sw_6stem_fp16.onnx` (336 MB, default) or `bs_roformer_sw_6stem_fp32.onnx` (669 MB). Cached after first download.
- **I/O spec (from model card):**
  - Inputs: `spec_real`, `spec_imag` — `float32 [1, 2, 1025, T]`
  - Outputs: `out_spec_real`, `out_spec_imag` — `float32 [1, 6, 2, 1025, T]`
  - Stem order: `0: bass, 1: drums, 2: other, 3: vocals, 4: guitar, 5: piano`
- **Preprocess:** `torch.stft(audio, n_fft=2048, hop_length=512, win_length=2048, window=hann, center=True, normalized=False)`. Input decoded + resampled to **44.1 kHz stereo** (soundfile handles wav/mp3/flac/ogg/m4a).
- **Chunking:** model is traced at a fixed chunk of 176,400 samples (4 s @ 44.1 kHz, T=345) — cannot flex the time dimension. Process in 4 s chunks with **crossfade overlap-add** (port the exact logic from the reference browser port `elicwhite/bs-roformer-web`).
- **Backend:** `onnxruntime-gpu` (CUDA provider) with CPU fallback on runtime error.
- **Masking:** apply the 6 masks to the input STFT, ISTFT each back to a WAV.

*Not exposed in GUI:* chunk size, STFT params, overlap-add crossfade, sample rate — fixed internal constants.

## Transcription engine (app/transcribe.py)

- **Model:** `muscriptor` (Kyutai + Mirelo). `TranscriptionModel.load_model("large")` default; small/medium/large selectable.
- **API:** `model.transcribe_to_midi("stem.wav")` returns MIDI bytes (used over the event stream because it corrects a ~25 ms timing lag). Progress reported via the `transcribe` generator's `ProgressEvent`s.
- **Instrument constraint:** per-stem instrument passed via `instruments=[...]` (hard-constrained decoding). Default auto-matches the stem name; user can override per stem.
- **Output:** `<song>_<stem>.mid` written to `output/<job_id>/midi/`.
- MuScriptor does **not** preserve velocity — timing, pitch, instrument only. Accepted limitation.

## GUI & settings

Single-page frontend (`index.html` + `app.js` + `style.css`, vanilla JS, no build step). Sections: drop zone, job progress, stem selection, results list (MIDI + stem preview), settings panel.

All settings live in `app/settings.json`, loaded on page load, persisted on save, overridable per-run. The same knobs are exposed as API query params so the pipeline is fully driven by the GUI.

### Separation settings
| Setting | Options | Default |
|---|---|---|
| Device | auto / CUDA / CPU | auto |
| Model precision | fp16 / fp32 | fp16 |

### Transcription settings
| Setting | Options | Default |
|---|---|---|
| Model size | small / medium / large | large |
| Instrument per stem | vocals / piano / guitar / bass / drums / other / auto | auto (matches stem) |
| Sampling temperature | 0–1.5 slider (0 = deterministic) | 0 |
| Beam size | 1–8 (1 = greedy) | 4 |
| Batch size | 1–8 | 4 |
| Device | auto / CUDA / CPU | auto |

### General settings
| Setting | Options | Default |
|---|---|---|
| Output folder | text field | `./output` |
| Keep stem WAVs after conversion | on / off | on |
| Remember last stem selection | on / off | on |

### Default stem selection
vocals, piano, guitar, bass **checked**; drums, other **unchecked** (MuScriptor on drums yields noisy piano-style hits; user can tick them).

## Error handling

- Unsupported file type / zero-length upload → rejected with a clear message before processing.
- Missing HF token or model download failure → setup hint ("run `uv run hf auth login`"), not a stack trace.
- GPU/ONNX runtime error → auto-fallback to CPU, noted in the UI.
- Per-stem transcription failure → job marked failed with which stem and why; other stems still downloadable.
- Job temp dirs cleaned up on cancellation.
- Every failure surfaces in the UI — no silent failures.

## Testing

pytest; model-heavy parts are mocked — tests do **not** need GPU or the HF token.

- `test_separator.py`
  - STFT → mask → ISTFT round-trips back to original
  - Chunk boundaries are seamless (no clicks)
  - Exactly 6 stems in the right order (bass, drums, other, vocals, guitar, piano)
  - fp16 vs fp32 path
- `test_pipeline.py`
  - Full job flow with fake separator (silence stems) + fake transcriber (bytes): correct naming, SSE event order, cancellation mid-job, error propagation
- Remaining code is thin glue; the models themselves verified manually on one real file.

## Dependencies

- `fastapi`, `uvicorn` — HTTP + SSE
- `onnxruntime-gpu` — separation inference
- `torch` (cu128 via uv `torch-backend`) — STFT wrapper + muscriptor backend
- `soundfile` — audio decode (wav/mp3/flac/ogg/m4a)
- `muscriptor` — transcription (brings its own torch/transformers deps)
- dev: `pytest`, `pytest-asyncio`, `httpx` (FastAPI TestClient/AsyncClient)

## Setup & first run

1. `uv sync`
2. Accept the MuScriptor license on Hugging Face, then `uv run hf auth login` (or set `HF_TOKEN`).
3. `uv run python -m app` → browser opens `http://localhost:8000`.
4. First run downloads the separation ONNX (~336 MB) and MuScriptor weights (large ~2.8 GB) into caches.

## Out of scope

- Batching multiple files / folder processing (batch CLI variant can come later).
- Real-time playback of MIDI (the check-mix / auralize feature).
- Any web-service deployment — this is a local personal tool.
