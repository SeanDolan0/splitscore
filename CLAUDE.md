# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uvx splitscore              # install + run: auto-detects GPU → correct CUDA torch (~2.5 GB first time)
uv run python sync.py       # dev install: auto-detects GPU, installs torch into local venv
uv run python -m app        # start server → opens http://127.0.0.1:8000
uv run pytest               # run all tests
uv run pytest tests/test_pipeline.py::test_separate_flow_and_events   # single test
uv run hf auth login        # required once: MuScriptor weights are HF-gated
```

No linter or formatter is configured; pytest with `asyncio_mode = "auto"` is the only check. Tests use fake Separator/Transcriber/session factories, never the real models — the suite runs in ~2s with no GPU.

## Architecture

A local web app: separate an audio file into 6 stems (BS-RoFormer-SW via ONNX), then transcribe selected stems to per-stem `.mid` files (MuScriptor). Vanilla JS frontend, no build step. One job at a time by design.

**Request flow:** `app/main.py` (FastAPI routes, SSE stream, static mount) → `app/pipeline.py` (job orchestration) → `app/separator.py` + `app/transcribe.py` (the two heavy model workers). `app/settings.py` is a dataclass persisted to `app/settings.json`; PUT `/api/settings` replaces `PIPELINE.settings` live.

**Concurrency model (the non-obvious part):** both model calls run in `asyncio.to_thread` (onnxruntime and muscriptor are blocking). Progress callbacks fire on the worker thread and are pushed to the job's `asyncio.Queue` via `loop.call_soon_threadsafe`; the SSE handler (`/api/jobs/{id}/events`) drains the queue with a 15s heartbeat comment keepalive. The `Pipeline._lock` serializes separation/transcription so only one job runs at once.

**Dependency injection for tests:** `Pipeline` takes `separator_factory`/`transcriber_factory`; `Separator` takes `session_factory`. Tests substitute fakes (identity masks, canned MIDI bytes) and assert on `job.events`/`job.status`/written files rather than real inference.

**Output layout:** every job gets `./output/<job_id>/` with `input/`, `stems/*.wav`, `midi/<song>_<stem>.mid`. Paths are validated with `Path.is_relative_to` against the output base on downloads (path-traversal guard).

## Key constraints

- **`STEMS` order is sacred.** `["bass", "drums", "other", "vocals", "guitar", "piano"]` must match the ONNX output channel order and is duplicated in `app/separator.py` and `app/settings.py`. Reorder or rename and every stem maps to the wrong channel.
- **`onnxruntime-gpu` must stay <1.27** (1.27+ is built against CUDA 13). `sync.py` auto-detects the CUDA version and pins torch to the matching wheel (cu118–cu130); without a GPU it installs CPU-only torch from PyPI. On macOS use `--cpu`.
- **MuScriptor weights are HF-gated** (CC BY-NC 4.0). A gated/401 error during separation or transcription gets `_hf_setup_hint` appended in `pipeline.py`, telling the user to run `hf auth login`.
- **CUDA OOM during transcription is job-fatal**, not per-stem — `pipeline.py` detects "out of memory"/"cuda error" and fails the whole job with a hint to lower beam/batch size, because continuing would OOM the next stem too. Any *other* per-stem transcription error is non-terminal (emits an `error` event, continues).
- **fp16 weights are used on CUDA** in `transcribe.py` deliberately: fp32 weights + beam-search KV cache OOMs a 12 GB laptop GPU at `beam_size > 1` (the default).
- **Separation model downloads on first run** to `~/.cache/audio-to-midi/separator/`. `_download_model` checks `local_files_only` first because a plain Hub call does a ~25s×2-retry HEAD check before any session is created.
- The approved design spec lives at `docs/superpowers/specs/2026-08-08-audio-to-midi-design.md`.

## Frontend

`app/static/` is hand-written vanilla JS/CSS (no framework, no build). `app.js` talks to the REST/SSE endpoints above; stems render as checkbox cards with per-stem instrument inputs and inline `<audio>` previews. There is no test coverage for it beyond `test_main.py` asserting the HTML contains the stem checkbox markup.
