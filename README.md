# SplitScore

Separate an audio file into 6 stems with **BS-RoFormer-SW**, then transcribe the stems
you choose to **MIDI** with **MuScriptor**. A local web console (FastAPI + vanilla JS).

```
audio ──▶ BS-RoFormer-SW ──▶ 6 stems ──▶ MuScriptor ──▶ per-stem .mid
```

## Quick start

```bash
uvx splitscore
```

This auto-detects your NVIDIA GPU and installs the correct CUDA torch backend.
First run downloads models (~3 GB total) into `~/.cache/`.

## Requirements

- Windows 11 / Linux, Python 3.13, [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU recommended — requires [CUDA 12.x Toolkit](https://developer.nvidia.com/cuda-downloads) installed for GPU acceleration (auto-detected; falls back to CPU if missing)
- Apple Silicon (MPS) and AMD (ROCm) GPUs are also supported

## Setup (development)

```bash
uv run python sync.py          # auto-detect GPU, install correct CUDA torch
uv run python -m app           # start server
```

MuScriptor weights are **gated**: you need a free Hugging Face account.

1. Open https://huggingface.co/muscriptor/muscriptor and accept the **CC BY-NC 4.0**
   (non-commercial) license.
2. Log in from the terminal:
   ```bash
   uv run hf auth login
   ```
   (or export `HF_TOKEN=hf_...` in your shell).

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

## License

- **Code:** MIT — see [LICENSE](LICENSE).
- **Models:** the separation model and the MuScriptor weights are separate and
  distributed under their own licenses. In particular, MuScriptor's weights are
  **CC BY-NC 4.0** (non-commercial) and gated on Hugging Face; that restriction
  applies to the models regardless of this project's MIT license.
