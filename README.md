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
