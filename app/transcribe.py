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
