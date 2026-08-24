"""MuScriptor wrapper: transcribe a stem audio file to MIDI bytes."""

from __future__ import annotations

import shutil
import subprocess

from muscriptor import TranscriptionModel
from muscriptor.events import ProgressEvent

from app.settings import STEMS

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


def _resolve_instrument(instruments: str | None, stem: str) -> str | None:
    """Map a user-typed instrument to a valid MuScriptor vocabulary name.

    The vocabulary is not the stem names: 'vocals' is 'voice', and there is
    no 'guitar'/'piano'/'bass'. Echoing a stem's own label means 'auto' —
    let the model detect the instrument. Anything else must be a real
    vocabulary name, or we fail loudly here instead of mid-generation.
    """
    if not instruments:
        return None
    if instruments == stem or instruments in STEMS:
        return None
    valid = list_instruments()
    if valid and instruments not in valid:
        raise ValueError(
            f"Unknown instrument {instruments!r}. Valid names: "
            f"{', '.join(sorted(valid))}")
    return instruments


class Transcriber:
    def __init__(self, model_size: str = "large", device: str = "auto"):
        self.model_size = model_size
        from app.gpu import resolve_torch_device
        load_device = resolve_torch_device(device)
        # fp16 weights halve VRAM: fp32 weights + beam-search KV cache OOM a
        # 12 GB laptop GPU at beam_size>1 (the app's default).
        # fp16 is only safe on CUDA; MPS/XPU/CPU use fp32.
        dtype = "float16" if load_device == "cuda" else None
        self.model = TranscriptionModel.load_model(model_size, device=load_device, dtype=dtype)

    def transcribe(self, stem_path, stem, instruments=None, temperature=0.0,
                   beam_size=4, batch_size=1, on_chunk=None) -> bytes:
        model = self.model
        instruments = _resolve_instrument(instruments, stem)
        beat_grid = model.detect_beat_grid_for(str(stem_path))
        events = model.transcribe(
            str(stem_path),
            instruments=[instruments] if instruments else None,
            use_sampling=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            beam_size=beam_size,
            batch_size=batch_size,
            prelude_forcing=(batch_size == 1),
        )

        def tee():
            # muscriptor yields a ProgressEvent(completed, total) per chunk;
            # forward those to on_chunk (called from the pipeline's worker
            # thread) and pass everything else through to the MIDI assembler.
            for ev in events:
                if isinstance(ev, ProgressEvent):
                    if on_chunk:
                        on_chunk(ev.completed, ev.total)
                else:
                    yield ev

        return model.events_to_midi_bytes(tee(), beat_grid=beat_grid)
