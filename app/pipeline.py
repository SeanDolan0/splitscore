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


def _hf_setup_hint(exc: Exception) -> str:
    """Append a setup hint when the error looks like a missing/rejected HF token."""
    text = str(exc).lower()
    if any(s in text for s in ("401", "gated", "authentication", "forbidden", "hf_token", "huggingface")):
        return f"{exc}\n\nSetup hint: run `uv run hf auth login` with a Hugging Face token to download the models."
    return str(exc)


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
                if job.cancel.is_set():
                    job.status = STATUS_CANCELLED
                    self._emit(job, {"type": "cancelled"})
                    return
                job.status = STATUS_READY
                emit({"type": "stems", "stems": STEMS})
            except asyncio.CancelledError:
                job.status = STATUS_CANCELLED
                self._emit(job, {"type": "cancelled"})
            except Exception as exc:
                msg = _hf_setup_hint(exc)
                job.status = STATUS_FAILED
                job.error = msg
                self._emit(job, {"type": "failed", "message": msg})

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
                msg = _hf_setup_hint(exc)
                job.status = STATUS_FAILED
                job.error = msg
                self._emit(job, {"type": "failed", "message": msg})
                return
        if job.cancel.is_set():
            job.status = STATUS_CANCELLED
            self._emit(job, {"type": "cancelled"})
        else:
            job.status = STATUS_DONE
            self._emit(job, {"type": "done"})
