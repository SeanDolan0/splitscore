"""Job orchestration: separate then transcribe, with SSE event stream + cancel."""

from __future__ import annotations

import asyncio
import shutil
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


def _is_cuda_oom(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda error" in text


_OOM_MESSAGE = (
    "CUDA ran out of memory while transcribing. Lower Beam size or Batch size in "
    "Settings (or pick a smaller model), then try again.")


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

    def _finish_cancelled(self, job: Job) -> None:
        """Idempotent cancel: mark cancelled, discard this job's output, notify SSE."""
        if job.status == STATUS_CANCELLED:
            return
        job.status = STATUS_CANCELLED
        job.error = "Cancelled"
        if job.output_dir:
            shutil.rmtree(job.output_dir, ignore_errors=True)
        self._emit(job, {"type": "cancelled", "message": "Cancelled"})

    async def separate(self, job: Job) -> None:
        if job.cancel.is_set():
            self._finish_cancelled(job)
            return
        job.status = STATUS_SEPARATING
        async with self._lock:
            try:
                if job.cancel.is_set():
                    self._finish_cancelled(job)
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
                    self._finish_cancelled(job)
                    return
                job.status = STATUS_READY
                emit({"type": "stems", "stems": STEMS})
            except asyncio.CancelledError:
                self._finish_cancelled(job)
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
                loop = asyncio.get_running_loop()
                for stem in stems:
                    if job.cancel.is_set():
                        self._finish_cancelled(job)
                        return
                    self._emit(job, {"type": "progress", "phase": "transcribing",
                                     "stem": stem, "pct": 0})
                    try:
                        def on_chunk(completed, total, stem=stem):
                            # Called from the worker thread -> thread-safe push.
                            loop.call_soon_threadsafe(
                                job.events.put_nowait,
                                {"type": "progress", "phase": "transcribing",
                                 "stem": stem, "pct": round(100.0 * completed / total)})
                        midi_bytes = await asyncio.to_thread(
                            tr.transcribe, job.output_dir / "stems" / f"{stem}.wav", stem,
                            instrument_by_stem.get(stem) or None, temperature, beam_size,
                            batch_size, on_chunk)
                        out = job.output_dir / "midi" / f"{job.song_name}_{stem}.mid"
                        out.write_bytes(midi_bytes)
                        self._emit(job, {"type": "midi", "stem": stem, "file": out.name})
                    except Exception as exc:
                        if _is_cuda_oom(exc):
                            # VRAM exhausted at the KV-cache stage: continuing to
                            # the next stem would OOM again, so fail the job.
                            job.status = STATUS_FAILED
                            job.error = _OOM_MESSAGE
                            self._emit(job, {"type": "failed", "message": _OOM_MESSAGE})
                            return
                        # One stem failing must not stop the others (non-terminal).
                        self._emit(job, {"type": "error", "message": f"{stem}: {exc}"})
            except Exception as exc:
                msg = _hf_setup_hint(exc)
                job.status = STATUS_FAILED
                job.error = msg
                self._emit(job, {"type": "failed", "message": msg})
                return
        if job.cancel.is_set():
            self._finish_cancelled(job)
        else:
            job.status = STATUS_DONE
            self._emit(job, {"type": "done"})
