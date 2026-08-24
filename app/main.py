"""FastAPI backend: routes, SSE event stream, static frontend."""

from __future__ import annotations

import asyncio
import io
import json
import webbrowser
import zipfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline import STATUS_CREATED, STATUS_DONE, STATUS_FAILED, STATUS_READY, Pipeline
from app.separator import STEMS
from app.settings import Settings, load_settings, save_settings
from app.transcribe import list_instruments
from app.gpu import detect_gpu

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aiff"}

app = FastAPI(title="SplitScore")
PIPELINE = Pipeline(load_settings())

_HEARTBEAT = ":" + " " * 15 + "\n\n"  # SSE comment keeps the connection alive

TERMINAL_EVENTS = {"done", "failed", "cancelled"}


class TranscribeBody(BaseModel):
    stems: list[str]


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    name = Path(file.filename or "").name  # basename only, strips any ../ or drive segments
    if not name:
        raise HTTPException(400, "Invalid filename")
    if "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid filename")
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
    # Idle job (post-separation, waiting to transcribe): no worker coroutine
    # observes the flag, so finish the cancel here and discard the stems.
    if job.status in (STATUS_CREATED, STATUS_READY):
        PIPELINE._finish_cancelled(job)
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


@app.get("/api/instruments")
async def get_instruments():
    # First call shells out to `muscriptor list-instruments` (~seconds); run it
    # off the event loop. Result is cached in transcribe.list_instruments.
    instruments = await asyncio.to_thread(list_instruments)
    return {"instruments": instruments}


@app.get("/api/hardware")
async def get_hardware():
    """Report what compute hardware is available and which provider the models use."""
    import importlib
    info = {"torch": None, "onnxruntime": None, "gpu": None,
            "settings_device": PIPELINE.settings.separation_device,
            "separator_actual": None}

    # torch
    try:
        torch = importlib.import_module("torch")
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda or None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        }
    except Exception:
        info["torch"] = {"version": "not installed", "cuda_available": False}

    # onnxruntime
    try:
        ort = importlib.import_module("onnxruntime")
        info["onnxruntime"] = {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        }
    except Exception:
        info["onnxruntime"] = {"version": "not installed", "providers": []}

    # gpu detection (runtime)
    gpu = detect_gpu()
    info["gpu"] = {"vendor": gpu.vendor, "name": gpu.name, "preferred_device": gpu.preferred_device}

    # actual device used by the separator (if any job has run)
    for job in PIPELINE.jobs.values():
        if hasattr(job, "_separator_device"):
            info["separator_actual"] = job._separator_device
            break

    return info


@app.get("/api/settings")
async def get_settings():
    return load_settings()


@app.put("/api/settings")
async def put_settings(body: dict):
    merged = asdict(load_settings())
    merged.update({k: v for k, v in body.items() if k in merged})
    updated = Settings(**merged)
    save_settings(updated)
    PIPELINE.settings = updated
    return merged


@app.get("/output/{job_id}/midi/{filename}")
async def download_midi(job_id: str, filename: str):
    base = Path(PIPELINE.settings.output_folder).resolve()
    path = (base / job_id / "midi" / filename).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, "MIDI not found")
    return FileResponse(path, media_type="audio/midi", filename=filename)


@app.get("/output/{job_id}/stems/{filename}")
async def download_stem(job_id: str, filename: str):
    base = Path(PIPELINE.settings.output_folder).resolve()
    path = (base / job_id / "stems" / filename).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, "Stem not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/output/{job_id}/stems")
async def download_stems_zip(job_id: str):
    """Return all stems as a zip archive for bulk download."""
    base = Path(PIPELINE.settings.output_folder).resolve()
    stems_dir = (base / job_id / "stems").resolve()
    if not stems_dir.is_relative_to(base) or not stems_dir.is_dir():
        raise HTTPException(404, "Stems not found")
    wav_files = sorted(stems_dir.glob("*.wav"))
    if not wav_files:
        raise HTTPException(404, "No stems found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav in wav_files:
            zf.write(wav, wav.name)
    buf.seek(0)
    zip_name = f"{stems_dir.parent.name}_stems.zip"
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


def _find_free_port(start: int) -> int:
    """Return the first available port starting from *start*."""
    import socket
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found near {start}")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    port = _find_free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port)
