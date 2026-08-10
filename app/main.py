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
from app.transcribe import list_instruments

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aiff"}

app = FastAPI(title="AudioToMIDI")
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


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


def main() -> None:
    import uvicorn
    url = "http://127.0.0.1:8000"
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=8000)
