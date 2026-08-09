import asyncio
import io
import wave
from pathlib import Path
from fastapi.testclient import TestClient
import app.main as main


def _make_upload_bytes() -> bytes:
    """Minimal valid WAV (1s of silence, 16 kHz mono) built entirely in memory."""
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"\x00\x00" * 16000)
    w.close()
    return buf.getvalue()


class FakePipeline:
    def __init__(self, settings, separator_factory=None, transcriber_factory=None):
        self.settings = settings
        self.jobs = {}
    def create_job(self, song_name, input_path):
        from app.pipeline import STATUS_READY, Job
        job = Job(id="abc", song_name=song_name, input_path=Path(input_path))
        job.output_dir = Path(self.settings.output_folder) / "abc"
        (job.output_dir / "stems").mkdir(parents=True, exist_ok=True)
        (job.output_dir / "midi").mkdir(parents=True, exist_ok=True)
        job.status = STATUS_READY
        self.jobs[job.id] = job
        return job
    async def separate(self, job):
        await asyncio.sleep(0)


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(main, "PIPELINE", FakePipeline(main.PIPELINE.settings))
    monkeypatch.setattr(main.PIPELINE.settings, "output_folder", str(tmp_path))
    return TestClient(main.app)


def test_upload_creates_job(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.wav", _make_upload_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["job_id"] == "abc"

def test_upload_rejects_bad_extension(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400
    assert "extension" in r.json()["detail"].lower()

def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.wav", b"", "audio/wav")})
    assert r.status_code == 400

def test_settings_roundtrip(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.put("/api/settings", json={"model_size": "small", "temperature": 0.5})
    assert r.json()["model_size"] == "small"
    r2 = client.get("/api/settings")
    assert r2.json()["model_size"] == "small"

def test_transcribe_requires_valid_stems(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("song.wav", _make_upload_bytes(), "audio/wav")})
    job_id = r.json()["job_id"]
    r = client.post(f"/api/jobs/{job_id}/transcribe", json={"stems": ["not-a-stem"]})
    assert r.status_code == 400

def test_index_html_served(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "audio-midi-app" in r.text
