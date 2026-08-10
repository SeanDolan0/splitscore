import asyncio
import io
import wave
from pathlib import Path
from unittest.mock import patch
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

def test_instruments_endpoint(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    with patch("app.main.list_instruments", return_value=["voice", "drums"]):
        r = client.get("/api/instruments")
        assert r.status_code == 200
        assert r.json() == {"instruments": ["voice", "drums"]}

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

def test_download_midi_rejects_path_traversal(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    # A real file outside the output base that an unchecked traversal would serve.
    outside = tmp_path.parent / "midi" / "secret.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("secret")
    r = client.get("/output/%2e%2e/midi/secret.txt")
    assert r.status_code == 404

def test_upload_stores_traversal_filename_inside_job_dir(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/jobs", files={"file": ("..\\evil.wav", b"x", "audio/wav")})
    assert r.status_code == 200
    assert (tmp_path / "abc" / "input" / "evil.wav").is_file()
    assert not (tmp_path / "abc" / "evil.wav").exists()

async def test_separation_hf_error_includes_setup_hint(tmp_path):
    from app.pipeline import Pipeline, Job
    from app.settings import Settings

    class GatedSeparator:
        def __init__(self, precision="fp16", device="auto", session_factory=None):
            pass
        def separate(self, in_path, out_dir, on_progress=None):
            raise Exception("gated repo elicwhite/bs-roformer-sw-6stem-onnx is gated")

    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    pipe = Pipeline(Settings(), separator_factory=GatedSeparator)
    await pipe.separate(job)
    assert job.status == "failed"
    assert "hf auth login" in job.error
    last = None
    while not job.events.empty():
        last = job.events.get_nowait()
    assert last["type"] == "failed"
    assert "hf auth login" in last["message"]


def test_frontend_marks_stem_checkboxes(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.get("/")
    html = r.text
    for stem in ["vocals", "piano", "guitar", "bass", "drums", "other"]:
        assert f'data-stem="{stem}"' in html
    assert "audio-midi-app" in html

def test_put_settings_applies_to_running_pipeline(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    r = client.put("/api/settings", json={"model_size": "small"})
    assert r.status_code == 200
    assert main.PIPELINE.settings.model_size == "small"
