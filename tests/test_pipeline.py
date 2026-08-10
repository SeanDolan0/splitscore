import asyncio
from pathlib import Path
from app.pipeline import Pipeline, Job
from app.settings import Settings

class FakeSeparator:
    def __init__(self, precision="fp16", device="auto", session_factory=None):
        pass
    def separate(self, in_path, out_dir, on_progress=None):
        for pct in (10, 50, 100):
            on_progress(pct)

class FakeTranscriber:
    def __init__(self, model_size="large", device="auto"):
        pass
    def transcribe(self, stem_path, stem, instruments=None, temperature=0.0,
                   beam_size=4, batch_size=4, on_chunk=None) -> bytes:
        return f"midi:{stem}".encode()


def _drain(q: asyncio.Queue) -> list[dict]:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


async def test_separate_flow_and_events(tmp_path):
    job = Job(id="j1", song_name="song", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)
    pipe = Pipeline(Settings(), separator_factory=FakeSeparator)
    await pipe.separate(job)
    await asyncio.sleep(0)
    assert job.status == "ready"
    events = _drain(job.events)
    assert [e["pct"] for e in events if e["type"] == "progress"] == [10, 50, 100]
    assert events[-1] == {"type": "stems",
                          "stems": ["bass", "drums", "other", "vocals", "guitar", "piano"]}


async def test_transcribe_writes_midi_per_stem(tmp_path):
    job = Job(id="j1", song_name="my song", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out" / "midi").mkdir(parents=True)
    pipe = Pipeline(Settings(), transcriber_factory=FakeTranscriber)
    await pipe.transcribe(job, ["vocals", "piano"], {"vocals": "", "piano": ""},
                          temperature=0.0, beam_size=4, batch_size=4)
    assert job.status == "done"
    files = sorted(p.name for p in (tmp_path / "out" / "midi").glob("*.mid"))
    assert files == ["my song_piano.mid", "my song_vocals.mid"]
    midi_events = [e for e in _drain(job.events) if e["type"] == "midi"]
    assert {e["stem"] for e in midi_events} == {"vocals", "piano"}


async def test_per_stem_failure_does_not_stop_pipeline(tmp_path):
    class Flaky(FakeTranscriber):
        def transcribe(self, *a, **k):
            if a[1] == "drums":
                raise RuntimeError("boom")
            return b"ok"
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out" / "midi").mkdir(parents=True)
    pipe = Pipeline(Settings(), transcriber_factory=Flaky)
    await pipe.transcribe(job, ["vocals", "drums"], {}, 0.0, 4, 4)
    assert job.status == "done"  # drums failed but vocals succeeded
    assert (tmp_path / "out" / "midi" / "s_vocals.mid").exists()


async def test_cancelled_separation_sets_cancelled(tmp_path):
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    job.cancel.set()
    pipe = Pipeline(Settings(), separator_factory=FakeSeparator)
    await pipe.separate(job)
    assert job.status == "cancelled"
    assert _drain(job.events)[-1] == {"type": "cancelled"}


async def test_cancel_during_separation_is_observed(tmp_path):
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")

    class CancelMidway(FakeSeparator):
        def separate(self, in_path, out_dir, on_progress=None):
            on_progress(10)
            job.cancel.set()  # user cancels mid-run
            on_progress(50)
            on_progress(100)

    pipe = Pipeline(Settings(), separator_factory=CancelMidway)
    await pipe.separate(job)
    assert job.status == "cancelled"
    events = _drain(job.events)
    assert events[-1] == {"type": "cancelled"}
    assert not any(e["type"] == "stems" for e in events)


async def test_separation_failure_sets_failed_with_error(tmp_path):
    class Boom:
        def __init__(self, precision="fp16", device="auto", session_factory=None):
            pass
        def separate(self, in_path, out_dir, on_progress=None):
            raise RuntimeError("no model weights")
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    pipe = Pipeline(Settings(), separator_factory=Boom)
    await pipe.separate(job)
    assert job.status == "failed"
    assert "model weights" in job.error
    assert _drain(job.events)[-1]["type"] == "failed"


async def test_cuda_oom_fails_job_not_stems(tmp_path):
    class Boom(FakeTranscriber):
        def transcribe(self, *a, **k):
            raise RuntimeError("CUDA error: out of memory (torch.AcceleratorError)")
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out" / "midi").mkdir(parents=True)
    pipe = Pipeline(Settings(), transcriber_factory=Boom)
    await pipe.transcribe(job, ["vocals", "drums"], {}, 0.0, 4, 4)
    assert job.status == "failed"
    assert "out of memory" in job.error.lower()
    assert _drain(job.events)[-1]["type"] == "failed"


async def test_transcribe_forwards_chunk_progress(tmp_path):
    class Progressing(FakeTranscriber):
        def transcribe(self, stem_path, stem, instruments=None, temperature=0.0,
                       beam_size=4, batch_size=4, on_chunk=None) -> bytes:
            for c in (0, 1, 2):
                on_chunk(c, 2)
            return b"ok"
    job = Job(id="j1", song_name="s", input_path=tmp_path / "in.wav",
              output_dir=tmp_path / "out")
    (tmp_path / "out" / "midi").mkdir(parents=True)
    pipe = Pipeline(Settings(), transcriber_factory=Progressing)
    await pipe.transcribe(job, ["vocals"], {}, 0.0, 4, 4)
    assert job.status == "done"
    progress = [e["pct"] for e in _drain(job.events)
                if e["type"] == "progress" and e["phase"] == "transcribing"]
    assert progress == [0, 0, 50, 100]


async def test_create_job_registers_and_makes_dirs(tmp_path):
    pipe = Pipeline(Settings())
    job = pipe.create_job("song", tmp_path / "in.wav")
    assert job.id in pipe.jobs
    assert job.output_dir is not None
    assert (job.output_dir / "stems").is_dir()
    assert (job.output_dir / "midi").is_dir()
