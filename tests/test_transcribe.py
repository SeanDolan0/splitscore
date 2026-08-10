import pytest
from unittest.mock import MagicMock, patch
import app.transcribe as tr
from app.transcribe import Transcriber, list_instruments
from muscriptor.events import ProgressEvent

def test_list_instruments_caches_and_returns():
    tr._INSTRUMENTS = None
    with patch("app.transcribe._fetch_instruments", return_value=["acoustic_piano", "drums"]) as f:
        assert list_instruments() == ["acoustic_piano", "drums"]
        assert list_instruments() == ["acoustic_piano", "drums"]
        assert f.call_count == 1  # cached after first call

def test_list_instruments_handles_failure():
    tr._INSTRUMENTS = None
    with patch("app.transcribe._fetch_instruments", side_effect=RuntimeError("no cli")):
        assert list_instruments() == []

def _fake_model(midi_bytes=b"\x00MIDI"):
    model = MagicMock()
    def _assemble(events, beat_grid=None):
        for _ in events:  # consume the tee generator, like real events_to_midi_bytes
            pass
        return midi_bytes
    model.events_to_midi_bytes.side_effect = _assemble
    model.transcribe.return_value = iter([])
    return model

def test_transcribe_passes_instruments_and_temperature():
    fake_model = _fake_model()
    with patch("app.transcribe.TranscriptionModel") as TM, \
         patch("app.transcribe.list_instruments", return_value=["voice", "drums"]):
        TM.load_model.return_value = fake_model
        t = Transcriber(model_size="large", device="cpu")
        out = t.transcribe("s.wav", "vocals", instruments="voice", temperature=0.8, beam_size=4, batch_size=4)
        assert out == b"\x00MIDI"
        kwargs = fake_model.transcribe.call_args[1]
        assert kwargs["instruments"] == ["voice"]
        assert kwargs["use_sampling"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["beam_size"] == 4
        assert kwargs["batch_size"] == 4
        assert kwargs["prelude_forcing"] is False  # batch_size>1 forces this

def test_stem_label_instrument_means_auto():
    # The MuScriptor vocabulary has no "vocals"/"guitar"/"piano"/"bass" —
    # echoing a stem's own label means "auto", letting the model detect it.
    fake_model = _fake_model()
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber(device="cpu")
        t.transcribe("s.wav", "vocals", instruments="vocals", batch_size=1)
        kwargs = fake_model.transcribe.call_args[1]
        assert kwargs["instruments"] is None

def test_unknown_instrument_raises_clear_error():
    with patch("app.transcribe.TranscriptionModel") as TM, \
         patch("app.transcribe.list_instruments", return_value=["voice", "drums"]):
        TM.load_model.return_value = _fake_model()
        t = Transcriber(device="cpu")
        with pytest.raises(ValueError, match="Unknown instrument 'xylophone'.*Valid names"):
            t.transcribe("s.wav", "vocals", instruments="xylophone", batch_size=1)

def test_transcribe_none_instruments_and_deterministic():
    fake_model = _fake_model()
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber()
        t.transcribe("s.wav", "piano", instruments=None, temperature=0.0, batch_size=1)
        kwargs = fake_model.transcribe.call_args[1]
        assert kwargs["instruments"] is None
        assert kwargs["use_sampling"] is False
        assert kwargs["temperature"] == 1.0
        assert kwargs["batch_size"] == 1
        assert kwargs["prelude_forcing"] is True

def test_transcribe_forwards_chunk_progress():
    fake_model = _fake_model()
    fake_model.transcribe.return_value = iter([
        ProgressEvent(completed=0, total=3),
        ProgressEvent(completed=3, total=3),
    ])
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber(device="cpu")
        seen = []
        t.transcribe("s.wav", "piano", batch_size=1, on_chunk=lambda c, n: seen.append((c, n)))
        assert seen == [(0, 3), (3, 3)]

def test_load_model_uses_fp16_on_cuda():
    with patch("app.transcribe.TranscriptionModel") as TM, patch("torch.cuda.is_available", return_value=True):
        Transcriber(device="cuda")
        assert TM.load_model.call_args.kwargs["device"] == "cuda"
        assert TM.load_model.call_args.kwargs["dtype"] == "float16"
