from unittest.mock import MagicMock, patch
import app.transcribe as tr
from app.transcribe import Transcriber, list_instruments

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

def test_transcribe_passes_instruments_and_temperature():
    fake_model = MagicMock()
    fake_model.transcribe_to_midi.return_value = b"\x00MIDI"
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber(model_size="large", device="cpu")
        out = t.transcribe("s.wav", "vocals", instruments="vocals", temperature=0.8, beam_size=4, batch_size=4)
        assert out == b"\x00MIDI"
        kwargs = fake_model.transcribe_to_midi.call_args[1]
        assert kwargs["instruments"] == ["vocals"]
        assert kwargs["use_sampling"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["beam_size"] == 4
        assert kwargs["batch_size"] == 4
        assert kwargs["prelude_forcing"] is False  # batch_size>1 forces this

def test_transcribe_none_instruments_and_deterministic():
    fake_model = MagicMock()
    fake_model.transcribe_to_midi.return_value = b"midi"
    with patch("app.transcribe.TranscriptionModel") as TM:
        TM.load_model.return_value = fake_model
        t = Transcriber()
        t.transcribe("s.wav", "piano", instruments=None, temperature=0.0, batch_size=1)
        kwargs = fake_model.transcribe_to_midi.call_args[1]
        assert kwargs["instruments"] is None
        assert kwargs["use_sampling"] is False
        assert kwargs["temperature"] == 1.0
        assert kwargs["batch_size"] == 1
        assert kwargs["prelude_forcing"] is True
