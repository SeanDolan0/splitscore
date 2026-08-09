import sys
from pathlib import Path

# The `pytest` console script does not place the project root on sys.path,
# so bootstrap it here to make `uv run pytest` resolve `app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import Settings, STEMS, DEFAULT_SETTINGS, load_settings, save_settings, resolve_device

def test_stems_order():
    assert STEMS == ["bass", "drums", "other", "vocals", "guitar", "piano"]

def test_defaults_match_spec():
    s = DEFAULT_SETTINGS
    assert s.separation_device == "auto"
    assert s.separation_precision == "fp16"
    assert s.model_size == "large"
    assert s.temperature == 0.0
    assert s.beam_size == 4
    assert s.batch_size == 4
    assert s.output_folder == "./output"
    assert s.keep_stems is True
    assert s.remember_selection is True

def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "settings.json")
    s = Settings(model_size="small", temperature=0.5)
    save_settings(s)
    loaded = load_settings()
    assert loaded.model_size == "small"
    assert loaded.temperature == 0.5
    assert loaded.output_folder == "./output"  # untouched fields keep defaults

def test_load_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.SETTINGS_FILE", tmp_path / "nope.json")
    assert load_settings() == DEFAULT_SETTINGS

def test_resolve_device_auto_and_explicit(monkeypatch):
    monkeypatch.setattr("app.settings.torch", _FakeTorch(cuda=True))
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr("app.settings.torch", _FakeTorch(cuda=False))
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"

class _FakeTorch:
    def __init__(self, cuda):
        # Mirror real torch: `torch.cuda` is a module-like attribute, so
        # `resolve_device`'s `torch.cuda.is_available()` call works.
        self.cuda = _FakeCuda(cuda)

class _FakeCuda:
    def __init__(self, available):
        self._available = available
    def is_available(self):
        return self._available
