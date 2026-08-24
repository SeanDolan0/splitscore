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
    assert s.batch_size == 1
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
    from app.gpu import GpuInfo
    monkeypatch.setattr("app.gpu.detect_gpu", lambda: GpuInfo("nvidia", "Fake GPU", "cuda"))
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr("app.gpu.detect_gpu", lambda: GpuInfo("none", "No GPU", "cpu"))
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
