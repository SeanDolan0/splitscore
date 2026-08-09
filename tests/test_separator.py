import torch
from app.separator import stft, istft, split_into_chunks, overlap_add, CHUNK_FRAMES, CHUNK_HOP_FRAMES

def _tone(seconds=0.5, freq=440.0, sr=44100):
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    return 0.5 * torch.sin(2 * torch.pi * freq * t)

def _stereo_tone(seconds=0.5):
    mono = _tone(seconds)
    return torch.stack([mono, mono])

def test_stft_istft_roundtrip():
    audio = _stereo_tone(0.4)
    spec = stft(audio)
    assert spec.shape[0] == 2 and spec.shape[1] == 1025
    recon = istft(spec, length=audio.shape[1])
    assert recon.shape == audio.shape
    err = (recon - audio).abs().mean().item()
    assert err < 1e-3, f"roundtrip error too high: {err}"

def test_split_into_chunks_counts():
    spec = stft(_stereo_tone(10.0))  # ~862 frames -> 4 chunks
    chunks = split_into_chunks(spec)
    expected = (spec.shape[-1] - CHUNK_FRAMES + CHUNK_HOP_FRAMES - 1) // CHUNK_HOP_FRAMES + 1
    assert len(chunks) == expected == 4
    for c in chunks:
        assert c.shape[-1] == CHUNK_FRAMES

def test_split_short_audio_single_padded_chunk():
    spec = stft(_stereo_tone(0.5))  # ~44 frames, well under 345
    chunks = split_into_chunks(spec)
    assert len(chunks) == 1
    assert chunks[0].shape[-1] == CHUNK_FRAMES

def test_overlap_add_reconstructs_input_within_tolerance():
    spec = stft(_stereo_tone(10.0))  # 4 chunks, exercises every crossfade branch
    chunks = split_into_chunks(spec)
    assert len(chunks) == 4
    back = overlap_add(chunks, spec.shape[-1])
    assert back.shape == spec.shape
    rel = (back - spec).abs().mean().item() / spec.abs().mean().item()
    assert rel < 0.04, f"reconstruction rel error too high: {rel}"

def test_overlap_add_total_frames_respected():
    spec = stft(_stereo_tone(10.0))
    chunks = split_into_chunks(spec)
    back = overlap_add(chunks, spec.shape[-1])
    assert back.shape[-1] == spec.shape[-1]

import numpy as np
import soundfile as sf
import torch
from app.separator import Separator, load_audio, STEMS

class FakeSession:
    """Identity separation: masks are all ones, so each stem = the original."""
    def __init__(self, precision, device):
        self.precision = precision
        self.device = device
        self.calls = 0
    def run(self, output_names, feeds):
        self.calls += 1
        r = feeds["spec_real"]  # [1,2,1025,345]
        i = feeds["spec_imag"]
        ones = np.ones_like(r)
        zeros = np.zeros_like(i)
        out_r = np.stack([ones] * 6, axis=1)   # [1,6,2,1025,345]
        out_i = np.stack([zeros] * 6, axis=1)
        return out_r, out_i


def _make_wav(path, seconds=1.0, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.4 * np.sin(2 * np.pi * 330.0 * t)
    stereo = np.stack([sig, sig], axis=1).astype(np.float32)
    sf.write(str(path), stereo, sr)
    return path


def test_load_audio_returns_stereo_44k(tmp_path):
    path = _make_wav(tmp_path / "a.wav", seconds=0.5)
    audio, sr = load_audio(path)
    assert sr == 44100
    assert audio.shape[0] == 2
    assert audio.shape[1] > 0
    assert audio.dtype == torch.float32


def test_separate_writes_six_stems_in_order(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    out = tmp_path / "out"
    sep = Separator(session_factory=FakeSession)
    results = sep.separate(src, out)
    assert [p.name for p in results] == [f"{s}.wav" for s in STEMS]
    for p in results:
        data, sr = sf.read(str(p))
        assert sr == 44100
        assert data.shape[0] > 0


def test_separate_reports_progress(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    sep = Separator(session_factory=FakeSession)
    seen = []
    sep.separate(src, tmp_path / "out", on_progress=lambda pct: seen.append(pct))
    assert seen
    assert seen[-1] == 100.0


def test_identity_separation_reconstructs_original(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    sep = Separator(session_factory=FakeSession)
    results = sep.separate(src, tmp_path / "out")
    orig, _ = sf.read(str(src))
    stem0, _ = sf.read(str(results[0]))
    # Ones mask -> stem 0 should equal the original (same length, close values).
    assert stem0.shape[0] == orig.shape[0]
    rel = np.abs(stem0 - orig).mean() / (np.abs(orig).mean() + 1e-9)
    assert rel < 0.05, f"identity separation drifted: {rel:.3f}"


def test_separate_uses_fake_session_precision_and_device(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=1.0)
    seen = {}

    def factory(precision, device):
        seen["precision"] = precision
        seen["device"] = device
        return FakeSession(precision, device)

    sep = Separator(precision="fp32", device="cpu", session_factory=factory)
    sep.separate(src, tmp_path / "out")
    assert seen == {"precision": "fp32", "device": "cpu"}


class ChannelDistinctFakeSession:
    """Distinct real mask per output channel: channel s scales input by (1 + 0.1*s)."""
    def __init__(self, precision, device):
        self.precision = precision
        self.device = device
    def run(self, output_names, feeds):
        r = feeds["spec_real"]  # [1,2,1025,345]
        i = feeds["spec_imag"]
        zeros = np.zeros_like(i)
        out_r = np.stack([np.ones_like(r) * (1.0 + 0.1 * s) for s in range(6)], axis=1)
        out_i = np.stack([zeros] * 6, axis=1)
        return out_r, out_i


def test_separate_maps_output_channels_to_stems(tmp_path):
    src = _make_wav(tmp_path / "song.wav", seconds=3.0)
    sep = Separator(session_factory=ChannelDistinctFakeSession)
    results = sep.separate(src, tmp_path / "out")
    orig, _ = sf.read(str(src))
    for s, p in enumerate(results):
        stem, _ = sf.read(str(p))
        scale = 1.0 + 0.1 * s
        rel = np.abs(stem - scale * orig).mean() / (scale * np.abs(orig).mean() + 1e-9)
        assert rel < 0.02, f"stem {s} not from output channel {s}: rel {rel:.3f}"
