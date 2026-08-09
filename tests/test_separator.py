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
