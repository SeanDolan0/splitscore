"""BS-RoFormer-SW separation: torch DSP around the ONNX model."""

from __future__ import annotations

import torch

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
CHUNK_FRAMES = 345        # model traced at 4 s @ 44.1 kHz (176400 samples)
CHUNK_HOP_FRAMES = 220    # 125-frame overlap (~1.45 s) for crossfading
OVERLAP_FRAMES = CHUNK_FRAMES - CHUNK_HOP_FRAMES  # 125

# Order must match the ONNX output channels.
STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"]


def _window():
    return torch.hann_window(WIN_LENGTH)


def stft(audio: torch.Tensor) -> torch.Tensor:
    """[2, N] float32 @44.1kHz -> [2, 1025, F] complex spectrogram."""
    return torch.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=_window(), center=True, return_complex=True,
    )


def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """[2, 1025, F] -> [2, N] audio. Pass length to reconstruct the exact N."""
    return torch.istft(
        spec, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=_window(), center=True, length=length,
    )


def split_into_chunks(spec: torch.Tensor) -> list[torch.Tensor]:
    """Split [2,1025,F] into [2,1025,345] chunks, stride CHUNK_HOP_FRAMES, pad last."""
    total = spec.shape[-1]
    if total <= CHUNK_FRAMES:
        return [torch.nn.functional.pad(spec, (0, CHUNK_FRAMES - total))]
    n_chunks = (total - CHUNK_FRAMES + CHUNK_HOP_FRAMES - 1) // CHUNK_HOP_FRAMES + 1
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_HOP_FRAMES
        piece = spec[..., start:start + CHUNK_FRAMES]
        if piece.shape[-1] < CHUNK_FRAMES:
            piece = torch.nn.functional.pad(piece, (0, CHUNK_FRAMES - piece.shape[-1]))
        chunks.append(piece)
    return chunks


def overlap_add(chunks: list[torch.Tensor], total_frames: int) -> torch.Tensor:
    """Reverse of split_into_chunks with a linear crossfade over the overlap.

    Chunk i starts at frame i*CHUNK_HOP_FRAMES. The overlap region gets a
    linear ramp 0->1 on the incoming chunk and 1->0 on existing content, which
    sums to exactly 1 for identical overlap content (exact reconstruction).
    """
    if not chunks:
        raise ValueError("no chunks to overlap-add")
    out = chunks[0][..., :total_frames].clone()
    for i in range(1, len(chunks)):
        chunk = chunks[i]
        start = i * CHUNK_HOP_FRAMES
        if start >= total_frames:
            break
        end = min(start + CHUNK_FRAMES, total_frames)
        over = min(OVERLAP_FRAMES, end - start)
        ramp = torch.linspace(0.0, 1.0, over).reshape(1, 1, over)  # float32
        out[..., start:start + over] = (
            out[..., start:start + over] * (1 - ramp) + chunk[..., :over] * ramp)
        if end > out.shape[-1]:
            tail = end - out.shape[-1]
            out = torch.cat([out, chunk[..., over:over + tail]], dim=-1)
    return out[..., :total_frames]


def mask_and_synthesize(input_spec: torch.Tensor, mask: torch.Tensor,
                        length: int | None = None) -> torch.Tensor:
    """Apply a per-stem mask [2,1025,F] to input_spec, ISTFT back to audio [2,N]."""
    return istft(input_spec * mask, length=length)
