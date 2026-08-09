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


from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


MODEL_ID = "elicwhite/bs-roformer-sw-6stem-onnx"
MODEL_FILES = {"fp16": "bs_roformer_sw_6stem_fp16.onnx", "fp32": "bs_roformer_sw_6stem_fp32.onnx"}
MODEL_CACHE = Path.home() / ".cache" / "audio-to-midi" / "separator"


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode any soundfile-readable file to stereo float32 @ 44.1 kHz."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")  # [N, ch]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    audio = torch.from_numpy(data.T).float()  # [ch, N]
    if sr != SR:
        target = int(audio.shape[1] * SR / sr)
        audio = torch.stack([
            torch.nn.functional.interpolate(
                audio[c:c+1, None, :], size=target, mode="linear", align_corners=False)[0, 0]
            for c in range(2)])
    return audio, SR


def _download_model(precision: str) -> Path:
    """Download the ONNX file to MODEL_CACHE if missing; return its path."""
    from huggingface_hub import hf_hub_download
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=MODEL_ID, filename=MODEL_FILES[precision], cache_dir=MODEL_CACHE))


def _default_session_factory(precision: str, device: str) -> ort.InferenceSession:
    model_path = _download_model(precision)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(model_path), providers=providers)


class Separator:
    def __init__(self, precision: str = "fp16", device: str = "auto", session_factory=None):
        self.precision = precision if precision in MODEL_FILES else "fp16"
        from app.settings import resolve_device
        self.device = resolve_device(device)
        factory = session_factory or _default_session_factory
        try:
            self.session = factory(self.precision, self.device)
        except Exception:
            # GPU runtime failed (missing provider, download hiccup) -> CPU.
            self.session = _default_session_factory(self.precision, "cpu")
            self.device = "cpu"

    def _run_chunk(self, spec_chunk: torch.Tensor):
        """[2,1025,345] complex -> ([6,2,1025,345] real, [6,2,1025,345] imag) as tensors."""
        r = spec_chunk[None, ...].real.float().numpy()
        i = spec_chunk[None, ...].imag.float().numpy()
        out_r, out_i = self.session.run(None, {"spec_real": r, "spec_imag": i})
        return torch.from_numpy(out_r), torch.from_numpy(out_i)

    def separate(self, in_path, out_dir, on_progress=None) -> list[Path]:
        audio, _ = load_audio(in_path)
        spec = stft(audio.to("cpu"))
        chunks = split_into_chunks(spec)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        acc = [None] * len(STEMS)  # per-stem masked chunk lists
        for idx, chunk in enumerate(chunks):
            masks_r, masks_i = self._run_chunk(chunk)
            for s in range(len(STEMS)):
                mask = masks_r[0, s] + 1j * masks_i[0, s]  # [2,1025,345]
                masked = chunk * mask
                acc[s] = acc[s] or []
                acc[s].append(masked)
            if on_progress:
                on_progress(100.0 * (idx + 1) / len(chunks))
        results = []
        for s, name in enumerate(STEMS):
            recon = overlap_add(acc[s], spec.shape[-1])
            audio_s = istft(recon, length=audio.shape[1]).clamp(-1.0, 1.0)
            path = out_dir / f"{name}.wav"
            sf.write(str(path), audio_s.T.numpy(), SR)
            results.append(path)
        return results
