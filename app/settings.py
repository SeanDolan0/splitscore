"""Persistent app settings."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Order must match the BS-RoFormer-SW ONNX output channels (0-5).
STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"]

SETTINGS_FILE = Path(__file__).parent / "settings.json"


@dataclass
class Settings:
    separation_device: str = "auto"      # auto | cuda | mps | xpu | cpu
    separation_precision: str = "fp16"   # fp16 | fp32
    model_size: str = "large"            # small | medium | large
    instrument_by_stem: dict = field(default_factory=dict)  # stem -> instrument name or "" (auto)
    temperature: float = 0.0             # 0 = deterministic
    beam_size: int = 4                   # 1 = greedy
    batch_size: int = 1                  # >1 disables prelude forcing & multiplies KV-cache VRAM
    transcription_device: str = "auto"   # auto | cuda | mps | xpu | cpu
    output_folder: str = "./output"
    keep_stems: bool = True
    remember_selection: bool = True


DEFAULT_SETTINGS = Settings()


def load_settings() -> Settings:
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS
    data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    merged = asdict(DEFAULT_SETTINGS)
    merged.update({k: v for k, v in data.items() if k in merged})
    return Settings(**merged)


def save_settings(settings: Settings) -> None:
    SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def resolve_device(requested: str) -> str:
    """Resolve 'auto' to the best available device string."""
    if requested != "auto":
        return requested
    from app.gpu import detect_gpu
    return detect_gpu().preferred_device
