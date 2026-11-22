from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import ConfigError

MODEL_ID = "google/embeddinggemma-2"
#: Pinned so that a Hub update cannot silently change your vectors.
PINNED_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"

MODALITIES = ("text", "image", "video", "audio")
SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)
DIMS = (768, 512, 256, 128)


@dataclass(frozen=True, kw_only=True)
class EngineConfig:
    """Everything that affects how the model is loaded and how items are turned into tokens.

    Media settings are config-level (not per call) on purpose: embeddings made with different
    settings are different vectors, so one engine produces one consistent kind.
    """

    model: str = MODEL_ID
    """Hub id (loaded from the HF cache) or a local directory in Hugging Face format."""
    revision: str | None = PINNED_REVISION
    """Hub commit. Ignored when `model` is a local directory."""
    local_files_only: bool = True
    """Never touch the network. Download once with `hf download` (see README)."""

    modalities: tuple[str, ...] = MODALITIES
    """Which towers to load. "text" is required. image/video share the vision tower."""
    device: str = "cuda"
    dtype: Literal["auto", "bfloat16", "float32"] = "auto"
    """auto = bfloat16 on GPUs that support it, else float32. float16 is refused (NaN)."""
    batch_size: int = 8

    max_tokens: int = 8192
    """Per-item context limit from the model card. Not enforced by the libraries."""
    image_tokens: int = 280
    """Soft-token budget per image (70, 140, 280, 560 or 1120)."""
    video_frame_tokens: int = 140
    """Soft-token budget per video frame (70, 140, 280, 560 or 1120)."""
    video_fps: float = 1.0
    video_max_frames: int = 32

    def __post_init__(self) -> None:
        mods = tuple(self.modalities)
        object.__setattr__(self, "modalities", mods)
        unknown = set(mods) - set(MODALITIES)
        if unknown:
            raise ConfigError(f"Unknown modalities {sorted(unknown)}. Valid: {MODALITIES}.")
        if "text" not in mods:
            raise ConfigError('modalities must include "text" (the text model is always needed).')
        if self.dtype not in ("auto", "bfloat16", "float32"):
            hint = " float16 makes the model return NaN on long inputs." if "16" in self.dtype else ""
            raise ConfigError(f"dtype must be 'auto', 'bfloat16' or 'float32', got {self.dtype!r}.{hint}")
        for name in ("image_tokens", "video_frame_tokens"):
            if getattr(self, name) not in SOFT_TOKEN_BUDGETS:
                raise ConfigError(f"{name} must be one of {SOFT_TOKEN_BUDGETS}, got {getattr(self, name)}.")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}.")
        if not self.video_fps > 0:
            raise ConfigError(f"video_fps must be > 0, got {self.video_fps}.")
        if self.video_max_frames < 1:
            raise ConfigError(f"video_max_frames must be >= 1, got {self.video_max_frames}.")

    @property
    def loads_vision(self) -> bool:
        return "image" in self.modalities or "video" in self.modalities

    @property
    def loads_audio(self) -> bool:
        return "audio" in self.modalities
