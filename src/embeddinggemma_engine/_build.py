"""Item -> chat message for the processor.

Rules (see docs/conventions.md):
- The task prefix is a leading text part, added only when the item has text or a title.
  Media-only items get no prefix (model card; confirmed in docs/model-facts.md).
- Placeholders are produced by the chat template from typed parts, never typed by the user.
- Audio is passed as decoded mono 16 kHz arrays; images and videos as paths (or PIL images).
"""

from __future__ import annotations

from typing import Any

from .config import EngineConfig
from .errors import FrameLimitError, MediaError, ModalityNotLoadedError, TaskError
from .tasks import Side, Task, prefix
from .types import Audio, Image, Item, Text, Video


def validate(item: Item, index: int, task: Task, side: Side, config: EngineConfig) -> None:
    """Cheap checks that run for every item before anything is decoded or embedded."""
    from . import _media

    where = f"item {index}"
    if item.title is not None and side != "document":
        raise TaskError(f"{where}: title is only used by embed_documents (this is the {side} side).")
    missing = item.modalities - set(config.modalities)
    if missing:
        raise ModalityNotLoadedError(
            f"{where} uses {sorted(missing)}, but the engine was configured with "
            f"modalities={config.modalities}."
        )
    for part in item.parts:
        if isinstance(part, Image) and not hasattr(part.source, "mode"):
            if not part.source.is_file():
                raise MediaError(f"{where}: image file not found: {part.source}")
        elif isinstance(part, Audio) and not hasattr(part.source, "shape"):
            if not part.source.is_file():
                raise MediaError(f"{where}: audio file not found: {part.source}")
        elif isinstance(part, Video):
            duration = _media.video_duration(part.source)
            frames = _media.video_frames(duration, config.video_fps)
            if frames > config.video_max_frames and part.overflow == "error":
                raise FrameLimitError(
                    f"{where}: {part.source} is {duration:.1f} s, which is {frames} frames at "
                    f"video_fps={config.video_fps}, more than video_max_frames="
                    f"{config.video_max_frames}. The processor would silently drop "
                    f"{frames - config.video_max_frames} frames. Use Video(..., overflow='subsample') "
                    "to accept uniform subsampling, or change video_fps / video_max_frames."
                )


def build_message(item: Item, task: Task, side: Side) -> list[dict[str, Any]]:
    from . import _media

    content: list[dict[str, Any]] = []
    if item.has_text or item.title is not None:
        content.append({"type": "text", "text": prefix(task, side, item.title)})
    for part in item.parts:
        if isinstance(part, Text):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, Image):
            if hasattr(part.source, "mode"):  # PIL image
                content.append({"type": "image", "image": part.source})
            else:
                content.append({"type": "image", "url": str(part.source)})
        elif isinstance(part, Video):
            content.append({"type": "video", "url": str(part.source)})
            if part.include_audio:
                content.append({"type": "audio", "audio": _media.load_audio_file(part.source)})
        elif isinstance(part, Audio):
            if hasattr(part.source, "shape"):
                wave = _media.audio_from_array(part.source, part.sampling_rate)
            else:
                wave = _media.load_audio_file(part.source)
            content.append({"type": "audio", "audio": wave})
    return [{"role": "user", "content": content}]


PLACEHOLDER = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}


def render(content: list[dict[str, Any]]) -> str:
    """The text the chat template produces for a message: text verbatim, one placeholder per
    media part (tested against the real template in tests/test_gpu.py). The tokenizer then adds
    <bos>/<eos> and the processor expands each placeholder into that media's soft tokens."""
    return "".join(c["text"] if c["type"] == "text" else PLACEHOLDER[c["type"]] for c in content)


def preview_content(item: Item, task: Task, side: Side) -> list[dict[str, Any]]:
    """Like build_message, but without opening any file (media parts are only typed)."""
    content: list[dict[str, Any]] = []
    if item.has_text or item.title is not None:
        content.append({"type": "text", "text": prefix(task, side, item.title)})
    for part in item.parts:
        if isinstance(part, Text):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, Image):
            content.append({"type": "image"})
        elif isinstance(part, Video):
            content.append({"type": "video"})
            if part.include_audio:
                content.append({"type": "audio"})
        else:
            content.append({"type": "audio"})
    return content


def processing_kwargs(config: EngineConfig) -> dict[str, dict[str, Any]]:
    return {
        "image": {"max_soft_tokens": config.image_tokens},
        "video": {
            "max_soft_tokens": config.video_frame_tokens,
            "fps": config.video_fps,
            "max_frames": config.video_max_frames,
        },
        # The feature extractor cuts audio at 30 s by default; the context check replaces that.
        "audio": {"truncation": False},
    }


def cost_estimate(item: Item) -> int:
    """Rough cost, only used to sort items into similar-sized batches (less padding)."""
    cost = 0
    for p in item.parts:
        if isinstance(p, Text):
            cost += len(p.text) // 4
        elif isinstance(p, Image):
            cost += 300
        elif isinstance(p, Video):
            cost += 3000
        else:
            cost += 500
    return cost
