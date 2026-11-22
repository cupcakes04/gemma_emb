"""Inputs (Item and its parts) and results (Embeddings, TokenCount, Hit, Pair, Label)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .errors import InputError
from .tasks import Side, Task

if TYPE_CHECKING:
    import numpy as np

PLACEHOLDERS = ("<|image|>", "<|video|>", "<|audio|>")


# ---------------------------------------------------------------- parts


@dataclass(frozen=True)
class Text:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InputError(f"Text must be a str, got {type(self.text).__name__}.")
        if not self.text.strip():
            raise InputError("Text is empty.")
        found = [p for p in PLACEHOLDERS if p in self.text]
        if found:
            raise InputError(
                f"Text contains the placeholder(s) {found}. Pass media as Image/Video/Audio parts "
                "instead: the chat template silently drops typed media when text has placeholders."
            )


@dataclass(frozen=True)
class Image:
    """An image file path or a PIL image."""

    source: Any

    def __post_init__(self) -> None:
        if isinstance(self.source, (str, Path)):
            object.__setattr__(self, "source", Path(self.source))
            return
        if type(self.source).__module__.startswith("PIL."):
            return
        raise InputError(f"Image source must be a path or PIL image, got {type(self.source).__name__}.")


@dataclass(frozen=True)
class Video:
    """A video file.

    include_audio: also embed the file's soundtrack (as an extra audio part right after the video).
        Off by default, which matches the model's default of ignoring a video's audio.
    overflow: what to do when the clip yields more than EngineConfig.video_max_frames frames.
        "error" (default) raises FrameLimitError; "subsample" lets the processor pick
        video_max_frames frames spread uniformly over the clip.
    """

    source: str | Path
    include_audio: bool = False
    overflow: Literal["error", "subsample"] = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.source, (str, Path)):
            raise InputError(f"Video source must be a file path, got {type(self.source).__name__}.")
        object.__setattr__(self, "source", Path(self.source))
        if self.overflow not in ("error", "subsample"):
            raise InputError(f"Video overflow must be 'error' or 'subsample', got {self.overflow!r}.")


@dataclass(frozen=True, eq=False)
class Audio:
    """An audio file (any format ffmpeg reads, incl. the soundtrack of a video file) or a raw array.

    Files are decoded, downmixed to mono and resampled to 16 kHz by the engine.
    Arrays need `sampling_rate` (the processor would silently assume 16 kHz). Shape (samples,)
    or (channels, samples); they are downmixed and resampled the same way.
    """

    source: Any
    sampling_rate: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source, (str, Path)):
            if self.sampling_rate is not None:
                raise InputError("sampling_rate is only for arrays; files carry their own rate.")
            object.__setattr__(self, "source", Path(self.source))
            return
        if not hasattr(self.source, "shape"):
            raise InputError(f"Audio source must be a path or array, got {type(self.source).__name__}.")
        if self.sampling_rate is None:
            raise InputError(
                "Audio arrays need sampling_rate=... (the processor would silently treat any array "
                "as 16 kHz, stretching or squashing it)."
            )
        if self.sampling_rate <= 0:
            raise InputError(f"sampling_rate must be > 0, got {self.sampling_rate}.")


Part = Text | Image | Video | Audio


def _as_part(p: Any) -> Part:
    if isinstance(p, (Text, Image, Video, Audio)):
        return p
    if isinstance(p, str):
        return Text(p)
    raise InputError(
        f"Item parts must be str, Text, Image, Video or Audio, got {type(p).__name__}. "
        "Wrap file paths in Image(...), Video(...) or Audio(...)."
    )


# ---------------------------------------------------------------- items


@dataclass(frozen=True)
class Item:
    """One thing to embed: an ordered sequence of parts, embedded into one vector.

    A plain `str` passed to an embed method is shorthand for `Item.text(str)`. Strings are always
    text, never sniffed for file extensions.
    `title` is only used on the document side of asymmetric tasks.
    """

    parts: tuple[Part, ...]
    title: str | None = None

    def __post_init__(self) -> None:
        parts = tuple(_as_part(p) for p in self.parts)
        if not parts:
            raise InputError("Item has no parts.")
        object.__setattr__(self, "parts", parts)
        if self.title is not None:
            if not isinstance(self.title, str) or not self.title.strip():
                raise InputError("title must be a non-empty str or None.")
            Text(self.title)  # same placeholder checks

    @classmethod
    def of(cls, *parts: Any, title: str | None = None) -> Item:
        return cls(parts=tuple(parts), title=title)

    @classmethod
    def text(cls, text: str, *, title: str | None = None) -> Item:
        return cls(parts=(Text(text),), title=title)

    @classmethod
    def image(cls, source: Any, *, title: str | None = None) -> Item:
        return cls(parts=(Image(source),), title=title)

    @classmethod
    def video(cls, source: str | Path, *, include_audio: bool = False,
              overflow: Literal["error", "subsample"] = "error", title: str | None = None) -> Item:
        return cls(parts=(Video(source, include_audio, overflow),), title=title)

    @classmethod
    def audio(cls, source: Any, *, sampling_rate: int | None = None,
              title: str | None = None) -> Item:
        return cls(parts=(Audio(source, sampling_rate),), title=title)

    @property
    def modalities(self) -> set[str]:
        mods: set[str] = set()
        for p in self.parts:
            if isinstance(p, Text):
                mods.add("text")
            elif isinstance(p, Image):
                mods.add("image")
            elif isinstance(p, Video):
                mods.add("video")
                if p.include_audio:
                    mods.add("audio")
            else:
                mods.add("audio")
        return mods

    @property
    def has_text(self) -> bool:
        return any(isinstance(p, Text) for p in self.parts)


def as_item(x: Any) -> Item:
    if isinstance(x, Item):
        return x
    if isinstance(x, str):
        return Item.text(x)
    raise InputError(
        f"Expected an Item or str, got {type(x).__name__}. Use Item.image(...), Item.video(...), "
        "Item.audio(...) or Item.of(...) for media."
    )


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class TokenCount:
    """Tokens one item uses. `other` is text plus special tokens (bos/eos, media begin/end)."""

    total: int
    image: int
    video: int
    audio: int
    other: int

    def to_json(self) -> dict[str, int]:
        return {"total": self.total, "image": self.image, "video": self.video,
                "audio": self.audio, "other": self.other}


@dataclass(frozen=True, eq=False)
class Embeddings:
    """Vectors plus everything needed to know what they can be compared with.

    vectors: float32 array (n, dim), every row L2-normalized.
    """

    vectors: np.ndarray
    task: Task
    side: Side
    dim: int
    model: str
    revision: str | None
    token_counts: tuple[TokenCount, ...] = field(default=())

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "vectors": self.vectors.tolist(),
            "task": self.task.value,
            "side": self.side,
            "dim": self.dim,
            "model": self.model,
            "revision": self.revision,
            "token_counts": [t.to_json() for t in self.token_counts],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Embeddings:
        import numpy as np

        vectors = np.asarray(data["vectors"], dtype=np.float32).reshape(-1, data["dim"])
        return cls(
            vectors=vectors,
            task=Task(data["task"]),
            side=data["side"],
            dim=data["dim"],
            model=data["model"],
            revision=data["revision"],
            token_counts=tuple(TokenCount(**t) for t in data.get("token_counts", [])),
        )


@dataclass(frozen=True)
class Hit:
    index: int
    score: float


@dataclass(frozen=True)
class Pair:
    i: int
    j: int
    score: float


@dataclass(frozen=True)
class Label:
    name: str
    score: float
    scores: dict[str, float]
