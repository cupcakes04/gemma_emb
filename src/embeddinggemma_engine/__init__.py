"""Explicit, validated engine around Google EmbeddingGemma 2.

Importing this package must stay cheap: torch, transformers and sentence-transformers are
imported only when the model is first loaded (enforced by tests/test_unit.py).
"""

from ._preview import preview
from .config import DIMS, MODEL_ID, PINNED_REVISION, SOFT_TOKEN_BUDGETS, EngineConfig
from .engine import EmbeddingEngine
from .errors import (
    ConfigError,
    ContextOverflowError,
    EngineError,
    FrameLimitError,
    IncompatibleEmbeddingsError,
    InputError,
    MediaError,
    ModalityNotLoadedError,
    ModelNotFoundError,
    NumericalError,
    TaskError,
)
from .ops import check_compatible, classify, near_duplicates, similarity, top_k
from .tasks import Task
from .types import Audio, Embeddings, Hit, Image, Item, Label, Pair, Text, TokenCount, Video

__version__ = "0.1.0"

__all__ = [
    "DIMS",
    "MODEL_ID",
    "PINNED_REVISION",
    "SOFT_TOKEN_BUDGETS",
    "Audio",
    "ConfigError",
    "ContextOverflowError",
    "EmbeddingEngine",
    "Embeddings",
    "EngineConfig",
    "EngineError",
    "FrameLimitError",
    "Hit",
    "Image",
    "IncompatibleEmbeddingsError",
    "InputError",
    "Item",
    "Label",
    "MediaError",
    "ModalityNotLoadedError",
    "ModelNotFoundError",
    "NumericalError",
    "Pair",
    "Task",
    "TaskError",
    "Text",
    "TokenCount",
    "Video",
    "check_compatible",
    "classify",
    "near_duplicates",
    "preview",
    "similarity",
    "top_k",
]
