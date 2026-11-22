"""Typed errors. Every rule the engine enforces fails with one of these, before the model runs."""


class EngineError(Exception):
    """Base class for every error raised by embeddinggemma_engine."""


class ConfigError(EngineError, ValueError):
    """Invalid EngineConfig (e.g. dtype float16, unknown modality, bad token budget)."""


class ModelNotFoundError(EngineError, OSError):
    """The weights are not in the Hugging Face cache (or local directory) and downloads are off."""


class TaskError(EngineError, ValueError):
    """Task used with the wrong method or side (e.g. a symmetric task passed to embed_queries)."""


class InputError(EngineError, ValueError):
    """An item is malformed: empty text, placeholder tokens in text, bad audio array, etc."""


class MediaError(EngineError, OSError):
    """A media file is missing, cannot be decoded, or lacks the requested stream."""


class ModalityNotLoadedError(EngineError, ValueError):
    """An item uses a modality that EngineConfig.modalities did not load."""


class ContextOverflowError(EngineError, ValueError):
    """An item needs more tokens than EngineConfig.max_tokens."""


class FrameLimitError(EngineError, ValueError):
    """A video would yield more frames than EngineConfig.video_max_frames (frames would be dropped)."""


class NumericalError(EngineError, RuntimeError):
    """The model produced NaN or inf."""


class IncompatibleEmbeddingsError(EngineError, ValueError):
    """Two Embeddings cannot be compared (different dim, task or model revision)."""
