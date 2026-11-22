"""Model loading. The only module that imports torch / sentence-transformers at module level."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import EngineConfig
from .errors import ConfigError, ModelNotFoundError

EXPECTED_MODEL_TYPE = "embedding_gemma2"


def resolve_dtype(config: EngineConfig) -> torch.dtype:
    if config.dtype == "float32":
        return torch.float32
    on_cuda = config.device.startswith("cuda")
    if config.dtype == "bfloat16":
        if on_cuda and not torch.cuda.is_bf16_supported():
            raise ConfigError("dtype='bfloat16' but this GPU does not support bf16. Use 'float32'.")
        return torch.bfloat16
    # auto
    if on_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model(config: EngineConfig):
    from sentence_transformers import SentenceTransformer

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"device={config.device!r} but CUDA is not available. Use device='cpu'.")

    config_kwargs = {}
    if not config.loads_vision:
        config_kwargs["vision_config"] = None
    if not config.loads_audio:
        config_kwargs["audio_config"] = None

    from transformers.utils import logging as hf_logging

    is_dir = Path(config.model).is_dir()
    bar_was_on = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()  # load quietly; restored below
    try:
        model = SentenceTransformer(
            config.model,
            revision=None if is_dir else config.revision,
            local_files_only=config.local_files_only,
            device=config.device,
            model_kwargs={"dtype": resolve_dtype(config)},
            config_kwargs=config_kwargs,
        )
    except OSError as ex:
        if is_dir:
            raise ModelNotFoundError(f"Cannot load the model from {config.model}: {ex}") from ex
        rev = f" --revision {config.revision}" if config.revision else ""
        raise ModelNotFoundError(
            f"{config.model}{' @ ' + config.revision if config.revision else ''} is not in the "
            f"Hugging Face cache (local_files_only=True). Download it once with:\n"
            f"    hf download {config.model}{rev}"
        ) from ex
    finally:
        if bar_was_on:
            hf_logging.enable_progress_bar()

    model_type = getattr(model[0].config, "model_type", None)
    if model_type != EXPECTED_MODEL_TYPE:
        raise ConfigError(
            f"{config.model} is a {model_type!r} model; this engine only supports "
            f"{EXPECTED_MODEL_TYPE!r} (google/embeddinggemma-2)."
        )
    model.eval()
    return model


def load_processor(config: EngineConfig):
    """The processor alone (tokenizer + media processors), for token counting without weights."""
    from transformers import AutoProcessor

    is_dir = Path(config.model).is_dir()
    try:
        return AutoProcessor.from_pretrained(
            config.model,
            revision=None if is_dir else config.revision,
            local_files_only=config.local_files_only,
        )
    except OSError as ex:
        rev = f" --revision {config.revision}" if config.revision and not is_dir else ""
        raise ModelNotFoundError(
            f"Cannot load the processor for {config.model}. Download the model once with:\n"
            f"    hf download {config.model}{rev}"
        ) from ex
