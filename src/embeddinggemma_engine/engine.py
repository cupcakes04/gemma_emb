from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from .config import DIMS, EngineConfig
from .errors import ContextOverflowError, InputError, NumericalError
from .tasks import Side, Task, resolve
from .types import Embeddings, Item, TokenCount, as_item


class EmbeddingEngine:
    """EmbeddingGemma 2 with its usage rules enforced.

    Asymmetric tasks (search, question_answering, fact_checking, code_retrieval) use
    `embed_queries` and `embed_documents`. Symmetric tasks (classification, clustering,
    similarity) use `embed`. The model loads on first use (or `load()`), and the engine is
    safe to share between threads: calls into the model are serialised by one lock.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config if config is not None else EngineConfig()
        self._model: Any = None
        self._processor: Any = None  # used by count_tokens when the model isn't loaded
        self._lock = threading.RLock()

    # ------------------------------------------------------------ lifecycle

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from ._runtime import load_model

                self._model = load_model(self.config)

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            import gc

            import torch

            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    # ------------------------------------------------------------ public API

    def embed_queries(self, items: Sequence[Item | str], *, task: Task | str,
                      dim: int = 768) -> Embeddings:
        """Embed queries for an asymmetric task."""
        return self._embed(items, task=self._task(task, "query"), side="query", dim=dim)

    def embed_documents(self, items: Sequence[Item | str], *, task: Task | str,
                        dim: int = 768) -> Embeddings:
        """Embed documents for an asymmetric task. Item titles are used here."""
        return self._embed(items, task=self._task(task, "document"), side="document", dim=dim)

    def embed(self, items: Sequence[Item | str], *, task: Task | str,
              dim: int = 768) -> Embeddings:
        """Embed items for a symmetric task (every item is treated the same way)."""
        return self._embed(items, task=self._task(task, "symmetric"), side="symmetric", dim=dim)

    def count_tokens(self, items: Sequence[Item | str], *, task: Task | str,
                     side: Side | None = None) -> list[TokenCount]:
        """Exact token usage per item, from the real processor output.

        Only the processor is loaded (not the model weights), and nothing is embedded.
        Items over max_tokens are reported, not rejected. `side` is required for asymmetric
        tasks ("query" or "document") because the prefix differs.
        """
        from . import _build

        t, side = resolve(task, side)
        parsed = self._items(items)
        for i, item in enumerate(parsed):
            _build.validate(item, i, t, side, self.config)
        pk = _build.processing_kwargs(self.config)
        processor_kwargs = {"images_kwargs": pk["image"], "videos_kwargs": pk["video"],
                            "audio_kwargs": pk["audio"]}
        counts = []
        with self._lock:
            processor = self._get_processor()
            for item in parsed:
                out = processor.apply_chat_template(
                    _build.build_message(item, t, side), tokenize=True, return_dict=True,
                    return_tensors="pt", processor_kwargs=processor_kwargs)
                counts.extend(_token_counts(out, processor))
        return counts

    # ------------------------------------------------------------ internals

    def _get_processor(self) -> Any:
        if self._model is not None:
            return self._model[0].processor
        if self._processor is None:
            from ._runtime import load_processor

            self._processor = load_processor(self.config)
        return self._processor

    @staticmethod
    def _task(task: Task | str, side: Side) -> Task:
        return resolve(task, side)[0]

    @staticmethod
    def _items(items: Sequence[Item | str]) -> list[Item]:
        if isinstance(items, (str, Item)):
            raise InputError("Pass a list of items, e.g. [item], not a single item.")
        return [as_item(x) for x in items]

    def _embed(self, items: Sequence[Item | str], *, task: Task, side: Side,
               dim: int) -> Embeddings:
        if dim not in DIMS:
            raise InputError(f"dim must be one of {DIMS}, got {dim}.")
        parsed = self._items(items)
        counts, vectors = self._run(parsed, task, side, dim=dim)
        is_dir = Path(self.config.model).is_dir()
        return Embeddings(
            vectors=vectors,
            task=task,
            side=side,
            dim=dim,
            model=self.config.model,
            revision=None if is_dir else self.config.revision,
            token_counts=tuple(counts),
        )

    def _run(self, items: list[Item], task: Task, side: Side, dim: int):
        """Validate, preprocess, check the budget, run the model."""
        import numpy as np

        from . import _build

        cfg = self.config
        for i, item in enumerate(items):
            _build.validate(item, i, task, side, cfg)
        if not items:
            return [], np.zeros((0, dim), dtype=np.float32)

        self.load()
        import torch
        from sentence_transformers.util import batch_to_device

        model = self._model
        pk = _build.processing_kwargs(cfg)
        order = sorted(range(len(items)), key=lambda i: -_build.cost_estimate(items[i]))
        counts: list[TokenCount | None] = [None] * len(items)
        vectors = np.zeros((len(items), dim), dtype=np.float32)

        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            messages = [_build.build_message(items[i], task, side) for i in idx]
            with self._lock:
                features = model.preprocess(messages, processing_kwargs=pk)
                batch_counts = _token_counts(features, model[0].processor)
                for i, c in zip(idx, batch_counts):
                    if c.total > cfg.max_tokens:
                        raise ContextOverflowError(
                            f"item {i} needs {c.total} tokens, more than max_tokens="
                            f"{cfg.max_tokens} (image {c.image}, video {c.video}, audio {c.audio}, "
                            f"text+special {c.other}). Shorten it, split it, or lower "
                            "image_tokens / video_frame_tokens / video_fps."
                        )
                    counts[i] = c
                features = batch_to_device(features, model.device)
                with torch.inference_mode():
                    out = model(features)["sentence_embedding"].float()
                if not torch.isfinite(out).all():
                    bad = [idx[r] for r in range(out.shape[0]) if not torch.isfinite(out[r]).all()]
                    raise NumericalError(f"The model returned NaN/inf for items {bad}.")
                out = torch.nn.functional.normalize(out[:, :dim], p=2, dim=1)
                vectors[idx] = out.cpu().numpy()
        return counts, vectors



def _token_counts(features: Any, processor: Any) -> list[TokenCount]:
    """Per-row token usage from processor output (input_ids + attention_mask)."""
    ids = features["input_ids"]
    mask = features.get("attention_mask")
    totals = mask.sum(dim=1) if mask is not None else (ids != -1).sum(dim=1)
    image = (ids == processor.image_token_id).sum(dim=1)
    video = (ids == processor.video_token_id).sum(dim=1)
    audio = (ids == processor.audio_token_id).sum(dim=1)
    result = []
    for r in range(ids.shape[0]):
        t, im, vi, au = int(totals[r]), int(image[r]), int(video[r]), int(audio[r])
        result.append(TokenCount(total=t, image=im, video=vi, audio=au, other=t - im - vi - au))
    return result
