"""Similarity helpers on Embeddings. Pure numpy; vectors are already L2-normalized."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .errors import IncompatibleEmbeddingsError, InputError
from .tasks import Task
from .types import Embeddings, Hit, Item, Label, Pair

if TYPE_CHECKING:
    from .engine import EmbeddingEngine


def check_compatible(a: Embeddings, b: Embeddings) -> None:
    """Raise IncompatibleEmbeddingsError unless a and b live in the same space."""
    problems = []
    if a.dim != b.dim:
        problems.append(f"dim {a.dim} vs {b.dim}")
    if a.task != b.task:
        problems.append(f"task {a.task.value} vs {b.task.value}")
    if (a.model, a.revision) != (b.model, b.revision):
        problems.append(f"model {a.model}@{a.revision} vs {b.model}@{b.revision}")
    if problems:
        raise IncompatibleEmbeddingsError("Cannot compare these embeddings: " + "; ".join(problems))


def similarity(a: Embeddings, b: Embeddings) -> np.ndarray:
    """Cosine similarity matrix, shape (len(a), len(b))."""
    check_compatible(a, b)
    return a.vectors @ b.vectors.T


def top_k(queries: Embeddings, documents: Embeddings, *, k: int) -> list[list[Hit]]:
    """The k best documents for each query, best first."""
    if not 1 <= k <= len(documents):
        raise InputError(f"k must be between 1 and len(documents)={len(documents)}, got {k}.")
    scores = similarity(queries, documents)
    result = []
    for row in scores:
        best = np.argsort(-row, kind="stable")[:k]
        result.append([Hit(index=int(i), score=float(row[i])) for i in best])
    return result


def near_duplicates(embeddings: Embeddings, *, threshold: float) -> list[Pair]:
    """All pairs (i < j) with cosine >= threshold, highest first."""
    if not -1.0 <= threshold <= 1.0:
        raise InputError(f"threshold must be in [-1, 1], got {threshold}.")
    scores = embeddings.vectors @ embeddings.vectors.T
    i, j = np.triu_indices(len(embeddings), k=1)
    keep = scores[i, j] >= threshold
    pairs = [Pair(i=int(a), j=int(b), score=float(s))
             for a, b, s in zip(i[keep], j[keep], scores[i, j][keep])]
    return sorted(pairs, key=lambda p: -p.score)


def classify(engine: EmbeddingEngine, items: Sequence[Item | str], *,
             labels: Mapping[str, str], task: Task | str = Task.SIMILARITY,
             dim: int = 768) -> list[Label]:
    """Zero-shot labelling: each item gets the label whose description is most similar.

    labels maps a label name to a short description, e.g. {"cat": "a photo of a cat"}.
    Items and descriptions are embedded with a symmetric task. The default is "similarity":
    on the bundled assets it matched or beat "classification" (audio: 1.00 vs 0.30 accuracy,
    see research/02_engine_evaluation). Validate on your own labels.
    """
    if len(labels) < 2:
        raise InputError("classify needs at least two labels.")
    names = list(labels)
    item_emb = engine.embed(items, task=task, dim=dim)
    label_emb = engine.embed([labels[n] for n in names], task=task, dim=dim)
    scores = similarity(item_emb, label_emb)
    result = []
    for row in scores:
        best = int(np.argmax(row))
        result.append(Label(name=names[best], score=float(row[best]),
                            scores={n: float(s) for n, s in zip(names, row)}))
    return result
