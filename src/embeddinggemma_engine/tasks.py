"""Tasks and their prefix templates. This is the only place prompt strings live.

Templates follow the model card. The aliases in the model's config_sentence_transformers.json
(e.g. "Summarization" -> sentence similarity) are deliberately not exposed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

Side = Literal["query", "document", "symmetric"]


class Task(StrEnum):
    # asymmetric: queries and documents are embedded differently
    SEARCH = "search"
    QUESTION_ANSWERING = "question_answering"
    FACT_CHECKING = "fact_checking"
    CODE_RETRIEVAL = "code_retrieval"
    # symmetric: every input is embedded the same way
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    SIMILARITY = "similarity"

    @property
    def is_asymmetric(self) -> bool:
        return self in _QUERY_PREFIX


_QUERY_PREFIX: dict[Task, str] = {
    Task.SEARCH: "task: search result | query: ",
    Task.QUESTION_ANSWERING: "task: question answering | query: ",
    Task.FACT_CHECKING: "task: fact checking | query: ",
    Task.CODE_RETRIEVAL: "task: code retrieval | query: ",
}

_SYMMETRIC_PREFIX: dict[Task, str] = {
    Task.CLASSIFICATION: "task: classification | query: ",
    Task.CLUSTERING: "task: clustering | query: ",
    Task.SIMILARITY: "task: sentence similarity | query: ",
}


def parse_task(task: Task | str) -> Task:
    from .errors import TaskError

    try:
        return Task(task)
    except ValueError:
        valid = ", ".join(t.value for t in Task)
        raise TaskError(f"Unknown task {task!r}. Valid tasks: {valid}.") from None


def resolve(task: Task | str, side: str | None) -> tuple[Task, Side]:
    """Check that task and side fit together. side=None means "symmetric" for symmetric tasks."""
    from .errors import TaskError

    t = parse_task(task)
    if side is None:
        if t.is_asymmetric:
            raise TaskError(f"{t.value} is asymmetric: pass side='query' or side='document'.")
        return t, "symmetric"
    if side not in ("query", "document", "symmetric"):
        raise TaskError(f"side must be 'query', 'document' or 'symmetric', got {side!r}.")
    if side == "symmetric" and t.is_asymmetric:
        raise TaskError(f"{t.value} is asymmetric: use embed_queries / embed_documents, not embed.")
    if side != "symmetric" and not t.is_asymmetric:
        method = "embed_queries" if side == "query" else "embed_documents"
        raise TaskError(f"{t.value} is symmetric: use embed, not {method}.")
    return t, side


def prefix(task: Task, side: Side, title: str | None) -> str:
    """The text prefix for one item. `title` is only meaningful on the document side."""
    if side == "document":
        return f"title: {title if title is not None else 'none'} | text: "
    if side == "query":
        return _QUERY_PREFIX[task]
    return _SYMMETRIC_PREFIX[task]
