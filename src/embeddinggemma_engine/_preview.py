from __future__ import annotations

from .tasks import Task, resolve
from .types import Item, as_item


def preview(item: Item | str, *, task: Task | str, side: str | None = None) -> str:
    """The exact text the model receives for one item, without loading anything.

    Media parts appear as their placeholders (<|image|>, <|video|>, <|audio|>); the processor
    later expands each one into that media's tokens. `side` is "query" or "document" for
    asymmetric tasks and can be left out for symmetric ones.

        >>> preview("how to brew coffee", task="search", side="query")
        'task: search result | query: how to brew coffee'
    """
    from ._build import preview_content, render
    from .errors import TaskError

    t, s = resolve(task, side)
    it = as_item(item)
    if it.title is not None and s != "document":
        raise TaskError(f"title is only used by embed_documents (this is the {s} side).")
    return render(preview_content(it, t, s))
