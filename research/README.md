# Research

Evidence, not a guide. These notebooks measure what the model and the libraries do, and they
back every claim in [`docs/model-facts.md`](../docs/model-facts.md). For how to *use* the engine,
see [`../notebooks`](../notebooks/README.md).

| Notebook | Measures |
|---|---|
| [01 · The raw model](01_raw_model.ipynb) | `sentence-transformers` / `transformers` **without** the engine: VRAM per tower, prefixes, normalization, fp16 NaN, token costs, the silent limits (frames, 30 s audio, sampling rate, length), media prefix choice |
| [02 · Engine evaluation](02_engine_evaluation.ipynb) | Through the engine, on the bundled assets: retrieval per modality and dim, which task to use for `classify`, audio quality, the modality gap, batch invariance |

**When to rerun:** after upgrading `transformers` or `sentence-transformers`, or after changing the
pinned model revision. Then update `docs/model-facts.md` if any number moves.

The test data lives in `../assets/text/`:
- `corpus.jsonl`: documents, code and two paraphrased duplicate pairs
- `captions.jsonl`: a caption per media file. Audio captions come from the file names.
- `queries.jsonl`: queries with their expected matches
- `labels.json`: label sets with the expected label for each file
