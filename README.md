# embeddinggemma_engine

An explicit, validated engine around Google's [**EmbeddingGemma 2**](https://huggingface.co/google/embeddinggemma-2). It embeds text, images, video and audio into one 768-d space.

The model is small (744M params, 1.5 GB in bf16) and easy to load, but easy to call wrongly without noticing:

- Task prefixes are needed. The Hub snippet leaves them out.
- `truncate_dim` returns vectors that are no longer normalized.
- Nothing enforces the 8,192-token context.
- Videos silently lose frames past 32.
- Audio is silently cut at 30 s.
- A raw audio array is assumed to be 16 kHz whatever its real rate.
- fp16 returns NaN on long inputs.

[docs/model-facts.md](docs/model-facts.md) has the measured evidence for each of these. `embeddinggemma_engine` turns each one into a typed input or a clear error:

- **Explicit tasks and sides.** `embed_queries` / `embed_documents` for asymmetric tasks, `embed` for symmetric ones. The engine builds the prefixes and titles itself.
- **Typed, ordered items.** `Item.of("caption", Image(p), Video(v), Audio(a))` becomes one vector. Strings are always text, never guessed to be file paths.
- **Validated up front.** Missing files, unloaded modalities, videos that would lose frames, and items over the context raise before the model runs, with exact token breakdowns.
- **Comparable results.** `Embeddings` carry their task, side, dim and model revision, and the helpers refuse to compare different spaces. Vectors are always unit-norm.
- **Reproducible.** Pinned model revision, offline by default, bf16 or fp32 only.

```python
import embeddinggemma_engine as eg

engine = eg.EmbeddingEngine()   # google/embeddinggemma-2, pinned revision, all modalities

docs = engine.embed_documents([
    eg.Item.text("Bloom the grounds for thirty seconds, then pour slowly.", title="Pour-over coffee"),
    eg.Item.image("assets/images/wildlife.jpg"),
    eg.Item.video("assets/videos/04_coffee.mp4", include_audio=True),
    eg.Item.audio("assets/audio/voicebosch-soda-can-opening-167415.mp3"),
], task="search")
queries = engine.embed_queries(["how to brew coffee", "giraffes and zebras"], task="search")

for hits in eg.top_k(queries, docs, k=2):
    print([(h.index, round(h.score, 3)) for h in hits])
```

## Which method?

| Goal | Call |
|---|---|
| Search / RAG over text or media | `embed_queries` + `embed_documents` with `task="search"`, then `eg.top_k` |
| Questions vs passages · claims vs evidence · text vs code | the same, with `question_answering` · `fact_checking` · `code_retrieval` |
| Zero-shot labels | `eg.classify(engine, items, labels={"name": "description", ...})` |
| Near-duplicates | `embed(items, task="similarity")`, then `eg.near_duplicates` |
| Clustering | `embed(items, task="clustering")`, then any clustering library |
| What exactly does the model read? | `eg.preview(item, task=..., side=...)` |
| How many tokens will this use? | `engine.count_tokens(items, task=..., side=...)` |

## Install

```bash
conda env create -f environment.yml && conda activate embeddinggemma_engine     # python 3.12 + FFmpeg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # or cu128, to match your driver
pip install -e ".[notebook,dev]"
hf download google/embeddinggemma-2 --revision f9972ac22e696a105c482202646979e33e4b5a1e
```

The weights live in the Hugging Face cache. The engine loads them offline at that pinned revision, and never puts weight files in the repo. To use another revision or a local copy, set `EngineConfig(model=..., revision=...)`.

### Choosing what to load

```python
eg.EngineConfig(
    modalities=("text", "image", "video", "audio"),  # drop towers you don't need: text-only is 0.54 GB
    image_tokens=280, video_frame_tokens=140,        # 70 / 140 / 280 / 560 / 1120
    video_fps=1.0, video_max_frames=32,
    max_tokens=8192, batch_size=8, device="cuda", dtype="auto",
)
```

## Learn more

- [`notebooks/`](notebooks/README.md): **start here.** A short user guide, read in order:
  quickstart → **tasks and prompts** → search → labelling → duplicates and clustering → media settings
- [`docs/conventions.md`](docs/conventions.md): every rule the engine enforces, and its errors
- [`research/`](research/README.md) and [`docs/model-facts.md`](docs/model-facts.md): what the model and libraries actually do, measured

## Tests

```bash
pytest tests/test_unit.py   # CPU only, no weights
pytest -m gpu               # end to end; needs CUDA and the weights in the HF cache
```
