# EmbeddingGemma 2: model facts

What the model and its libraries actually do. Each fact is tagged by its source:
**card** (the HF model card), **config** (files in the model repo), or **measured**.

"Measured" means run by [`research/01_raw_model.ipynb`](../research/01_raw_model.ipynb) (the raw libraries) or [`research/02_engine_evaluation.ipynb`](../research/02_engine_evaluation.ipynb) (through the engine) at revision `f9972ac22e696a105c482202646979e33e4b5a1e`, with sentence-transformers 6.1.0, transformers 5.19.0, torch 2.14 (cu130) and bf16 on an RTX 4070 Ti SUPER (16 GB).

Facts that the engine turns into rules link to the rule in [conventions](conventions.md).

## Model

| Fact | Value | Source |
|---|---|---|
| Parameters | 744M full; text 271M, text+vision 439M, text+audio 577M | measured |
| VRAM (weights, bf16) | full 1.50 GB; text 0.54; text+vision 0.89; text+audio 1.15 | measured |
| Output | 768-d, mean pooling (prompt tokens included), then Normalize | config |
| Matryoshka dims | 768, 512, 256, 128 | card |
| dtype | bf16 or fp32. **fp16 returns NaN on long inputs without any error**; short inputs look fine | card, measured |
| Context | 8,192 tokens per item, shared by text and media | card |
| `max_position_embeddings` | 262,144 (the config doesn't enforce the card's 8,192) | config |

## Things that go wrong silently in sentence-transformers / transformers

1. **No length limit.** `max_seq_length` and `tokenizer.model_max_length` are about 1e30. A 12k-token text embeds without warning (measured). The engine enforces 8,192.
2. **`truncate_dim` does not re-normalize.** Norms come out at 0.885, 0.718 and 0.609 for 512, 256 and 128 (measured). Cosine through `model.similarity` still works; raw dot products don't. Even 768-d outputs have norms around 1.001 because normalization runs in bf16.
3. **Some strings are guessed as media.** A string that is an existing file path, an `http(s)` URL or a data URI with a media extension (`.jpg`, `.mp4`, `.mp3`…) is loaded as that media, not embedded as text (measured). A string that merely mentions one (`"photo.jpg"`, `"see https://…/a.jpg"`) stays text. The engine always builds explicit messages, so its strings are always text.
4. **The prefix matters.** With and without `task: search result | query: `, cosine is 0.91 for the same text (measured). A manually built prefix string is bit-identical to `prompt_name=` (measured).
5. **`prompt_name="Document"` always adds `title: none`.** Titles must be built by hand (card, config).
6. **Video frames are capped at `max_frames=32`** (`overflow_strategy="uniform"`). A 34.8 s clip at 1 fps is silently subsampled to 32 frames (measured). This is not the card's "~58 frames".
7. **Audio is cut at 30 s.** The feature extractor defaults to `max_length=480000, truncation=True`, so 120 s and 400 s both give 750 tokens (measured). `{"truncation": False}` lifts the cap: 60 s gives 1,500 tokens. The card's "~327 s of audio" needs this.
8. **The sampling rate of a raw audio array is ignored.** A 10 s array at 44.1 kHz is read as 16 kHz, giving 689 tokens, as if it were 27.6 s. This happens even when `sampling_rate=44100` is passed (measured). A stereo `(2, N)` array fails with a numpy broadcast error.
9. **A video's soundtrack is ignored by default** (`load_audio_from_video=False`). Enabling it on a video with no audio stream crashes inside torchcodec with "This should never happen".
10. **Placeholders in text plus typed media.** The chat template drops typed media items when the text already contains `<|image|>`, `<|video|>` or `<|audio|>` (config `chat_template.jinja`).
11. **A missing tower is only reported after preprocessing** (e.g. images on a text-only load). The full video decode runs before the error.

## Token costs (measured)

Each item also gets `<bos>` and `<eos>`.

| Input | Tokens |
|---|---|
| Image, default budget 280 | 276 + 2 (boi/eoi). Depends on aspect ratio |
| Image, `max_soft_tokens` 70 / 1120 | 66 / 1,104 |
| Video frame (16:9), default budget 140 | 120 per frame + framing |
| Video frame, `max_soft_tokens` 70 | 66 per frame |
| Video frames | floor(duration × fps): 10.3 s gives 10, 9.6 s gives 9 |
| Audio | 25 per second (`audio_ms_per_token` 40) |

Image and video budgets (`max_soft_tokens`), `fps` and `max_frames` can all be overridden per call through the processor kwargs (measured).

## Prefix rule for media (measured)

This is a cross-modal retrieval test on the assets: text queries with the `search` query prefix against media documents.

| Document form | Image MRR / margin | Video MRR / margin |
|---|---|---|
| media only, no prefix | 1.00 / **0.142** | 1.00 / **0.123** |
| `title: none \| text: ` + media | 1.00 / 0.121 | 1.00 / 0.121 |

No prefix gives the clearest separation, which agrees with the card ("prefixes apply to text only"). The engine adds no prefix to media-only items.

## Audio quality (measured, a model limitation)

- Audio embeddings sit close together: the mean cosine between unrelated clips (music, sound effects, ambient) is 0.87.
- Text→audio retrieval on the asset clips is weak: top-1 1/4, margin about 0.006.
- The engine's own decode (torchcodec, mono 16 kHz) matches the processor's loading of the same file (cosine ≈ 1.00), so the weakness is not a pipeline bug.
- Expect audio to work better audio↔audio than text↔audio, and validate on your own data.

## Cross-modal structure (measured, research/02_engine_evaluation)

- **Zero-shot labelling:** the `similarity` task matched or beat `classification` for labelling.
  Accuracy on images: 1.00 vs 1.00; on videos: 0.83 vs 0.83; on audio: **1.00 vs 0.30**.
  This is why `classify` defaults to `similarity`.
- **Smaller vectors:** at 512 and 256 dims, retrieval on the assets is unchanged. At 128 every target still ranks correctly, but score margins drop by roughly 40% (e.g. images 0.143 → 0.093).
- **Modality gap:** pairwise, each image or video is closer to its own caption (mean cosine 0.775) than to other images (0.651). But k-means over a mixed image + video + text set splits largely by modality.
  Cluster one modality at a time; connect modalities with search.

## Card vs config conflicts

| Topic | Card | Config / measured |
|---|---|---|
| Context | 8,192 | `max_position_embeddings` 262,144; nothing enforced |
| Sliding window | 1,024 | `sliding_window` 512 |
| Video frames | ~58 at default budget | `max_frames` 32 |
| Audio length | ~327 s | 30 s cap unless `truncation=False` |
