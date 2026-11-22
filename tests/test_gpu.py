"""End-to-end tests on the real model. Needs CUDA and google/embeddinggemma-2 in the HF cache.

    pytest -m gpu
"""

import json
from pathlib import Path

import numpy as np
import pytest

import embeddinggemma_engine as eg

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TEXT = ASSETS / "text"


def _jsonl(name):
    return [json.loads(line) for line in (TEXT / name).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def engine():
    with eg.EmbeddingEngine() as e:
        e.load()
        yield e


def _media_item(path):
    p = ROOT / path
    if "/images/" in path:
        return eg.Item.image(p)
    if "/videos/" in path:
        return eg.Item.video(p, overflow="subsample")
    return eg.Item.audio(p)


def test_loads_pinned_revision_offline(engine):
    assert engine.is_loaded
    assert engine._model[0].config.model_type == "embedding_gemma2"


def test_manual_prefix_equals_prompt_name(engine):
    q = "how do I make naan without a tandoor"
    ours = engine.embed_queries([q], task="search").vectors[0]
    ref = engine._model.encode(q, prompt_name="SearchQuery")
    ref = ref / np.linalg.norm(ref)
    assert float(ours @ ref) > 0.9999


@pytest.mark.parametrize("dim", eg.DIMS)
def test_vectors_are_unit_norm(engine, dim):
    e = engine.embed(["a", "some longer text about kettles"], task="clustering", dim=dim)
    assert e.vectors.shape == (2, dim)
    assert np.allclose(np.linalg.norm(e.vectors, axis=1), 1.0, atol=1e-5)


def test_smaller_dim_is_prefix_of_larger(engine):
    full = engine.embed(["kettle"], task="clustering", dim=768).vectors[0]
    small = engine.embed(["kettle"], task="clustering", dim=128).vectors[0]
    assert float(small @ (full[:128] / np.linalg.norm(full[:128]))) > 0.9999


def test_batching_does_not_change_vectors(engine):
    texts = ["short", "a much longer sentence about pour-over coffee " * 5, "medium text"]
    batched = engine.embed(texts, task="clustering").vectors
    single = np.stack([engine.embed([t], task="clustering").vectors[0] for t in texts])
    assert (batched * single).sum(axis=1).min() > 0.999


def test_text_retrieval(engine):
    corpus = _jsonl("corpus.jsonl")
    docs = [eg.Item.text(d["text"], title=d["title"]) for d in corpus]
    for q in (q for q in _jsonl("queries.jsonl") if q["target"] == "text"):
        qe = engine.embed_queries([q["query"]], task=q["task"])
        de = engine.embed_documents(docs, task=q["task"])
        best = corpus[eg.top_k(qe, de, k=1)[0][0].index]["id"]
        assert best in q["expected"], (q["id"], best)


@pytest.mark.parametrize("target", ["image", "video"])
def test_cross_modal_retrieval(engine, target):
    paths = [c["path"] for c in _jsonl("captions.jsonl") if f"/{target}" in c["path"]]
    docs = engine.embed_documents([_media_item(p) for p in paths], task="search")
    for q in (q for q in _jsonl("queries.jsonl") if q["target"] == target):
        qe = engine.embed_queries([q["query"]], task="search")
        best = paths[eg.top_k(qe, docs, k=1)[0][0].index]
        assert best in q["expected"], (q["id"], best)


def test_audio_is_not_cut_at_30s(engine):
    path = ASSETS / "audio" / "van_wiese-gitarren-loop-136bpm-stereo-8969.mp3"  # 56.5 s
    count = engine.count_tokens([eg.Item.audio(path)], task="clustering")[0]
    assert abs(count.audio - 56.5 * 25) < 5


def test_audio_array_matches_file(engine):
    from torchcodec.decoders import AudioDecoder

    path = ASSETS / "audio" / "ahmadmousavipour-1throw-the-chord-flamenco-stock-299645.mp3"
    raw = AudioDecoder(str(path)).get_all_samples()  # stereo 44.1 kHz
    from_array = eg.Item.audio(raw.data.numpy(), sampling_rate=raw.sample_rate)
    e = engine.embed([from_array, eg.Item.audio(path)], task="clustering").vectors
    assert float(e[0] @ e[1]) > 0.95  # different resamplers (scipy vs ffmpeg), same content


def test_video_with_soundtrack(engine):
    item = eg.Item.video(ASSETS / "videos" / "04_coffee.mp4", include_audio=True)
    count = engine.count_tokens([item], task="clustering")[0]
    assert count.video > 0 and count.audio > 200


def test_token_budgets_apply(engine):
    img = eg.Item.image(ASSETS / "images" / "wildlife.jpg")
    with eg.EmbeddingEngine(eg.EngineConfig(image_tokens=70)) as small:
        assert small.count_tokens([img], task="clustering")[0].image <= 70
    assert engine.count_tokens([img], task="clustering")[0].image > 200


def test_context_overflow_raises(engine):
    with pytest.raises(eg.ContextOverflowError):
        engine.embed(["word " * 9000], task="clustering")


def test_pil_equals_path(engine):
    from PIL import Image

    path = ASSETS / "images" / "kitchen.jpg"
    e = engine.embed([eg.Item.image(path), eg.Item.image(Image.open(path))], task="clustering")
    assert float(e.vectors[0] @ e.vectors[1]) > 0.9999


def test_text_only_engine():
    with eg.EmbeddingEngine(eg.EngineConfig(modalities=("text",))) as e:
        assert e.embed(["hello"], task="clustering").vectors.shape == (1, 768)
        with pytest.raises(eg.ModalityNotLoadedError):
            e.embed([eg.Item.image(ASSETS / "images" / "kitchen.jpg")], task="clustering")


def test_preview_matches_chat_template(engine):
    from embeddinggemma_engine import _build

    processor = engine._model[0].processor
    img = ASSETS / "images" / "kitchen.jpg"
    cases = [
        (eg.Item.text("brew coffee"), "search", "query"),
        (eg.Item.text("Bloom first.", title="Pour-over"), "search", "document"),
        (eg.Item.image(img), "search", "document"),
        (eg.Item.of("A", eg.Image(img), "B", eg.Audio(np.zeros(16_000, np.float32), 16_000)),
         "similarity", "symmetric"),
    ]
    for item, task, side in cases:
        message = _build.build_message(item, eg.Task(task), side)
        rendered = processor.apply_chat_template(message, tokenize=False)
        assert rendered == eg.preview(item, task=task, side=None if side == "symmetric" else side)


def test_count_tokens_matches_embedding_and_skips_weights(engine):
    items = [
        eg.Item.text("Bloom first.", title="Pour-over"),
        eg.Item.image(ASSETS / "images" / "kitchen.jpg"),
        eg.Item.video(ASSETS / "videos" / "04_coffee.mp4", include_audio=True),
        eg.Item.audio(ASSETS / "audio" / "van_wiese-gitarren-loop-136bpm-stereo-8969.mp3"),
    ]
    embedded = engine.embed_documents(items, task="search").token_counts
    fresh = eg.EmbeddingEngine()
    counted = fresh.count_tokens(items, task="search", side="document")
    assert list(embedded) == counted
    assert not fresh.is_loaded  # only the processor was loaded


def test_count_tokens_reports_overflow_instead_of_raising(engine):
    count = engine.count_tokens(["word " * 9000], task="clustering")[0]
    assert count.total > engine.config.max_tokens
