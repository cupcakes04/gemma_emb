"""CPU-only tests. No weights, no GPU: everything here runs before the model would load."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import embeddinggemma_engine as eg
from embeddinggemma_engine import _build, _media
from embeddinggemma_engine.tasks import prefix

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
HEAVY = ("torch", "transformers", "sentence_transformers")


def test_import_is_lazy():
    code = (
        "import sys, embeddinggemma_engine\n"
        f"loaded = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("kwargs", [
    {"dtype": "float16"},
    {"modalities": ("image",)},
    {"modalities": ("text", "smell")},
    {"image_tokens": 300},
    {"video_frame_tokens": 0},
    {"batch_size": 0},
    {"video_fps": 0},
])
def test_config_rejects(kwargs):
    with pytest.raises(eg.ConfigError):
        eg.EngineConfig(**kwargs)


def test_config_towers():
    assert not eg.EngineConfig(modalities=("text",)).loads_vision
    assert eg.EngineConfig(modalities=("text", "video")).loads_vision
    assert not eg.EngineConfig(modalities=("text", "image")).loads_audio


# ---------------------------------------------------------------- tasks and prefixes


def test_prefixes_match_model_card():
    assert prefix(eg.Task.SEARCH, "query", None) == "task: search result | query: "
    assert prefix(eg.Task.CODE_RETRIEVAL, "query", None) == "task: code retrieval | query: "
    assert prefix(eg.Task.SEARCH, "document", None) == "title: none | text: "
    assert prefix(eg.Task.SEARCH, "document", "Shoe A") == "title: Shoe A | text: "
    assert prefix(eg.Task.SIMILARITY, "symmetric", None) == "task: sentence similarity | query: "


def test_task_sides():
    assert eg.Task.SEARCH.is_asymmetric
    assert not eg.Task.CLUSTERING.is_asymmetric
    engine = eg.EmbeddingEngine()
    with pytest.raises(eg.TaskError):
        engine.embed(["x"], task="search")
    with pytest.raises(eg.TaskError):
        engine.embed_queries(["x"], task="clustering")
    with pytest.raises(eg.TaskError):
        engine.embed(["x"], task="nope")
    with pytest.raises(eg.TaskError):
        engine.count_tokens(["x"], task="search")
    assert not engine.is_loaded


# ---------------------------------------------------------------- items


def test_item_validation():
    with pytest.raises(eg.InputError):
        eg.Item.text("")
    with pytest.raises(eg.InputError):
        eg.Item.text("see <|image|> here")
    with pytest.raises(eg.InputError):
        eg.Item.of()
    with pytest.raises(eg.InputError):
        eg.Item.of("text", 42)
    with pytest.raises(eg.InputError):
        eg.Audio(np.zeros(10))  # no sampling rate
    with pytest.raises(eg.InputError):
        eg.Audio("a.mp3", sampling_rate=16000)
    with pytest.raises(eg.InputError):
        eg.Video("a.mp4", overflow="drop")


def test_strings_are_always_text():
    item = eg.EmbeddingEngine._items(["photo.jpg"])[0]
    assert item.parts == (eg.Text("photo.jpg"),)


def test_single_item_is_rejected():
    with pytest.raises(eg.InputError):
        eg.EmbeddingEngine().embed("hello", task="clustering")


def test_item_modalities():
    item = eg.Item.of("caption", eg.Image("x.jpg"), eg.Video("v.mp4", include_audio=True))
    assert item.modalities == {"text", "image", "video", "audio"}


# ---------------------------------------------------------------- message building


def _content(item, task="search", side="document"):
    return _build.build_message(item, eg.Task(task), side)[0]["content"]


def test_text_gets_prefix():
    content = _content(eg.Item.text("hello", title="T"))
    assert content == [{"type": "text", "text": "title: T | text: "},
                       {"type": "text", "text": "hello"}]


def test_media_only_gets_no_prefix():
    img = str(ASSETS / "images" / "kitchen.jpg")
    assert _content(eg.Item.image(img)) == [{"type": "image", "url": img}]
    assert _content(eg.Item.image(img), task="clustering", side="symmetric") == [
        {"type": "image", "url": img}]


def test_media_with_title_gets_prefix():
    img = str(ASSETS / "images" / "kitchen.jpg")
    content = _content(eg.Item.image(img, title="Oven"))
    assert content[0] == {"type": "text", "text": "title: Oven | text: "}


def test_mixed_item_keeps_order():
    img = str(ASSETS / "images" / "kitchen.jpg")
    content = _content(eg.Item.of("before", eg.Image(img), "after"), side="query")
    assert [c["type"] for c in content] == ["text", "text", "image", "text"]
    assert content[0]["text"] == "task: search result | query: "


# ---------------------------------------------------------------- validation before loading


def _validate(item, side="document", config=None):
    _build.validate(item, 0, eg.Task.SEARCH, side, config or eg.EngineConfig())


def test_long_video_raises_unless_subsample():
    long = ASSETS / "videos" / "30s_vid.mp4"
    with pytest.raises(eg.FrameLimitError):
        _validate(eg.Item.video(long))
    _validate(eg.Item.video(long, overflow="subsample"))
    _validate(eg.Item.video(long), config=eg.EngineConfig(video_max_frames=40))


def test_missing_files():
    with pytest.raises(eg.MediaError):
        _validate(eg.Item.image("missing.jpg"))
    with pytest.raises(eg.MediaError):
        _validate(eg.Item.audio("missing.mp3"))
    with pytest.raises(eg.MediaError):
        _validate(eg.Item.video("missing.mp4"))


def test_modality_not_loaded():
    with pytest.raises(eg.ModalityNotLoadedError):
        _validate(eg.Item.image(ASSETS / "images" / "kitchen.jpg"),
                  config=eg.EngineConfig(modalities=("text",)))
    with pytest.raises(eg.ModalityNotLoadedError):
        _validate(eg.Item.video(ASSETS / "videos" / "04_coffee.mp4", include_audio=True),
                  config=eg.EngineConfig(modalities=("text", "video")))


def test_title_only_on_documents():
    with pytest.raises(eg.TaskError):
        _validate(eg.Item.text("x", title="t"), side="query")


# ---------------------------------------------------------------- media


def test_video_frames_floor():
    assert _media.video_frames(10.29, 1.0) == 10
    assert _media.video_frames(9.58, 1.0) == 9
    assert _media.video_frames(0.4, 1.0) == 1
    assert _media.video_frames(10.0, 2.0) == 20


def test_audio_array_downmix_and_resample():
    stereo = np.stack([np.ones(44_100), -np.ones(44_100)]).astype(np.float32)
    mono = _media.audio_from_array(stereo, 44_100)
    assert mono.ndim == 1 and abs(len(mono) - 16_000) <= 1
    assert np.allclose(mono, 0.0, atol=1e-5)
    same = _media.audio_from_array(np.ones(16_000, np.float32), 16_000)
    assert len(same) == 16_000


@pytest.mark.parametrize("bad", [np.zeros((16_000, 2)), np.zeros((2, 2, 2)), np.zeros(0),
                                 np.array([np.nan, 1.0])])
def test_audio_array_rejects(bad):
    with pytest.raises(eg.InputError):
        _media.audio_from_array(bad, 16_000)


def test_audio_file_decodes_to_mono_16k():
    wave = _media.load_audio_file(ASSETS / "audio" / "voicebosch-soda-can-opening-167415.mp3")
    assert wave.ndim == 1 and wave.dtype == np.float32
    assert abs(len(wave) / 16_000 - 1.15) < 0.05


def test_video_without_audio_track():
    with pytest.raises(eg.MediaError, match="no audio stream"):
        _media.load_audio_file(ASSETS / "videos" / "05_juggle.mp4")


# ---------------------------------------------------------------- ops


def _emb(vectors, task="search", side="query", dim=None, revision="r"):
    v = np.asarray(vectors, dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return eg.Embeddings(vectors=v, task=eg.Task(task), side=side, dim=dim or v.shape[1],
                         model="m", revision=revision)


def test_similarity_and_top_k():
    q = _emb([[1, 0], [0, 1]])
    d = _emb([[0, 1], [1, 0], [1, 1]], side="document")
    assert np.allclose(eg.similarity(q, d)[0], [0, 1, 2 ** -0.5])
    hits = eg.top_k(q, d, k=2)
    assert [h.index for h in hits[0]] == [1, 2]
    assert [h.index for h in hits[1]] == [0, 2]
    with pytest.raises(eg.InputError):
        eg.top_k(q, d, k=4)


@pytest.mark.parametrize("other", [
    {"task": "code_retrieval"}, {"revision": "other"},
])
def test_incompatible(other):
    a = _emb([[1, 0]])
    b = _emb([[1, 0]], **other)
    with pytest.raises(eg.IncompatibleEmbeddingsError):
        eg.similarity(a, b)


def test_incompatible_dim():
    with pytest.raises(eg.IncompatibleEmbeddingsError):
        eg.similarity(_emb([[1, 0]]), _emb([[1, 0, 0]]))


def test_near_duplicates():
    e = _emb([[1, 0], [0.99, 0.01], [0, 1]], task="clustering", side="symmetric")
    pairs = eg.near_duplicates(e, threshold=0.95)
    assert [(p.i, p.j) for p in pairs] == [(0, 1)]


def test_embeddings_json_roundtrip():
    e = _emb([[1, 0], [0, 1]])
    back = eg.Embeddings.from_json(e.to_json())
    assert np.allclose(back.vectors, e.vectors)
    assert (back.task, back.side, back.dim, back.revision) == (e.task, e.side, e.dim, e.revision)


# ---------------------------------------------------------------- preview


def test_preview_text():
    assert eg.preview("brew coffee", task="search", side="query") == (
        "task: search result | query: brew coffee")
    assert eg.preview(eg.Item.text("Bloom first.", title="Pour-over"), task="search",
                      side="document") == "title: Pour-over | text: Bloom first."
    assert eg.preview("Bloom first.", task="search", side="document") == (
        "title: none | text: Bloom first.")
    assert eg.preview("hi", task="clustering") == "task: clustering | query: hi"


def test_preview_media():
    img = ASSETS / "images" / "kitchen.jpg"
    assert eg.preview(eg.Item.image(img), task="search", side="document") == "<|image|>"
    assert eg.preview(eg.Item.image(img, title="Oven"), task="search", side="document") == (
        "title: Oven | text: <|image|>")
    item = eg.Item.of("A", eg.Image(img), "B", eg.Video("v.mp4", include_audio=True))
    assert eg.preview(item, task="similarity") == (
        "task: sentence similarity | query: A<|image|>B<|video|><|audio|>")


def test_preview_checks_task_and_side():
    with pytest.raises(eg.TaskError):
        eg.preview("x", task="search")
    with pytest.raises(eg.TaskError):
        eg.preview("x", task="clustering", side="query")
    with pytest.raises(eg.TaskError):
        eg.preview(eg.Item.text("x", title="t"), task="search", side="query")
