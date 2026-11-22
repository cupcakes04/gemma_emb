"""Media loading. Audio always ends up as float32 mono 16 kHz numpy; videos are only probed here
(the processor decodes frames itself)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .errors import InputError, MediaError

SAMPLE_RATE = 16_000
MAX_CHANNELS = 8


def _require_file(path: Path, kind: str) -> None:
    if not path.is_file():
        raise MediaError(f"{kind} file not found: {path}")


def load_audio_file(path: Path) -> np.ndarray:
    """Decode the (first) audio stream of any media file to mono 16 kHz."""
    from torchcodec.decoders import AudioDecoder

    _require_file(path, "Audio")
    try:
        decoder = AudioDecoder(str(path), sample_rate=SAMPLE_RATE, num_channels=1)
        samples = decoder.get_all_samples()
    except Exception as ex:  # torchcodec raises ValueError/RuntimeError with opaque messages
        msg = str(ex)
        if "audio stream" in msg.lower() or "best audio stream" in msg.lower():
            raise MediaError(f"{path} has no audio stream.") from ex
        raise MediaError(f"Cannot decode audio from {path}: {msg.splitlines()[0]}") from ex
    wave = samples.data[0].numpy().astype(np.float32, copy=False)
    if wave.size == 0:
        raise MediaError(f"{path} decoded to zero audio samples.")
    return wave


def audio_from_array(array, sampling_rate: int) -> np.ndarray:
    """Downmix (channels, samples) to mono and resample to 16 kHz."""
    wave = np.asarray(array.detach().cpu() if hasattr(array, "detach") else array, dtype=np.float32)
    if wave.ndim == 2:
        if wave.shape[0] > MAX_CHANNELS:
            raise InputError(
                f"Audio array has shape {wave.shape}; expected (samples,) or (channels, samples) "
                f"with at most {MAX_CHANNELS} channels. Transpose (samples, channels) arrays first."
            )
        wave = wave.mean(axis=0)
    elif wave.ndim != 1:
        raise InputError(f"Audio array must be 1-D or 2-D, got shape {wave.shape}.")
    if wave.size == 0:
        raise InputError("Audio array is empty.")
    if not np.isfinite(wave).all():
        raise InputError("Audio array contains NaN or inf.")
    if sampling_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = math.gcd(int(sampling_rate), SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // g, int(sampling_rate) // g).astype(np.float32)
    return wave


def video_duration(path: Path) -> float:
    from torchcodec.decoders import VideoDecoder

    _require_file(path, "Video")
    try:
        meta = VideoDecoder(str(path), seek_mode="approximate").metadata
    except Exception as ex:
        raise MediaError(f"Cannot open video {path}: {str(ex).splitlines()[0]}") from ex
    duration = meta.duration_seconds
    if duration is None:
        if meta.num_frames and meta.average_fps:
            duration = meta.num_frames / meta.average_fps
        else:
            raise MediaError(f"Cannot determine the duration of {path}.")
    return float(duration)


def video_frames(duration: float, fps: float) -> int:
    """Frames the processor samples: floor(duration * fps), at least 1 (measured, model-facts.md)."""
    return max(1, math.floor(duration * fps + 1e-9))
