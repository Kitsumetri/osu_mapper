"""Audio feature extraction (log-mel spectrogram) aligned to signal frames."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from ..config import AUDIO, AudioConfig


def load_audio(path: str | Path, cfg: AudioConfig = AUDIO) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=cfg.sample_rate, mono=True)
    return y


def log_mel(y: np.ndarray, cfg: AudioConfig = AUDIO) -> np.ndarray:
    """Return (n_mels, T) log-mel spectrogram, per-feature normalised."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # normalise roughly to ~[-1, 1]; dB range is about [-80, 0]
    logmel = (logmel + 40.0) / 40.0
    return logmel


def audio_to_mel(path: str | Path, cfg: AudioConfig = AUDIO) -> np.ndarray:
    return log_mel(load_audio(path, cfg), cfg)
