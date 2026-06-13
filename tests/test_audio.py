"""Tests for log-mel feature extraction. Uses a synthetic waveform (no files)."""

import numpy as np

from src.config import AUDIO
from src.data.audio import log_mel


def _sine(seconds=2.0, freq=440.0):
    t = np.linspace(0, seconds, int(AUDIO.sample_rate * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_log_mel_shape():
    y = _sine(2.0)
    mel = log_mel(y)
    assert mel.shape[0] == AUDIO.n_mels
    # frame count ~= samples / hop_length
    expected = len(y) // AUDIO.hop_length
    assert abs(mel.shape[1] - expected) <= 1
    assert mel.dtype == np.float32


def test_log_mel_normalised_range():
    mel = log_mel(_sine(1.0))
    # normalisation maps dB (~[-80,0]) to roughly [-1, 1]; allow some slack
    assert mel.min() >= -2.5 and mel.max() <= 1.5


def test_log_mel_frame_alignment_with_config():
    """Mel frame count should match config.time_to_frame for the clip length."""
    seconds = 3.0
    mel = log_mel(_sine(seconds))
    ms = seconds * 1000
    assert abs(mel.shape[1] - AUDIO.time_to_frame(ms)) <= 2
