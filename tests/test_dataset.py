"""Tests for the random-crop dataset over .npz shards."""

import numpy as np
import torch

from src.config import AUDIO, N_SIGNAL_CHANNELS
from src.data.dataset import OsuSignalDataset


def _make_shard(path, T):
    mel = np.random.rand(AUDIO.n_mels, T).astype(np.float16)
    sig = np.full((N_SIGNAL_CHANNELS, T), -1.0, dtype=np.float16)
    sig[4:6] = 0.0
    np.savez_compressed(
        path, mel=mel, signal=sig, title="t", version="v", n_objects=10, audio="a.mp3"
    )


def test_dataset_crops_long_sample(tmp_path):
    _make_shard(tmp_path / "a.npz", T=4000)
    ds = OsuSignalDataset(tmp_path, crop_frames=1024)
    sig, mel = ds[0]
    assert sig.shape == (N_SIGNAL_CHANNELS, 1024)
    assert mel.shape == (AUDIO.n_mels, 1024)
    assert sig.dtype == torch.float32 and mel.dtype == torch.float32


def test_dataset_pads_short_sample(tmp_path):
    _make_shard(tmp_path / "b.npz", T=300)
    ds = OsuSignalDataset(tmp_path, crop_frames=1024)
    sig, mel = ds[0]
    assert sig.shape[1] == 1024 and mel.shape[1] == 1024
    # padded onset/hold region stays at the silent baseline (-1)
    assert float(sig[0, -1]) == -1.0
    # padded cursor channels are centred (0)
    assert float(sig[4, -1]) == 0.0


def test_dataset_len_and_missing_dir(tmp_path):
    _make_shard(tmp_path / "a.npz", T=2000)
    _make_shard(tmp_path / "b.npz", T=2000)
    assert len(OsuSignalDataset(tmp_path, crop_frames=512)) == 2
    import pytest

    with pytest.raises(FileNotFoundError):
        OsuSignalDataset(tmp_path / "nope", crop_frames=512)


def test_dataset_crop_within_bounds(tmp_path):
    """Cropped region must be a real slice (values within original range)."""
    _make_shard(tmp_path / "a.npz", T=2048)
    ds = OsuSignalDataset(tmp_path, crop_frames=1024)
    for _ in range(5):
        sig, mel = ds[0]
        assert torch.isfinite(sig).all() and torch.isfinite(mel).all()
        assert mel.min() >= 0.0 and mel.max() <= 1.0
