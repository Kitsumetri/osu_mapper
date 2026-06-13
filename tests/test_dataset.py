"""Tests for the manifest-indexed, mel-deduped dataset."""
import json

import numpy as np
import torch

from src.config import AUDIO, N_SIGNAL_CHANNELS
from src.data.dataset import OsuSignalDataset


def _make_dataset(root, items):
    """items: list of (item_id, audio_id, T, n_objects)."""
    (root / "mels").mkdir(parents=True, exist_ok=True)
    (root / "items").mkdir(parents=True, exist_ok=True)
    manifest = []
    seen_audio = set()
    for item_id, audio_id, T, n_obj in items:
        if audio_id not in seen_audio:
            mel = np.random.rand(AUDIO.n_mels, T).astype(np.float16)
            np.save(root / "mels" / f"{audio_id}.npy", mel)
            seen_audio.add(audio_id)
        sig = np.full((N_SIGNAL_CHANNELS, T), -1.0, dtype=np.float16)
        sig[4:6] = 0.0
        np.savez_compressed(root / "items" / f"{item_id}.npz", signal=sig)
        manifest.append({"item_id": item_id, "audio_id": audio_id, "n_objects": n_obj})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dataset_crops_long_sample(tmp_path):
    _make_dataset(tmp_path, [("a", "aud0", 4000, 300)])
    ds = OsuSignalDataset(tmp_path, crop_frames=1024)
    sig, mel = ds[0]
    assert sig.shape == (N_SIGNAL_CHANNELS, 1024)
    assert mel.shape == (AUDIO.n_mels, 1024)
    assert sig.dtype == torch.float32 and mel.dtype == torch.float32


def test_dataset_pads_short_sample(tmp_path):
    _make_dataset(tmp_path, [("b", "aud0", 300, 100)])
    ds = OsuSignalDataset(tmp_path, crop_frames=1024)
    sig, mel = ds[0]
    assert sig.shape[1] == 1024 and mel.shape[1] == 1024
    assert float(sig[0, -1]) == -1.0   # padded onset stays at baseline
    assert float(sig[4, -1]) == 0.0    # padded cursor centred


def test_dataset_shared_mel_dedup(tmp_path):
    # two difficulties share one audio_id -> only one mel file
    _make_dataset(tmp_path, [("a", "aud0", 2000, 200), ("b", "aud0", 2000, 200)])
    ds = OsuSignalDataset(tmp_path, crop_frames=512)
    assert len(ds) == 2
    assert len(list((tmp_path / "mels").glob("*.npy"))) == 1
    _ = ds[0], ds[1]  # both load without error


def test_dataset_min_objects_filter_and_missing(tmp_path):
    _make_dataset(tmp_path, [("a", "aud0", 2000, 40), ("b", "aud1", 2000, 300)])
    assert len(OsuSignalDataset(tmp_path, crop_frames=512, min_objects=50)) == 1
    import pytest
    with pytest.raises(FileNotFoundError):
        OsuSignalDataset(tmp_path / "nope", crop_frames=512)
