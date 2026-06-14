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
    sig, mel, ctx = ds[0]
    assert sig.shape == (N_SIGNAL_CHANNELS, 1024)
    assert mel.shape == (AUDIO.n_mels, 1024)
    assert sig.dtype == torch.float32 and mel.dtype == torch.float32
    from src.conditioning import CONTEXT_DIM
    assert ctx.shape == (CONTEXT_DIM,)


def test_dataset_pads_short_sample(tmp_path):
    _make_dataset(tmp_path, [("b", "aud0", 300, 100)])
    ds = OsuSignalDataset(tmp_path, crop_frames=1024)
    sig, mel, ctx = ds[0]
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


def test_augment_flips_cursor_channels_only(tmp_path):
    np.random.seed(0)
    (tmp_path / "mels").mkdir(parents=True)
    (tmp_path / "items").mkdir(parents=True)
    T = 512
    np.save(tmp_path / "mels" / "aud0.npy", np.random.rand(AUDIO.n_mels, T).astype(np.float16))
    sig = np.full((N_SIGNAL_CHANNELS, T), -1.0, dtype=np.float16)
    cx = np.linspace(-1, 1, T).astype(np.float16)
    cy = np.full(T, 0.4, dtype=np.float16)
    sig[4], sig[5] = cx, cy
    sig[0, 10] = 1.0  # an onset (position-independent channel)
    np.savez_compressed(tmp_path / "items" / "a.npz", signal=sig)
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"item_id": "a", "audio_id": "aud0", "n_objects": 100}]), encoding="utf-8")

    # augment off: cursor channels untouched
    s0 = OsuSignalDataset(tmp_path, crop_frames=T, augment=False)[0][0]
    assert np.allclose(s0[4].numpy(), cx, atol=1e-2)
    assert np.allclose(s0[5].numpy(), cy, atol=1e-2)

    ds = OsuSignalDataset(tmp_path, crop_frames=T, augment=True)
    seen_x, seen_y = set(), set()
    for _ in range(60):
        s = ds[0][0]
        x, y = s[4].numpy(), s[5].numpy()
        assert np.allclose(x, cx, atol=1e-2) or np.allclose(x, -cx, atol=1e-2)
        assert np.allclose(y, cy, atol=1e-2) or np.allclose(y, -cy, atol=1e-2)
        assert float(s[0, 10]) == 1.0          # non-cursor channel never flipped
        seen_x.add(bool(np.allclose(x, -cx, atol=1e-2)))
        seen_y.add(bool(np.allclose(y, -cy, atol=1e-2)))
    assert seen_x == {True, False}             # both horizontal orientations appear
    assert seen_y == {True, False}             # both vertical orientations appear


def test_augment_flips_slider_anchor_channels(tmp_path):
    from src.config import CH_SLIDER_ANCHORS, CH_SLIDES
    np.random.seed(0)
    (tmp_path / "mels").mkdir(parents=True)
    (tmp_path / "items").mkdir(parents=True)
    T = 256
    np.save(tmp_path / "mels" / "aud0.npy", np.random.rand(AUDIO.n_mels, T).astype(np.float16))
    sig = np.full((N_SIGNAL_CHANNELS, T), -1.0, dtype=np.float16)
    sig[4:6] = 0.0
    sig[CH_SLIDER_ANCHORS:CH_SLIDES] = 0.0
    sig[CH_SLIDER_ANCHORS + 0] = 0.5   # dx1
    sig[CH_SLIDER_ANCHORS + 1] = 0.3   # dy1
    sig[CH_SLIDES] = -0.33             # slides (flip-invariant)
    np.savez_compressed(tmp_path / "items" / "a.npz", signal=sig)
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"item_id": "a", "audio_id": "aud0", "n_objects": 100}]), encoding="utf-8")

    ds = OsuSignalDataset(tmp_path, crop_frames=T, augment=True)
    seen_dx, seen_dy = set(), set()
    for _ in range(60):
        s = ds[0][0].numpy()
        dx, dy = s[CH_SLIDER_ANCHORS + 0], s[CH_SLIDER_ANCHORS + 1]
        assert np.allclose(dx, 0.5, atol=1e-2) or np.allclose(dx, -0.5, atol=1e-2)
        assert np.allclose(dy, 0.3, atol=1e-2) or np.allclose(dy, -0.3, atol=1e-2)
        assert np.allclose(s[CH_SLIDES], -0.33, atol=1e-2)   # slides never flipped
        seen_dx.add(bool(np.allclose(dx, -0.5, atol=1e-2)))
        seen_dy.add(bool(np.allclose(dy, -0.3, atol=1e-2)))
    assert seen_dx == {True, False}   # dx flips horizontally
    assert seen_dy == {True, False}   # dy flips vertically


def test_dataset_min_objects_filter_and_missing(tmp_path):
    _make_dataset(tmp_path, [("a", "aud0", 2000, 40), ("b", "aud1", 2000, 300)])
    assert len(OsuSignalDataset(tmp_path, crop_frames=512, min_objects=50)) == 1
    import pytest
    with pytest.raises(FileNotFoundError):
        OsuSignalDataset(tmp_path / "nope", crop_frames=512)
