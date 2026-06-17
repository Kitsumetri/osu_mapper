"""PyTorch dataset over a manifest-indexed, mel-deduped processed dataset.

Reads ``<dir>/manifest.json`` and loads each item's signal (``items/<id>.npz``)
together with its shared mel (``mels/<audio_id>.npy``), with an LRU cache so the
mel is decoded from disk once per audio even though many difficulties share it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..conditioning import context_from_manifest
from ..config import (
    CH_CORNER,
    CH_CURVE,
    CH_CURX,
    CH_CURY,
    CH_SLIDER_ANCHORS,
    CH_SLIDES,
    CH_SV,
    N_SIGNAL_CHANNELS,
    N_SLIDER_ANCHORS,
)


@lru_cache(maxsize=256)
def _load_mel_cached(path_str: str) -> np.ndarray:
    """Module-level LRU cache (picklable; one cache per DataLoader worker)."""
    return np.load(path_str).astype(np.float32)


class OsuSignalDataset(Dataset):
    def __init__(self, processed_dir: str | Path, crop_frames: int = 1024,
                 min_objects: int = 0, augment: bool = False):
        self.dir = Path(processed_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json in {self.dir}")
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.items = [it for it in items if it["n_objects"] >= min_objects]
        if not self.items:
            raise FileNotFoundError(f"no items in {manifest_path} (min_objects={min_objects})")
        self.crop = crop_frames
        # playfield-mirror augmentation: osu!std is symmetric under horizontal /
        # vertical flips, so mirroring the cursor channels yields a valid alternate
        # map for the same audio. Free 4x spatial variety, train-time only.
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = _load_mel_cached(str(self.dir / "mels" / f"{it['audio_id']}.npy"))  # (n_mels, T)
        sig = np.load(self.dir / "items" / f"{it['item_id']}.npz")["signal"].astype(np.float32)
        # pad the channel dim for data preprocessed before a channel was added (e.g. the
        # v7 SV channel): new trailing channels have baseline 0 (SV 1.0), so zero-pad.
        if sig.shape[0] < N_SIGNAL_CHANNELS:
            extra = np.zeros((N_SIGNAL_CHANNELS - sig.shape[0], sig.shape[1]), dtype=np.float32)
            sig = np.concatenate([sig, extra], axis=0)
        T = min(mel.shape[1], sig.shape[1])
        mel, sig = mel[:, :T], sig[:, :T]
        L = self.crop
        if T >= L:
            start = np.random.randint(0, T - L + 1)
            mel, sig = mel[:, start:start + L], sig[:, start:start + L]
        else:
            pad = L - T
            # pad mel with -1.0 = true silence (after audio.py's (dB+40)/40 norm);
            # the np.pad default 0.0 is ~-40 dB (audible) -> teaches "audio -> empty map".
            mel = np.pad(mel, ((0, 0), (0, pad)), constant_values=-1.0)
            sigpad = np.full((sig.shape[0], pad), -1.0, dtype=np.float32)
            sigpad[CH_CURX:CH_CURY + 1] = 0.0  # cursor centre
            if sig.shape[0] > CH_SLIDES:
                sigpad[CH_SLIDER_ANCHORS:CH_SLIDES] = 0.0  # anchor offsets baseline 0
            sigpad[CH_SV] = 0.0  # SV baseline = SV 1.0
            sigpad[CH_CURVE] = 0.0  # curvature baseline = straight
            sigpad[CH_CORNER] = 0.0  # corner baseline = smooth
            sig = np.concatenate([sig, sigpad], axis=1)
        if self.augment:
            # cursor + slider-anchor channels are normalised so playfield centre is
            # 0; negating mirrors positions about the centre. Flipping must hit the
            # anchor dx/dy channels too, or slider geometry is corrupted.
            sig = sig.copy()
            has_anchors = sig.shape[0] > CH_SLIDES
            if np.random.rand() < 0.5:  # horizontal flip -> negate all x channels
                sig[CH_CURX] = -sig[CH_CURX]
                if has_anchors:
                    for i in range(N_SLIDER_ANCHORS):
                        sig[CH_SLIDER_ANCHORS + 2 * i] = -sig[CH_SLIDER_ANCHORS + 2 * i]
            if np.random.rand() < 0.5:  # vertical flip -> negate all y channels
                sig[CH_CURY] = -sig[CH_CURY]
                if has_anchors:
                    for i in range(N_SLIDER_ANCHORS):
                        sig[CH_SLIDER_ANCHORS + 2 * i + 1] = -sig[CH_SLIDER_ANCHORS + 2 * i + 1]
        ctx = torch.tensor(context_from_manifest(it), dtype=torch.float32)
        return torch.from_numpy(sig), torch.from_numpy(mel), ctx
