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


@lru_cache(maxsize=256)
def _load_mel_cached(path_str: str) -> np.ndarray:
    """Module-level LRU cache (picklable; one cache per DataLoader worker)."""
    return np.load(path_str).astype(np.float32)


class OsuSignalDataset(Dataset):
    def __init__(self, processed_dir: str | Path, crop_frames: int = 1024,
                 min_objects: int = 0):
        self.dir = Path(processed_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json in {self.dir}")
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.items = [it for it in items if it["n_objects"] >= min_objects]
        if not self.items:
            raise FileNotFoundError(f"no items in {manifest_path} (min_objects={min_objects})")
        self.crop = crop_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = _load_mel_cached(str(self.dir / "mels" / f"{it['audio_id']}.npy"))  # (n_mels, T)
        sig = np.load(self.dir / "items" / f"{it['item_id']}.npz")["signal"].astype(np.float32)
        T = min(mel.shape[1], sig.shape[1])
        mel, sig = mel[:, :T], sig[:, :T]
        L = self.crop
        if T >= L:
            start = np.random.randint(0, T - L + 1)
            mel, sig = mel[:, start:start + L], sig[:, start:start + L]
        else:
            pad = L - T
            mel = np.pad(mel, ((0, 0), (0, pad)))
            sigpad = np.full((sig.shape[0], pad), -1.0, dtype=np.float32)
            sigpad[4:6] = 0.0  # cursor centre
            sig = np.concatenate([sig, sigpad], axis=1)
        return torch.from_numpy(sig), torch.from_numpy(mel)
