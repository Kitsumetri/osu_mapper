"""Crawl the osu! Songs library into a deduped, manifest-indexed dataset.

Layout (see STORAGE.md):
  <out>/mels/<audio_id>.npy    log-mel per *audio file* (shared by difficulties)
  <out>/items/<item_id>.npz    signal (float16) per difficulty
  <out>/manifest.json          index: every item + metadata for filtering/stats

Metadata captured per item (future-proofs difficulty/style/kiai conditioning):
  creator, title, version, n_objects, cs/ar/od/hp, bpm, n_timing_points,
  has_kiai, duration_s, frames, audio_id.

Usage:
  python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/std-v1 --limit 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import AUDIO
from ..parsing.beatmap import parse_beatmap
from .audio import audio_to_mel
from .signal import encode_beatmap


def _audio_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).lower().encode()).hexdigest()[:16]


def _safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "._- ")[:120]


def find_sets(songs_dir: Path):
    for d in sorted(p for p in songs_dir.iterdir() if p.is_dir()):
        osus = list(d.glob("*.osu"))
        if osus:
            yield d, osus


def process_library(songs_dir: Path, out_dir: Path, limit: int | None = None,
                    min_objects: int = 50, max_seconds: float = 240.0):
    mels_dir = out_dir / "mels"
    items_dir = out_dir / "items"
    mels_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    written = skipped = errors = mels_cached = 0
    mel_T: dict[str, int] = {}

    sets = list(find_sets(songs_dir))
    pbar = tqdm(sets, desc="sets")
    for set_dir, osu_paths in pbar:
        by_audio: dict[str, list] = {}
        for op in osu_paths:
            try:
                bm = parse_beatmap(op)
            except Exception:
                errors += 1
                continue
            if bm.mode != 0 or len(bm.hit_objects) < min_objects:
                skipped += 1
                continue
            by_audio.setdefault(bm.audio_filename.lower(), []).append(bm)

        for bms in by_audio.values():
            audio_path = bms[0].audio_path
            if not audio_path.exists():
                skipped += len(bms)
                continue
            aid = _audio_id(audio_path)
            mel_npy = mels_dir / f"{aid}.npy"
            if aid in mel_T:
                T = mel_T[aid]
            else:
                try:
                    mel = audio_to_mel(audio_path)
                except Exception:
                    errors += 1
                    continue
                T = mel.shape[1]
                cap = int(AUDIO.time_to_frame(max_seconds * 1000))
                if cap < T:
                    mel, T = mel[:, :cap], cap
                np.save(mel_npy, mel.astype(np.float16))
                mel_T[aid] = T
                mels_cached += 1

            for bm in bms:
                try:
                    sig = encode_beatmap(bm, T).astype(np.float16)
                except Exception:
                    errors += 1
                    continue
                item_id = _safe(f"{set_dir.name}__{bm.version}").replace(" ", "_")
                np.savez_compressed(items_dir / f"{item_id}.npz", signal=sig)
                manifest.append({
                    "item_id": item_id, "audio_id": aid,
                    "creator": bm.creator, "title": bm.title, "version": bm.version,
                    "n_objects": len(bm.hit_objects),
                    "cs": bm.circle_size, "ar": bm.approach_rate,
                    "od": bm.overall_difficulty, "hp": bm.hp,
                    "slider_multiplier": bm.slider_multiplier,
                    "bpm": bm.bpm, "n_timing_points": len(bm.timing_points),
                    "has_kiai": len(bm.kiai_spans()) > 0,
                    "duration_s": round(AUDIO.frame_to_time(T) / 1000, 1),
                    "frames": T,
                })
                written += 1
            pbar.set_postfix(items=written, mels=mels_cached, skip=skipped, err=errors)
        if limit is not None and written >= limit:
            break

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=0), encoding="utf-8")
    print(f"\nDone. items={written} mels={mels_cached} skipped={skipped} errors={errors}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--out", default="data/processed/std-v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-objects", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=240.0)
    args = ap.parse_args()
    process_library(Path(args.songs), Path(args.out), args.limit,
                    args.min_objects, args.max_seconds)


if __name__ == "__main__":
    main()
