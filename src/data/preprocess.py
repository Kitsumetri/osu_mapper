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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import AUDIO
from ..difficulty import star_rating
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


def _process_set(args):
    """Process one beatmap set (runs in a worker process). Decodes each audio
    once, writes mel .npy + per-difficulty signal .npz, returns manifest rows +
    counts. Audio decoding (the bottleneck) parallelises cleanly across sets."""
    set_dir, osu_paths, mels_dir, items_dir, min_objects, max_seconds, max_sr = args
    entries: list[dict] = []
    skipped = errors = mels = 0
    cap = int(AUDIO.time_to_frame(max_seconds * 1000))

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
        try:
            mel = audio_to_mel(audio_path)          # the expensive step
        except Exception:
            errors += 1
            continue
        T = mel.shape[1]
        if cap < T:
            mel, T = mel[:, :cap], cap
        np.save(mels_dir / f"{aid}.npy", mel.astype(np.float16))
        mels += 1
        for bm in bms:
            sr = star_rating(bm.path)               # curation
            if sr is not None and sr > max_sr:
                skipped += 1
                continue
            try:
                sig = encode_beatmap(bm, T).astype(np.float16)
            except Exception:
                errors += 1
                continue
            item_id = _safe(f"{set_dir.name}__{bm.version}").replace(" ", "_")
            np.savez_compressed(items_dir / f"{item_id}.npz", signal=sig)
            entries.append({
                "item_id": item_id, "audio_id": aid,
                "creator": bm.creator, "title": bm.title, "version": bm.version,
                "n_objects": len(bm.hit_objects),
                "star_rating": round(sr, 3) if sr is not None else 0.0,
                "cs": bm.circle_size, "ar": bm.approach_rate,
                "od": bm.overall_difficulty, "hp": bm.hp,
                "slider_multiplier": bm.slider_multiplier,
                "bpm": bm.bpm, "n_timing_points": len(bm.timing_points),
                "has_kiai": len(bm.kiai_spans()) > 0,
                "duration_s": round(AUDIO.frame_to_time(T) / 1000, 1),
                "frames": T,
            })
    return entries, skipped, errors, mels


def process_library(songs_dir: Path, out_dir: Path, limit: int | None = None,
                    min_objects: int = 50, max_seconds: float = 240.0,
                    max_sr: float = 12.0, workers: int | None = None):
    mels_dir = out_dir / "mels"
    items_dir = out_dir / "items"
    mels_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)

    sets = list(find_sets(songs_dir))
    tasks = [(sd, ops, mels_dir, items_dir, min_objects, max_seconds, max_sr)
             for sd, ops in sets]

    manifest: list[dict] = []
    skipped = errors = mels_cached = 0

    def _collect(res):
        nonlocal skipped, errors, mels_cached
        e, s, er, m = res
        manifest.extend(e)
        skipped += s
        errors += er
        mels_cached += m

    if workers <= 1:
        for t in tqdm(tasks, desc="sets"):
            _collect(_process_set(t))
            if limit is not None and len(manifest) >= limit:
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_set, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"sets x{workers}"):
                _collect(f.result())
                if limit is not None and len(manifest) >= limit:
                    for fut in futs:
                        fut.cancel()
                    break

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=0), encoding="utf-8")
    print(f"\nDone. items={len(manifest)} mels={mels_cached} skipped={skipped} errors={errors}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    return len(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--out", default="data/processed/std-v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-objects", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=240.0)
    ap.add_argument("--max-sr", type=float, default=12.0,
                    help="skip maps with star rating above this (junk/joke maps)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: cpu_count-1)")
    args = ap.parse_args()
    process_library(Path(args.songs), Path(args.out), args.limit,
                    args.min_objects, args.max_seconds, args.max_sr, args.workers)


if __name__ == "__main__":
    main()
