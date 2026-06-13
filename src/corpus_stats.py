"""Aggregate pattern metrics over real maps into reference distributions.

Crawls a sample of the osu! library, computes `metrics.compute_metrics` per
std-mode difficulty, buckets by object density (a coarse difficulty proxy), and
reports mean / std / p10 / p90 per metric. The output is the "what real maps look
like" baseline to compare generated maps against.

  python -m src.corpus_stats --songs "C:/osu!/Songs" --limit 400 \
      --out artifacts/reference_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .metrics import compute_metrics
from .parsing.beatmap import parse_beatmap

# density (objects/sec) -> coarse difficulty bucket
DENSITY_BINS = [(0, 2, "Easy"), (2, 3, "Normal"), (3, 4.5, "Hard"),
                (4.5, 6, "Insane"), (6, 99, "Extra")]

# metrics to summarise (numeric, comparable across maps)
KEYS = ["density_per_s", "circle_ratio", "slider_ratio", "spinner_ratio",
        "bezier_slider_ratio", "new_combo_ratio", "mean_spacing_px",
        "std_spacing_px", "stream_ratio", "jump_ratio", "on_quarter_grid_ratio",
        "mean_turn_angle_deg", "reversal_ratio", "sv_changes_per_min"]


def _bucket(density: float) -> str:
    for lo, hi, name in DENSITY_BINS:
        if lo <= density < hi:
            return name
    return "Extra"


def _summary(values: list[float]) -> dict:
    if not values:
        return {}
    vs = sorted(values)
    n = len(vs)
    mean = sum(vs) / n
    std = (sum((v - mean) ** 2 for v in vs) / n) ** 0.5
    return {"n": n, "mean": round(mean, 3), "std": round(std, 3),
            "p10": round(vs[int(0.1 * (n - 1))], 3),
            "p90": round(vs[int(0.9 * (n - 1))], 3)}


def collect(songs_dir: Path, limit: int, seed: int = 0) -> dict:
    files = []
    for p in songs_dir.rglob("*.osu"):
        files.append(p)
        if len(files) >= max(limit * 8, 4000):
            break
    random.seed(seed)
    random.shuffle(files)

    buckets: dict[str, dict[str, list[float]]] = {}
    n_used = 0
    for f in files:
        if n_used >= limit:
            break
        try:
            bm = parse_beatmap(f)
        except Exception:
            continue
        if bm.mode != 0 or len(bm.hit_objects) < 50:
            continue
        m = compute_metrics(bm)
        if m.get("n_objects", 0) < 50:
            continue
        b = _bucket(m["density_per_s"])
        store = buckets.setdefault(b, {k: [] for k in KEYS})
        for k in KEYS:
            if k in m:
                store[k].append(m[k])
        n_used += 1

    out = {"n_maps": n_used, "buckets": {}}
    for b, store in buckets.items():
        out["buckets"][b] = {k: _summary(v) for k, v in store.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default="artifacts/reference_stats.json")
    args = ap.parse_args()
    stats = collect(Path(args.songs), args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"sampled {stats['n_maps']} std maps -> {out}")
    for b in ("Normal", "Hard", "Insane", "Extra"):
        if b not in stats["buckets"]:
            continue
        s = stats["buckets"][b]
        print(f"\n[{b}] (n={s['density_per_s']['n']})")
        for k in ("density_per_s", "on_quarter_grid_ratio", "stream_ratio",
                  "jump_ratio", "reversal_ratio", "bezier_slider_ratio"):
            if s.get(k):
                print(f"  {k:24} mean {s[k]['mean']:>6} ± {s[k]['std']:<6} "
                      f"(p10 {s[k]['p10']}, p90 {s[k]['p90']})")


if __name__ == "__main__":
    main()
