"""Aggregate pattern metrics over real maps into reference distributions.

Crawls the osu! library, computes `metrics.compute_metrics` per std-mode
difficulty, and buckets by **star rating** (the official lazer difficulty, via
rosu-pp — see `difficulty.py`), reporting mean / std / p10 / p90 per metric.
The output is the "what real maps look like" baseline to compare generated maps
against. Star rating is used because mappers' difficulty *names* are arbitrary.

  python -m src.corpus_stats --songs "C:/osu!/Songs"      # all maps
  python -m src.corpus_stats --songs "C:/osu!/Songs" --limit 400
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .difficulty import SR_BUCKET_ORDER, sr_bucket, star_rating
from .metrics import compute_metrics
from .parsing.beatmap import parse_beatmap

# metrics to summarise (numeric, comparable across maps). star_rating is added
# per map and bucketed on, but also summarised so each bucket shows its SR range.
KEYS = ["star_rating", "density_per_s", "circle_ratio", "slider_ratio",
        "spinner_ratio", "bezier_slider_ratio", "new_combo_ratio",
        "mean_spacing_px", "std_spacing_px", "stream_ratio", "jump_ratio",
        "on_quarter_grid_ratio", "mean_turn_angle_deg", "reversal_ratio",
        "sv_changes_per_min"]


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


def collect(songs_dir: Path, limit: int | None = None, seed: int = 0,
            progress_every: int = 2000) -> dict:
    """Compute metrics over (a sample of) the library, bucketed by star rating.

    ``limit=None`` processes every .osu in the library.
    """
    files = list(songs_dir.rglob("*.osu"))
    random.seed(seed)
    random.shuffle(files)

    buckets: dict[str, dict[str, list[float]]] = {}
    n_used = n_seen = 0
    for f in files:
        if limit is not None and n_used >= limit:
            break
        n_seen += 1
        if n_seen % progress_every == 0:
            print(f"  scanned {n_seen}/{len(files)} files, used {n_used} std maps", flush=True)
        sr = star_rating(f)            # also filters to std mode
        if sr is None:
            continue
        try:
            bm = parse_beatmap(f)
        except Exception:
            continue
        if bm.mode != 0 or len(bm.hit_objects) < 50:
            continue
        m = compute_metrics(bm)
        if m.get("n_objects", 0) < 50:
            continue
        m["star_rating"] = round(sr, 3)
        store = buckets.setdefault(sr_bucket(sr), {k: [] for k in KEYS})
        for k in KEYS:
            if k in m:
                store[k].append(m[k])
        n_used += 1

    out = {"n_maps": n_used, "bucket_by": "star_rating", "buckets": {}}
    for b, store in buckets.items():
        out["buckets"][b] = {k: _summary(v) for k, v in store.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--limit", type=int, default=None,
                    help="max std maps to use (default: all)")
    ap.add_argument("--out", default="artifacts/reference_stats.json")
    args = ap.parse_args()
    stats = collect(Path(args.songs), args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nprocessed {stats['n_maps']} std maps (bucketed by star rating) -> {out}")
    for b in SR_BUCKET_ORDER:
        if b not in stats["buckets"]:
            continue
        s = stats["buckets"][b]
        sr = s["star_rating"]
        print(f"\n[{b}] (n={sr['n']}, SR {sr['mean']} +/- {sr['std']})")
        for k in ("density_per_s", "on_quarter_grid_ratio", "stream_ratio",
                  "jump_ratio", "reversal_ratio", "bezier_slider_ratio",
                  "sv_changes_per_min"):
            if s.get(k):
                print(f"  {k:24} mean {s[k]['mean']:>7} +/- {s[k]['std']:<7} "
                      f"(p10 {s[k]['p10']}, p90 {s[k]['p90']})")


if __name__ == "__main__":
    main()
