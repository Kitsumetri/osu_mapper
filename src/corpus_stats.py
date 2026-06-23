"""Aggregate pattern metrics over real maps into reference distributions.

Crawls the osu! library, computes `metrics.compute_metrics` per std-mode
difficulty, and buckets by **star rating** (the official lazer difficulty, via
rosu-pp — see `difficulty.py`), reporting mean / std / p10 / p90 per metric.
The output is the "what real maps look like" baseline to compare generated maps
against. Star rating is used because mappers' difficulty *names* are arbitrary.

  python -m src.corpus_stats --songs "C:/osu!/Songs"                 # all maps, parallel
  python -m src.corpus_stats --songs "C:/osu!/Songs" --limit 400
  python -m src.corpus_stats --songs "C:/osu!/Songs" --workers 1     # force serial
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        # v9 flow/slider distributional traits (reward.py FAMILIES.flow /
        # .slider_shape) — registered here so a future gold-stats refresh
        # produces p10/p90 bands for them; reward.py ignores them until then.
        "stream_spacing_cv", "slider_anchor_spread_px",
        "sv_changes_per_min", "kiai_ratio", "hitsound_ratio", "clap_ratio",
        "whistle_ratio", "finish_ratio"]


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


def _process_file(path: Path) -> tuple[str, dict[str, float]] | None:
    """Parse one .osu and return ``(sr_bucket, {metric: value})`` for a usable
    std map, else ``None``. Module-level + picklable so it runs in a worker
    process (Windows ``spawn``). The per-file work — parse, the cheap mode/object
    filter, the expensive rosu SR call, metrics — is independent across files, so
    this is the unit the pool distributes.
    """
    # parse + cheap mode/object filter FIRST, then the expensive rosu SR call
    # (avoids parsing std maps twice and skips rosu on taiko/mania/ctb).
    try:
        bm = parse_beatmap(path)
    except Exception:
        return None
    if bm.mode != 0 or len(bm.hit_objects) < 50:
        return None
    sr = star_rating(path)
    if sr is None:
        return None
    m = compute_metrics(bm)
    if m.get("n_objects", 0) < 50:
        return None
    m["star_rating"] = round(sr, 3)
    return sr_bucket(sr), {k: m[k] for k in KEYS if k in m}


def collect(songs_dir: Path, limit: int | None = None, seed: int = 0,
            workers: int | None = None, progress_every: int = 2000) -> dict:
    """Compute metrics over (a sample of) the library, bucketed by star rating.

    ``limit=None`` processes every .osu in the library. ``workers`` parallel
    processes (default ``cpu_count-1``; ``workers<=1`` forces the serial path) —
    the rosu SR call dominates and parallelises cleanly. Aggregation is order-
    independent (``_summary`` sorts), so the result is identical to the serial run
    regardless of worker count or completion order.
    """
    files = list(songs_dir.rglob("*.osu"))
    random.seed(seed)
    random.shuffle(files)
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    buckets: dict[str, dict[str, list[float]]] = {}
    n_used = n_seen = 0

    def _accumulate(res: tuple[str, dict[str, float]] | None) -> None:
        nonlocal n_used
        if res is None:
            return
        bucket, vals = res
        store = buckets.setdefault(bucket, {k: [] for k in KEYS})
        for k in KEYS:
            if k in vals:
                store[k].append(vals[k])
        n_used += 1

    def _tick() -> None:
        if n_seen % progress_every == 0:
            print(f"  scanned {n_seen}/{len(files)} files, used {n_used} std maps", flush=True)

    if workers <= 1:
        for f in files:
            if limit is not None and n_used >= limit:
                break
            n_seen += 1
            _tick()
            _accumulate(_process_file(f))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_file, f) for f in files]
            for fut in as_completed(futs):
                n_seen += 1
                _tick()
                _accumulate(fut.result())
                if limit is not None and n_used >= limit:
                    for f2 in futs:
                        f2.cancel()
                    break

    out = {"n_maps": n_used, "bucket_by": "star_rating", "workers": workers, "buckets": {}}
    for b, store in buckets.items():
        out["buckets"][b] = {k: _summary(v) for k, v in store.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--limit", type=int, default=None,
                    help="max std maps to use (default: all)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: cpu_count-1; 1 = serial)")
    ap.add_argument("--out", default="artifacts/reference_stats.json")
    args = ap.parse_args()
    stats = collect(Path(args.songs), args.limit, workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nprocessed {stats['n_maps']} std maps (bucketed by star rating, "
          f"{stats.get('workers', 1)} workers) -> {out}")
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
