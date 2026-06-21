"""Measure the ranked-map reward on a sample of REAL gold maps — the calibration
sanity check for ``src/eval/reward.py``.

A *general* "is this a ranked map?" reward must score genuinely ranked maps HIGH
across every style (jump, stream, tech/slider, balanced). This tool samples K real
ranked/approved/loved std maps (via ``data/osu_db.py``; falls back to a random
sample of real maps if osu!.db is absent), scores each at its OWN rosu star rating
as the target — so ``sr_closeness`` is ~1.0 by construction and ``quality`` is the
thing under test — and reports the reward distribution overall, per SR bucket, and
per pattern family. A family that scores low on real maps is a mis-weight or a
mis-calibrated metric (note the known ``on_quarter_grid_ratio`` single-BPM caveat).

  uv run python -m src.eval.measure_reward --limit 400
  uv run python -m src.eval.measure_reward --songs "C:/osu!/Songs" \
      --db "C:/osu!/osu!.db" --ref-stats artifacts/reference_stats.json --limit 500
  uv run python -m src.eval.measure_reward --limit 400 --json out.json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..difficulty import SR_BUCKET_ORDER, star_rating
from .reward import FAMILIES, reward_from_osu


def _percentile(vs: list[float], q: float) -> float:
    if not vs:
        return 0.0
    s = sorted(vs)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def _summary(vs: list[float]) -> dict:
    if not vs:
        return {"n": 0}
    n = len(vs)
    mean = sum(vs) / n
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(_percentile(vs, 0.5), 4),
        "p10": round(_percentile(vs, 0.1), 4),
        "p90": round(_percentile(vs, 0.9), 4),
        "min": round(min(vs), 4),
        "max": round(max(vs), 4),
    }


def sample_gold_paths(songs_dir: Path, db_path: Path | None, limit: int,
                      seed: int = 0) -> list[Path]:
    """Up to ``limit`` real .osu paths, preferring osu!.db-confirmed ranked maps.

    Falls back to a random sample of every .osu in the library when osu!.db is
    missing/unreadable (callers should note the sample is then "real" not "gold").
    """
    rng = random.Random(seed)
    if db_path and Path(db_path).exists():
        try:
            from ..data.osu_db import ranked_osu_paths
            paths = list(ranked_osu_paths(songs_dir, db_path))
            if paths:
                rng.shuffle(paths)
                return paths[:limit]
        except Exception as e:  # pragma: no cover - depends on a real osu!.db
            print(f"  [warn] osu!.db parse failed ({e}); falling back to random real maps")
    paths = list(Path(songs_dir).rglob("*.osu"))
    rng.shuffle(paths)
    return paths[:limit]


def measure(songs_dir: Path, db_path: Path | None, ref_stats: dict, limit: int,
            seed: int = 0, sr_weight: float = 0.35) -> dict:
    """Score a gold sample at each map's own SR; aggregate overall / per-bucket /
    per-family. Scoring at own SR makes ``sr_closeness`` ~1.0, isolating quality.
    """
    paths = sample_gold_paths(songs_dir, db_path, limit, seed)
    overall = {"reward": [], "quality": [], "sr_closeness": []}
    per_bucket: dict[str, dict[str, list[float]]] = {}
    per_family: dict[str, list[float]] = {f: [] for f in FAMILIES}
    per_metric: dict[str, list[float]] = {}
    n_scored = n_seen = 0

    for p in paths:
        n_seen += 1
        sr = star_rating(p)
        if sr is None:                     # not std / unparseable -> skip
            continue
        try:
            bd = reward_from_osu(p, ref_stats, target_sr=sr, sr_weight=sr_weight)
        except Exception:
            continue
        if not bd.per_metric:              # no bucket / no comparable metrics
            continue
        n_scored += 1
        overall["reward"].append(bd.reward)
        overall["quality"].append(bd.quality)
        overall["sr_closeness"].append(bd.sr_closeness)
        b = bd.bucket
        pb = per_bucket.setdefault(b, {"reward": [], "quality": []})
        pb["reward"].append(bd.reward)
        pb["quality"].append(bd.quality)
        for fam, v in bd.family_breakdown.items():
            per_family[fam].append(v)
        for k, v in bd.per_metric.items():
            per_metric.setdefault(k, []).append(v)
        if n_scored % 100 == 0:
            print(f"  scored {n_scored} (seen {n_seen})", flush=True)

    return {
        "n_seen": n_seen,
        "n_scored": n_scored,
        "sr_weight": sr_weight,
        "ref_stats_n": ref_stats.get("n_maps"),
        "overall": {k: _summary(v) for k, v in overall.items()},
        "per_bucket": {
            b: {"reward": _summary(per_bucket[b]["reward"]),
                "quality": _summary(per_bucket[b]["quality"])}
            for b in SR_BUCKET_ORDER if b in per_bucket
        },
        "per_family": {f: _summary(per_family[f]) for f in FAMILIES if per_family[f]},
        "per_metric": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_metric.items())},
    }


def _print_report(rep: dict) -> None:
    o = rep["overall"]
    print(f"\n=== gold-map reward measurement (n_scored={rep['n_scored']}, "
          f"ref n={rep['ref_stats_n']}, sr_weight={rep['sr_weight']}) ===")
    print(f"  reward       mean {o['reward']['mean']:.4f}  median {o['reward']['median']:.4f}"
          f"  p10 {o['reward']['p10']:.4f}  p90 {o['reward']['p90']:.4f}")
    print(f"  quality      mean {o['quality']['mean']:.4f}  median {o['quality']['median']:.4f}"
          f"  p10 {o['quality']['p10']:.4f}  p90 {o['quality']['p90']:.4f}")
    print(f"  sr_closeness mean {o['sr_closeness']['mean']:.4f}  (own-SR target -> ~1.0)")
    print("\n  per SR bucket (reward mean / quality mean / n):")
    for b, s in rep["per_bucket"].items():
        print(f"    {b:8} R {s['reward']['mean']:.4f}  Q {s['quality']['mean']:.4f}"
              f"  (n={s['reward']['n']})")
    print("\n  per family (mean band-membership over gold maps):")
    for f, s in rep["per_family"].items():
        print(f"    {f:14} {s['mean']:.4f}  (median {s['median']:.4f}, p10 {s['p10']:.4f})")
    print("\n  per metric (mean band-membership):")
    for k, v in rep["per_metric"].items():
        print(f"    {k:24} {v:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the ranked-map reward on a sample of real gold maps.")
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--db", default=r"C:/osu!/osu!.db",
                    help="osu!.db for ranked-status filtering (omit -> random real maps)")
    ap.add_argument("--ref-stats", default="artifacts/reference_stats.json")
    ap.add_argument("--limit", type=int, default=400, help="max gold maps to score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr-weight", type=float, default=0.35)
    ap.add_argument("--json", default=None, help="also write the full report JSON here")
    args = ap.parse_args()

    ref_path = Path(args.ref_stats)
    if not ref_path.exists():
        raise SystemExit(f"reference stats not found: {ref_path} (build with src.corpus_stats)")
    ref_stats = json.loads(ref_path.read_text(encoding="utf-8"))
    db = Path(args.db) if args.db and Path(args.db).exists() else None
    if db is None:
        print(f"  [note] no osu!.db at {args.db}; sampling RANDOM real maps (not ranked-filtered)")

    rep = measure(Path(args.songs), db, ref_stats, args.limit,
                  seed=args.seed, sr_weight=args.sr_weight)
    _print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
