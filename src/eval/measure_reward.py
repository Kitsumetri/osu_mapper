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
  # brute-force EVERY map, parallel, dump the worst-50 tail:
  uv run python -m src.eval.measure_reward --all --workers 10 \
      --json artifacts/reward_audit.json --bottom-n 50
  # ...restricted to the single-BPM GOLD subset (matches the training distribution):
  uv run python -m src.eval.measure_reward --all --gold --workers 10 \
      --json artifacts/reward_audit_gold.json --bottom-n 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..difficulty import SR_BUCKET_ORDER, star_rating
from ..parsing.beatmap import parse_beatmap
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
    """Real .osu paths, preferring osu!.db-confirmed ranked maps.

    ``limit <= 0`` means **no cap** — return EVERY candidate (the ``--all`` /
    ``--limit 0`` brute-force path). Otherwise return up to ``limit`` (shuffled).
    Falls back to a random sample of every .osu in the library when osu!.db is
    missing/unreadable (callers should note the sample is then "real" not "gold").
    """
    rng = random.Random(seed)
    cap = None if limit <= 0 else limit
    if db_path and Path(db_path).exists():
        try:
            from ..data.osu_db import ranked_osu_paths
            paths = list(ranked_osu_paths(songs_dir, db_path))
            if paths:
                rng.shuffle(paths)
                return paths if cap is None else paths[:cap]
        except Exception as e:  # pragma: no cover - depends on a real osu!.db
            print(f"  [warn] osu!.db parse failed ({e}); falling back to random real maps")
    paths = list(Path(songs_dir).rglob("*.osu"))
    rng.shuffle(paths)
    return paths if cap is None else paths[:cap]


def _worst_family(family_breakdown: dict[str, float]) -> str | None:
    """The family with the lowest band-membership score (the map's weakest axis)."""
    if not family_breakdown:
        return None
    return min(family_breakdown, key=family_breakdown.get)


def _is_gold(bm, sr: float | None) -> bool:
    """The preprocess ``--gold`` gates: std + single-BPM + has-kiai + >=10%%
    hitsounds + 1 <= SR <= 10. ``--gold`` restricts the scan to this subset so the
    calibration matches the model's TRAINING distribution (the gold maps), and so
    the single-BPM ``on_quarter_grid_ratio`` metric isn't dinged by variable-BPM
    maps (the multi-BPM rhythm artifact in the audit tail).
    """
    if sr is None or bm.mode != 0 or not (1.0 <= sr <= 10.0):
        return False
    if sum(1 for tp in bm.timing_points if tp.uninherited) > 1:   # single BPM only
        return False
    if not bm.kiai_spans():                                        # require kiai
        return False
    n = len(bm.hit_objects) or 1
    return sum(1 for o in bm.hit_objects if o.hit_sound) / n >= 0.1


def _score_one(args: tuple) -> dict | str | None:
    """Score ONE gold map at its OWN rosu SR (so sr_closeness ~1.0, quality is the
    thing under test). Module-level + picklable so it runs in a worker process
    (Windows ``spawn``). Returns a compact dict; ``"filtered"`` when ``gold`` is on
    and the map fails the gold gates; or ``None`` to skip (not std / unparseable /
    no comparable metrics). The per-file work (parse + rosu SR + metrics + reward)
    is independent across files — the unit the pool distributes.
    """
    path, ref_stats, sr_weight, gold = args
    sr = star_rating(path)
    if sr is None:                         # not std / unparseable -> skip
        return None
    bm = None
    if gold:
        try:
            bm = parse_beatmap(path)
        except Exception:
            return None
        if not _is_gold(bm, sr):
            return "filtered"              # ranked but not in the gold subset
    try:
        # reuse the already-parsed map (gold path) AND the SR we just computed, so
        # neither the parse nor the (dominant) rosu star-rating call runs twice.
        bd = reward_from_osu(path, ref_stats, target_sr=sr, sr_weight=sr_weight,
                             bm=bm, achieved_sr=sr)
    except Exception:
        return None
    if not bd.per_metric:                  # no bucket / no comparable metrics
        return None
    return {
        "path": str(path),
        "reward": bd.reward,
        "quality": bd.quality,
        "sr_closeness": bd.sr_closeness,
        "bucket": bd.bucket,
        "family_breakdown": bd.family_breakdown,
        "per_metric": bd.per_metric,
        "playability": bd.playability,
        "defects": bd.defects,
        "worst_family": _worst_family(bd.family_breakdown),
    }


def measure(songs_dir: Path, db_path: Path | None, ref_stats: dict, limit: int,
            seed: int = 0, sr_weight: float = 0.35, workers: int | None = None,
            bottom_n: int = 50, progress_every: int = 200, gold: bool = False) -> dict:
    """Score a gold sample at each map's own SR; aggregate overall / per-bucket /
    per-family, and collect the BOTTOM-N lowest-reward maps. ``limit <= 0`` scores
    EVERY candidate (the brute-force ``--all`` path). ``gold=True`` keeps only maps
    passing the preprocess gold gates (``_is_gold``) so calibration matches the
    training distribution. ``workers`` parallel processes (default ``cpu_count-1``;
    ``<=1`` forces serial) — the rosu SR call dominates and parallelises cleanly,
    exactly like ``corpus_stats.collect``. Aggregation is order-independent
    (``_summary`` sorts; the bottom-N is sorted at the end), so the result is
    identical regardless of worker count or completion order.
    """
    paths = sample_gold_paths(songs_dir, db_path, limit, seed)
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    overall = {"reward": [], "quality": [], "sr_closeness": []}
    per_bucket: dict[str, dict[str, list[float]]] = {}
    per_family: dict[str, list[float]] = {f: [] for f in FAMILIES}
    per_metric: dict[str, list[float]] = {}
    per_defect: dict[str, list[float]] = {}
    playabilities: list[float] = []
    scored_rows: list[dict] = []           # compact per-map rows (for the tail)
    n_scored = n_seen = n_filtered = 0

    def _accumulate(res: dict | None) -> None:
        nonlocal n_scored
        if res is None:
            return
        n_scored += 1
        overall["reward"].append(res["reward"])
        overall["quality"].append(res["quality"])
        overall["sr_closeness"].append(res["sr_closeness"])
        pb = per_bucket.setdefault(res["bucket"], {"reward": [], "quality": []})
        pb["reward"].append(res["reward"])
        pb["quality"].append(res["quality"])
        for fam, v in res["family_breakdown"].items():
            per_family[fam].append(v)
        for k, v in res["per_metric"].items():
            per_metric.setdefault(k, []).append(v)
        for k, v in res["defects"].items():
            per_defect.setdefault(k, []).append(v)
        playabilities.append(res["playability"])
        scored_rows.append({
            "path": res["path"], "reward": res["reward"],
            "quality": res["quality"], "bucket": res["bucket"],
            "worst_family": res["worst_family"], "playability": res["playability"],
            "defects": res["defects"],
        })

    def _handle(res: dict | str | None) -> None:
        nonlocal n_filtered
        if res == "filtered":              # gold gate rejected it (not scored)
            n_filtered += 1
        else:
            _accumulate(res)

    def _tick() -> None:
        if n_scored and n_scored % progress_every == 0:
            print(f"  scored {n_scored} (seen {n_seen}/{len(paths)})", flush=True)

    jobs = ((p, ref_stats, sr_weight, gold) for p in paths)
    if workers <= 1:
        for job in jobs:
            n_seen += 1
            _handle(_score_one(job))
            _tick()
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_score_one, job) for job in jobs]
            for fut in as_completed(futs):
                n_seen += 1
                _handle(fut.result())
                _tick()

    # the lowest-reward tail (path + reward + worst family) — the audit target.
    bottom = sorted(scored_rows, key=lambda r: r["reward"])[:max(0, bottom_n)]

    return {
        "n_seen": n_seen,
        "n_scored": n_scored,
        "n_filtered": n_filtered,
        "gold_filter": gold,
        "sr_weight": sr_weight,
        "workers": workers,
        "ref_stats_n": ref_stats.get("n_maps"),
        "overall": {k: _summary(v) for k, v in overall.items()},
        "playability": _summary(playabilities),
        "per_defect": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_defect.items())},
        "per_bucket": {
            b: {"reward": _summary(per_bucket[b]["reward"]),
                "quality": _summary(per_bucket[b]["quality"])}
            for b in SR_BUCKET_ORDER if b in per_bucket
        },
        "per_family": {f: _summary(per_family[f]) for f in FAMILIES if per_family[f]},
        "per_metric": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_metric.items())},
        "bottom_n": bottom,
    }


def _print_report(rep: dict) -> None:
    o = rep["overall"]
    gold_tag = ", GOLD-filtered" if rep.get("gold_filter") else ""
    print(f"\n=== gold-map reward measurement (n_scored={rep['n_scored']}, "
          f"ref n={rep['ref_stats_n']}, sr_weight={rep['sr_weight']}, "
          f"workers={rep.get('workers')}{gold_tag}) ===")
    if rep.get("gold_filter"):
        print(f"  gold filter: kept {rep['n_scored']}, skipped {rep.get('n_filtered', 0)} "
              f"non-gold (multi-BPM / no-kiai / low-hitsound / SR outside 1-10)")
    print(f"  reward       mean {o['reward']['mean']:.4f}  median {o['reward']['median']:.4f}"
          f"  p10 {o['reward']['p10']:.4f}  p90 {o['reward']['p90']:.4f}")
    print(f"  quality      mean {o['quality']['mean']:.4f}  median {o['quality']['median']:.4f}"
          f"  p10 {o['quality']['p10']:.4f}  p90 {o['quality']['p90']:.4f}")
    print(f"  sr_closeness mean {o['sr_closeness']['mean']:.4f}  (own-SR target -> ~1.0)")
    pl = rep.get("playability", {})
    if pl.get("n"):
        print(f"  playability  mean {pl['mean']:.4f}  median {pl['median']:.4f}"
              f"  min {pl['min']:.4f}  (1.0 = no objective defects)")
    pd = rep.get("per_defect", {})
    if pd:
        print("  per defect (mean rate over maps; want ~0 on gold — TUNE the threshold):")
        for k, v in pd.items():
            print(f"    {k:18} {v:.4f}")
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
    bottom = rep.get("bottom_n", [])
    if bottom:
        print(f"\n  lowest-reward tail (worst {len(bottom)}; audit these):")
        for r in bottom[:25]:
            print(f"    R {r['reward']:.4f}  [{r['bucket']:8}] worst={str(r['worst_family']):12} "
                  f"play={r['playability']:.3f}  {Path(r['path']).name}")
        if len(bottom) > 25:
            print(f"    ... (+{len(bottom) - 25} more in the --json output)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the ranked-map reward on a sample of real gold maps.")
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--db", default=r"C:/osu!/osu!.db",
                    help="osu!.db for ranked-status filtering (omit -> random real maps)")
    ap.add_argument("--ref-stats", default="artifacts/reference_stats.json")
    ap.add_argument("--limit", type=int, default=400,
                    help="max gold maps to score (0 = ALL, same as --all)")
    ap.add_argument("--all", action="store_true",
                    help="score EVERY candidate map with no cap (== --limit 0)")
    ap.add_argument("--gold", action="store_true",
                    help="restrict to the preprocess gold subset (std + single-BPM + "
                         "kiai + >=10%% hitsounds + 1<=SR<=10) so calibration matches the "
                         "training distribution and avoids the multi-BPM grid artifact")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default cpu_count-1; 1 = serial)")
    ap.add_argument("--bottom-n", type=int, default=50,
                    help="dump the N lowest-reward maps (path + reward + worst family)")
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

    limit = 0 if args.all else args.limit
    rep = measure(Path(args.songs), db, ref_stats, limit,
                  seed=args.seed, sr_weight=args.sr_weight,
                  workers=args.workers, bottom_n=args.bottom_n, gold=args.gold)
    _print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
