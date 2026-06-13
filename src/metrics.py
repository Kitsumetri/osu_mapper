"""Pattern/quality metrics for osu! beatmaps.

Turns a beatmap into interpretable numbers so generated maps can be compared to
real ones and to each other across experiments. Not a difficulty model — these
are descriptive statistics tied to the pattern vocabulary in RESEARCH.md.

  python -m src.metrics --osu map.osu
  python -m src.metrics --osu gen.osu --ref real.osu     # side-by-side
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .parsing.beatmap import Beatmap, parse_beatmap

# spacing thresholds (osu! pixels); playfield is 512x384
STREAM_MAX_SPACING = 120.0
JUMP_MIN_SPACING = 200.0


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def compute_metrics(bm: Beatmap) -> dict:
    objs = sorted(bm.hit_objects, key=lambda o: o.time)
    n = len(objs)
    if n < 2:
        return {"n_objects": n}

    duration_s = (objs[-1].end_time - objs[0].time) / 1000 or 1.0
    beat = 60000.0 / bm.bpm if bm.bpm > 0 else 0.0

    gaps_ms, spacings, beats, ongrid = [], [], [], 0
    streams = jumps = 0
    for a, b in zip(objs, objs[1:]):
        dt = b.time - a.time
        if dt <= 0:
            continue
        d = _dist(a, b)
        gaps_ms.append(dt)
        spacings.append(d)
        if beat > 0:
            nb = dt / beat
            beats.append(nb)
            # distance (in beats) to nearest 1/4 subdivision
            q = nb * 4
            if abs(q - round(q)) <= 0.12:
                ongrid += 1
            if nb <= 0.30 and d <= STREAM_MAX_SPACING:   # ~1/4 beat, tight
                streams += 1
        if d >= JUMP_MIN_SPACING:
            jumps += 1

    m = len(gaps_ms)

    # turning angle at each interior object: angle between (b-a) and (c-b).
    # ~0 deg = straight flow, ~180 deg = full reversal (back-and-forth / 1-2).
    turn_angles, reversals = [], 0
    for a, b, c in zip(objs, objs[1:], objs[2:]):
        v1 = (b.x - a.x, b.y - a.y)
        v2 = (c.x - b.x, c.y - b.y)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1 or n2 < 1:
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        turn = math.degrees(math.acos(cosang))  # 0=straight, 180=reverse
        turn_angles.append(turn)
        if turn >= 150.0:
            reversals += 1

    # slider-velocity changes: inherited timing points with a distinct SV.
    sv_changes = 0
    last_sv = 1.0
    for tp in bm.timing_points:
        sv = tp.sv if not tp.uninherited else 1.0
        if abs(sv - last_sv) > 1e-3:
            sv_changes += 1
        last_sv = sv

    def _mean(x):
        return sum(x) / len(x) if x else 0.0

    def _std(x):
        if len(x) < 2:
            return 0.0
        mu = _mean(x)
        return math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))

    return {
        "n_objects": n,
        "duration_s": round(duration_s, 1),
        "density_per_s": round(n / duration_s, 2),
        "bpm": bm.bpm,
        "circle_ratio": round(sum(o.is_circle for o in objs) / n, 3),
        "slider_ratio": round(sum(o.is_slider for o in objs) / n, 3),
        "spinner_ratio": round(sum(o.is_spinner for o in objs) / n, 3),
        "bezier_slider_ratio": round(
            sum(o.is_slider and o.curve_type == "B" for o in objs)
            / max(1, sum(o.is_slider for o in objs)), 3),
        "new_combo_ratio": round(sum(o.is_new_combo for o in objs) / n, 3),
        "mean_spacing_px": round(_mean(spacings), 1),
        "std_spacing_px": round(_std(spacings), 1),
        "stream_ratio": round(streams / m, 3) if m else 0.0,
        "jump_ratio": round(jumps / m, 3) if m else 0.0,
        "on_quarter_grid_ratio": round(ongrid / m, 3) if m else 0.0,
        "mean_turn_angle_deg": round(_mean(turn_angles), 1),
        "reversal_ratio": round(reversals / len(turn_angles), 3) if turn_angles else 0.0,
        "sv_changes_per_min": round(sv_changes / (duration_s / 60), 2) if duration_s else 0.0,
    }


def compute_metrics_for_osu(path: str | Path) -> dict:
    return compute_metrics(parse_beatmap(path))


def score_against_reference(m: dict, ref_stats: dict, bucket: str) -> tuple[str, list]:
    """Compare a map's metrics to a named reference bucket (e.g. an SR band).

    Returns (bucket_name, rows) where each row is
    (metric, value, ref_mean, ref_std, z_score, in_p10_p90).
    """
    ref = ref_stats.get("buckets", {}).get(bucket, {})
    rows = []
    for k, v in m.items():
        s = ref.get(k)
        if not s or not isinstance(v, (int, float)):
            continue
        std = s["std"] or 1e-9
        z = (v - s["mean"]) / std
        in_range = s["p10"] <= v <= s["p90"]
        rows.append((k, v, s["mean"], s["std"], round(z, 2), in_range))
    return bucket, rows


def _print(title, m):
    print(f"== {title} ==")
    for k, v in m.items():
        print(f"  {k:24} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", required=True)
    ap.add_argument("--ref", default=None, help="optional reference map to compare")
    ap.add_argument("--ref-stats", default=None,
                    help="reference_stats.json from corpus_stats, for z-scoring")
    args = ap.parse_args()
    m = compute_metrics_for_osu(args.osu)
    if args.ref:
        r = compute_metrics_for_osu(args.ref)
        print(f"{'metric':24} {'generated':>12} {'reference':>12}")
        for k in m:
            print(f"{k:24} {str(m[k]):>12} {str(r.get(k, '')):>12}")
    elif args.ref_stats:
        from .difficulty import sr_bucket, star_rating
        ref_stats = json.loads(Path(args.ref_stats).read_text(encoding="utf-8"))
        sr = star_rating(args.osu)
        bucket = sr_bucket(sr) if sr is not None else "Hard"
        sr_str = f"{sr:.2f}*" if sr is not None else "n/a"
        bucket, rows = score_against_reference(m, ref_stats, bucket)
        print(f"{Path(args.osu).name}  (star rating {sr_str} -> {bucket} bucket)\n")
        print(f"{'metric':24}{'value':>10}{'real_mean':>11}{'real_std':>10}{'z':>7}  range")
        for k, v, mu, sd, z, ok in rows:
            flag = "ok" if ok else "OUT"
            print(f"{k:24}{v:>10}{mu:>10}{sd:>9}{z:>7}  {flag}")
    else:
        _print(Path(args.osu).name, m)


if __name__ == "__main__":
    main()
