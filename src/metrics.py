"""Pattern/quality metrics for osu! beatmaps.

Turns a beatmap into interpretable numbers so generated maps can be compared to
real ones and to each other across experiments. Not a difficulty model — these
are descriptive statistics tied to the pattern vocabulary in RESEARCH.md.

  python -m src.metrics --osu map.osu
  python -m src.metrics --osu gen.osu --ref real.osu     # side-by-side
"""
from __future__ import annotations

import argparse
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
    }


def compute_metrics_for_osu(path: str | Path) -> dict:
    return compute_metrics(parse_beatmap(path))


def _print(title, m):
    print(f"== {title} ==")
    for k, v in m.items():
        print(f"  {k:24} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", required=True)
    ap.add_argument("--ref", default=None, help="optional reference map to compare")
    args = ap.parse_args()
    m = compute_metrics_for_osu(args.osu)
    if args.ref:
        r = compute_metrics_for_osu(args.ref)
        print(f"{'metric':24} {'generated':>12} {'reference':>12}")
        for k in m:
            print(f"{k:24} {str(m[k]):>12} {str(r.get(k, '')):>12}")
    else:
        _print(Path(args.osu).name, m)


if __name__ == "__main__":
    main()
