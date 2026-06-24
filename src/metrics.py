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
# a slider counts as *visibly* curved if any control point bows this far (px) off
# the head->tail chord. Tracks perceived curvature, unlike the B/L type flag (a
# bezier with near-collinear anchors looks straight): real ranked ~0.38, v6 ~0.13.
CURVE_SAGITTA_PX = 10.0

# osu! stack leniency: two objects whose gap and (head) distance are below the
# stack threshold are *stacked* by the editor (drawn with a small per-object
# offset). The stack radius is ~3 px (objects within this are visually "the same
# spot"). We use it to detect when consecutive notes sit on top of each other.
STACK_RADIUS_PX = 3.0


def circle_radius_px(cs: float) -> float:
    """osu!standard hit-circle radius in playfield px from Circle Size.

    Official formula: r = 54.4 - 4.48 * CS  (CS 4 -> 36.5 px, CS 5 -> 32.0 px).
    The follow-circle / hittable region of a slider is ~2.4 r, but for "do two
    objects occupy the same spot" we use the circle radius itself.
    """
    return 54.4 - 4.48 * cs


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def slider_sagitta(o) -> float:
    """Max perpendicular bow (px) of a slider's control polygon off its chord.

    0 for a straight slider; grows with visible curvature. Uses the control
    points, so it reflects shape regardless of the stored curve-type letter.
    """
    poly = [(o.x, o.y), *(o.curve_points or [])]
    if len(poly) < 3:
        return 0.0
    (ax, ay), (bx, by) = poly[0], poly[-1]
    dx, dy = bx - ax, by - ay
    chord = math.hypot(dx, dy)
    if chord < 1e-6:
        return max(math.hypot(px - ax, py - ay) for px, py in poly)
    return max(abs((px - ax) * dy - (py - ay) * dx) / chord for px, py in poly)


def slider_anchor_min_gap(o) -> float:
    """Smallest distance (px) between consecutive points of a slider's control
    polygon (head + anchors). Small values mean anchors are clustered on top of
    each other — a *degenerate* control polygon that decodes to a kink/spike or a
    zero-length segment. 0.0 if the slider has no anchors. ``inf`` sentinel folded
    to a large number so callers can compare cleanly.
    """
    poly = [(o.x, o.y), *(o.curve_points or [])]
    if len(poly) < 2:
        return float("inf")
    return min(math.hypot(poly[i + 1][0] - poly[i][0],
                          poly[i + 1][1] - poly[i][1])
               for i in range(len(poly) - 1))


def slider_polyline(o, n: int = 12) -> list[tuple[float, float]]:
    """Coarse polyline approximation of a slider body, head->tail.

    Not the exact bezier/perfect-circle path osu! renders — for *overlap* and
    *proximity* tests a piecewise-linear walk over (head, anchors..., last anchor)
    is enough and is shape-agnostic (works for L/B/P/C). Resamples to ``n`` evenly
    spaced points along that control polygon so two sliders are compared on
    comparable resolutions regardless of anchor count.
    """
    poly = [(float(o.x), float(o.y)), *((float(a), float(b)) for a, b in (o.curve_points or []))]
    if len(poly) < 2:
        return poly
    seg_len = [math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
               for i in range(len(poly) - 1)]
    total = sum(seg_len) or 1.0
    out, target, acc, j = [], 0.0, 0.0, 0
    for k in range(n + 1):
        target = total * k / n
        while j < len(seg_len) - 1 and acc + seg_len[j] < target:
            acc += seg_len[j]
            j += 1
        t = (target - acc) / (seg_len[j] or 1.0)
        t = max(0.0, min(1.0, t))
        out.append((poly[j][0] + t * (poly[j + 1][0] - poly[j][0]),
                    poly[j][1] + t * (poly[j + 1][1] - poly[j][1])))
    return out


def stack_ratio_of(objs: list) -> float:
    """Fraction of consecutive object pairs sitting on the same spot (head-to-head
    distance <= STACK_RADIUS_PX, positive time gap).

    Stacking is a normal, intentional osu! pattern (notes drawn on one spot with a
    small editor offset), so this is a DISTRIBUTIONAL trait scored by a gold band,
    NOT a defect: a map that stacks like ranked maps sits in the band. (An earlier
    design penalised *all* stacking, which wrongly dinged real maps.)
    """
    pairs = [(a, b) for a, b in zip(objs, objs[1:]) if b.time - a.time > 0]
    if not pairs:
        return 0.0
    stacked = sum(1 for a, b in pairs if math.hypot(a.x - b.x, a.y - b.y) <= STACK_RADIUS_PX)
    return stacked / len(pairs)


def slider_overlap_ratio_of(objs: list, radius: float) -> float:
    """Fraction of sliders whose body passes within ``radius`` px of a NON-adjacent
    object's head (an immediate neighbour is allowed — that is normal follow-through).

    Overlapping a slider with nearby objects is a common stylistic device, so this
    is a DISTRIBUTIONAL trait scored by a gold band, NOT a defect — only a map that
    overlaps far more than any ranked map drifts out of band.
    """
    sliders = [(i, o) for i, o in enumerate(objs) if o.is_slider and (o.curve_points or [])]
    if not sliders:
        return 0.0
    bad = 0
    for i, o in sliders:
        body = slider_polyline(o)
        for j, other in enumerate(objs):
            if abs(j - i) <= 1:                 # skip self + immediate neighbours
                continue
            ox, oy = float(other.x), float(other.y)
            if any(math.hypot(px - ox, py - oy) <= radius for px, py in body):
                bad += 1
                break
    return bad / len(sliders)


def compute_metrics(bm: Beatmap) -> dict:
    objs = sorted(bm.hit_objects, key=lambda o: o.time)
    n = len(objs)
    if n < 2:
        return {"n_objects": n}

    duration_s = (objs[-1].end_time - objs[0].time) / 1000 or 1.0
    beat = 60000.0 / bm.bpm if bm.bpm > 0 else 0.0

    gaps_ms, spacings, beats, ongrid = [], [], [], 0
    streams = jumps = 0
    stream_spacings = []   # circle->circle spacing of pairs that ARE in a stream
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
                stream_spacings.append(d)
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

    # --- new distributional flow / slider-shape traits ----------------------
    # (i) stream-spacing regularity: coefficient of variation (std/mean) of the
    #     circle->circle spacing of pairs that form a stream. A clean stream has
    #     near-constant spacing (CV ~ 0); a messy/varying stream has a high CV.
    #     This is a *distributional* trait (style varies, so it needs a gold band),
    #     so it goes in compute_metrics + corpus_stats, NOT the defect penalty.
    mu_ss = _mean(stream_spacings)
    stream_spacing_cv = round(_std(stream_spacings) / mu_ss, 3) if mu_ss > 1e-6 else 0.0

    # (ii) slider anchor spread: mean over sliders of the smallest gap between
    #     consecutive control points (px), capped so a few huge sliders don't
    #     dominate. Low = anchors bunched (kinky/degenerate control polygons);
    #     real ranked sliders place anchors with sane spacing. Distributional.
    sliders = [o for o in objs if o.is_slider and (o.curve_points or [])]
    anchor_gaps = [min(slider_anchor_min_gap(o), 200.0) for o in sliders]
    slider_anchor_spread_px = round(_mean(anchor_gaps), 1) if anchor_gaps else 0.0

    # (iii) stacking rate + (iv) slider/object overlap rate: also distributional
    #     traits (both are intentional osu! patterns at a ranked-typical rate, so
    #     they are band-scored — NOT defects; see the helper docstrings + reward.py).
    radius = max(1.0, circle_radius_px(bm.circle_size))
    stack_ratio = stack_ratio_of(objs)
    slider_overlap_ratio = slider_overlap_ratio_of(objs, radius)

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
        # visibly-curved sliders (sagitta-based) — the meaningful curvature measure;
        # bezier_slider_ratio over-counts near-straight beziers (see CURVE_SAGITTA_PX).
        "curved_slider_ratio": round(
            sum(o.is_slider and slider_sagitta(o) >= CURVE_SAGITTA_PX for o in objs)
            / max(1, sum(o.is_slider for o in objs)), 3),
        "new_combo_ratio": round(sum(o.is_new_combo for o in objs) / n, 3),
        "mean_spacing_px": round(_mean(spacings), 1),
        "std_spacing_px": round(_std(spacings), 1),
        "stream_ratio": round(streams / m, 3) if m else 0.0,
        "jump_ratio": round(jumps / m, 3) if m else 0.0,
        "on_quarter_grid_ratio": round(ongrid / m, 3) if m else 0.0,
        "mean_turn_angle_deg": round(_mean(turn_angles), 1),
        "reversal_ratio": round(reversals / len(turn_angles), 3) if turn_angles else 0.0,
        # NEW distributional flow trait: stream-spacing regularity (CV).
        "stream_spacing_cv": stream_spacing_cv,
        "sv_changes_per_min": round(sv_changes / (duration_s / 60), 2) if duration_s else 0.0,
        # NEW distributional slider-shape trait: mean min-anchor-gap (px).
        "slider_anchor_spread_px": slider_anchor_spread_px,
        # NEW distributional traits: stacking rate (flow) + slider/object overlap
        # rate (slider_shape) — intentional patterns, band-scored not penalised.
        "stack_ratio": round(stack_ratio, 3),
        "slider_overlap_ratio": round(slider_overlap_ratio, 3),
        # v3 outputs: kiai coverage and hitsound usage
        "kiai_ratio": round(
            sum(e - s for s, e in bm.kiai_spans()) / 1000 / duration_s, 3) if duration_s else 0.0,
        "hitsound_ratio": round(sum(o.hit_sound > 0 for o in objs) / n, 3),
        "clap_ratio": round(sum(o.hit_sound & 8 > 0 for o in objs) / n, 3),
        "whistle_ratio": round(sum(o.hit_sound & 2 > 0 for o in objs) / n, 3),
        "finish_ratio": round(sum(o.hit_sound & 4 > 0 for o in objs) / n, 3),
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
