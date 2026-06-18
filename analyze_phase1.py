"""Phase 1 analysis (v7 patterns plan): real ranked maps vs v6 generated.

Measures the things the eval metrics don't: slider *curvature magnitude* (not just
B/L type), spacing/flow distributions, and SV-timeline structure. Throwaway probe
to decide whether the slider/pattern fixes are decode-side or model-side.

  uv run python analyze_phase1.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from src.data.osu_db import ranked_osu_paths
from src.generate import generate, load_model, prepare_audio
from src.metrics import compute_metrics
from src.parsing.beatmap import parse_beatmap

SONGS = "C:/osu!/Songs"
DB = "C:/osu!/osu!.db"
CKPT = "runs/20260616-013932-ranked-v6/ckpt/best.pt"
AUDIO_2MIN = "C:/osu!/Songs/986934 JIN feat LiSA - Headphone Actor/audio.mp3"
N_REAL = 400
GEN_SRS = [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]


def _perp(p, a, b) -> float:
    """Perpendicular distance of point p from the line a->b (px)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return math.hypot(px - ax, py - ay)
    return abs((px - ax) * dy - (py - ay) * dx) / L


def slider_curvatures(bm):
    """Per-slider (chord_px, path/chord ratio, sagitta_px) from the control polygon."""
    out = []
    for o in bm.hit_objects:
        if not o.is_slider:
            continue
        poly = [(o.x, o.y), *[(cx, cy) for cx, cy in (o.curve_points or [])]]
        if len(poly) < 2:
            continue
        head, tail = poly[0], poly[-1]
        chord = math.hypot(tail[0] - head[0], tail[1] - head[1])
        path = sum(math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
                   for i in range(len(poly) - 1))
        ratio = path / chord if chord > 1 else 1.0
        sagitta = max(_perp(p, head, tail) for p in poly)
        out.append((chord, ratio, sagitta))
    return out


def sv_structure(bm):
    """Per-map SV timeline: (#inherited points, #distinct SV vals, #changes, min, max)."""
    svs = [tp.sv for tp in bm.timing_points if not tp.uninherited]
    if not svs:
        return None
    changes = sum(1 for a, b in zip(svs, svs[1:]) if abs(a - b) > 1e-3)
    distinct = len({round(s, 2) for s in svs})
    return (len(svs), distinct, changes, min(svs), max(svs))


def summarize(name, beatmaps):
    curvs, metrics_rows, sv_rows = [], [], []
    for bm in beatmaps:
        curvs.extend(slider_curvatures(bm))
        m = compute_metrics(bm)
        if "stream_ratio" in m:
            metrics_rows.append(m)
        sv = sv_structure(bm)
        if sv:
            sv_rows.append(sv)

    print(f"\n================ {name}  (maps={len(beatmaps)}, sliders={len(curvs)}) ===========")
    if curvs:
        ratios = np.array([c[1] for c in curvs])
        sags = np.array([c[2] for c in curvs])
        print("  SLIDER CURVATURE")
        print(f"    path/chord ratio:  mean {ratios.mean():.3f}  med {np.median(ratios):.3f}"
              f"  p90 {np.percentile(ratios, 90):.3f}")
        print(f"    sagitta_px:        mean {sags.mean():.1f}   med {np.median(sags):.1f}"
              f"   p90 {np.percentile(sags, 90):.1f}")
        for thr in (1.06, 1.15, 1.30):
            print(f"    %% ratio>={thr:.2f} (curved): {100 * (ratios >= thr).mean():.1f}%")
        for thr in (10, 20, 40):
            print(f"    %% sagitta>={thr}px (visibly bent): {100 * (sags >= thr).mean():.1f}%")

    if metrics_rows:
        def avg(k):
            return np.mean([r[k] for r in metrics_rows])
        print("  PATTERN METRICS (map-mean)")
        print(f"    mean_spacing_px {avg('mean_spacing_px'):.1f}  std_spacing_px "
              f"{avg('std_spacing_px'):.1f}  density/s {avg('density_per_s'):.2f}")
        print(f"    stream_ratio {avg('stream_ratio'):.3f}  jump_ratio {avg('jump_ratio'):.3f}"
              f"  reversal_ratio {avg('reversal_ratio'):.3f}"
              f"  turn_deg {avg('mean_turn_angle_deg'):.1f}")
        print(f"    bezier_slider_ratio (type B) {avg('bezier_slider_ratio'):.3f}")

    if sv_rows:
        pts = np.array([r[0] for r in sv_rows])
        dis = np.array([r[1] for r in sv_rows])
        chg = np.array([r[2] for r in sv_rows])
        mins = np.array([r[3] for r in sv_rows])
        maxs = np.array([r[4] for r in sv_rows])
        print(f"  SV TIMELINE  (maps with green lines: {len(sv_rows)}/{len(beatmaps)})")
        print(f"    inherited pts/map: med {np.median(pts):.0f}  p90 {np.percentile(pts, 90):.0f}")
        print(f"    distinct SV vals/map: med {np.median(dis):.0f}"
              f"  p90 {np.percentile(dis, 90):.0f}")
        print(f"    SV changes/map: med {np.median(chg):.0f}  p90 {np.percentile(chg, 90):.0f}")
        print(f"    SV range across maps: min {mins.min():.2f}  max {maxs.max():.2f}"
              f"  median-max {np.median(maxs):.2f}")
        nontrivial = np.mean([(r[1] > 1 or abs(r[3] - 1) > 0.05 or abs(r[4] - 1) > 0.05)
                              for r in sv_rows])
        print(f"    %% maps with non-trivial SV (varies or !=1): {100 * nontrivial:.1f}%")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--label", default=None, help="name for the generated block")
    ap.add_argument("--rescale", type=float, default=0.0, help="guidance_rescale (v/zero-SNR)")
    ap.add_argument("--no-real", action="store_true", help="skip the real-map baseline")
    args = ap.parse_args()
    label = args.label or Path(args.ckpt).parts[-3]  # the run-id folder

    random.seed(0)
    if not args.no_real:
        print("loading real ranked maps from osu!.db ...")
        paths = sorted(ranked_osu_paths(SONGS, DB))
        sample = random.sample(paths, min(N_REAL, len(paths)))
        real = []
        for p in sample:
            try:
                bm = parse_beatmap(p)
                if len(bm.hit_objects) >= 50:
                    real.append(bm)
            except Exception:
                continue
        summarize("REAL ranked", real)

    print(f"\ngenerating SR sweep for {label} (load once, rescale={args.rescale}) ...")
    loaded = load_model(args.ckpt)
    prepared = prepare_audio(AUDIO_2MIN, loaded.device)
    gen_dir = Path("artifacts/eval/phase1")
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen = []
    for sr in GEN_SRS:
        out = gen_dir / f"gen_sr{sr}.osu"
        generate(AUDIO_2MIN, out_path=out, sr=sr, loaded=loaded, prepared=prepared,
                 guidance_rescale=args.rescale)
        gen.append(parse_beatmap(out))
    summarize(f"GENERATED [{label}]", gen)


if __name__ == "__main__":
    main()
