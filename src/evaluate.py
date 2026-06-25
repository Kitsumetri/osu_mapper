"""Evaluate a trained model: generate at a sweep of target star ratings and
report how well it tracks the target and matches real maps.

For each target SR it generates a map, measures the *achieved* SR (rosu-pp on the
generated file), computes pattern metrics, and counts how many land within the
real p10-p90 band for that SR bucket.

  python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt \
      --srs 2,3,4,5,6 --ref-stats artifacts/reference_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .difficulty import sr_bucket, star_rating
from .generate import generate, load_model, prepare_audio
from .metrics import compute_metrics_for_osu, score_against_reference

REPORT_KEYS = ["density_per_s", "stream_ratio", "jump_ratio", "on_quarter_grid_ratio",
               "bezier_slider_ratio", "kiai_ratio", "hitsound_ratio"]


def evaluate(audio, ckpt, srs, ref_stats_path=None, out_dir="artifacts/eval",
             guidance=2.0, steps=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = json.loads(Path(ref_stats_path).read_text()) if ref_stats_path else None

    # load the checkpoint + decode the audio once, reuse for every target SR
    loaded = load_model(ckpt)
    prepared = prepare_audio(audio, loaded.device)

    rows = []
    for sr in srs:
        out = out_dir / f"sr{sr}.osu"
        generate(audio, out_path=out, steps=steps, sr=sr, guidance=guidance,
                 loaded=loaded, prepared=prepared)
        m = compute_metrics_for_osu(out)
        achieved = star_rating(out)
        in_range = None
        if ref is not None and achieved is not None:
            _, scored = score_against_reference(m, ref, sr_bucket(achieved))
            in_range = (sum(r[5] for r in scored), len(scored))
        rows.append((sr, achieved, m, in_range))

    print("\n================ EVALUATION ================")
    hdr = (f"{'target':>7}{'got SR':>8}{'dens':>7}{'strm':>6}{'jump':>6}"
           f"{'grid':>6}{'bez':>6}{'kiai':>6}{'hs':>6}")
    if ref is not None:
        hdr += "  in-range"
    print(hdr)
    for sr, achieved, m, in_range in rows:
        # .get with 0.0 defaults: a <2-object generation returns only {"n_objects": n}
        line = (f"{sr:>7.1f}{(achieved or 0):>8.2f}{m.get('density_per_s', 0):>7.2f}"
                f"{m.get('stream_ratio', 0):>6.2f}{m.get('jump_ratio', 0):>6.2f}"
                f"{m.get('on_quarter_grid_ratio', 0):>6.2f}"
                f"{m.get('bezier_slider_ratio', 0):>6.2f}"
                f"{m.get('kiai_ratio', 0):>6.2f}{m.get('hitsound_ratio', 0):>6.2f}")
        if in_range is not None:
            line += f"  {in_range[0]}/{in_range[1]}"
        print(line)
    # does achieved SR track target?
    valid = [(sr, a) for sr, a, _, _ in rows if a]
    if len(valid) >= 2:
        mono = all(valid[i][1] <= valid[i + 1][1] + 0.3 for i in range(len(valid) - 1))
        print(f"\nachieved SR monotonic with target: {mono}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--srs", default="2,3,4,5,6")
    ap.add_argument("--ref-stats", default="artifacts/reference_stats.json")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out-dir", default="artifacts/eval")
    args = ap.parse_args()
    srs = [float(s) for s in args.srs.split(",")]
    ref = args.ref_stats if Path(args.ref_stats).exists() else None
    evaluate(args.audio, args.ckpt, srs, ref, args.out_dir, args.guidance, args.steps)


if __name__ == "__main__":
    main()
