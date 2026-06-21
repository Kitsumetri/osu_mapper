"""Best-of-N reward-ranked sampling — pick the best of N generated candidates.

The model under-*produces* (but does not lack) the per-song extremes; the headline
quality gap is that a single sample regresses toward the SR-average (HANDOFF §7,
RESEARCH §10.11/§10.12). Best-of-N is the cheapest lever against that: sample N maps
for a (song, SR), score each with the "ranked-map" reward (``src/eval/reward.py``),
and keep the highest — i.e. *select* the high-reward tail the model already samples
with low probability. No training; it is also the data source for the later RWR/DPO
phases (RESEARCH §10.12.3 Phase 0).

One model load + one audio prepare are reused across all N samples (and all SRs);
only the per-candidate noise seed changes, so the candidates are reproducible.

PREFERRED USER ENTRYPOINT
  For everyday use (reward-ranked generation + auto-packaging into osu! Songs):

    uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 5 6 --best-of-n 8

  This module's CLI (``bestofn``) is the *no-package / debug* path — useful when you
  want to inspect all N candidates or skip the packaging step:

    uv run python -m src.best_of_n --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt \\
        --sr 5 6 7 --n 8 --timing-from ref.osu
    uv run python main.py bestofn --audio song.mp3 --sr 5 --n 12 --keep-candidates
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from .eval.reward import RewardBreakdown, reward_from_osu
from .generate import generate, load_model, prepare_audio

DEFAULT_REF_STATS = "artifacts/reference_stats.json"


def select_winner(breakdowns: list[RewardBreakdown]) -> int:
    """Index of the highest-reward candidate; ties broken by lowest index.

    Pure (no I/O) so the ranking is hermetically testable. Empty list -> -1.
    """
    best_i, best_r = -1, float("-inf")
    for i, bd in enumerate(breakdowns):
        if bd.reward > best_r:
            best_i, best_r = i, bd.reward
    return best_i


def _candidate_row(idx: int, bd: RewardBreakdown) -> dict:
    """One audit row for the report JSON (full breakdown so hacking is visible)."""
    return {"idx": idx, "reward": bd.reward, "quality": bd.quality,
            "sr_closeness": bd.sr_closeness, "achieved_sr": bd.achieved_sr,
            "bucket": bd.bucket, "n_objects": bd.n_objects, "per_metric": bd.per_metric}


def best_of_n(audio_path: str, sr: float, ref_stats: dict, out_path: str = "best.osu",
              ckpt_path: str | None = None, n: int = 8, seed: int = 0,
              loaded=None, prepared=None, timing_ref: str | None = None,
              work_dir: str | None = None, keep_candidates: bool = False,
              **gen_kwargs) -> tuple[str, RewardBreakdown, list[RewardBreakdown]]:
    """Sample ``n`` candidates for ``(audio_path, sr)``, score with the reward, keep
    the best at ``out_path``; write a ``<out>.bon.json`` audit report.

    ``gen_kwargs`` pass straight through to ``generate`` (guidance, density,
    onset_threshold, spacing_scale, steps, amp, batch_cfg, snap, ...). Returns
    ``(winner_path, winner_breakdown, all_breakdowns)``.
    """
    if loaded is None:
        loaded = load_model(ckpt_path)
    if prepared is None:
        prepared = prepare_audio(audio_path, loaded.device, timing_ref)

    work = Path(work_dir) if work_dir else Path(out_path).with_suffix("").parent / (
        "_bon_" + Path(out_path).stem)
    work.mkdir(parents=True, exist_ok=True)

    breakdowns: list[RewardBreakdown] = []
    cand_paths: list[Path] = []
    for i in range(n):
        torch.manual_seed(seed + i)            # reproducible per-candidate variety
        cand = work / f"cand_{i:02d}.osu"
        generate(audio_path, out_path=str(cand), sr=sr,
                 loaded=loaded, prepared=prepared, **gen_kwargs)
        bd = reward_from_osu(cand, ref_stats, target_sr=sr)
        breakdowns.append(bd)
        cand_paths.append(cand)
        print(f"  [bon] cand {i:02d}: R={bd.reward:.4f}  "
              f"quality={bd.quality:.3f}  achieved_sr={bd.achieved_sr}"
              f" (sr_close {bd.sr_closeness:.3f})  n_obj={bd.n_objects}")

    win = select_winner(breakdowns)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cand_paths[win], out_path)
    win_bd = breakdowns[win]
    print(f"  [bon] winner = cand {win:02d}  R={win_bd.reward:.4f}  -> {out_path}")

    report = {
        "audio": str(audio_path), "sr": sr, "n": n, "seed": seed,
        "winner": win, "out": str(out_path),
        "ref_stats_n": ref_stats.get("n_maps"),
        "reward_mean": round(sum(b.reward for b in breakdowns) / len(breakdowns), 4),
        "reward_max": win_bd.reward,
        "candidates": [_candidate_row(i, b) for i, b in enumerate(breakdowns)],
    }
    Path(str(out_path) + ".bon.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not keep_candidates:
        shutil.rmtree(work, ignore_errors=True)
    return out_path, win_bd, breakdowns


def _load_ref_stats(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"reward needs reference stats but '{path}' is missing. Build them with:\n"
            f'  uv run python -m src.corpus_stats --songs "C:/osu!/Songs" --out {path}')
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Best-of-N reward-ranked map generation (sample N, keep the best).")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default=None, help="default: latest runs/*/ckpt/best.pt")
    ap.add_argument("--sr", type=float, nargs="+", required=True,
                    help="target star rating(s); one best-of-N run per SR")
    ap.add_argument("--n", type=int, default=8, help="candidates per SR (default 8)")
    ap.add_argument("--out-dir", default="artifacts/generated",
                    help="where to write the per-SR winners")
    ap.add_argument("--ref-stats", default=DEFAULT_REF_STATS,
                    help="reference_stats.json for the reward (corpus_stats output)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timing-from", default=None,
                    help="read exact BPM+offset from this reference .osu")
    ap.add_argument("--keep-candidates", action="store_true",
                    help="keep all N candidate .osu (default: only the winner)")
    # generation knobs (pass-through to generate)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--guidance-rescale", type=float, default=0.0)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--onset-threshold", type=float, default=0.3)
    ap.add_argument("--spacing-scale", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-batch-cfg", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        from .run_inference import _find_latest_ckpt
        ckpt = _find_latest_ckpt()
        if ckpt is None:
            raise SystemExit("no --ckpt and no runs/*/ckpt/best.pt found")
        print(f"using latest checkpoint: {ckpt}")

    ref_stats = _load_ref_stats(args.ref_stats)
    print(f"reward calibrated on {ref_stats.get('n_maps')} ranked maps ({args.ref_stats})")

    loaded = load_model(ckpt, compile_model=args.compile)
    prepared = prepare_audio(args.audio, loaded.device, timing_ref=args.timing_from)

    gen_kwargs = dict(steps=args.steps, guidance=args.guidance,
                      guidance_rescale=args.guidance_rescale, density=args.density,
                      onset_threshold=args.onset_threshold, spacing_scale=args.spacing_scale,
                      batch_cfg=not args.no_batch_cfg, amp=args.amp)

    out_dir = Path(args.out_dir)
    summary = []
    for sr in args.sr:
        print(f"\n=== best-of-{args.n} @ SR {sr:g} ===")
        out_path = out_dir / f"{Path(args.audio).stem}_bon_sr{sr:g}.osu"
        t0 = time.time()
        _, win_bd, bds = best_of_n(
            args.audio, sr=sr, ref_stats=ref_stats, out_path=str(out_path),
            n=args.n, seed=args.seed, loaded=loaded, prepared=prepared,
            keep_candidates=args.keep_candidates, **gen_kwargs)
        rewards = [b.reward for b in bds]
        summary.append((sr, win_bd.reward, sum(rewards) / len(rewards), out_path))
        print(f"  [bon] SR {sr:g}: best {win_bd.reward:.4f} vs mean "
              f"{sum(rewards) / len(rewards):.4f} (lift "
              f"{win_bd.reward - sum(rewards) / len(rewards):+.4f}) in {time.time() - t0:.0f}s")

    print("\n=== best-of-N summary ===")
    for sr, best, mean, path in summary:
        print(f"  SR {sr:g}: R best {best:.4f} / mean {mean:.4f}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
