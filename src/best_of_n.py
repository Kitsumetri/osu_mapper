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

import numpy as np
import torch

from .difficulty import sr_bucket
from .eval.reward import RewardBreakdown, quality_score, reward_from_osu
from .generate import generate, load_model, prepare_audio

DEFAULT_REF_STATS = "artifacts/reference_stats.json"

# --- early-abort defaults (all opt-in; the feature defaults to OFF) ----------
# Monitor only the LATE steps: early x0 is blurry and its decoded quality is
# meaningless, so we never abort before this fraction of the reverse process is
# done (the user's "only the later denoising steps" constraint).
EA_MONITOR_FRAC = 0.70          # arm the monitor over the last ~30% of steps
# Step-relative margin: a candidate is a kill-candidate only if, at a monitored
# step, its proxy quality is more than this far BELOW the best COMPLETED candidate
# at the same step. Bigger = more conservative (aborts fewer).
EA_MARGIN = 0.15
# Absolute floor: NEVER abort a candidate whose proxy quality is still >= this,
# even if it trails the best-so-far. A second safety so a merely-behind (but
# respectable) candidate is never thrown away — we only kill genuine bottom-feeders.
EA_ABS_FLOOR = 0.55
# Don't arm the step-relative rule until this many candidates have COMPLETED, so
# the best-so-far per-step cache is trustworthy (the first candidate(s) are never
# aborted blindly — the user's "stop if the reward curve approaches the lowest
# values" needs a population to define "lowest").
EA_MIN_CANDIDATES = 2


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


# --- early-abort: cheap quality proxy + step-relative monitor ----------------
# The whole point is to spend best-of-N's compute on viable candidates: at LATE
# DDIM steps, decode the partial x0 IN MEMORY (no file I/O) into a *cheap* quality
# proxy and abort candidates that are heading to the bottom of the reward curve, so
# the remaining steps aren't wasted on a sample that would have lost anyway. This is
# an efficiency gate, NOT steering (ddim_sample is eta=0 deterministic, the reward
# is non-differentiable) — we only ever ABORT, never resample. Correctness property:
# the selected winner is IDENTICAL with and without early-abort, because we only kill
# candidates that trail the best COMPLETED candidate by a margin AND sit below an
# absolute floor (a would-be loser).


def cheap_quality(sig: np.ndarray, ref_stats: dict, bucket: str, bpm: float,
                  *, decode=None, metrics_fn=None, quality_fn=None,
                  onset_threshold: float = 0.3) -> float:
    """The cheap quality proxy for a partial (or final) decoded signal.

    Decodes ``sig`` (C, T) into hit objects, wraps them in a minimal in-memory
    ``Beatmap`` (one uninherited timing point so ``compute_metrics`` has a BPM, and
    so the metrics' grid/stream/spacing terms are meaningful), computes the metric
    vector, and returns ``quality_score`` band-membership ONLY — deliberately NOT the
    rosu star rating (that dominates the per-step cost; aborting cheaply is the whole
    point). No file I/O. Returns 0.0 for an undecodable / too-short signal (a strong
    "this is going nowhere"). The ``decode``/``metrics_fn``/``quality_fn`` hooks let
    tests inject deterministic fakes without a model or rosu.
    """
    decode = decode or _default_decode
    metrics_fn = metrics_fn or _default_metrics
    quality_fn = quality_fn or quality_score
    objs = decode(sig, onset_threshold=onset_threshold)
    bm = _bm_from_objects(objs, bpm)
    metrics = metrics_fn(bm)
    if metrics.get("n_objects", 0) < 2:
        return 0.0
    q, _per_metric, _families = quality_fn(metrics, ref_stats, bucket)
    return float(q)


def _default_decode(sig, onset_threshold: float = 0.3):
    from .data.signal import decode_signal
    return decode_signal(sig, onset_threshold=onset_threshold)


def _default_metrics(bm):
    from .metrics import compute_metrics
    return compute_metrics(bm)


def _bm_from_objects(objs, bpm: float):
    """Wrap decoded hit objects in a throwaway in-memory Beatmap for metrics.

    A single uninherited timing point gives ``Beatmap.bpm`` the value
    ``compute_metrics`` needs for its beat-relative (grid/stream) terms; nothing is
    written to disk.
    """
    from .parsing.beatmap import Beatmap, TimingPoint
    beat_len = 60000.0 / bpm if bpm > 0 else 500.0
    bm = Beatmap(path=Path("_inmem_.osu"))
    bm.timing_points = [TimingPoint(0.0, beat_len, 4, True)]
    bm.hit_objects = list(objs)
    return bm


class EarlyAbortMonitor:
    """Per-candidate ``ddim_sample`` monitor: aborts a doomed candidate at a late step.

    Holds a reference to a SHARED ``best_by_step`` cache (best proxy quality among
    COMPLETED candidates, keyed by the remaining-step index ``k``) so the abort rule
    is step-relative: "stop if this candidate's reward curve approaches the lowest
    values" becomes "abort if, at a monitored late step, its proxy quality trails the
    best completed candidate at that step by more than ``margin`` AND is below the
    absolute ``floor``." Both conditions must hold (err strongly toward NOT aborting:
    a false abort throws away a possible winner). The rule is disarmed until
    ``min_candidates`` have completed (the cache must be trustworthy first).

    The monitor records ``aborted`` / ``abort_k`` / ``last_quality`` / per-step
    qualities on itself; ``best_of_n`` reads them after the call and — only for a
    candidate that COMPLETED — folds its per-step qualities into ``best_by_step``.
    Keeping the cache to completed candidates makes "best-so-far at this step"
    well-defined and independent of which candidates happened to abort.
    """

    def __init__(self, ref_stats: dict, bucket: str, bpm: float,
                 best_by_step: dict[int, float], n_completed: int,
                 *, monitor_frac: float = EA_MONITOR_FRAC, margin: float = EA_MARGIN,
                 abs_floor: float = EA_ABS_FLOOR, min_candidates: int = EA_MIN_CANDIDATES,
                 onset_threshold: float = 0.3, quality_fn=None, decode=None,
                 metrics_fn=None):
        self.ref_stats = ref_stats
        self.bucket = bucket
        self.bpm = bpm
        self.best_by_step = best_by_step
        self.n_completed = n_completed
        self.monitor_frac = monitor_frac
        self.margin = margin
        self.abs_floor = abs_floor
        self.min_candidates = min_candidates
        self.onset_threshold = onset_threshold
        self.quality_fn = quality_fn
        self.decode = decode
        self.metrics_fn = metrics_fn
        # recorded state (read by best_of_n after sampling)
        self.aborted = False
        self.abort_k: int | None = None
        self.abort_frac: float | None = None
        self.last_quality: float | None = None
        self.step_quality: dict[int, float] = {}

    def _quality(self, x0) -> float:
        sig = x0[0].float().cpu().numpy() if hasattr(x0, "cpu") else np.asarray(x0)[0]
        return cheap_quality(sig, self.ref_stats, self.bucket, self.bpm,
                             decode=self.decode, metrics_fn=self.metrics_fn,
                             quality_fn=self.quality_fn,
                             onset_threshold=self.onset_threshold)

    def __call__(self, k: int, frac_done: float, x0) -> bool:
        # Only the late steps: early x0 is blurry and its decoded quality is noise.
        if frac_done < self.monitor_frac:
            return False
        q = self._quality(x0)
        self.last_quality = q
        self.step_quality[k] = q
        # Disarmed until enough candidates have completed -> cache is trustworthy.
        if self.n_completed < self.min_candidates:
            return False
        best_at = self.best_by_step.get(k)
        if best_at is None:
            return False
        # Step-relative AND absolute: kill only a genuine bottom-feeder.
        if q < best_at - self.margin and q < self.abs_floor:
            self.aborted = True
            self.abort_k = k
            self.abort_frac = frac_done
            return True
        return False


def _bpm_of(prepared) -> float:
    """BPM from the prepared timing point (for the in-memory monitor metrics)."""
    bl = getattr(getattr(prepared, "tp", None), "beat_length", 0.0) or 0.0
    return 60000.0 / bl if bl > 0 else 0.0


def _patched_monitor(diff, monitor):
    """Context-manager-ish helper: temporarily inject ``monitor`` into this diff's
    ``ddim_sample`` so the unmodified ``generate`` path runs WITH the monitor.

    Returns ``(install, restore)`` callables. We monkey-patch the bound method on the
    instance (not the class) for exactly one ``generate`` call, then restore it — so
    we never edit generate.py yet the monitor rides the *production* sampling pass.
    With no abort the monitor cannot change x0 (proven byte-identical in diffusion.py),
    so a candidate that completes is byte-for-byte identical to a plain ``generate``.
    """
    orig = diff.ddim_sample

    def install():
        def wrapped(*a, **kw):
            kw.setdefault("monitor", monitor)
            return orig(*a, **kw)
        diff.ddim_sample = wrapped

    def restore():
        diff.ddim_sample = orig

    return install, restore


def best_of_n(audio_path: str, sr: float, ref_stats: dict, out_path: str = "best.osu",
              ckpt_path: str | None = None, n: int = 8, seed: int = 0,
              loaded=None, prepared=None, timing_ref: str | None = None,
              work_dir: str | None = None, keep_candidates: bool = False,
              early_abort: bool = False, ea_monitor_frac: float = EA_MONITOR_FRAC,
              ea_margin: float = EA_MARGIN, ea_abs_floor: float = EA_ABS_FLOOR,
              ea_min_candidates: int = EA_MIN_CANDIDATES,
              **gen_kwargs) -> tuple[str, RewardBreakdown, list[RewardBreakdown]]:
    """Sample ``n`` candidates for ``(audio_path, sr)``, score with the reward, keep
    the best at ``out_path``; write a ``<out>.bon.json`` audit report.

    ``gen_kwargs`` pass straight through to ``generate`` (guidance, density,
    onset_threshold, spacing_scale, steps, amp, batch_cfg, snap, ...). Returns
    ``(winner_path, winner_breakdown, all_breakdowns)``.

    ``early_abort`` (opt-in, default OFF — existing behaviour is unchanged): monitor
    the LATE DDIM steps of each candidate with a cheap decode->quality proxy and abort
    a candidate that is heading to the bottom of the reward curve, so best-of-N's
    compute is spent on viable candidates. Aborted candidates are EXCLUDED from the
    ranking (they would have lost), so the selected winner is identical with and
    without early-abort. The abort rule is step-relative (trails the best COMPLETED
    candidate at the same step by > ``ea_margin``) AND absolute (below ``ea_abs_floor``),
    armed only after ``ea_min_candidates`` complete; ``ea_monitor_frac`` sets how late
    monitoring begins. ``<out>.bon.json`` surfaces ``n_aborted`` / ``steps_saved``.
    """
    if loaded is None:
        loaded = load_model(ckpt_path)
    if prepared is None:
        prepared = prepare_audio(audio_path, loaded.device, timing_ref)

    work = Path(work_dir) if work_dir else Path(out_path).with_suffix("").parent / (
        "_bon_" + Path(out_path).stem)
    work.mkdir(parents=True, exist_ok=True)

    bucket = sr_bucket(sr)
    bpm = _bpm_of(prepared)
    onset_threshold = gen_kwargs.get("onset_threshold", 0.3)
    steps = gen_kwargs.get("steps", 100)
    best_by_step: dict[int, float] = {}        # shared per-step best of COMPLETED cands
    n_completed = 0
    n_aborted = 0
    steps_saved = 0
    abort_info: dict[int, dict] = {}           # idx -> {abort_k, abort_frac, last_quality}

    breakdowns: list[RewardBreakdown | None] = []
    cand_paths: list[Path | None] = []
    for i in range(n):
        torch.manual_seed(seed + i)            # reproducible per-candidate variety
        cand = work / f"cand_{i:02d}.osu"
        monitor = None
        if early_abort:
            monitor = EarlyAbortMonitor(
                ref_stats, bucket, bpm, best_by_step, n_completed,
                monitor_frac=ea_monitor_frac, margin=ea_margin, abs_floor=ea_abs_floor,
                min_candidates=ea_min_candidates, onset_threshold=onset_threshold)
            install, restore = _patched_monitor(loaded.diff, monitor)
            install()
            try:
                generate(audio_path, out_path=str(cand), sr=sr,
                         loaded=loaded, prepared=prepared, **gen_kwargs)
            finally:
                restore()
        else:
            generate(audio_path, out_path=str(cand), sr=sr,
                     loaded=loaded, prepared=prepared, **gen_kwargs)

        if monitor is not None and monitor.aborted:
            # A doomed candidate: skip the EXPENSIVE reward (rosu SR + full parse) and
            # exclude it from ranking. steps_saved = the remaining steps we didn't run
            # (we broke at remaining-step index abort_k, so abort_k steps were skipped).
            n_aborted += 1
            steps_saved += int(monitor.abort_k or 0)
            abort_info[i] = {"abort_k": monitor.abort_k, "abort_frac": monitor.abort_frac,
                             "last_quality": round(monitor.last_quality or 0.0, 4)}
            breakdowns.append(None)
            cand_paths.append(None)
            print(f"  [bon] cand {i:02d}: ABORTED at step k={monitor.abort_k} "
                  f"(frac {monitor.abort_frac:.2f}, proxy_q={monitor.last_quality:.3f}) "
                  f"-> {monitor.abort_k} steps saved")
            continue

        bd = reward_from_osu(cand, ref_stats, target_sr=sr)
        breakdowns.append(bd)
        cand_paths.append(cand)
        if monitor is not None:
            # fold this COMPLETED candidate's late-step proxy qualities into the shared
            # best-so-far cache (max per step) so the step-relative rule is well-defined.
            for k, q in monitor.step_quality.items():
                if q > best_by_step.get(k, float("-inf")):
                    best_by_step[k] = q
            n_completed += 1
        print(f"  [bon] cand {i:02d}: R={bd.reward:.4f}  "
              f"quality={bd.quality:.3f}  achieved_sr={bd.achieved_sr}"
              f" (sr_close {bd.sr_closeness:.3f})  n_obj={bd.n_objects}")

    # rank only the COMPLETED candidates; aborted ones (None) are excluded.
    completed = [(i, bd) for i, bd in enumerate(breakdowns) if bd is not None]
    if not completed:
        raise SystemExit(
            "best-of-N: every candidate was aborted — loosen --ea-margin / --ea-abs-floor "
            "or raise --ea-min-candidates (this should be near-impossible by design).")
    win = max(completed, key=lambda t: t[1].reward)[0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cand_paths[win], out_path)
    win_bd = breakdowns[win]
    if early_abort:
        print(f"  [bon] early-abort: {n_aborted}/{n} aborted, {steps_saved} DDIM steps saved")
    print(f"  [bon] winner = cand {win:02d}  R={win_bd.reward:.4f}  -> {out_path}")

    rewards = [bd.reward for _, bd in completed]
    report = {
        "audio": str(audio_path), "sr": sr, "n": n, "seed": seed,
        "winner": win, "out": str(out_path),
        "ref_stats_n": ref_stats.get("n_maps"),
        "reward_mean": round(sum(rewards) / len(rewards), 4),
        "reward_max": win_bd.reward,
        "early_abort": early_abort,
        "n_aborted": n_aborted,
        "n_completed": len(completed),
        "steps_saved": steps_saved,
        "steps_per_candidate": steps,
        "aborted": abort_info,
        "candidates": [_candidate_row(i, b) for i, b in enumerate(breakdowns) if b is not None],
    }
    Path(str(out_path) + ".bon.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not keep_candidates:
        shutil.rmtree(work, ignore_errors=True)
    # return the full per-index list (aborted entries as None) so callers can see the
    # full slate; the public contract (winner_path, winner_bd, breakdowns) is kept.
    return out_path, win_bd, [bd for _, bd in completed]


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
    # early-abort (opt-in; default OFF leaves best-of-N behaviour unchanged)
    ap.add_argument("--early-abort", action="store_true",
                    help="abort doomed candidates at late DDIM steps via a cheap "
                         "decode->quality proxy (saves compute; winner unchanged)")
    ap.add_argument("--ea-monitor-frac", type=float, default=EA_MONITOR_FRAC,
                    help="only monitor steps past this fraction of the reverse process")
    ap.add_argument("--ea-margin", type=float, default=EA_MARGIN,
                    help="abort if proxy quality trails the best completed by > this")
    ap.add_argument("--ea-abs-floor", type=float, default=EA_ABS_FLOOR,
                    help="never abort a candidate whose proxy quality is still >= this")
    ap.add_argument("--ea-min-candidates", type=int, default=EA_MIN_CANDIDATES,
                    help="arm the abort rule only after this many candidates complete")
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
            keep_candidates=args.keep_candidates, early_abort=args.early_abort,
            ea_monitor_frac=args.ea_monitor_frac, ea_margin=args.ea_margin,
            ea_abs_floor=args.ea_abs_floor, ea_min_candidates=args.ea_min_candidates,
            **gen_kwargs)
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
