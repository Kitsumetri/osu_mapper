"""A "does this play like a ranked map?" reward over the existing metrics.

Pure / hermetic prototype of the reward in ``docs/v9/task3_rl_alignment.md`` (part A).
Built entirely on machinery we already trust:

- ``metrics.compute_metrics`` — the descriptive pattern vector (spacing, jumps,
  streams, flow, grid-snap, curvature, ...);
- the per-SR-bucket reference distribution from ``corpus_stats`` (mean/std/p10/p90
  of every metric over real ranked maps) — the "what ranked maps look like" gold;
- ``difficulty.star_rating`` (rosu-pp) for the SR-closeness term.

Design stance (see the doc): **band membership, not z-maximisation.** We learned
the hard way (``--spacing-scale`` play feedback) that *maximising* a single metric
makes maps play WORSE. So the per-metric reward is a smooth tent that PEAKS inside
the real p10-p90 band and only falls off OUTSIDE it — there is no gradient that
rewards pushing a metric past the real distribution. The reward saturates at
"indistinguishable from ranked"; it cannot be farmed by going more extreme.

This is a *cheap, gameable-aware* surrogate. The doc argues it must always be
paired with held-out in-game play feedback (trust the mapper over the metric).

  from src.eval.reward import reward_from_osu
  r = reward_from_osu("gen.osu", ref_stats, target_sr=5.0)   # -> RewardBreakdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..difficulty import sr_bucket
from ..metrics import compute_metrics_for_osu

# Metrics that actually distinguish a ranked map from a bad one, with a mapper's
# weighting (rhythm / spacing / flow / grid-snap matter most; cosmetic ratios
# matter least). Weights need not sum to 1 — they are renormalised over whatever
# metrics are present in BOTH the map and the reference bucket. Anything not
# listed here is ignored (e.g. bpm, duration, n_objects are not quality signals).
METRIC_WEIGHTS: dict[str, float] = {
    # rhythm / timing — the spine of a playable map
    "on_quarter_grid_ratio": 2.0,
    "density_per_s": 1.5,
    "stream_ratio": 1.5,
    # spacing / aim — the v8/v9 crux (jumps under-produced); weighted but NOT
    # maximised (band membership caps the upside, see _band_score).
    "mean_spacing_px": 1.5,
    "std_spacing_px": 1.5,
    "jump_ratio": 1.5,
    # flow — already ~real, kept so the reward notices if it regresses
    "mean_turn_angle_deg": 1.0,
    "reversal_ratio": 0.75,
    # shape / structure
    "curved_slider_ratio": 1.0,
    "slider_ratio": 0.75,
    "new_combo_ratio": 0.5,
    "sv_changes_per_min": 0.5,
    # cosmetic (down-weighted; hitsounds are handled by a separate v9 head)
    "kiai_ratio": 0.25,
    "hitsound_ratio": 0.25,
}

# how far OUTSIDE the p10-p90 band (measured in band half-widths) a metric may
# stray before its sub-score hits 0. 1.0 = one extra band-width past the edge.
BAND_FALLOFF = 1.0


@dataclass
class RewardBreakdown:
    """Full, inspectable reward decomposition (so reward hacking is auditable)."""

    reward: float
    quality: float                       # band-membership quality in [0, 1]
    sr_closeness: float                  # SR term in [0, 1]
    achieved_sr: float | None
    target_sr: float
    bucket: str
    per_metric: dict[str, float] = field(default_factory=dict)  # metric -> [0,1]
    n_objects: int = 0


def _band_score(value: float, mean: float, p10: float, p90: float) -> float:
    """Smooth tent in [0, 1]: 1.0 anywhere inside [p10, p90], linearly falling to
    0 by ``BAND_FALLOFF`` band-half-widths outside it.

    Crucially flat-topped inside the real band: there is NO reward gradient for
    pushing a metric further once it already looks like ranked maps. This is the
    anti-reward-hacking core — you cannot farm reward by going more extreme.
    """
    if p10 <= value <= p90:
        return 1.0
    half = max((p90 - p10) / 2.0, 1e-9)
    dist = (p10 - value) if value < p10 else (value - p90)
    return max(0.0, 1.0 - dist / (BAND_FALLOFF * half))


def quality_score(metrics: dict, ref_stats: dict, bucket: str) -> tuple[float, dict]:
    """Weighted mean band-membership of the map's metric vector vs a ref bucket.

    Returns (quality in [0,1], per_metric sub-scores). Robust to a partial /
    refreshed reference schema: only metrics present in both, with a positive
    band, contribute (so this is written against the SCHEMA, not specific values).
    """
    ref = ref_stats.get("buckets", {}).get(bucket, {})
    per_metric: dict[str, float] = {}
    num = den = 0.0
    for k, w in METRIC_WEIGHTS.items():
        v = metrics.get(k)
        s = ref.get(k)
        if v is None or not s:
            continue
        p10, p90 = s.get("p10"), s.get("p90")
        if p10 is None or p90 is None or p90 <= p10:
            continue
        sub = _band_score(float(v), s.get("mean", (p10 + p90) / 2), p10, p90)
        per_metric[k] = round(sub, 3)
        num += w * sub
        den += w
    return (num / den if den else 0.0), per_metric


def sr_closeness(achieved_sr: float | None, target_sr: float, tol: float = 0.5) -> float:
    """1.0 when the achieved (rosu) SR hits the target; decays with |error|.

    ``tol`` is the SR error at which the term is ~0.37 (one e-folding). Returns
    0.0 if SR couldn't be computed (a parse failure is a strong "not ranked").
    """
    if achieved_sr is None:
        return 0.0
    err = abs(achieved_sr - target_sr)
    # exp-style decay without importing math.exp for one call: 1/(1+(err/tol)^2)
    # is smooth, bounded, and cheaper; ~0.5 at err=tol, ~0.2 at err=2*tol.
    return 1.0 / (1.0 + (err / max(tol, 1e-6)) ** 2)


def combine(quality: float, sr_close: float, sr_weight: float = 0.35) -> float:
    """Blend quality and SR-closeness into a single scalar in [0, 1].

    A convex blend (not a product) so a momentary SR miss doesn't zero out an
    otherwise-ranked-looking map, but SR still pulls meaningfully. ``sr_weight``
    is the SR share; the rest is pattern quality.
    """
    sr_weight = min(1.0, max(0.0, sr_weight))
    return (1.0 - sr_weight) * quality + sr_weight * sr_close


def reward_from_metrics(metrics: dict, ref_stats: dict, target_sr: float,
                        achieved_sr: float | None, sr_weight: float = 0.35,
                        sr_tol: float = 0.5) -> RewardBreakdown:
    """Compute the full reward from an already-parsed metric dict (hermetic core)."""
    bucket = sr_bucket(target_sr)
    quality, per_metric = quality_score(metrics, ref_stats, bucket)
    sr_close = sr_closeness(achieved_sr, target_sr, tol=sr_tol)
    total = combine(quality, sr_close, sr_weight=sr_weight)
    return RewardBreakdown(
        reward=round(total, 4), quality=round(quality, 4),
        sr_closeness=round(sr_close, 4), achieved_sr=achieved_sr,
        target_sr=target_sr, bucket=bucket, per_metric=per_metric,
        n_objects=int(metrics.get("n_objects", 0)),
    )


def reward_from_osu(osu_path: str | Path, ref_stats: dict, target_sr: float,
                    sr_weight: float = 0.35, sr_tol: float = 0.5) -> RewardBreakdown:
    """End-to-end reward for a generated .osu file (parses + rosu SR + reward).

    The rosu SR call (non-differentiable) lives here, isolated from the pure core
    so the core stays testable without rosu installed.
    """
    from ..difficulty import star_rating
    metrics = compute_metrics_for_osu(osu_path)
    achieved = star_rating(osu_path)
    return reward_from_metrics(metrics, ref_stats, target_sr, achieved,
                               sr_weight=sr_weight, sr_tol=sr_tol)
