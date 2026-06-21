"""A "does this play like a ranked map?" reward over the existing metrics.

Pure / hermetic prototype of the reward in ``docs/v9/task3_rl_alignment.md`` (part A),
made **pattern-balanced** in ``docs/v9/task_general_reward.md`` (the general-reward
task). Built entirely on machinery we already trust:

- ``metrics.compute_metrics`` — the descriptive pattern vector (spacing, jumps,
  streams, flow, grid-snap, curvature, ...);
- the per-SR-bucket reference distribution from ``corpus_stats`` (mean/std/p10/p90
  of every metric over real ranked maps) — the "what ranked maps look like" gold;
- ``difficulty.star_rating`` (rosu-pp) for the SR-closeness term.

Design stance (see the docs): **band membership, not z-maximisation.** We learned
the hard way (``--spacing-scale`` play feedback) that *maximising* a single metric
makes maps play WORSE. So the per-metric reward is a smooth tent that PEAKS inside
the real p10-p90 band and only falls off OUTSIDE it — there is no gradient that
rewards pushing a metric past the real distribution. The reward saturates at
"indistinguishable from ranked"; it cannot be farmed by going more extreme.

**General, not jump-biased.** The old reward took a flat weighted mean over 14
metrics, so the *spacing/aim* family (3 metrics, mean_spacing+std_spacing+jump)
out-voted *slider-shape* (whose single off-band metric had a tiny share). A great
stream map or a great tech/slider map could not score as high as a jump map. The
fix: group metrics into **pattern families** (rhythm, spacing/aim, flow,
slider-shape, accents) and score quality as a weighted mean **over families**, so a
family's contribution no longer depends on how many metrics it happens to contain.
Each family is itself a band-membership mean, so the anti-hacking flat-top is
preserved end to end.

  from src.eval.reward import reward_from_osu
  r = reward_from_osu("gen.osu", ref_stats, target_sr=5.0)   # -> RewardBreakdown
  r.family_breakdown   # {"rhythm": 1.0, "spacing_aim": 1.0, "slider_shape": 0.0, ...}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..difficulty import sr_bucket
from ..metrics import compute_metrics_for_osu

# --- pattern families -------------------------------------------------------
# Metrics grouped by what they describe about play. Quality is a weighted mean
# *over families*, and within a family a weighted mean over its metrics, so each
# family contributes by its FAMILY weight regardless of how many metrics it has
# (the old flat mean let spacing's 3 metrics out-vote slider-shape's — the core
# jump-bias this redesign fixes). Per-metric weights only set RELATIVE importance
# *inside* a family; they do not change the family's overall share.
#
# Anything not listed is ignored (bpm, duration, n_objects are scene facts, not
# quality signals — and rewarding n_objects would invite spam).

# (family_weight, {metric: within_family_weight})
FAMILIES: dict[str, tuple[float, dict[str, float]]] = {
    # WHEN you click — the spine of a playable map. Grid-snap is the single
    # strongest ranked/non-ranked discriminator; density sets the right map for
    # the song. (on_quarter_grid assumes a single BPM — see the variable-BPM
    # caveat below; it is one metric of one family, so it can't dominate.)
    "rhythm": (1.0, {
        "on_quarter_grid_ratio": 2.0,
        "density_per_s": 1.5,
    }),
    # HOW FAR the cursor travels — the v8/v9 crux (jumps under-produced). Three
    # metrics, but as ONE family it no longer out-votes the rest. Band-capped,
    # never maximised (flat top, see _band_score).
    "spacing_aim": (1.0, {
        "mean_spacing_px": 1.0,
        "std_spacing_px": 1.0,
        "jump_ratio": 1.0,
    }),
    # HOW the cursor moves — directional character. stream_ratio lives here (a
    # stream is a flow/movement axis), so a great *stream* map is rewarded as a
    # flow family rather than being lumped against jumps. This is what lets a
    # stream map and a jump map both score high: they trade off WITHIN families
    # the song's real bucket already accounts for.
    "flow": (1.0, {
        "stream_ratio": 1.5,
        "mean_turn_angle_deg": 1.0,
        "reversal_ratio": 0.75,
    }),
    # SLIDER & combo structure — the family the old reward drowned out. Given
    # full family weight so a tech/slider map is judged on its sliders, not
    # masked by perfect spacing 1.0s.
    "slider_shape": (1.0, {
        "slider_ratio": 1.0,
        "curved_slider_ratio": 1.0,
        "sv_changes_per_min": 0.75,
        "new_combo_ratio": 0.5,
    }),
    # cosmetic accents (hitsounds/kiai handled by a separate v9 head) — low
    # family weight so they nudge, never decide.
    "accents": (0.4, {
        "kiai_ratio": 0.5,
        "hitsound_ratio": 0.5,
    }),
}

# Back-compat: a flat metric->effective-weight view some callers / tests read.
# Effective weight = family_weight * (within_weight / sum_of_within_weights),
# i.e. the metric's true share of the final quality. Kept so anything importing
# METRIC_WEIGHTS keeps working; the reward itself uses FAMILIES.
METRIC_WEIGHTS: dict[str, float] = {
    m: fam_w * (w / sum(metrics.values()))
    for fam_w, metrics in FAMILIES.values()
    for m, w in metrics.items()
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
    family_breakdown: dict[str, float] = field(default_factory=dict)  # family -> [0,1]
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


def _metric_band_score(value, stat: dict) -> float | None:
    """Band-membership of one metric value vs its reference summary, or None if
    the summary lacks a usable band. Written against the SCHEMA, not values."""
    if value is None or not stat:
        return None
    p10, p90 = stat.get("p10"), stat.get("p90")
    if p10 is None or p90 is None or p90 <= p10:
        return None
    return _band_score(float(value), stat.get("mean", (p10 + p90) / 2), p10, p90)


def quality_score(metrics: dict, ref_stats: dict, bucket: str
                  ) -> tuple[float, dict, dict]:
    """Family-balanced band-membership quality of the map's metric vector.

    Returns ``(quality in [0,1], per_metric sub-scores, family sub-scores)``.

    Each family's score is the within-family weighted mean band-membership over
    the metrics present in BOTH the map and the ref bucket; ``quality`` is the
    family-weighted mean over the families that had at least one usable metric.
    Because the outer mean is over families (not metrics), a family with three
    metrics (spacing) and a family with one usable metric contribute by their
    *family* weight — the fix for the jump/spacing bias. Robust to a partial or
    refreshed reference schema: missing metrics/families just drop out.
    """
    ref = ref_stats.get("buckets", {}).get(bucket, {})
    per_metric: dict[str, float] = {}
    family_scores: dict[str, float] = {}
    q_num = q_den = 0.0
    for fam, (fam_w, fam_metrics) in FAMILIES.items():
        f_num = f_den = 0.0
        for k, w in fam_metrics.items():
            sub = _metric_band_score(metrics.get(k), ref.get(k))
            if sub is None:
                continue
            per_metric[k] = round(sub, 3)
            f_num += w * sub
            f_den += w
        if f_den == 0.0:
            continue                      # no usable metric in this family
        fam_score = f_num / f_den
        family_scores[fam] = round(fam_score, 3)
        q_num += fam_w * fam_score
        q_den += fam_w
    quality = q_num / q_den if q_den else 0.0
    return quality, per_metric, family_scores


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
    quality, per_metric, family_scores = quality_score(metrics, ref_stats, bucket)
    sr_close = sr_closeness(achieved_sr, target_sr, tol=sr_tol)
    total = combine(quality, sr_close, sr_weight=sr_weight)
    return RewardBreakdown(
        reward=round(total, 4), quality=round(quality, 4),
        sr_closeness=round(sr_close, 4), achieved_sr=achieved_sr,
        target_sr=target_sr, bucket=bucket, per_metric=per_metric,
        family_breakdown=family_scores,
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
