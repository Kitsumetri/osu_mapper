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
from ..metrics import compute_metrics

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
#
# v9 rhythm>>flow reweight (task_reward_flow.md): WRONG RHYTHM = UNPLAYABLE,
# wrong flow = merely stylistic. So rhythm is the heaviest family (2.0) and flow
# the lightest pattern family (0.6); within rhythm, grid-snap is up-weighted
# (2.0 -> 3.0) so snapping dominates the rhythm score. on_quarter_grid_ratio now
# credits the standard {1/4, 1/8, 1/6} grid (round-3a-fix), not 1/4 alone, so
# correctly-snapped bursts/triplets are no longer mis-scored as off-grid. (CAVEAT:
# it still assumes a single BPM and under-measures variable-BPM ranked maps;
# up-weighting amplifies that bias in GOLD calibration / real-map validation, but
# it is SAFE for GENERATED maps which are single-BPM by construction — see the doc.)
FAMILIES: dict[str, tuple[float, dict[str, float]]] = {
    # WHEN you click — the spine of a playable map. Grid-snap is the single
    # strongest ranked/non-ranked discriminator; density sets the right map for
    # the song. HEAVIEST family (rhythm>>flow): a wrong-rhythm map is unplayable.
    "rhythm": (2.0, {
        "on_quarter_grid_ratio": 3.0,   # up-weighted: {1/4,1/8,1/6} snap dominates
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
    # flow family rather than being lumped against jumps. LIGHTEST pattern family
    # (rhythm>>flow): flow is stylistic, not playability. stream_spacing_cv is a
    # NEW distributional trait (stream regularity) — band-less until the next
    # gold-stats refresh, so reward.py ignores it gracefully for now.
    "flow": (0.6, {
        "stream_ratio": 1.5,
        "mean_turn_angle_deg": 1.0,
        "reversal_ratio": 0.75,
        "stream_spacing_cv": 1.0,       # NEW (band-less until gold refresh)
        "stack_ratio": 0.5,             # NEW: stacking rate (band-less until refresh)
    }),
    # SLIDER & combo structure — the family the old reward drowned out. Given
    # full family weight so a tech/slider map is judged on its sliders, not
    # masked by perfect spacing 1.0s. slider_anchor_spread_px is a NEW
    # distributional trait (sane anchor spacing) — band-less until gold refresh.
    "slider_shape": (1.0, {
        "slider_ratio": 1.0,
        "curved_slider_ratio": 1.0,
        "sv_changes_per_min": 0.75,
        "new_combo_ratio": 0.5,
        "slider_anchor_spread_px": 0.75,    # NEW (band-less until gold refresh)
        "slider_overlap_ratio": 0.5,        # NEW: overlap rate (band-less until refresh)
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

# --- playability penalty (ONE objective defect, NO gold band needed) --------
# v9 round-3a recalibration (audit n=87176): the earlier penalty also flagged
# stacks, slider/object overlaps, and tight ("degenerate") slider anchors. But
# those are INTENTIONAL osu! patterns — real ranked maps do all three — so the
# penalty fired on gold maps (playability averaged 0.76, dragging reward 0.95 ->
# 0.72). Those three are now DISTRIBUTIONAL band metrics instead (stack_ratio +
# slider_overlap_ratio in metrics.compute_metrics; slider_anchor_spread_px was
# already there): a ranked-typical amount scores 1.0, only a far-out-of-band
# amount costs — exactly like every other metric, and anti-hackable by the flat
# top.
#
# What remains here is the one thing NO ranked map of any style does and that has
# no meaningful distribution to sit inside: a cursor move that is physically
# impossible at the map's own rhythm (the decode-glitch teleport failure mode of
# a GENERATED map). It needs no band — a clean map has ~zero — so we keep it as a
# bounded fraction of affected pairs, folded MULTIPLICATIVELY into the reward
# (clean -> playability 1.0 -> reward unchanged; glitchy -> < 1.0). Anti-hackable:
# it can only LOWER the reward and is bounded in [0, 1].

# A jump is unhittable if the cursor must move faster than this. ~150 BPM 1/2
# jump of 300 px / 200 ms = 1.5 px/ms is hard-but-fine; even the hardest ranked
# 1/4 jump-bursts top out ~6-7 px/ms. The earlier 4.0 clipped real aim maps (the
# audit's tech/jump tail); 10.0 clears the hardest ranked aim while still catching
# teleport glitches (full playfield ~600 px in < 60 ms -> 10+ px/ms). TUNE to the
# gold p99: re-run `measure_reward --gold` and read the per-defect unhittable_jump
# rate — it should be ~0 on gold; raise this if it is not.
UNHITTABLE_PX_PER_MS = 10.0


def _unhittable_jump_rate(objs: list) -> float:
    """Fraction of consecutive pairs whose required cursor velocity
    (distance / Δt, px per ms) exceeds the human ceiling — a jump that cannot be
    hit at the map's own rhythm."""
    pairs = [(a, b) for a, b in zip(objs, objs[1:]) if (b.time - a.time) > 0]
    if not pairs:
        return 0.0
    bad = 0
    for a, b in pairs:
        dt = b.time - a.time
        dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
        if dist / dt > UNHITTABLE_PX_PER_MS:
            bad += 1
    return bad / len(pairs)


# per-defect weights (relative importance inside the penalty). Only the
# physically-impossible-velocity defect remains — stacks/overlaps/anchors became
# distributional band metrics (see the note above). Kept as a dict so the penalty
# stays trivially extensible if a genuinely band-less defect turns up later.
DEFECT_WEIGHTS = {
    "unhittable_jump": 1.0,
}


def playability_penalty(bm) -> tuple[float, dict[str, float]]:
    """Bounded objective-defect penalty in [0, 1] for a parsed Beatmap.

    Returns ``(penalty, per_defect_rates)``. The penalty is the DEFECT_WEIGHTS-
    weighted mean of the defect rates (each a fraction of affected pairs in
    [0, 1]) — currently just the physically-impossible cursor-velocity defect, so
    the penalty is itself in [0, 1]. A clean ranked map scores ~0; glitchy maps
    raise it toward 1. ``playability = 1 - penalty`` is what folds into the reward.
    """
    objs = sorted(bm.hit_objects, key=lambda o: o.time)
    if len(objs) < 2:
        return 0.0, {}
    rates = {
        "unhittable_jump": _unhittable_jump_rate(objs),
    }
    num = sum(DEFECT_WEIGHTS[k] * v for k, v in rates.items())
    den = sum(DEFECT_WEIGHTS.values())
    penalty = num / den if den else 0.0
    return min(1.0, max(0.0, penalty)), {k: round(v, 4) for k, v in rates.items()}


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
    playability: float = 1.0             # 1 - defect penalty, in [0, 1]
    defects: dict[str, float] = field(default_factory=dict)  # defect -> rate [0,1]


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
                        sr_tol: float = 0.5, playability: float = 1.0,
                        defects: dict[str, float] | None = None) -> RewardBreakdown:
    """Compute the full reward from an already-parsed metric dict (hermetic core).

    ``playability`` in [0, 1] is the objective-defect multiplier (1.0 = clean;
    see ``playability_penalty``). It folds in MULTIPLICATIVELY *after* the convex
    quality/SR blend, so it can only pull a defective map DOWN — it cannot lift a
    map above its band-membership ceiling (anti-hackable), and the product of two
    [0, 1] numbers stays in [0, 1]. Defaults to 1.0 so the pure core is testable
    without a Beatmap and old callers see no change on clean maps.
    """
    bucket = sr_bucket(target_sr)
    quality, per_metric, family_scores = quality_score(metrics, ref_stats, bucket)
    sr_close = sr_closeness(achieved_sr, target_sr, tol=sr_tol)
    blended = combine(quality, sr_close, sr_weight=sr_weight)
    playability = min(1.0, max(0.0, playability))
    total = blended * playability
    return RewardBreakdown(
        reward=round(total, 4), quality=round(quality, 4),
        sr_closeness=round(sr_close, 4), achieved_sr=achieved_sr,
        target_sr=target_sr, bucket=bucket, per_metric=per_metric,
        family_breakdown=family_scores,
        n_objects=int(metrics.get("n_objects", 0)),
        playability=round(playability, 4), defects=defects or {},
    )


def reward_from_osu(osu_path: str | Path, ref_stats: dict, target_sr: float,
                    sr_weight: float = 0.35, sr_tol: float = 0.5,
                    bm=None) -> RewardBreakdown:
    """End-to-end reward for a generated .osu file (parses + rosu SR + reward).

    The rosu SR call (non-differentiable) lives here, isolated from the pure core
    so the core stays testable without rosu installed. The beatmap is parsed ONCE
    and reused for metrics, the rosu SR call, and the playability penalty; pass a
    pre-parsed ``bm`` (e.g. a caller that already parsed it to gold-filter) to skip
    the re-parse.
    """
    from ..difficulty import star_rating
    from ..parsing.beatmap import parse_beatmap
    if bm is None:
        bm = parse_beatmap(osu_path)
    metrics = compute_metrics(bm)
    achieved = star_rating(osu_path)
    penalty, defect_rates = playability_penalty(bm)
    return reward_from_metrics(metrics, ref_stats, target_sr, achieved,
                               sr_weight=sr_weight, sr_tol=sr_tol,
                               playability=1.0 - penalty, defects=defect_rates)
