"""Hermetic tests for the v9 rhythm>>flow reweight + playability penalty wiring.

Exercise the PURE reward core (no rosu): the family weights changed as intended,
the playability multiplier pulls defective maps down and is a no-op on clean ones,
the reward stays in [0, 1], and band-less NEW metrics don't break anything.
"""
from src.eval.reward import (
    DEFECT_WEIGHTS,
    FAMILIES,
    METRIC_WEIGHTS,
    quality_score,
    reward_from_metrics,
)

# a ref bucket with bands for the OLD metrics only — deliberately NO bands for the
# new distributional metrics (stream_spacing_cv, slider_anchor_spread_px), to prove
# reward.py ignores band-less metrics until a future gold-stats refresh adds them.
REF = {"buckets": {"Insane": {
    "mean_spacing_px": {"mean": 150.0, "std": 40.0, "p10": 100.0, "p90": 200.0},
    "std_spacing_px": {"mean": 80.0, "std": 12.0, "p10": 66.0, "p90": 96.0},
    "jump_ratio": {"mean": 0.25, "std": 0.1, "p10": 0.1, "p90": 0.4},
    "on_quarter_grid_ratio": {"mean": 0.8, "std": 0.1, "p10": 0.65, "p90": 0.95},
    "density_per_s": {"mean": 4.0, "std": 0.9, "p10": 3.0, "p90": 5.0},
    "stream_ratio": {"mean": 0.3, "std": 0.2, "p10": 0.1, "p90": 0.6},
    "mean_turn_angle_deg": {"mean": 90.0, "std": 20.0, "p10": 60.0, "p90": 120.0},
    "reversal_ratio": {"mean": 0.2, "std": 0.1, "p10": 0.05, "p90": 0.4},
    "slider_ratio": {"mean": 0.44, "std": 0.11, "p10": 0.3, "p90": 0.58},
    "curved_slider_ratio": {"mean": 0.4, "std": 0.2, "p10": 0.2, "p90": 0.6},
}}}


# --- Part A: the rhythm >> flow reweight ------------------------------------

def test_rhythm_family_is_heaviest_flow_is_lightest_pattern():
    rhythm_w = FAMILIES["rhythm"][0]
    flow_w = FAMILIES["flow"][0]
    spacing_w = FAMILIES["spacing_aim"][0]
    # rhythm must outweigh flow (wrong rhythm = unplayable; wrong flow = stylistic)
    assert rhythm_w > flow_w
    assert rhythm_w > spacing_w
    # flow is the lightest PATTERN family (accents is the cosmetic one below it)
    assert flow_w < spacing_w
    assert flow_w < FAMILIES["slider_shape"][0]


def test_grid_snap_within_rhythm_up_weighted():
    grid_w = FAMILIES["rhythm"][1]["on_quarter_grid_ratio"]
    density_w = FAMILIES["rhythm"][1]["density_per_s"]
    # 1/4 grid-snap dominates the rhythm family (the heaviest within-rhythm metric)
    assert grid_w >= 3.0
    assert grid_w > density_w


def test_effective_share_rhythm_exceeds_flow():
    # the TRUE share of quality: rhythm's effective weight must beat flow's, and
    # grid-snap must be the single biggest contributing metric.
    fam_share = {f: 0.0 for f in FAMILIES}
    fam_of = {m: f for f, (_, ms) in FAMILIES.items() for m in ms}
    tot = sum(METRIC_WEIGHTS.values())
    for m, w in METRIC_WEIGHTS.items():
        fam_share[fam_of[m]] += w / tot
    assert fam_share["rhythm"] > fam_share["flow"]
    # on_quarter_grid_ratio is the largest single effective metric weight
    top_metric = max(METRIC_WEIGHTS, key=METRIC_WEIGHTS.get)
    assert top_metric == "on_quarter_grid_ratio"


def test_grid_snap_swings_quality_more_than_flow():
    # CONCRETE: tanking grid-snap costs more quality than tanking flow.
    base = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
            "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0,
            "stream_ratio": 0.3, "mean_turn_angle_deg": 90.0, "reversal_ratio": 0.2,
            "slider_ratio": 0.44, "curved_slider_ratio": 0.4}
    q_base, _, _ = quality_score(base, REF, "Insane")
    assert q_base == 1.0
    bad_grid = dict(base, on_quarter_grid_ratio=0.0)          # rhythm wrecked
    bad_flow = dict(base, stream_ratio=2.0, mean_turn_angle_deg=300.0,
                    reversal_ratio=2.0)                        # flow wrecked
    q_bad_grid, _, _ = quality_score(bad_grid, REF, "Insane")
    q_bad_flow, _, _ = quality_score(bad_flow, REF, "Insane")
    # wrecking rhythm (grid-snap) hurts MORE than wrecking flow
    assert q_bad_grid < q_bad_flow < 1.0


# --- band-less new metrics don't break the reward ---------------------------

def test_bandless_new_metrics_are_ignored():
    # the new distributional metrics have NO band in REF -> they must drop out
    # silently (no crash, no zero-ing of quality). A map in-band on the OLD
    # metrics but carrying values for the new ones still scores 1.0.
    m = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
         "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0,
         "stream_ratio": 0.3, "mean_turn_angle_deg": 90.0, "reversal_ratio": 0.2,
         "slider_ratio": 0.44, "curved_slider_ratio": 0.4,
         "stream_spacing_cv": 0.9, "slider_anchor_spread_px": 1.0}  # extreme, band-less
    q, per, _ = quality_score(m, REF, "Insane")
    assert q == 1.0
    assert "stream_spacing_cv" not in per          # never scored (no band)
    assert "slider_anchor_spread_px" not in per


# --- Part B: the playability penalty folds in -------------------------------

GOOD = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
        "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0,
        "stream_ratio": 0.3, "mean_turn_angle_deg": 90.0, "reversal_ratio": 0.2,
        "slider_ratio": 0.44, "curved_slider_ratio": 0.4, "n_objects": 400}


def test_clean_map_playability_is_noop():
    # playability 1.0 (no defects) leaves the reward at the un-penalised value
    r = reward_from_metrics(GOOD, REF, 4.5, achieved_sr=4.5, playability=1.0)
    assert r.quality == 1.0
    assert r.playability == 1.0
    assert r.reward == 1.0                          # quality 1, sr_close 1, no penalty


def test_playability_penalty_pulls_reward_down():
    clean = reward_from_metrics(GOOD, REF, 4.5, achieved_sr=4.5, playability=1.0)
    defective = reward_from_metrics(GOOD, REF, 4.5, achieved_sr=4.5,
                                    playability=0.5, defects={"unhittable_jump": 0.5})
    assert defective.reward < clean.reward
    # multiplicative fold: reward == blended * playability
    assert abs(defective.reward - clean.reward * 0.5) < 1e-3
    assert defective.defects == {"unhittable_jump": 0.5}


def test_playability_cannot_lift_above_quality_ceiling():
    # ANTI-HACK: a defect (playability<1) can only LOWER the reward; even a
    # perfect-quality map can't exceed its blend, and playability is capped at 1.
    over = reward_from_metrics(GOOD, REF, 4.5, achieved_sr=4.5, playability=5.0)
    assert over.playability == 1.0                  # clamped
    assert over.reward <= 1.0


def test_reward_stays_in_unit_interval():
    # across a sweep of qualities, sr-misses and penalties, reward stays in [0,1]
    for q_metrics in (GOOD, {"jump_ratio": 0.99, "on_quarter_grid_ratio": 0.0}):
        for sr in (4.5, 9.0, 1.0):
            for play in (1.0, 0.5, 0.0):
                r = reward_from_metrics(q_metrics, REF, 4.5, achieved_sr=sr,
                                        playability=play)
                assert 0.0 <= r.reward <= 1.0
                assert 0.0 <= r.quality <= 1.0
                assert 0.0 <= r.playability <= 1.0


# --- preserved prior behaviour ----------------------------------------------

def test_defect_weights_present_and_positive():
    assert set(DEFECT_WEIGHTS) == {
        "unhittable_jump", "unintended_stack", "slider_overlap", "degenerate_anchor"}
    assert all(w > 0 for w in DEFECT_WEIGHTS.values())


def test_back_compat_default_playability_is_one():
    # old callers that don't pass playability see the un-penalised reward
    r = reward_from_metrics(GOOD, REF, 4.5, achieved_sr=4.5)
    assert r.playability == 1.0
    assert r.defects == {}
    # the family_breakdown / per_metric API is intact
    assert r.family_breakdown and r.per_metric
