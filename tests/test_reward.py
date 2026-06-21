"""Hermetic tests for the ranked-map reward prototype (src/eval/reward.py).

Exercise only the pure core (reward_from_metrics + the band/SR/combine pieces);
the rosu SR call in reward_from_osu is out of scope for a hermetic test.
"""
from src.eval.reward import (
    _band_score,
    combine,
    quality_score,
    reward_from_metrics,
    sr_closeness,
)

# a tiny stand-in reference bucket (schema matches corpus_stats output)
REF = {"buckets": {"Insane": {
    "mean_spacing_px": {"mean": 150.0, "std": 40.0, "p10": 100.0, "p90": 200.0},
    "jump_ratio": {"mean": 0.25, "std": 0.1, "p10": 0.1, "p90": 0.4},
    "on_quarter_grid_ratio": {"mean": 0.8, "std": 0.1, "p10": 0.65, "p90": 0.95},
    "density_per_s": {"mean": 4.0, "std": 0.9, "p10": 3.0, "p90": 5.0},
}}}


def test_band_score_flat_top_no_overshoot_reward():
    # inside the band -> exactly 1.0 everywhere (flat top: no farming gradient)
    assert _band_score(120, 150, 100, 200) == 1.0
    assert _band_score(100, 150, 100, 200) == 1.0   # at the edge
    assert _band_score(200, 150, 100, 200) == 1.0
    # going MORE extreme past p90 does not raise the score above 1 -> un-farmable
    below_far = _band_score(260, 150, 100, 200)
    below_near = _band_score(210, 150, 100, 200)
    assert below_far < below_near < 1.0
    # far enough outside -> clamps to 0
    assert _band_score(1000, 150, 100, 200) == 0.0


def test_quality_in_band_is_near_one():
    # a metric vector squarely inside every band -> quality ~1
    m = {"mean_spacing_px": 150, "jump_ratio": 0.25,
         "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0}
    q, per = quality_score(m, REF, "Insane")
    assert q == 1.0
    assert all(v == 1.0 for v in per.values())


def test_quality_penalises_out_of_band():
    # spacing way over the real p90, jumps way over -> quality drops below in-band
    m = {"mean_spacing_px": 320, "jump_ratio": 0.9,
         "on_quarter_grid_ratio": 0.3, "density_per_s": 9.0}
    q, _ = quality_score(m, REF, "Insane")
    assert q < 0.5


def test_reward_hacking_overshoot_not_better_than_ranked():
    # the anti-hacking property: an EXTREME map cannot out-score a ranked-looking one
    ranked = {"mean_spacing_px": 150, "jump_ratio": 0.25,
              "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0, "n_objects": 500}
    extreme = {"mean_spacing_px": 400, "jump_ratio": 0.99,
               "on_quarter_grid_ratio": 0.8, "density_per_s": 8.0, "n_objects": 500}
    r_ranked = reward_from_metrics(ranked, REF, 4.5, achieved_sr=4.5)
    r_extreme = reward_from_metrics(extreme, REF, 4.5, achieved_sr=4.5)
    assert r_ranked.reward > r_extreme.reward
    assert r_ranked.quality == 1.0


def test_sr_closeness_decays_and_handles_none():
    assert sr_closeness(5.0, 5.0) == 1.0
    assert sr_closeness(5.5, 5.0, tol=0.5) == 0.5      # one tol away -> 0.5
    assert sr_closeness(6.0, 5.0, tol=0.5) < 0.3
    assert sr_closeness(None, 5.0) == 0.0              # unparseable -> not ranked


def test_combine_is_convex_blend():
    assert combine(1.0, 1.0) == 1.0
    assert combine(0.0, 0.0) == 0.0
    # SR miss doesn't zero an otherwise-perfect map, but pulls it down
    blended = combine(1.0, 0.0, sr_weight=0.35)
    assert abs(blended - 0.65) < 1e-9


def test_reward_robust_to_missing_metrics_and_bucket():
    # empty ref bucket -> quality 0 (nothing to compare), no crash
    r = reward_from_metrics({"mean_spacing_px": 150}, {"buckets": {}}, 4.5, 4.5)
    assert r.quality == 0.0
    assert r.bucket == "Insane"
    # partial metric dict still scores on the metrics that ARE present
    r2 = reward_from_metrics({"jump_ratio": 0.25}, REF, 4.5, 4.5)
    assert r2.per_metric == {"jump_ratio": 1.0}
