"""Hermetic tests for the ranked-map reward (src/eval/reward.py).

Exercise only the pure core (reward_from_metrics + the band/SR/combine pieces +
the family-balanced quality); the rosu SR call in reward_from_osu is out of scope
for a hermetic test.
"""
from src.eval.reward import (
    FAMILIES,
    METRIC_WEIGHTS,
    _band_score,
    combine,
    quality_score,
    reward_from_metrics,
    sr_closeness,
)

# a stand-in reference bucket (schema matches corpus_stats output). Covers metrics
# from several families so the family-balancing can be exercised: rhythm
# (on_quarter_grid, density), spacing_aim (mean_spacing, std_spacing, jump),
# slider_shape (slider_ratio, curved_slider_ratio).
REF = {"buckets": {"Insane": {
    "mean_spacing_px": {"mean": 150.0, "std": 40.0, "p10": 100.0, "p90": 200.0},
    "std_spacing_px": {"mean": 80.0, "std": 12.0, "p10": 66.0, "p90": 96.0},
    "jump_ratio": {"mean": 0.25, "std": 0.1, "p10": 0.1, "p90": 0.4},
    "on_quarter_grid_ratio": {"mean": 0.8, "std": 0.1, "p10": 0.65, "p90": 0.95},
    "density_per_s": {"mean": 4.0, "std": 0.9, "p10": 3.0, "p90": 5.0},
    "slider_ratio": {"mean": 0.44, "std": 0.11, "p10": 0.3, "p90": 0.58},
    "curved_slider_ratio": {"mean": 0.4, "std": 0.2, "p10": 0.2, "p90": 0.6},
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
    m = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
         "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0,
         "slider_ratio": 0.44, "curved_slider_ratio": 0.4}
    q, per, fams = quality_score(m, REF, "Insane")
    assert q == 1.0
    assert all(v == 1.0 for v in per.values())
    assert all(v == 1.0 for v in fams.values())


def test_quality_penalises_out_of_band():
    # spacing way over the real p90, jumps way over -> quality drops below in-band
    m = {"mean_spacing_px": 320, "jump_ratio": 0.9,
         "on_quarter_grid_ratio": 0.3, "density_per_s": 9.0}
    q, _, _ = quality_score(m, REF, "Insane")
    assert q < 0.5


def test_quality_score_returns_family_breakdown():
    # only spacing in-band, slider off-band -> the two families score differently
    m = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
         "slider_ratio": 0.0, "curved_slider_ratio": 0.0}
    _, _, fams = quality_score(m, REF, "Insane")
    assert fams["spacing_aim"] == 1.0
    assert fams["slider_shape"] < 0.6      # both slider metrics far below p10
    # families with no usable metric in REF (flow, rhythm partly, accents) drop out
    assert "spacing_aim" in fams and "slider_shape" in fams


def test_family_balance_slider_not_masked_by_spacing():
    # THE bias fix: a map with perfect spacing/aim but broken sliders must NOT
    # score as high as one good in BOTH families. Under the old flat mean, the
    # 3 spacing metrics out-voted the slider metrics and masked the breakage.
    good_both = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
                 "slider_ratio": 0.44, "curved_slider_ratio": 0.4}
    perfect_spacing_broken_sliders = {
        "mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
        "slider_ratio": 0.0, "curved_slider_ratio": 0.0}
    q_good, _, _ = quality_score(good_both, REF, "Insane")
    q_broken, _, _ = quality_score(perfect_spacing_broken_sliders, REF, "Insane")
    assert q_good == 1.0
    # slider_shape is a full family, so wrecking it costs ~half the quality, not
    # the tiny share it carried under the old flat mean.
    assert q_broken < 0.75


def test_family_share_tracks_family_weight():
    # A family's share of quality depends ONLY on its family weight, never on how
    # many metrics it contains (the anti jump-bias core). spacing_aim and
    # slider_shape carry different metric counts but the same family weight, so
    # their shares stay equal. After the v9 round-3 reweight the pattern families
    # are no longer all-equal (rhythm >> flow), but each family's share must still
    # be exactly proportional to its family weight.
    fam_w = {f: w for f, (w, _) in FAMILIES.items()}
    tot_w = sum(fam_w.values())
    fam_share = {f: 0.0 for f in FAMILIES}
    fam_of = {m: f for f, (_, ms) in FAMILIES.items() for m in ms}
    tot = sum(METRIC_WEIGHTS.values())
    for m, w in METRIC_WEIGHTS.items():
        fam_share[fam_of[m]] += w / tot
    for f in FAMILIES:
        assert abs(fam_share[f] - fam_w[f] / tot_w) < 1e-6
    # count-independence: equal family weight -> equal share despite metric count
    assert abs(fam_share["spacing_aim"] - fam_share["slider_shape"]) < 1e-6
    # the reweight's intent: rhythm is the heaviest, flow the lightest pattern family
    assert fam_share["rhythm"] > fam_share["spacing_aim"] > fam_share["flow"]
    assert fam_share["accents"] < fam_share["flow"]


def test_reward_hacking_overshoot_not_better_than_ranked():
    # the anti-hacking property: an EXTREME map cannot out-score a ranked-looking one
    ranked = {"mean_spacing_px": 150, "std_spacing_px": 80, "jump_ratio": 0.25,
              "on_quarter_grid_ratio": 0.8, "density_per_s": 4.0,
              "slider_ratio": 0.44, "curved_slider_ratio": 0.4, "n_objects": 500}
    extreme = {"mean_spacing_px": 400, "std_spacing_px": 200, "jump_ratio": 0.99,
               "on_quarter_grid_ratio": 0.8, "density_per_s": 8.0,
               "slider_ratio": 0.44, "curved_slider_ratio": 0.4, "n_objects": 500}
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
    assert r.family_breakdown == {}
    # partial metric dict still scores on the metrics that ARE present
    r2 = reward_from_metrics({"jump_ratio": 0.25}, REF, 4.5, 4.5)
    assert r2.per_metric == {"jump_ratio": 1.0}
    # a single-metric family still resolves to that family's score
    assert r2.family_breakdown.get("spacing_aim") == 1.0


def test_reward_breakdown_back_compat_fields():
    # best_of_n + other agents read these exact attributes; keep them present.
    r = reward_from_metrics(
        {"mean_spacing_px": 150, "jump_ratio": 0.25, "n_objects": 300},
        REF, 4.5, achieved_sr=4.5)
    for attr in ("reward", "quality", "sr_closeness", "achieved_sr", "target_sr",
                 "bucket", "per_metric", "n_objects", "family_breakdown"):
        assert hasattr(r, attr)
    assert r.n_objects == 300
