"""Hermetic tests for the v9 flow/slider distributional metrics + the (velocity-
only) playability penalty.

Each test builds a SYNTHETIC beatmap whose geometry we KNOW (a clean stream, an
irregular stream, a sane vs degenerate slider, a stack, an overlapping slider, an
impossible teleport) and asserts the metric/defect reads it as expected. No real
osu! library, no GPU, no rosu.

After the v9 round-3a recalibration, stacks / slider overlaps / tight anchors are
DISTRIBUTIONAL band metrics (they are intentional osu! patterns), NOT defects —
only a physically-impossible cursor velocity is still a hard penalty. The USER
still runs ``measure_reward --all --gold`` on the real corpus to confirm gold maps
score high (synthetic tests can't calibrate against real maps).
"""
from pathlib import Path

from src.eval.reward import (
    DEFECT_WEIGHTS,
    UNHITTABLE_PX_PER_MS,
    _unhittable_jump_rate,
    playability_penalty,
)
from src.metrics import (
    STACK_RADIUS_PX,
    circle_radius_px,
    compute_metrics,
    slider_anchor_min_gap,
    slider_overlap_ratio_of,
    slider_polyline,
    stack_ratio_of,
)
from src.parsing.beatmap import (
    TYPE_CIRCLE,
    TYPE_SLIDER,
    Beatmap,
    HitObject,
    TimingPoint,
)

# 150 BPM -> beat 400 ms, 1/4 = 100 ms (nb = 0.25, a stream-eligible gap)
BPM150 = TimingPoint(0, 400.0, 4, True)


def _bm(objs, cs=4.0, tp=BPM150):
    bm = Beatmap(path=Path("x.osu"), circle_size=cs, timing_points=[tp])
    bm.hit_objects = objs
    return bm


def _circle(x, y, t):
    return HitObject(x=int(x), y=int(y), time=int(t), type=TYPE_CIRCLE, end_time=int(t))


def _slider(x, y, t, anchors, length=120.0, dt=100):
    return HitObject(x=int(x), y=int(y), time=int(t), type=TYPE_SLIDER,
                     curve_type="B", curve_points=[(int(a), int(b)) for a, b in anchors],
                     length=length, end_time=int(t + dt))


# --- osu! domain helpers ----------------------------------------------------

def test_circle_radius_from_cs():
    # official r = 54.4 - 4.48*CS
    assert abs(circle_radius_px(4.0) - 36.48) < 1e-6
    assert abs(circle_radius_px(5.0) - 32.0) < 1e-6


def test_slider_polyline_walks_head_to_tail():
    s = _slider(0, 0, 0, [(100, 0)])     # straight 100 px slider
    poly = slider_polyline(s, n=10)
    assert poly[0] == (0.0, 0.0)
    assert abs(poly[-1][0] - 100.0) < 1e-6 and abs(poly[-1][1]) < 1e-6


# --- NEW metric (i): stream-spacing regularity (CV) -------------------------

def test_stream_spacing_cv_clean_stream_near_zero():
    # constant 30 px spacing, all 1/4 gaps -> a clean stream -> CV ~ 0
    objs = [_circle(100 + i * 30, 100, i * 100) for i in range(10)]
    m = compute_metrics(_bm(objs))
    assert m["stream_ratio"] > 0.8                 # detected as a stream
    assert m["stream_spacing_cv"] < 0.05           # near-constant spacing


def test_stream_spacing_cv_irregular_stream_positive():
    # same 1/4 timing but jittered spacing (10,60,10,60,...) -> high CV
    xs, x = [], 100
    for i in range(10):
        x += 10 if i % 2 == 0 else 60
        xs.append(x)
    objs = [_circle(xs[i], 100, i * 100) for i in range(10)]
    m = compute_metrics(_bm(objs))
    assert m["stream_ratio"] > 0.8
    assert m["stream_spacing_cv"] > 0.3            # clearly irregular


def test_on_grid_credits_quarter_eighth_and_sixth():
    # round-3a-fix: on_quarter_grid_ratio now credits the standard {1/4,1/8,1/6}
    # grid (150 BPM beat 400 ms: 1/4=100, 1/8=50, 1/6~=66.7 ms), not 1/4 alone.
    quarter = [_circle(i * 20, 100, i * 100) for i in range(8)]
    eighth = [_circle(i * 20, 100, i * 50) for i in range(8)]
    sixth = [_circle(i * 20, 100, round(i * 400 / 6)) for i in range(8)]
    assert compute_metrics(_bm(quarter))["on_quarter_grid_ratio"] == 1.0
    assert compute_metrics(_bm(eighth))["on_quarter_grid_ratio"] == 1.0
    assert compute_metrics(_bm(sixth))["on_quarter_grid_ratio"] >= 0.8
    # an off-grid gap (1/5 = 80 ms) is still NOT credited
    fifth = [_circle(i * 20, 100, i * 80) for i in range(8)]
    assert compute_metrics(_bm(fifth))["on_quarter_grid_ratio"] < 0.2


def test_stream_spacing_cv_zero_when_no_stream():
    # pure jumps, no stream pairs -> metric defined as 0.0
    objs = [_circle(0 if i % 2 == 0 else 400, 192, i * 200) for i in range(5)]
    m = compute_metrics(_bm(objs))
    assert m["stream_ratio"] == 0.0
    assert m["stream_spacing_cv"] == 0.0


# --- NEW metric (ii): slider anchor spread ----------------------------------

def test_slider_anchor_spread_sane_vs_degenerate():
    # a sane slider: anchors ~50 px apart; a degenerate one: anchors ~1 px apart
    sane = _slider(0, 0, 0, [(50, 0), (100, 0)])
    degen = _slider(0, 0, 200, [(1, 1), (2, 0)])
    assert slider_anchor_min_gap(sane) >= 40.0
    assert slider_anchor_min_gap(degen) < 4.0
    m_sane = compute_metrics(_bm([sane, _circle(300, 0, 400)]))
    m_degen = compute_metrics(_bm([degen, _circle(300, 0, 400)]))
    assert m_sane["slider_anchor_spread_px"] > m_degen["slider_anchor_spread_px"]
    assert m_degen["slider_anchor_spread_px"] < 4.0


# --- NEW metric (iii): stack_ratio (intentional pattern -> band, not defect) -

def test_stack_ratio_detects_stacked_notes():
    # three circles on the EXACT same spot -> every consecutive pair is stacked
    objs = [_circle(256, 192, 0), _circle(256, 192, 200), _circle(256, 192, 400)]
    assert stack_ratio_of(objs) == 1.0
    assert compute_metrics(_bm(objs))["stack_ratio"] == 1.0


def test_stack_ratio_zero_when_spaced():
    objs = [_circle(0, 0, 0), _circle(300, 0, 200), _circle(0, 200, 400)]
    assert stack_ratio_of(objs) == 0.0
    assert compute_metrics(_bm(objs))["stack_ratio"] == 0.0
    # boundary: just over the stack radius is not a stack
    just_apart = [_circle(0, 0, 0), _circle(int(STACK_RADIUS_PX) + 2, 0, 200)]
    assert stack_ratio_of(just_apart) == 0.0


def test_stacking_is_not_penalised():
    # the whole point of round-3a: a stack-heavy map has NO playability penalty
    objs = [_circle(256, 192, i * 200) for i in range(8)]   # all stacked
    penalty, rates = playability_penalty(_bm(objs))
    assert penalty == 0.0
    assert rates["unhittable_jump"] == 0.0


# --- NEW metric (iv): slider_overlap_ratio (intentional -> band, not defect) -

def test_slider_overlap_ratio_fires_on_non_adjacent_circle():
    # a slider (0,0)->(200,0); a NON-adjacent circle sits ON its body at (100,0)
    s = _slider(0, 0, 0, [(200, 0)])
    near_neighbour = _circle(220, 0, 200)           # adjacent (index 1) -> ignored
    on_body = _circle(100, 0, 400)                  # non-adjacent (index 2), on the body
    radius = circle_radius_px(4.0)
    assert slider_overlap_ratio_of([s, near_neighbour, on_body], radius) == 1.0
    m = compute_metrics(_bm([s, near_neighbour, on_body]))
    assert m["slider_overlap_ratio"] == 1.0


def test_slider_overlap_ratio_clean_when_far():
    s = _slider(0, 0, 0, [(50, 0)])
    far = [_circle(400, 300, 200), _circle(450, 300, 400)]
    radius = circle_radius_px(4.0)
    assert slider_overlap_ratio_of([s, *far], radius) == 0.0


def test_overlap_is_not_penalised():
    # an overlapping slider is no longer a defect -> no playability penalty
    s = _slider(0, 0, 0, [(200, 0)])
    on_body = _circle(100, 0, 400)
    penalty, _ = playability_penalty(_bm([s, _circle(220, 0, 200), on_body]))
    assert penalty == 0.0


# --- the ONE remaining defect: physically-impossible cursor velocity --------

def test_unhittable_jump_fires_on_impossible_velocity():
    # 600 px in 30 ms -> 20 px/ms, way past the 10 px/ms ceiling -> defect
    objs = [_circle(0, 192, 0), _circle(600, 192, 30)]
    assert _unhittable_jump_rate(objs) == 1.0


def test_hard_but_ranked_jump_not_flagged():
    # the recalibration's point: a hard 1/4 jump-burst is NOT flagged.
    # 250 px over 50 ms (200 BPM 1/4) = 5 px/ms < 10 ceiling
    assert _unhittable_jump_rate([_circle(0, 192, 0), _circle(250, 192, 50)]) == 0.0
    # a hard-but-fair 1/2 jump: 300 px / 200 ms = 1.5 px/ms
    assert _unhittable_jump_rate([_circle(0, 192, 0), _circle(300, 192, 200)]) == 0.0
    # a tight stream is trivially hittable
    stream = [_circle(100 + i * 30, 100, i * 100) for i in range(6)]
    assert _unhittable_jump_rate(stream) == 0.0
    # the recalibrated ceiling is the documented value
    assert UNHITTABLE_PX_PER_MS == 10.0


# --- the combined penalty (velocity-only now) -------------------------------

def test_playability_penalty_zero_on_clean_map():
    # a clean stream: hittable, no impossible velocity -> no penalty
    objs = [_circle(100 + i * 30, 100, i * 100) for i in range(10)]
    penalty, rates = playability_penalty(_bm(objs))
    assert penalty == 0.0
    assert set(rates) == {"unhittable_jump"}
    assert rates["unhittable_jump"] == 0.0


def test_playability_penalty_fires_only_on_velocity():
    # one impossible teleport among hittable pairs -> a positive bounded penalty
    objs = [
        _circle(0, 192, 0), _circle(30, 192, 100),        # hittable
        _circle(0, 192, 200), _circle(600, 192, 230),     # teleport (20 px/ms)
    ]
    penalty, rates = playability_penalty(_bm(objs))
    assert 0.0 < penalty <= 1.0
    assert rates["unhittable_jump"] > 0.0


def test_playability_penalty_bounded_on_worst_case():
    # every pair an impossible teleport -> penalty saturates but stays in [0, 1]
    objs = [_circle(0, 0, i * 30) if i % 2 == 0 else _circle(600, 0, i * 30)
            for i in range(8)]
    penalty, _ = playability_penalty(_bm(objs))
    assert 0.0 <= penalty <= 1.0


def test_defect_weights_velocity_only():
    # stacks/overlaps/anchors are no longer defects (they became band metrics)
    assert set(DEFECT_WEIGHTS) == {"unhittable_jump"}
    assert all(w > 0 for w in DEFECT_WEIGHTS.values())
