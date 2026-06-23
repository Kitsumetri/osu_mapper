"""Hermetic tests for the v9 flow/slider metrics + objective playability defects.

Each test builds a SYNTHETIC beatmap whose geometry we KNOW (a clean stream, an
irregular stream, a sane vs degenerate slider, an unintended stack, an unhittable
jump, an overlapping slider) and asserts the metric/defect reads it as expected.
No real osu! library, no GPU, no rosu. See ``test_reward_flow.py`` for the reward
wiring; the USER must still run ``measure_reward --all`` on the real corpus to
confirm gold maps score high (synthetic tests can't calibrate against real maps).
"""
from pathlib import Path

from src.eval.reward import (
    DEGENERATE_ANCHOR_PX,
    STACK_GAP_MS,
    UNHITTABLE_PX_PER_MS,
    _degenerate_anchor_rate,
    _slider_overlap_rate,
    _stack_defect_rate,
    _unhittable_jump_rate,
    playability_penalty,
)
from src.metrics import (
    STACK_RADIUS_PX,
    circle_radius_px,
    compute_metrics,
    slider_anchor_min_gap,
    slider_polyline,
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
    assert slider_anchor_min_gap(degen) < DEGENERATE_ANCHOR_PX
    m_sane = compute_metrics(_bm([sane, _circle(300, 0, 400)]))
    m_degen = compute_metrics(_bm([degen, _circle(300, 0, 400)]))
    assert m_sane["slider_anchor_spread_px"] > m_degen["slider_anchor_spread_px"]
    assert m_degen["slider_anchor_spread_px"] < DEGENERATE_ANCHOR_PX


def test_slider_polyline_walks_head_to_tail():
    s = _slider(0, 0, 0, [(100, 0)])     # straight 100 px slider
    poly = slider_polyline(s, n=10)
    assert poly[0] == (0.0, 0.0)
    assert abs(poly[-1][0] - 100.0) < 1e-6 and abs(poly[-1][1]) < 1e-6


# --- DEFECT: unintended stacks ----------------------------------------------

def test_unintended_stack_fires_on_surprise_stack():
    # two circles on the EXACT same spot, 200 ms apart (>> stack window) -> defect
    objs = [_circle(256, 192, 0), _circle(256, 192, 200), _circle(256, 192, 400)]
    rate = _stack_defect_rate(objs)
    assert rate == 1.0                              # every pair is a surprise stack


def test_deliberate_stack_not_flagged():
    # same spot but a tiny gap (<= STACK_GAP_MS) -> deliberate stack, NOT a defect
    objs = [_circle(256, 192, 0), _circle(256, 192, int(STACK_GAP_MS))]
    assert _stack_defect_rate(objs) == 0.0
    # spaced-out notes (> stack radius apart) are obviously not stacks
    spaced = [_circle(0, 0, 0), _circle(300, 0, 200)]
    assert _stack_defect_rate(spaced) == 0.0
    # boundary: just over the stack radius is not a stack
    just_apart = [_circle(0, 0, 0), _circle(int(STACK_RADIUS_PX) + 2, 0, 200)]
    assert _stack_defect_rate(just_apart) == 0.0


# --- DEFECT: unhittable jump velocity ---------------------------------------

def test_unhittable_jump_fires_on_impossible_velocity():
    # 500 px in 50 ms -> 10 px/ms, way past the ~4 px/ms ceiling -> defect
    objs = [_circle(0, 192, 0), _circle(500, 192, 50)]
    assert _unhittable_jump_rate(objs) == 1.0


def test_hittable_jump_not_flagged():
    # a hard-but-fair jump: 300 px in 200 ms -> 1.5 px/ms < ceiling -> no defect
    objs = [_circle(0, 192, 0), _circle(300, 192, 200)]
    assert _unhittable_jump_rate(objs) == 0.0
    # a tight stream is trivially hittable
    stream = [_circle(100 + i * 30, 100, i * 100) for i in range(6)]
    assert _unhittable_jump_rate(stream) == 0.0
    # sanity: the ceiling constant is the documented value
    assert UNHITTABLE_PX_PER_MS == 4.0


# --- DEFECT: degenerate slider anchors --------------------------------------

def test_degenerate_anchor_rate():
    good = _slider(0, 0, 0, [(50, 0), (100, 0)])
    bad = _slider(0, 0, 200, [(1, 0), (2, 0)])      # anchors 1-2 px apart
    assert _degenerate_anchor_rate([good]) == 0.0
    assert _degenerate_anchor_rate([bad]) == 1.0
    assert _degenerate_anchor_rate([good, bad]) == 0.5


# --- DEFECT: slider body overlapping a non-adjacent object ------------------

def test_slider_overlap_fires_on_non_adjacent_circle():
    # a slider from (0,0)->(200,0); a NON-adjacent circle sits ON its body at (100,0)
    s = _slider(0, 0, 0, [(200, 0)])
    near_neighbour = _circle(220, 0, 200)           # adjacent (index 1) -> ignored
    on_body = _circle(100, 0, 400)                  # non-adjacent (index 2), on the body
    radius = circle_radius_px(4.0)
    rate = _slider_overlap_rate([s, near_neighbour, on_body], radius)
    assert rate == 1.0                              # the one slider overlaps a far object


def test_slider_overlap_clean_when_far():
    # slider on the left, all other objects far to the right -> no overlap
    s = _slider(0, 0, 0, [(50, 0)])
    far = [_circle(400, 300, 200), _circle(450, 300, 400)]
    radius = circle_radius_px(4.0)
    assert _slider_overlap_rate([s, *far], radius) == 0.0


# --- the combined penalty ---------------------------------------------------

def test_playability_penalty_zero_on_clean_map():
    # a clean stream: hittable, no stacks, no sliders to be degenerate/overlap
    objs = [_circle(100 + i * 30, 100, i * 100) for i in range(10)]
    penalty, rates = playability_penalty(_bm(objs))
    assert penalty == 0.0
    assert all(v == 0.0 for v in rates.values())


def test_playability_penalty_fires_on_defective_map():
    # surprise stacks + an unhittable jump -> a clearly positive penalty in (0,1]
    objs = [
        _circle(256, 192, 0), _circle(256, 192, 200),     # surprise stack
        _circle(0, 192, 400), _circle(500, 192, 450),     # unhittable jump
    ]
    penalty, rates = playability_penalty(_bm(objs))
    assert 0.0 < penalty <= 1.0
    assert rates["unintended_stack"] > 0.0
    assert rates["unhittable_jump"] > 0.0


def test_playability_penalty_bounded_on_worst_case():
    # every pair an unhittable surprise stack-or-jump + degenerate overlapping
    # sliders -> penalty must stay bounded in [0, 1]
    s1 = _slider(0, 0, 0, [(1, 0)], dt=10)          # degenerate anchors
    s2 = _slider(0, 0, 50, [(2, 0)], dt=10)         # degenerate, on top of s1
    objs = [s1, s2, _circle(0, 0, 100), _circle(500, 0, 110)]
    penalty, _ = playability_penalty(_bm(objs))
    assert 0.0 <= penalty <= 1.0
