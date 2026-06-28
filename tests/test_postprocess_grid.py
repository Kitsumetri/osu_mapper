"""Grid-precision + playfield-guard tests for postprocess (v9 task1 audit).

These pin two fixes:
  1. ``snap_to_grid`` lands every moved object on the *editor's* integer-ms tick
     (round(grid)), not ``time + round(delta)`` — the latter collapses a half-tie
     delta to 0 and leaves the note 1 ms off the grid line, which the osu! editor
     flags as "unsnapped" (the user's "I have to nudge notes to 1/4" complaint).
  2. ``clamp_objects_to_playfield`` keeps every object head inside the playfield.
"""

from math import isfinite

from src.parsing.beatmap import (
    PLAYFIELD_H,
    PLAYFIELD_W,
    TYPE_CIRCLE,
    TYPE_SLIDER,
    TYPE_SPINNER,
    HitObject,
    TimingPoint,
)
from src.postprocess import (
    clamp_objects_to_playfield,
    clamp_slider_endpoints,
    snap_to_grid,
)


def _circles(times):
    return [HitObject(x=100, y=100, time=t, type=TYPE_CIRCLE, end_time=t) for t in times]


def _nearest_tick_int(t, beat, offset, divisors=(4, 8, 6)):
    """The integer tick the osu! editor would draw nearest to ``t`` (best divisor)."""
    best = None
    for d in divisors:
        iv = beat / d
        k = round((t - offset) / iv)
        grid = offset + k * iv
        err = abs(grid - t)
        if best is None or err < best[0]:
            best = (err, int(round(grid)))
    return best[1]


def test_snap_half_tie_lands_on_editor_tick():
    """240 BPM, offset 22: the 1/4 tick at t=1585 is the float 1584.5. The old
    ``time+round(delta)`` left it at 1585 (delta -0.5 -> round 0). The fix stores
    round(1584.5)=1584, which is exactly the editor's tick."""
    beat, offset = 250.0, 22.0          # 240 BPM
    tp = TimingPoint(offset, beat, 4, True)
    o = _circles([1585])
    moved = snap_to_grid(o, tp, divisors=(4, 8, 6))
    assert moved == 1
    assert o[0].time == 1584            # == round(1584.5), the editor's int tick
    assert o[0].time == _nearest_tick_int(1585, beat, offset)


def test_snap_every_moved_object_equals_editor_tick():
    """For a sweep of off-grid notes at a non-integer-1/4 BPM, every note that snap
    leaves within the bound must equal the editor's integer tick (no residual 1 ms
    offset that would show as 'unsnapped')."""
    beat, offset = 250.0, 22.0          # 240 BPM, 1/4 = 62.5 ms (half-integer ticks)
    tp = TimingPoint(offset, beat, 4, True)
    # notes sitting right on the float ticks (rounded to int) — the case that broke
    times = [int(round(offset + k * (beat / 4))) for k in range(1, 80)]
    objs = _circles(times)
    snap_to_grid(objs, tp, divisors=(4, 8, 6))
    for o in objs:
        assert o.time == _nearest_tick_int(o.time, beat, offset), o.time


def test_snap_preserves_slider_duration_on_half_tie():
    beat, offset = 250.0, 22.0
    tp = TimingPoint(offset, beat, 4, True)
    s = HitObject(x=0, y=0, time=1585, type=TYPE_SLIDER, end_time=1585 + 200,
                  curve_type="L", curve_points=[(100, 0)], slides=1, length=50.0)
    snap_to_grid([s], tp, divisors=(4, 8, 6))
    assert s.time == 1584
    assert s.end_time - s.time == 200   # duration unchanged (end shifted by same delta)


def test_snap_clean_bpm_unchanged():
    """At a BPM whose 1/4 is an integer, the fix is a no-op (notes already on int ticks)."""
    tp = TimingPoint(0, 400.0, 4, True)   # 150 BPM, 1/4 = 100 ms
    objs = _circles([0, 100, 200, 300])
    assert snap_to_grid(objs, tp, divisors=(4, 8, 6)) == 0
    assert [o.time for o in objs] == [0, 100, 200, 300]


def test_snap_no_double_count_when_grid_equals_time():
    """A note already exactly on the grid (delta rounds to 0 AND round(grid)==time)
    is not reported as moved."""
    tp = TimingPoint(0, 400.0, 4, True)
    objs = _circles([100])
    assert snap_to_grid(objs, tp) == 0
    assert objs[0].time == 100


# ---- playfield guard for ALL objects --------------------------------------
def test_clamp_objects_to_playfield_pulls_in_circle():
    objs = [
        HitObject(x=-30, y=200, time=0, type=TYPE_CIRCLE, end_time=0),
        HitObject(x=600, y=500, time=100, type=TYPE_CIRCLE, end_time=100),
        HitObject(x=256, y=192, time=200, type=TYPE_SPINNER, end_time=2000),
    ]
    changed = clamp_objects_to_playfield(objs)
    assert changed == 2                  # the two OOB circles; spinner head in-bounds
    for o in objs:
        assert 0 <= o.x <= PLAYFIELD_W and 0 <= o.y <= PLAYFIELD_H


def test_clamp_objects_to_playfield_noop_when_in_bounds():
    objs = _circles([0, 100, 200])
    assert clamp_objects_to_playfield(objs) == 0
    assert [(o.x, o.y) for o in objs] == [(100, 100)] * 3


def test_clamp_slider_endpoints_nan_safe():
    """A NaN control point (never from decode, but a defensive guard) must not crash
    int(round(nan)); it collapses to the playfield's lower bound."""
    s = HitObject(x=10, y=10, time=0, type=TYPE_SLIDER, end_time=200,
                  curve_type="L", curve_points=[(float("nan"), 10)], slides=1, length=50.0)
    clamp_slider_endpoints([s])          # must not raise
    for cx, cy in s.curve_points:
        assert isfinite(cx) and isfinite(cy)
        assert 0 <= cx <= PLAYFIELD_W and 0 <= cy <= PLAYFIELD_H
