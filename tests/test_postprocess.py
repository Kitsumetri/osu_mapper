
from src.parsing.beatmap import TYPE_CIRCLE, HitObject, TimingPoint
from src.postprocess import snap_to_grid


def _circles(times):
    return [HitObject(x=100, y=100, time=t, type=TYPE_CIRCLE, end_time=t) for t in times]


def test_snaps_near_grid_objects():
    # 150 BPM -> beat 400ms, 1/4 = 100ms. Times near multiples of 100 should snap.
    tp = TimingPoint(0, 400.0, 4, True)
    objs = _circles([8, 103, 197, 305])   # all within ~8ms of the 1/4 grid
    moved = snap_to_grid(objs, tp)
    assert moved == 4
    assert [o.time for o in objs] == [0, 100, 200, 300]


def test_does_not_snap_far_objects():
    tp = TimingPoint(0, 400.0, 4, True)
    objs = _circles([50])                 # 50ms off a 100ms grid -> beyond bound
    moved = snap_to_grid(objs, tp, max_snap_ms=20)
    assert moved == 0
    assert objs[0].time == 50


def test_triplet_aware():
    # 1/3 of a 400ms beat ~= 133.3ms; a note near 133 should snap via divisor 3
    tp = TimingPoint(0, 400.0, 4, True)
    objs = _circles([131])
    snap_to_grid(objs, tp, divisors=(4, 3), max_snap_ms=10)
    assert abs(objs[0].time - 133) <= 1


def test_trim_trailing_isolated_note():
    from src.postprocess import trim_isolated_ends
    objs = _circles([0, 200, 400, 600]) + _circles([20000])  # lone note 19s later
    removed = trim_isolated_ends(objs, max_gap_ms=3000)
    assert removed == 1
    assert max(o.time for o in objs) == 600


def test_trim_keeps_dense_tail():
    from src.postprocess import trim_isolated_ends
    objs = _circles([0, 200, 400, 600, 800])  # all close together
    assert trim_isolated_ends(objs, max_gap_ms=3000) == 0


def test_snap_slider_ends_to_grid():
    from src.parsing.beatmap import TYPE_SLIDER
    from src.postprocess import snap_slider_ends
    tp = TimingPoint(0, 400.0, 4, True)   # 1/4 = 100ms
    # slider starting at 0 with an off-grid 230ms duration -> should snap to 200ms
    s = HitObject(x=0, y=0, time=0, type=TYPE_SLIDER, end_time=230, length=322.0,
                  curve_type="L", curve_points=[(100, 0)], slides=1)
    nxt = HitObject(x=0, y=0, time=1000, type=TYPE_CIRCLE, end_time=1000)
    snap_slider_ends([s, nxt], tp, slider_multiplier=1.4)
    dur = s.end_time - s.time
    assert dur % 100 == 0          # on the 1/4 grid
    assert abs(dur - 200) <= 1     # nearest multiple of 100 to 230 is 200


def test_preserves_slider_duration():
    from src.parsing.beatmap import TYPE_SLIDER
    tp = TimingPoint(0, 400.0, 4, True)
    o = HitObject(x=0, y=0, time=98, type=TYPE_SLIDER, end_time=298)  # 200ms long
    snap_to_grid([o], tp)
    assert o.time == 100
    assert o.end_time - o.time == 200  # duration preserved
