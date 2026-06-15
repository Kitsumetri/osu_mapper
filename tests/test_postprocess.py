
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


def test_snaps_eighth_notes_to_eighth_grid():
    # 200 BPM -> beat 300ms, 1/8 = 37.5ms. A note near a 1/8 line (not a 1/4 line)
    # must snap to 1/8, not be dragged to the nearest 1/4 line (the rhythm fix).
    tp = TimingPoint(0, 300.0, 4, True)
    objs = _circles([34])              # ~1/8 (37.5ms), 3.5ms off
    snap_to_grid(objs, tp, divisors=(4, 8, 6))
    assert objs[0].time == 38          # snapped to the 1/8 line, NOT 0 (the 1/4 line)


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


def test_trim_trailing_more_aggressive_than_leading():
    from src.postprocess import trim_isolated_ends
    # a 2.5s trailing gap is trimmed (trail default 2200) but a 2.5s leading gap
    # is kept (lead default 3000).
    objs = _circles([0, 200, 400, 600]) + _circles([3100])  # 2500ms after 600
    assert trim_isolated_ends(objs) == 1
    assert max(o.time for o in objs) == 600
    objs2 = _circles([0]) + _circles([2500, 2700, 2900, 3100])  # 2500ms lead gap
    assert trim_isolated_ends(objs2) == 0


def test_trim_drops_circle_after_trailing_spinner():
    from src.parsing.beatmap import TYPE_SPINNER, HitObject
    from src.postprocess import trim_isolated_ends
    objs = _circles([0, 200, 400])
    objs.append(HitObject(x=256, y=192, time=600, type=TYPE_SPINNER, end_time=2000))
    objs.append(HitObject(x=100, y=100, time=2300, type=TYPE_CIRCLE, end_time=2300))  # +300ms
    removed = trim_isolated_ends(objs)
    assert removed == 1
    assert not objs[-1].is_circle or objs[-1].is_spinner  # last object is the spinner
    assert objs[-1].is_spinner


def test_trim_keeps_circle_well_after_spinner():
    from src.parsing.beatmap import TYPE_SPINNER, HitObject
    from src.postprocess import trim_isolated_ends
    objs = _circles([0, 200, 400])
    objs.append(HitObject(x=256, y=192, time=600, type=TYPE_SPINNER, end_time=2000))
    objs.append(HitObject(x=100, y=100, time=3500, type=TYPE_CIRCLE, end_time=3500))  # 1.5s after
    # 1500ms > spinner_tail default (1200) but < trail (2200) -> kept
    assert trim_isolated_ends(objs) == 0


def test_clamp_slider_endpoint_caps_overlong_length():
    from src.parsing.beatmap import TYPE_SLIDER
    from src.postprocess import clamp_slider_endpoints
    # head near the right edge, single anchor 20px to the right; a 400px length
    # would extrapolate ~380px past the edge -> must be capped so the tail stays
    # inside the 512-wide playfield.
    s = HitObject(x=490, y=192, time=0, type=TYPE_SLIDER, end_time=400,
                  curve_type="L", curve_points=[(510, 192)], length=400.0)
    assert clamp_slider_endpoints([s]) == 1
    # poly_len 20 + max extra to the right edge (512-510=2) = 22
    assert s.length <= 22.5
    assert s.end_time < 400          # duration scaled down with the length


def test_clamp_slider_leaves_in_bounds_slider():
    from src.parsing.beatmap import TYPE_SLIDER
    from src.postprocess import clamp_slider_endpoints
    s = HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, end_time=200,
                  curve_type="L", curve_points=[(180, 100)], length=80.0)
    assert clamp_slider_endpoints([s]) == 0
    assert s.length == 80.0


def test_compute_breaks_marks_big_gaps_only():
    from src.postprocess import compute_breaks
    objs = _circles([0, 200, 400]) + _circles([6000, 6200, 6400])  # ~5.6s gap
    breaks = compute_breaks(objs)
    assert len(breaks) == 1
    start, end = breaks[0]
    assert start > 400 and end < 6000        # padded inside the gap
    # a map with no big gaps yields no breaks
    assert compute_breaks(_circles([0, 200, 400, 600])) == []


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
