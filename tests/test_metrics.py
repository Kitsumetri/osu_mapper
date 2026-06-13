from pathlib import Path

from src.metrics import compute_metrics, compute_metrics_for_osu
from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject, TimingPoint


def test_metrics_on_sample(sample_osu):
    m = compute_metrics_for_osu(sample_osu)
    assert m["n_objects"] == 3
    assert 0 <= m["slider_ratio"] <= 1
    total = m["circle_ratio"] + m["slider_ratio"] + m["spinner_ratio"]
    assert abs(total - 1.0) < 0.01  # rounding


def test_stream_detection():
    # 150 BPM -> beat 400ms, 1/4 = 100ms; tight spacing -> streams
    bm = Beatmap(path=Path("x.osu"), timing_points=[TimingPoint(0, 400.0, 4, True)])
    bm.hit_objects = [
        HitObject(x=100 + i * 10, y=100, time=i * 100, type=TYPE_CIRCLE, end_time=i * 100)
        for i in range(8)
    ]
    m = compute_metrics(bm)
    assert m["bpm"] == 150.0
    assert m["stream_ratio"] > 0.8           # almost all gaps are tight 1/4s
    assert m["on_quarter_grid_ratio"] > 0.8


def test_jump_detection():
    bm = Beatmap(path=Path("x.osu"), timing_points=[TimingPoint(0, 400.0, 4, True)])
    # large spatial jumps every half beat
    objs = []
    xs = [0, 400, 0, 400, 0]
    for i, x in enumerate(xs):
        objs.append(HitObject(x=x, y=192, time=i * 200, type=TYPE_CIRCLE, end_time=i * 200))
    bm.hit_objects = objs
    m = compute_metrics(bm)
    assert m["jump_ratio"] > 0.8
    assert m["mean_spacing_px"] > JUMP_FLOOR
    # back-and-forth (0,400,0,400,0) is a 1-2 pattern -> near-180 reversals
    assert m["reversal_ratio"] > 0.8
    assert m["mean_turn_angle_deg"] > 150


def test_score_against_reference():
    from src.metrics import score_against_reference
    m = {"density_per_s": 3.5, "stream_ratio": 0.5, "jump_ratio": 0.2}
    ref = {"buckets": {"Hard": {
        "stream_ratio": {"mean": 0.15, "std": 0.1, "p10": 0.0, "p90": 0.3},
        "jump_ratio": {"mean": 0.25, "std": 0.1, "p10": 0.1, "p90": 0.4},
    }}}
    bucket, rows = score_against_reference(m, ref)
    assert bucket == "Hard"                       # density 3.5 -> Hard bin
    by_key = {r[0]: r for r in rows}
    # stream_ratio 0.5 is way above the Hard mean 0.15 -> high z, out of p10-p90
    assert by_key["stream_ratio"][4] > 3 and by_key["stream_ratio"][5] is False
    # jump_ratio 0.2 sits inside p10-p90
    assert by_key["jump_ratio"][5] is True


def test_straight_flow_low_turn():
    from src.parsing.beatmap import TimingPoint
    bm = Beatmap(path=Path("x.osu"), timing_points=[TimingPoint(0, 400.0, 4, True)])
    # monotonic line -> ~0 deg turns, no reversals
    bm.hit_objects = [
        HitObject(x=i * 40, y=100, time=i * 100, type=TYPE_CIRCLE, end_time=i * 100)
        for i in range(6)
    ]
    m = compute_metrics(bm)
    assert m["mean_turn_angle_deg"] < 10
    assert m["reversal_ratio"] == 0.0


JUMP_FLOOR = 200.0
