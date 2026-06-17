from pathlib import Path

from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject, TimingPoint
from src.timing_model.labels import beat_grid, beatmap_duration_ms, primary_timing
from src.timing_model.metrics import (
    bpm_offset_metrics,
    f_measure,
    grid_drift,
    summarize,
)


def test_primary_timing_single_bpm():
    tps = [TimingPoint(1000, 500.0, 4, True)]  # 120 BPM, offset 1000
    lab = primary_timing(tps)
    assert abs(lab.bpm - 120.0) < 1e-6
    assert lab.offset_ms == 1000.0 and lab.meter == 4
    assert primary_timing([]) is None
    # inherited (green) points are ignored
    assert primary_timing([TimingPoint(0, -100.0, 4, False)]) is None


def test_beat_grid_single_and_variable_bpm():
    beats, downs = beat_grid([TimingPoint(1000, 500.0, 4, True)], duration_ms=5000)
    assert beats == [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    assert downs == [1000, 3000]                       # every 4th beat from the offset
    # variable BPM: each red point restarts the measure
    var = [TimingPoint(1000, 500.0, 4, True), TimingPoint(3000, 400.0, 4, True)]
    beats, downs = beat_grid(var, duration_ms=5000)
    assert 3000 in downs and 4600 in downs             # downbeat resets at the 2nd red point
    assert 1500 in beats and 3400 in beats


def test_beatmap_duration():
    bm = Beatmap(path=Path("x.osu"))
    bm.hit_objects = [HitObject(x=0, y=0, time=t, type=TYPE_CIRCLE, end_time=t)
                      for t in (0, 500, 2000)]
    assert beatmap_duration_ms(bm) == 2000.0


def test_f_measure():
    assert f_measure([1000, 2000, 3000], [1000, 2000, 3000])["f"] == 1.0
    # one prediction far off -> precision/recall drop
    m = f_measure([1000, 2000, 9000], [1000, 2000, 3000])
    assert m["precision"] < 1.0 and m["recall"] < 1.0
    # within tolerance counts as a hit
    assert f_measure([1050], [1000], tol_ms=70)["f"] == 1.0


def test_bpm_offset_metrics():
    exact = bpm_offset_metrics(120.0, 1000.0, 120.0, 1000.0)
    assert exact["exact_bpm"] and not exact["octave"] and exact["exact"]
    assert exact["offset_bucket"] == "excellent"
    # double-tempo octave error
    oct_ = bpm_offset_metrics(240.0, 1000.0, 120.0, 1000.0)
    assert (not oct_["exact_bpm"]) and oct_["octave"]
    # small offset error (7 ms, beat 500) -> good + still exact (playable)
    off = bpm_offset_metrics(120.0, 1007.0, 120.0, 1000.0)
    assert off["offset_bucket"] == "good" and off["exact"]
    # half-beat phase error -> off, not exact
    half = bpm_offset_metrics(120.0, 1250.0, 120.0, 1000.0)
    assert half["offset_bucket"] == "off" and not half["exact"]


def test_grid_drift_and_summarize():
    assert grid_drift(120.0, 1000.0, [1000, 1500, 2000])["max_ms"] == 0.0
    drift = grid_drift(120.0, 1010.0, [1000, 1500, 2000])
    assert abs(drift["max_ms"] - 10.0) < 1e-6
    rows = [bpm_offset_metrics(120, 1000, 120, 1000),
            bpm_offset_metrics(240, 1000, 120, 1000)]
    s = summarize(rows)
    assert s["n"] == 2 and s["exact_bpm_rate"] == 0.5 and s["octave_rate"] == 0.5
