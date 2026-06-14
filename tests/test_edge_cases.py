"""Edge cases / degenerate inputs across modules — guard against crashes and
undefined behaviour on empty, tiny, or malformed data."""
import pathlib

import numpy as np
import pytest

from src.config import N_SIGNAL_CHANNELS
from src.data.signal import decode_kiai, decode_signal, encode_beatmap
from src.metrics import compute_metrics
from src.parsing.beatmap import (
    TYPE_CIRCLE,
    Beatmap,
    HitObject,
    TimingPoint,
    parse_beatmap,
)
from src.postprocess import snap_to_grid, trim_isolated_ends


# ---- parser ----------------------------------------------------------------
def test_parse_empty_file(tmp_path):
    p = tmp_path / "empty.osu"
    p.write_text("", encoding="utf-8")
    bm = parse_beatmap(p)
    assert bm.hit_objects == [] and bm.timing_points == []


def test_parse_no_hitobjects(tmp_path):
    p = tmp_path / "nohit.osu"
    p.write_text("osu file format v14\n[General]\nMode: 0\n[TimingPoints]\n0,500,4,2,0,50,1,0\n",
                 encoding="utf-8")
    bm = parse_beatmap(p)
    assert bm.hit_objects == [] and len(bm.timing_points) == 1


def test_parse_slider_missing_params(tmp_path):
    # a slider line lacking curve/length fields should not crash
    p = tmp_path / "badslider.osu"
    p.write_text("osu file format v14\n[General]\nMode: 0\n[HitObjects]\n100,100,0,2,0\n",
                 encoding="utf-8")
    bm = parse_beatmap(p)
    assert len(bm.hit_objects) == 1  # parsed, just without slider body


def test_parse_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_beatmap("does_not_exist_xyz.osu")


# ---- signal encode/decode --------------------------------------------------
def test_encode_empty_beatmap():
    bm = Beatmap(path=pathlib.Path("x.osu"))
    sig = encode_beatmap(bm, 200)
    assert sig.shape == (N_SIGNAL_CHANNELS, 200)
    # no objects -> onset baseline (-1), no peaks
    assert sig[0].max() <= -0.99


def test_decode_baseline_signal_has_no_objects():
    sig = np.full((N_SIGNAL_CHANNELS, 300), -1.0, dtype=np.float32)
    sig[4:6] = 0.0
    assert decode_signal(sig) == []


def test_decode_kiai_on_6channel_signal():
    sig = np.full((6, 300), -1.0, dtype=np.float32)  # pre-v3 signal, no kiai chan
    assert decode_kiai(sig) == []


def test_encode_single_object():
    bm = Beatmap(path=pathlib.Path("x.osu"))
    bm.hit_objects = [HitObject(x=256, y=192, time=100, type=TYPE_CIRCLE, end_time=100)]
    sig = encode_beatmap(bm, 100)
    assert sig.shape[1] == 100


# ---- postprocess -----------------------------------------------------------
def test_snap_empty_and_zero_beat():
    assert snap_to_grid([], TimingPoint(0, 400.0, 4, True)) == 0
    objs = [HitObject(x=0, y=0, time=10, type=TYPE_CIRCLE, end_time=10)]
    assert snap_to_grid(objs, TimingPoint(0, 0.0, 4, True)) == 0  # beat<=0 -> no-op


def test_trim_too_few_objects():
    objs = [HitObject(x=0, y=0, time=0, type=TYPE_CIRCLE, end_time=0),
            HitObject(x=0, y=0, time=99999, type=TYPE_CIRCLE, end_time=99999)]
    assert trim_isolated_ends(objs) == 0  # needs >=3 to trim


# ---- metrics ---------------------------------------------------------------
def test_metrics_too_few_objects():
    bm = Beatmap(path=pathlib.Path("x.osu"))
    bm.hit_objects = [HitObject(x=0, y=0, time=0, type=TYPE_CIRCLE, end_time=0)]
    m = compute_metrics(bm)
    assert m == {"n_objects": 1}


def test_metrics_zero_bpm_no_crash():
    # no timing points -> bpm 0 -> grid/stream metrics fall back gracefully
    bm = Beatmap(path=pathlib.Path("x.osu"))
    bm.hit_objects = [HitObject(x=i * 10, y=100, time=i * 200, type=TYPE_CIRCLE, end_time=i * 200)
                      for i in range(6)]
    m = compute_metrics(bm)
    assert m["bpm"] == 0.0 and m["on_quarter_grid_ratio"] == 0.0


# ---- difficulty ------------------------------------------------------------
def test_star_rating_missing_file():
    from src.difficulty import sr_bucket, star_rating
    assert star_rating("nope_xyz.osu") is None
    assert sr_bucket(0.0) == "Easy" and sr_bucket(50.0) == "Expert+"
