from src.parsing.beatmap import (
    parse_beatmap, write_osu, Beatmap, TimingPoint,
    TYPE_CIRCLE, TYPE_SLIDER, TYPE_SPINNER, TYPE_NEW_COMBO,
)


def test_parse_counts(sample_osu):
    bm = parse_beatmap(sample_osu)
    assert bm.mode == 0
    assert bm.title == "Test Song"
    assert bm.artist == "Tester"
    assert bm.slider_multiplier == 1.4
    assert len(bm.hit_objects) == 3
    kinds = [o for o in bm.hit_objects]
    assert sum(o.is_circle for o in kinds) == 1
    assert sum(o.is_slider for o in kinds) == 1
    assert sum(o.is_spinner for o in kinds) == 1


def test_type_bitflags(sample_osu):
    bm = parse_beatmap(sample_osu)
    circle, slider, spinner = bm.hit_objects
    assert circle.is_circle and not circle.is_slider
    assert slider.is_slider and not slider.is_circle
    assert spinner.is_spinner
    # spinner in the fixture is type 12 = spinner(8) | new_combo(4)
    assert spinner.is_new_combo


def test_uninherited_bool_fix(sample_osu):
    """Regression: prototype used bool(str) which is always True."""
    bm = parse_beatmap(sample_osu)
    assert bm.timing_points[0].uninherited is True   # "1"
    assert bm.timing_points[1].uninherited is False  # "0"


def test_inherited_sv(sample_osu):
    bm = parse_beatmap(sample_osu)
    # inherited point has beat_length -50 -> SV = -100/-50 = 2.0
    assert abs(bm.timing_points[1].sv - 2.0) < 1e-6
    # uninherited point reports SV 1.0
    assert bm.timing_points[0].sv == 1.0


def test_slider_duration(sample_osu):
    bm = parse_beatmap(sample_osu)
    slider = bm.hit_objects[1]
    # velocity = SM*100*SV = 1.4*100*2 = 280 px/beat; beats = 200/280
    # beat_length = 500 ms -> duration ~= 357 ms
    expected = 200.0 / (1.4 * 100 * 2.0) * 500.0
    assert abs((slider.end_time - slider.time) - expected) < 2.0


def test_spinner_end(sample_osu):
    bm = parse_beatmap(sample_osu)
    spinner = bm.hit_objects[2]
    assert spinner.end_time == 5000


def test_audio_path(sample_osu):
    bm = parse_beatmap(sample_osu)
    assert bm.audio_path.name == "audio.mp3"
    assert bm.audio_path.parent == sample_osu.parent


def test_malformed_lines_are_skipped(tmp_path):
    p = tmp_path / "bad.osu"
    p.write_text(
        "osu file format v14\n[General]\nMode: 0\n[HitObjects]\n"
        "garbage,line\n256,192,0,1,0\n", encoding="utf-8")
    bm = parse_beatmap(p)
    assert len(bm.hit_objects) == 1  # only the valid line survives
