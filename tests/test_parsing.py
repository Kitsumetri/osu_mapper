from src.parsing.beatmap import (
    parse_beatmap,
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


def test_sv_at_green_wins_over_red_at_same_offset(tmp_path):
    """A#6: a green (inherited, sets SV) and a red (uninherited, resets SV->1) stacked
    at the SAME offset — the green's SV is the one in effect. The parser sorts
    uninherited-first, so _sv_at (last point <= time) returns the green even when the
    file lists the green FIRST (the order that used to make the red win)."""
    osu = tmp_path / "sv.osu"
    osu.write_text(
        "osu file format v14\n\n"
        "[General]\nAudioFilename: a.mp3\nMode: 0\n\n"
        "[Metadata]\nTitle:T\nArtist:A\nCreator:c\nVersion:N\n\n"
        "[Difficulty]\nHPDrainRate:5\nCircleSize:4\nOverallDifficulty:6\n"
        "ApproachRate:7\nSliderMultiplier:1.4\nSliderTickRate:1\n\n"
        "[TimingPoints]\n"
        "0,500,4,2,0,50,1,0\n"        # base red @0 (120 BPM)
        "1000,-50,4,2,0,50,0,0\n"     # GREEN @1000 (SV 2.0) -- listed FIRST
        "1000,400,4,2,0,50,1,0\n\n"   # RED @1000 -- listed SECOND
        "[HitObjects]\n256,192,1000,1,0,0:0:0:0:\n",
        encoding="utf-8")
    bm = parse_beatmap(osu)
    assert bm._sv_at(1000) == 2.0    # green's SV, not the red's reset-to-1.0


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
    assert bm.timing_points[0].uninherited is True  # "1"
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


def test_kiai_and_bpm(tmp_path):
    p = tmp_path / "kiai.osu"
    p.write_text(
        "osu file format v14\n[General]\nMode: 0\n[TimingPoints]\n"
        "0,400,4,2,0,50,1,0\n"          # 150 BPM, no kiai
        "2000,-100,4,2,0,50,0,1\n"      # inherited, kiai ON
        "4000,-100,4,2,0,50,0,0\n"      # kiai OFF
        "[HitObjects]\n256,192,0,1,0\n256,192,5000,1,0\n",
        encoding="utf-8",
    )
    bm = parse_beatmap(p)
    assert bm.bpm == 150.0
    assert bm.timing_points[1].kiai is True
    assert bm.timing_points[2].kiai is False
    spans = bm.kiai_spans()
    assert spans == [(2000.0, 4000.0)]


def test_malformed_lines_are_skipped(tmp_path):
    p = tmp_path / "bad.osu"
    p.write_text(
        "osu file format v14\n[General]\nMode: 0\n[HitObjects]\ngarbage,line\n256,192,0,1,0\n",
        encoding="utf-8",
    )
    bm = parse_beatmap(p)
    assert len(bm.hit_objects) == 1  # only the valid line survives
