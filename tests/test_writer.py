from src.parsing.beatmap import TimingPoint, parse_beatmap, write_osu


def test_write_then_reparse_preserves_objects(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "out.osu"
    write_osu(bm, bm.hit_objects, out, timing_points=[TimingPoint(0, 500.0, 4, True)])
    bm2 = parse_beatmap(out)
    assert len(bm2.hit_objects) == len(bm.hit_objects)
    assert sum(o.is_circle for o in bm2.hit_objects) == sum(o.is_circle for o in bm.hit_objects)
    assert sum(o.is_slider for o in bm2.hit_objects) == sum(o.is_slider for o in bm.hit_objects)
    assert sum(o.is_spinner for o in bm2.hit_objects) == sum(o.is_spinner for o in bm.hit_objects)


def test_written_file_has_required_sections(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "out.osu"
    write_osu(bm, bm.hit_objects, out)
    text = out.read_text(encoding="utf-8")
    for section in ("[General]", "[Metadata]", "[Difficulty]", "[TimingPoints]", "[HitObjects]"):
        assert section in text
    assert text.startswith("osu file format v")


def test_writes_kiai_timing_points(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "kiai.osu"
    tps = [TimingPoint(0, 400.0, 4, True),
           TimingPoint(2000, -100.0, 4, False, effects=1),   # kiai on
           TimingPoint(5000, -100.0, 4, False, effects=0)]   # kiai off
    write_osu(bm, bm.hit_objects, out, timing_points=tps)
    bm2 = parse_beatmap(out)
    assert bm2.kiai_spans() == [(2000.0, 5000.0)]


def test_writes_hitsounds(sample_osu, tmp_path):
    from src.parsing.beatmap import TYPE_CIRCLE, HitObject
    bm = parse_beatmap(sample_osu)
    objs = [HitObject(x=100, y=100, time=0, type=TYPE_CIRCLE, hit_sound=8, end_time=0),
            HitObject(x=120, y=100, time=300, type=TYPE_CIRCLE, hit_sound=2, end_time=300)]
    out = tmp_path / "hs.osu"
    write_osu(bm, objs, out, timing_points=[TimingPoint(0, 400.0, 4, True)])
    bm2 = parse_beatmap(out)
    assert sorted(o.hit_sound for o in bm2.hit_objects) == [2, 8]


def test_writes_break_events(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "brk.osu"
    write_osu(bm, bm.hit_objects, out, timing_points=[TimingPoint(0, 400.0, 4, True)],
              breaks=[(1200, 4800)])
    text = out.read_text(encoding="utf-8")
    assert "2,1200,4800" in text
    # the break line sits in the [Events] section
    events = text.split("[Events]")[1].split("[TimingPoints]")[0]
    assert "2,1200,4800" in events


def test_writer_handles_empty_timing(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "out.osu"
    write_osu(bm, bm.hit_objects, out, timing_points=[])
    bm2 = parse_beatmap(out)
    assert len(bm2.timing_points) >= 1  # a default point is injected
