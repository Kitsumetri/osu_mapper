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


def test_writer_handles_empty_timing(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "out.osu"
    write_osu(bm, bm.hit_objects, out, timing_points=[])
    bm2 = parse_beatmap(out)
    assert len(bm2.timing_points) >= 1  # a default point is injected
