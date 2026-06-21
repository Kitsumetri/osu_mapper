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


def test_slider_extras_are_spec_correct(sample_osu, tmp_path):
    from src.parsing.beatmap import TYPE_SLIDER, HitObject
    bm = parse_beatmap(sample_osu)
    objs = [
        HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, hit_sound=0,
                  curve_type="B", curve_points=[(150, 80), (200, 120)], slides=1, length=120.0),
        HitObject(x=200, y=200, time=600, type=TYPE_SLIDER, hit_sound=0,
                  curve_type="L", curve_points=[(300, 200)], slides=2, length=100.0),  # reverse
    ]
    out = tmp_path / "sl.osu"
    write_osu(bm, objs, out, timing_points=[TimingPoint(0, 400.0, 4, True)])
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines()
             if "|" in ln and "," in ln and not ln.startswith("[")]
    for ln, slides in zip(lines, (1, 2)):
        f = ln.split(",")
        # x,y,time,type,hitSound,curve|pts,slides,length,edgeSounds,edgeSets,hitSample
        assert len(f) == 11, ln
        edge_sounds, edge_sets, hit_sample = f[8], f[9], f[10]
        parts_snd = edge_sounds.split("|")
        assert len(parts_snd) == slides + 1
        assert all(p.isdigit() for p in parts_snd)               # integers (spec)
        assert all(":" in p for p in edge_sets.split("|"))        # set:set pairs
        assert hit_sample.count(":") == 4                         # set:set:idx:vol:file


def test_slider_without_curve_points_is_written_as_circle(sample_osu, tmp_path):
    from src.parsing.beatmap import TYPE_SLIDER, HitObject
    bm = parse_beatmap(sample_osu)
    # a slider that lost its curve points must not be written with the slider bit set
    objs = [HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, curve_points=[], end_time=0)]
    out = tmp_path / "deg.osu"
    write_osu(bm, objs, out, timing_points=[TimingPoint(0, 400.0, 4, True)])
    o = parse_beatmap(out).hit_objects[0]
    assert not o.is_slider and o.is_circle


def test_write_does_not_mutate_caller_objects(sample_osu, tmp_path):
    """Regression: _clamp_slider_lengths used to shorten each slider's .length IN
    PLACE, so a second write (or metrics computed after a write) saw an already-
    clamped length. Writing must leave the caller's HitObjects untouched."""
    from src.parsing.beatmap import TYPE_SLIDER, HitObject
    bm = parse_beatmap(sample_osu)
    # a slider with a deliberately over-long length that the clamp will shorten
    s = HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, curve_type="L",
                  curve_points=[(400, 100)], slides=1, length=5000.0)
    nxt = HitObject(x=200, y=200, time=300, type=TYPE_SLIDER, curve_type="L",
                    curve_points=[(300, 200)], slides=1, length=100.0)
    objs = [s, nxt]
    tps = [TimingPoint(0, 400.0, 4, True)]
    write_osu(bm, objs, tmp_path / "a.osu", timing_points=tps)
    assert s.length == 5000.0          # caller's object NOT mutated
    # writing twice must produce identical files (no progressive over-clamping)
    write_osu(bm, objs, tmp_path / "b.osu", timing_points=tps)
    assert (tmp_path / "a.osu").read_text() == (tmp_path / "b.osu").read_text()


def test_written_slider_no_overlap_is_independent_of_caller_state(tmp_path):
    """The clamp still fits each slider inside the gap to the next object, and the
    same object list packaged into two difficulties yields the same clamped length."""
    from src.parsing.beatmap import TYPE_SLIDER, Beatmap, HitObject
    bm = Beatmap(path=tmp_path / "x.osu", slider_multiplier=1.4)
    objs = [HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, curve_type="L",
                      curve_points=[(450, 100)], slides=1, length=9000.0),
            HitObject(x=200, y=200, time=400, type=TYPE_SLIDER, curve_type="L",
                      curve_points=[(300, 200)], slides=1, length=80.0)]
    tps = [TimingPoint(0, 400.0, 4, True)]
    write_osu(bm, objs, tmp_path / "d1.osu", timing_points=tps)
    write_osu(bm, objs, tmp_path / "d2.osu", timing_points=tps)
    for name in ("d1.osu", "d2.osu"):
        parsed = sorted(parse_beatmap(tmp_path / name).hit_objects, key=lambda o: o.time)
        for a, b in zip(parsed, parsed[1:]):
            assert a.end_time <= b.time


def test_writer_handles_empty_timing(sample_osu, tmp_path):
    bm = parse_beatmap(sample_osu)
    out = tmp_path / "out.osu"
    write_osu(bm, bm.hit_objects, out, timing_points=[])
    bm2 = parse_beatmap(out)
    assert len(bm2.timing_points) >= 1  # a default point is injected


def test_package_set_one_folder_multiple_difficulties(tmp_path):
    """All generated SRs land in ONE beatmapset folder with a shared audio file, each as
    its own [Version] — not a separate folder per difficulty."""
    from src.package_map import package_set
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject

    song = tmp_path / "orig_song"
    song.mkdir()
    (song / "audio.mp3").write_bytes(b"fake audio")
    orig_path = song / "original.osu"
    orig = Beatmap(path=orig_path, artist="Artist", title="Song", audio_filename="audio.mp3")
    orig.hit_objects = [HitObject(x=100, y=100, time=0, type=TYPE_CIRCLE, end_time=0)]
    write_osu(orig, orig.hit_objects, orig_path)

    gens = []
    for i in range(2):
        g = tmp_path / f"gen{i}.osu"
        bm = Beatmap(path=g, audio_filename="audio.mp3")
        bm.hit_objects = [HitObject(x=50 + 10 * i, y=60, time=t, type=TYPE_CIRCLE, end_time=t)
                          for t in range(0, 600, 150)]
        write_osu(bm, bm.hit_objects, g, timing_points=[TimingPoint(0, 400.0, 4, True)])
        gens.append(g)

    songs = tmp_path / "Songs"
    songs.mkdir()
    out = package_set(gens, orig_path, songs, set_prefix="[AI]",
                      diff_names=["AI 4star", "AI 5star"])

    assert len(list(songs.iterdir())) == 1       # ONE set folder, not one per difficulty
    assert (out / "audio.mp3").exists()          # shared audio copied once
    osus = sorted(out.glob("*.osu"))
    assert len(osus) == 2                         # two difficulties live in it
    assert {parse_beatmap(o).version for o in osus} == {"AI 4star", "AI 5star"}
