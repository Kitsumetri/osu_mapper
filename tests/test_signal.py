import numpy as np

from src.config import AUDIO, N_SIGNAL_CHANNELS
from src.data.signal import decode_signal, encode_beatmap
from src.parsing.beatmap import parse_beatmap


def _n_frames(bm):
    last = max(o.end_time for o in bm.hit_objects)
    return int(AUDIO.time_to_frame(last)) + 10


def test_encode_shape_and_range(sample_osu):
    bm = parse_beatmap(sample_osu)
    n = _n_frames(bm)
    sig = encode_beatmap(bm, n)
    assert sig.shape == (N_SIGNAL_CHANNELS, n)
    assert sig.dtype == np.float32
    # all channels within a small margin of [-1, 1]
    assert sig.min() >= -1.3 and sig.max() <= 1.3


def test_onset_channel_has_peaks(sample_osu):
    bm = parse_beatmap(sample_osu)
    sig = encode_beatmap(bm, _n_frames(bm))
    # one peak per object start (3 objects) near +1
    assert (sig[0] > 0.9).any()


def test_roundtrip_object_counts(sample_osu):
    bm = parse_beatmap(sample_osu)
    sig = encode_beatmap(bm, _n_frames(bm))
    dec = decode_signal(sig, min_spinner_frames=5)
    assert sum(o.is_spinner for o in dec) == 1
    assert sum(o.is_slider for o in dec) == 1
    assert sum(o.is_circle for o in dec) == 1


def test_onset_timing_recall_on_real_shapes():
    """Synthetic dense map: every onset should be recovered within ~1 frame."""
    import pathlib

    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject

    bm = Beatmap(path=pathlib.Path("x.osu"))
    times = list(range(0, 4000, 200))
    bm.hit_objects = [
        HitObject(x=100 + i, y=100, time=t, type=TYPE_CIRCLE, end_time=t)
        for i, t in enumerate(times)
    ]
    n = int(AUDIO.time_to_frame(times[-1])) + 10
    sig = encode_beatmap(bm, n)
    dec = decode_signal(sig)
    dt = np.array([o.time for o in dec])
    matched = sum(np.min(np.abs(dt - t)) <= AUDIO.ms_per_frame * 1.5 for t in times)
    assert matched == len(times)


def test_decode_threshold_filters_noise():
    # flat baseline signal with low-amplitude wiggle -> no objects
    sig = np.full((N_SIGNAL_CHANNELS, 500), -1.0, dtype=np.float32)
    sig[0] += np.random.RandomState(0).rand(500) * 0.2  # up to -0.8
    dec = decode_signal(sig, onset_threshold=0.3)
    assert len(dec) == 0


def test_slider_does_not_overlap_next_onset():
    """A slider whose hold bleeds past the next onset must be clamped."""
    n = 80
    sig = np.full((N_SIGNAL_CHANNELS, n), -1.0, dtype=np.float32)
    sig[4:6] = 0.0
    for f in (10, 30):
        sig[0, f] = 1.0
    sig[1, 10:51] = 1.0  # slider hold overlaps the onset at frame 30
    dec = decode_signal(sig, onset_threshold=0.3, min_gap_frames=2)
    dec.sort(key=lambda o: o.time)
    for a, b in zip(dec, dec[1:]):
        assert a.end_time <= b.time


def test_kiai_and_hitsound_roundtrip():
    """v3 channels: kiai span and clap hitsound survive encode->decode."""
    import pathlib

    from src.data.signal import decode_kiai
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject, TimingPoint
    bm = Beatmap(path=pathlib.Path("x.osu"),
                 timing_points=[TimingPoint(0, 400.0, 4, True, effects=0),
                                TimingPoint(2000, 400.0, 4, True, effects=1),   # kiai on
                                TimingPoint(6000, 400.0, 4, True, effects=0)])  # kiai off
    bm.hit_objects = [HitObject(x=100, y=100, time=t, type=TYPE_CIRCLE,
                                hit_sound=(8 if t % 800 == 0 else 0), end_time=t)
                      for t in range(0, 8000, 200)]
    n = int(AUDIO.time_to_frame(8000)) + 10
    sig = encode_beatmap(bm, n)
    assert sig.shape[0] == N_SIGNAL_CHANNELS
    spans = decode_kiai(sig, min_ms=1000.0)
    assert len(spans) == 1
    assert abs(spans[0][0] - 2000) < 60 and abs(spans[0][1] - 6000) < 60
    dec = decode_signal(sig)
    orig_clap = sum(1 for o in bm.hit_objects if o.hit_sound & 8)
    dec_clap = sum(1 for o in dec if o.hit_sound & 8)
    assert abs(orig_clap - dec_clap) <= 1


def test_accent_threshold_filters_weak_hitsounds():
    # two onsets: one with a strong clap accent (+1), one with a weak one (+0.2).
    n = 40
    sig = np.full((N_SIGNAL_CHANNELS, n), -1.0, dtype=np.float32)
    sig[4:6] = 0.0
    sig[0, 5] = 1.0
    sig[0, 20] = 1.0
    sig[9, 5] = 1.0       # strong clap on first onset
    sig[9, 20] = 0.2      # weak clap on second onset
    strong = decode_signal(sig, onset_threshold=0.3, accent_threshold=0.4)
    assert sum(1 for o in strong if o.hit_sound & 8) == 1   # only the strong one
    loose = decode_signal(sig, onset_threshold=0.3, accent_threshold=0.0)
    assert sum(1 for o in loose if o.hit_sound & 8) == 2     # both fire at 0


def test_curved_slider_from_anchor_channels():
    """v5: a slider whose dedicated anchor channels hold distinct offsets decodes
    to a multi-point Bezier (shape no longer rides the cursor channels)."""
    from src.config import CH_SLIDER_ANCHORS, CH_SLIDES
    n = 60
    sig = np.full((N_SIGNAL_CHANNELS, n), -1.0, dtype=np.float32)
    sig[4:6] = 0.0           # cursor centred (head at playfield centre)
    sig[CH_SLIDER_ANCHORS:CH_SLIDES] = 0.0  # anchors baseline 0
    sig[0, 5] = 1.0          # one onset
    sig[1, 5:40] = 1.0       # long slider hold over frames 5..39
    # three distinct held anchor offsets -> a wave (dx1,dy1),(dx2,dy2),(dx3,dy3)
    span = slice(5, 40)
    sig[CH_SLIDER_ANCHORS + 0, span] = 0.2    # dx1
    sig[CH_SLIDER_ANCHORS + 1, span] = 0.3    # dy1
    sig[CH_SLIDER_ANCHORS + 2, span] = 0.4    # dx2
    sig[CH_SLIDER_ANCHORS + 3, span] = -0.2   # dy2
    sig[CH_SLIDER_ANCHORS + 4, span] = 0.6    # dx3
    sig[CH_SLIDER_ANCHORS + 5, span] = 0.1    # dy3
    dec = decode_signal(sig, onset_threshold=0.3, min_spinner_frames=100)
    sliders = [o for o in dec if o.is_slider]
    assert sliders, "expected a slider"
    s = sliders[0]
    assert s.curve_type == "B"
    assert len(s.curve_points) >= 2


def test_nearby_spinners_are_merged():
    n = 200
    sig = np.full((N_SIGNAL_CHANNELS, n), -1.0, dtype=np.float32)
    sig[4:6] = 0.0
    sig[2, 10:40] = 1.0   # spinner run 1
    sig[2, 50:80] = 1.0   # spinner run 2, ~10-frame gap -> within SPINNER_MERGE_MS
    dec = decode_signal(sig, min_spinner_frames=20, spinner_min_mean=0.3)
    assert sum(o.is_spinner for o in dec) == 1   # merged into one


def test_collinear_anchors_decode_to_linear_slider():
    """v5: near-collinear anchor channels should decode to a *linear* slider, not
    a wasteful 3-point bezier (fb: a line built from curve points looks bad)."""
    from src.config import CH_SLIDER_ANCHORS, CH_SLIDES

    def _slider_sig(anchors):
        n = 40
        sig = np.full((N_SIGNAL_CHANNELS, n), -1.0, dtype=np.float32)
        sig[4:6] = 0.0
        sig[CH_SLIDER_ANCHORS:CH_SLIDES] = 0.0
        sig[0, 5] = 1.0
        sig[1, 5:40] = 1.0
        sp = slice(5, 40)
        for i, (dx, dy) in enumerate(anchors):
            sig[CH_SLIDER_ANCHORS + 2 * i, sp] = dx
            sig[CH_SLIDER_ANCHORS + 2 * i + 1, sp] = dy
        return sig

    # collinear anchors along the diagonal -> linear
    lin = decode_signal(_slider_sig([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]),
                        onset_threshold=0.3, min_spinner_frames=100)
    s = [o for o in lin if o.is_slider][0]
    assert s.curve_type == "L"
    # a zig-zag -> curved bezier
    cur = decode_signal(_slider_sig([(0.3, -0.2), (0.1, 0.3), (0.4, 0.1)]),
                        onset_threshold=0.3, min_spinner_frames=100)
    s = [o for o in cur if o.is_slider][0]
    assert s.curve_type == "B"


def test_slider_shape_and_reverse_roundtrip():
    """encode->decode preserves a curved slider's shape and a reverse slider's
    repeat count (the two v5 representation fixes)."""
    import pathlib

    from src.parsing.beatmap import TYPE_SLIDER, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    # a curved (wave) slider with 3 control points, and a reverse slider (slides=2)
    curved = HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, end_time=400,
                       curve_type="B", curve_points=[(180, 40), (260, 160), (340, 60)],
                       slides=1, length=300.0)
    reverse = HitObject(x=200, y=300, time=800, type=TYPE_SLIDER, end_time=1400,
                        curve_type="L", curve_points=[(320, 300)], slides=2, length=120.0)
    bm.hit_objects = [curved, reverse]
    n = int(AUDIO.time_to_frame(1400)) + 10
    sig = encode_beatmap(bm, n)
    dec = sorted((o for o in decode_signal(sig, min_spinner_frames=100) if o.is_slider),
                 key=lambda o: o.time)
    assert len(dec) == 2
    assert dec[0].curve_type == "B" and len(dec[0].curve_points) >= 2   # curve survived
    assert dec[1].slides == 2                                           # reverse survived


def test_no_time_overlap_after_write_reparse(sample_osu, tmp_path):
    """The written .osu must have no time-overlapping objects after osu! derives
    slider durations from length (the real overlap source)."""
    from src.parsing.beatmap import TimingPoint, write_osu

    bm = parse_beatmap(sample_osu)
    sig = encode_beatmap(bm, _n_frames(bm))
    dec = decode_signal(sig, min_spinner_frames=5)
    out = tmp_path / "o.osu"
    write_osu(bm, dec, out, timing_points=[TimingPoint(0, 300.0, 4, True)])
    objs = sorted(parse_beatmap(out).hit_objects, key=lambda o: o.time)
    for a, b in zip(objs, objs[1:]):
        assert a.end_time <= b.time


def test_cursor_channels_track_positions(sample_osu):
    bm = parse_beatmap(sample_osu)
    sig = encode_beatmap(bm, _n_frames(bm))
    # cursor channels are continuous in [-1.2, 1.2]
    assert sig[4].min() >= -1.21 and sig[4].max() <= 1.21
    assert sig[5].min() >= -1.21 and sig[5].max() <= 1.21
