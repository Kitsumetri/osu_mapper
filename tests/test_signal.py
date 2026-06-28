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


def test_reverse_slider_cursor_flow_out_uses_gameplay_end():
    """A#3: the cursor flow-out key-frame is where the player ENDS the slider — the
    tail for an odd slide count, but back at the HEAD for an even (single-reverse)
    count. (Keying it to the tail unconditionally mis-encoded reverse sliders.)
    """
    import pathlib

    from src.config import CH_CURX
    from src.data.signal import _norm_x
    from src.parsing.beatmap import TYPE_SLIDER, Beatmap, HitObject, TimingPoint

    def _end_cursor_x(slides):
        end = 200 + 300 * slides
        bm = Beatmap(path=pathlib.Path("x.osu"),
                     timing_points=[TimingPoint(0, 500.0, 4, True)])
        bm.hit_objects = [HitObject(x=100, y=192, time=200, type=TYPE_SLIDER,
                                    curve_type="L", curve_points=[(300, 192)],
                                    slides=slides, length=200.0, end_time=end)]
        n = int(AUDIO.time_to_frame(end)) + 5
        # read past f_end -> np.interp right-fill returns the end key-frame exactly
        return encode_beatmap(bm, n)[CH_CURX, n - 1]

    assert abs(_end_cursor_x(1) - _norm_x(300)) < 0.05   # odd slides -> tail (x=300)
    assert abs(_end_cursor_x(2) - _norm_x(100)) < 0.05   # even slides -> head (x=100)


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
    from src.config import CH_CURVE, CH_SLIDER_ANCHORS, CH_SLIDES
    from src.data.signal import _enc_curve
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
    sig[CH_CURVE, span] = _enc_curve(45.0)    # v7 cue: this slider is curved
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
    from src.config import CH_CURVE, CH_SLIDER_ANCHORS, CH_SLIDES
    from src.data.signal import _enc_curve

    def _slider_sig(anchors, curve_px=0.0):
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
        sig[CH_CURVE, sp] = _enc_curve(curve_px)  # v7 cue drives straight/curved
        return sig

    # cue ~0 -> linear (the cue is authoritative over the collapsed anchors)
    lin = decode_signal(_slider_sig([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)], curve_px=0.0),
                        onset_threshold=0.3, min_spinner_frames=100)
    s = [o for o in lin if o.is_slider][0]
    assert s.curve_type == "L"
    # high cue -> curved bezier
    cur = decode_signal(_slider_sig([(0.3, -0.2), (0.1, 0.3), (0.4, 0.1)], curve_px=45.0),
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


def test_sv_value_encode_decode_roundtrip():
    from src.data.signal import _dec_sv, _enc_sv
    assert abs(_enc_sv(1.0) - 0.0) < 1e-6        # SV 1.0 -> baseline 0
    assert abs(_enc_sv(2.0) - 0.5) < 1e-6        # log2(2)/2
    assert abs(_enc_sv(0.5) + 0.5) < 1e-6
    for sv in (0.5, 1.0, 1.5, 2.0, 3.5):
        assert abs(_dec_sv(_enc_sv(sv)) - sv) < 1e-3
    assert _enc_sv(100.0) <= 1.0 and _enc_sv(0.001) >= -1.0  # clamped


def test_sv_channel_roundtrip():
    """A green-line SV section survives encode -> decode_sv as a stable section."""
    import pathlib

    from src.config import CH_SV
    from src.data.signal import decode_sv
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject, TimingPoint
    bm = Beatmap(path=pathlib.Path("x.osu"),
                 timing_points=[TimingPoint(0, 400.0, 4, True),            # red, SV 1.0
                                TimingPoint(4000, -50.0, 4, False),        # green, SV 2.0
                                TimingPoint(8000, -100.0, 4, False)])      # green, SV 1.0
    bm.hit_objects = [HitObject(x=100, y=100, time=t, type=TYPE_CIRCLE, end_time=t)
                      for t in range(0, 10000, 200)]
    n = int(AUDIO.time_to_frame(10000)) + 10
    sig = encode_beatmap(bm, n)
    assert sig.shape[0] == N_SIGNAL_CHANNELS
    mid = int(AUDIO.time_to_frame(6000))
    assert abs(sig[CH_SV, mid] - 0.5) < 1e-3      # SV 2.0 region encoded
    secs = decode_sv(sig)
    svs = [s for _, s in secs]
    assert max(svs) >= 1.8                          # the 2.0 section recovered
    on = [t for t, s in secs if s >= 1.8][0]
    assert abs(on - 4000) < 700                     # ~right place (median-smoothed)


def test_decode_sv_stable_and_capped():
    from src.config import CH_SV
    from src.data.signal import decode_sv
    rng = np.random.default_rng(0)
    sig = np.zeros((N_SIGNAL_CHANNELS, 6000), dtype=np.float32)
    sig[CH_SV] = rng.uniform(-1, 1, 6000)          # pure noise
    secs = decode_sv(sig, max_sections=8)
    assert len(secs) <= 8                            # noise collapses, never per-frame
    # a flat SV=1 channel yields a single (or no) trivial section
    sig[CH_SV] = 0.0
    assert len(decode_sv(sig)) <= 1


def test_decode_sv_empty_for_pre_v7_signal():
    from src.config import CH_SV
    from src.data.signal import decode_sv
    sig17 = np.zeros((CH_SV, 2000), dtype=np.float32)  # 17 ch, no SV channel
    assert decode_sv(sig17) == []                    # no SV channel -> no sections


def test_merge_green_lines_sv_and_kiai():
    from src.generate import _merge_green_lines
    from src.parsing.beatmap import TimingPoint
    base = TimingPoint(0, 400.0, 4, True)
    kiai = [(2000, 6000)]
    sv = [(0, 1.0), (4000, 2.0)]
    tps = _merge_green_lines(base, kiai, sv)
    greens = [t for t in tps if not t.uninherited]
    # kiai-on@2000 (SV1), SV2@4000 (kiai on), kiai-off@6000 (SV2)
    by_t = {int(t.time): t for t in greens}
    assert by_t[2000].kiai and abs(by_t[2000].sv - 1.0) < 1e-3
    assert by_t[4000].kiai and abs(by_t[4000].sv - 2.0) < 1e-3
    assert (not by_t[6000].kiai) and abs(by_t[6000].sv - 2.0) < 1e-3
    # empty SV reduces to kiai-only green lines (back-compat)
    only_kiai = [t for t in _merge_green_lines(base, kiai, []) if not t.uninherited]
    assert len(only_kiai) == 2 and all(abs(t.sv - 1.0) < 1e-3 for t in only_kiai)


def test_curve_value_encode_decode():
    from src.data.signal import _dec_curve, _enc_curve
    assert _enc_curve(0.0) == 0.0
    assert _dec_curve(_enc_curve(40.0)) == 40.0
    assert _enc_curve(10_000.0) <= 1.2          # clamped


def test_curve_cue_encode_reflects_sagitta():
    import pathlib

    from src.config import CH_CURVE
    from src.data.signal import _dec_curve
    from src.parsing.beatmap import TYPE_SLIDER, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    bm.hit_objects = [HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, end_time=400,
                               curve_type="B", curve_points=[(150, 80), (220, 40), (300, 100)],
                               slides=1, length=220.0)]
    n = int(AUDIO.time_to_frame(400)) + 10
    sig = encode_beatmap(bm, n)
    sag = _dec_curve(sig[CH_CURVE, int(AUDIO.time_to_frame(200))])
    assert sag > 15                              # the slider's bow was encoded


def test_recover_stream_gaps():
    from src.data.signal import _recover_stream_gaps
    ch = np.full(60, -1.0, dtype=np.float32)
    for f in (0, 8, 16, 32):
        ch[f] = 1.0
    ch[24] = 0.2                       # weak sub-threshold bump = a dropped 1/4 note
    out = _recover_stream_gaps([0, 8, 16, 32], ch, threshold=0.3, min_gap=2)
    assert any(22 <= p <= 26 for p in out)            # hole recovered
    # genuine 1/2 gap (no bump) must NOT be filled
    ch2 = np.full(60, -1.0, dtype=np.float32)
    for f in (0, 8, 16, 32):
        ch2[f] = 1.0
    out2 = _recover_stream_gaps([0, 8, 16, 32], ch2, threshold=0.3, min_gap=2)
    assert not any(20 <= p <= 28 for p in out2 if p not in (0, 8, 16, 32))


def test_has_red_points():
    from src.data.signal import _has_red_points
    assert _has_red_points([(1, 1), (1, 1), (2, 2)])        # doubled point = red corner
    assert not _has_red_points([(1, 1), (2, 2), (3, 3)])


def test_corner_cue_encode():
    import pathlib

    from src.config import CH_CORNER
    from src.parsing.beatmap import TYPE_SLIDER, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    mid = int(AUDIO.time_to_frame(200))
    n = int(AUDIO.time_to_frame(400)) + 10
    # red-point slider (doubled control point) -> corner cue 1
    bm.hit_objects = [HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, end_time=400,
                               curve_type="B", curve_points=[(200, 200), (200, 200), (300, 100)],
                               slides=1, length=300.0)]
    assert encode_beatmap(bm, n)[CH_CORNER, mid] == 1.0
    # smooth slider -> corner cue 0
    bm.hit_objects = [HitObject(x=100, y=100, time=0, type=TYPE_SLIDER, end_time=400,
                               curve_type="B", curve_points=[(150, 80), (250, 120), (300, 100)],
                               slides=1, length=220.0)]
    assert encode_beatmap(bm, n)[CH_CORNER, mid] == 0.0


def test_corner_cue_decode_makes_red_corner():
    from src.config import N_SLIDER_ANCHORS
    from src.data.signal import _slider_from_anchors
    # an L-shaped (90 deg) anchor polygon: head (256,192) -> (356,192) -> (356,292)
    anchor_ch = np.zeros((2 * N_SLIDER_ANCHORS, 12), dtype=np.float32)
    anchor_ch[0, :] = 100 / 512    # a1 = (356, 192)  (the corner)
    anchor_ch[2, :] = 100 / 512    # a2 dx
    anchor_ch[3, :] = 100 / 384    # a2 dy = (356, 292)  (the end)
    anchor_ch[4, :] = 100 / 512    # a3 = a2 -> deduped (avoid looping back to head)
    anchor_ch[5, :] = 100 / 384
    start = (256, 192)
    ct, pts, _ = _slider_from_anchors(start, anchor_ch, 0, 10, curve_cue=None, corner_cue=1.0)
    assert ct == "B"
    assert any(pts[i] == pts[i + 1] for i in range(len(pts) - 1))   # doubled = red corner
    # corner cue off -> no doubling
    _, pts2, _ = _slider_from_anchors(start, anchor_ch, 0, 10, curve_cue=None, corner_cue=0.0)
    assert not any(pts2[i] == pts2[i + 1] for i in range(len(pts2) - 1))


def test_curve_cue_bows_flat_anchors():
    """The decode realises a high cue as a visible bow even when anchors are collinear."""
    from src.config import N_SLIDER_ANCHORS
    from src.data.signal import _polygon_sagitta, _slider_from_anchors
    anchor_ch = np.zeros((2 * N_SLIDER_ANCHORS, 12), dtype=np.float32)
    anchor_ch[0, :] = 0.2   # dx1  (all dy=0 -> collinear horizontal)
    anchor_ch[2, :] = 0.4   # dx2
    anchor_ch[4, :] = 0.4   # dx3
    start = (256, 192)
    ctype, pts, _ = _slider_from_anchors(start, anchor_ch, 0, 10, curve_cue=40.0)
    assert ctype == "B"
    assert _polygon_sagitta([start, *pts]) >= 30     # bowed to ~the cue
    # a low cue keeps the same flat anchors straight
    ctype2, _, _ = _slider_from_anchors(start, anchor_ch, 0, 10, curve_cue=2.0)
    assert ctype2 == "L"


def test_spacing_value_encode_decode():
    from src.data.signal import SPACING_ENC_CLAMP, _dec_spacing, _enc_spacing
    assert _enc_spacing(0.0) == 0.0
    assert _dec_spacing(0.0) == 0.0
    assert abs(_dec_spacing(_enc_spacing(150.0)) - 150.0) < 1e-3   # round-trips below the cap
    assert _enc_spacing(10_000.0) == SPACING_ENC_CLAMP             # clamped


def test_spacing_channel_holds_distance():
    import pathlib

    from src.config import CH_SPACING
    from src.data.signal import _dec_spacing
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    # two circles 200px apart (x), 1s apart -> the gap holds spacing ~200px
    bm.hit_objects = [HitObject(x=100, y=100, time=0, type=TYPE_CIRCLE, end_time=0),
                      HitObject(x=300, y=100, time=1000, type=TYPE_CIRCLE, end_time=1000)]
    n = int(AUDIO.time_to_frame(1000)) + 10
    sig = encode_beatmap(bm, n)
    mid = int(AUDIO.time_to_frame(500))
    assert abs(_dec_spacing(sig[CH_SPACING, mid]) - 200.0) < 2.0


def test_pick_peaks_default_min_gap_collapses_plateau():
    """Invariant: the model can emit a flat-topped onset; with the decode default
    min_gap_frames>=2 a width-2 plateau must collapse to a single object (no
    duplicate notes 1 frame apart). Locks the picker against a min_gap regression."""
    from src.data.signal import _pick_peaks
    ch = np.full(20, -1.0, dtype=np.float32)
    ch[5] = ch[6] = 1.0                       # flat-topped onset (width 2)
    assert _pick_peaks(ch, threshold=0.3, min_gap=2) == [5]   # one object
    # a wide plateau (width 3, ~35 ms) is genuinely two onsets at the editor grid
    ch3 = np.full(20, -1.0, dtype=np.float32)
    ch3[5:8] = 1.0
    assert _pick_peaks(ch3, threshold=0.3, min_gap=2) == [5, 7]


def test_pick_peaks_handles_boundary_onsets():
    """An onset on the first or last frame (object at time 0 / end) is a valid peak."""
    from src.data.signal import _pick_peaks
    ch = np.full(10, -1.0, dtype=np.float32)
    ch[0] = 1.0          # object at time 0
    ch[9] = 1.0          # object at the end
    assert _pick_peaks(ch, threshold=0.3, min_gap=2) == [0, 9]


def test_spinner_time_roundtrip():
    """A spinner's start/end times survive encode -> decode within a frame or two."""
    import pathlib

    from src.parsing.beatmap import TYPE_SPINNER, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    bm.hit_objects = [HitObject(x=256, y=192, time=1000, type=TYPE_SPINNER, end_time=3000)]
    n = int(AUDIO.time_to_frame(3000)) + 10
    dec = [o for o in decode_signal(encode_beatmap(bm, n), min_spinner_frames=10)
           if o.is_spinner]
    assert len(dec) == 1
    assert abs(dec[0].time - 1000) < 2 * AUDIO.ms_per_frame
    assert abs(dec[0].end_time - 3000) < 2 * AUDIO.ms_per_frame


def test_sv_stacked_red_green_green_wins():
    """A red + green timing point at the SAME time: the green's SV must win for the
    section (matches osu! ordering), not be overwritten back to 1.0 by the red."""
    import pathlib

    from src.config import CH_SV
    from src.data.signal import _dec_sv
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject, TimingPoint
    bm = Beatmap(path=pathlib.Path("x.osu"),
                 timing_points=[TimingPoint(0, 400.0, 4, True),        # red, SV 1.0
                                TimingPoint(0, -50.0, 4, False)])      # green, SV 2.0 (same t)
    bm.hit_objects = [HitObject(x=100, y=100, time=t, type=TYPE_CIRCLE, end_time=t)
                      for t in range(0, 4000, 200)]
    n = int(AUDIO.time_to_frame(4000)) + 10
    sig = encode_beatmap(bm, n)
    assert abs(_dec_sv(sig[CH_SV, 10]) - 2.0) < 1e-2     # green won


def test_decode_spacing_all_zero_distance_skips():
    """Stacked objects (all gaps 0 px) -> the channel is baseline, so decode_spacing
    returns [] (skip respacing) instead of emitting bogus zero magnitudes."""
    import pathlib

    from src.data.signal import decode_spacing
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    bm.hit_objects = [HitObject(x=100, y=100, time=0, type=TYPE_CIRCLE, end_time=0),
                      HitObject(x=100, y=100, time=500, type=TYPE_CIRCLE, end_time=500)]
    n = int(AUDIO.time_to_frame(500)) + 10
    assert decode_spacing(encode_beatmap(bm, n), bm.hit_objects) == []


def test_decode_spacing_aligns_and_guards():
    import pathlib

    from src.config import CH_SPACING
    from src.data.signal import decode_spacing
    from src.parsing.beatmap import TYPE_CIRCLE, Beatmap, HitObject
    bm = Beatmap(path=pathlib.Path("x.osu"))
    objs = [HitObject(x=100, y=100, time=0, type=TYPE_CIRCLE, end_time=0),
            HitObject(x=300, y=100, time=1000, type=TYPE_CIRCLE, end_time=1000),
            HitObject(x=300, y=300, time=2000, type=TYPE_CIRCLE, end_time=2000)]
    bm.hit_objects = objs
    n = int(AUDIO.time_to_frame(2000)) + 10
    sig = encode_beatmap(bm, n)
    mags = decode_spacing(sig, objs)
    assert len(mags) == 3 and mags[0] == 0.0
    assert abs(mags[1] - 200.0) < 3.0 and abs(mags[2] - 200.0) < 3.0   # both gaps ~200px
    # guard: an all-baseline (untrained) channel -> [] (skip respacing, never collapse)
    sig0 = sig.copy()
    sig0[CH_SPACING] = 0.0
    assert decode_spacing(sig0, objs) == []
    # guard: a pre-v8 signal with no spacing channel -> []
    assert decode_spacing(sig[:CH_SPACING], objs) == []
