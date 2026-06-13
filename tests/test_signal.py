import numpy as np

from src.config import AUDIO, N_SIGNAL_CHANNELS
from src.parsing.beatmap import parse_beatmap
from src.data.signal import encode_beatmap, decode_signal


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
    from src.parsing.beatmap import Beatmap, HitObject, TYPE_CIRCLE
    import pathlib
    bm = Beatmap(path=pathlib.Path("x.osu"))
    times = list(range(0, 4000, 200))
    bm.hit_objects = [HitObject(x=100 + i, y=100, time=t, type=TYPE_CIRCLE,
                                end_time=t) for i, t in enumerate(times)]
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


def test_cursor_channels_track_positions(sample_osu):
    bm = parse_beatmap(sample_osu)
    sig = encode_beatmap(bm, _n_frames(bm))
    # cursor channels are continuous in [-1.2, 1.2]
    assert sig[4].min() >= -1.21 and sig[4].max() <= 1.21
    assert sig[5].min() >= -1.21 and sig[5].max() <= 1.21
