"""Tests for BPM/offset estimation, using a synthetic click track (no files)."""
import numpy as np

from src.config import AUDIO
from src.data.timing import _normalise_octave, estimate_timing, estimate_timing_point


def _click_track(bpm=150.0, seconds=12.0, offset_s=0.2):
    """A metronome: short noise bursts at a fixed BPM starting at offset_s."""
    sr = AUDIO.sample_rate
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    beat = 60.0 / bpm
    t = offset_s
    rng = np.random.RandomState(0)
    while t < seconds:
        i = int(t * sr)
        click = (rng.rand(int(0.01 * sr)) - 0.5).astype(np.float32)  # 10 ms burst
        y[i:i + len(click)] += click
        t += beat
    return y


def test_normalise_octave_folds_into_range():
    from src.data.timing import OSU_BPM_HI, OSU_BPM_LO
    assert OSU_BPM_LO <= _normalise_octave(45) < OSU_BPM_HI    # doubled into band
    assert OSU_BPM_LO <= _normalise_octave(480) < OSU_BPM_HI   # halved into band
    assert _normalise_octave(0) == 0


def test_normalise_octave_keeps_plausible_slow_bpm():
    """Regression: the old [125,250) floor DOUBLED genuinely-slow songs (a real
    120-BPM map became 240, an 89-BPM map 178 -> the doubled-red-line bug). A tempo
    already inside the plausible band must be returned unchanged."""
    for bpm in (90.0, 100.0, 110.0, 120.0, 124.0, 128.0, 150.0, 175.0, 200.0):
        assert _normalise_octave(bpm) == bpm
    # the specific reported failure: 120 must NOT become 240
    assert _normalise_octave(120.0) == 120.0


def test_normalise_octave_idempotent_and_in_band():
    from src.data.timing import OSU_BPM_HI, OSU_BPM_LO
    for raw in (30.0, 55.0, 60.0, 89.0, 130.0, 204.0, 205.0, 240.0, 300.0, 440.0):
        folded = _normalise_octave(raw)
        assert OSU_BPM_LO <= folded < OSU_BPM_HI       # always lands in band
        assert _normalise_octave(folded) == folded     # idempotent (no oscillation)


def test_estimate_bpm_on_click_track():
    bpm, offset = estimate_timing(_click_track(bpm=150.0, offset_s=0.2))
    # allow octave-equivalence; the family should be 150
    assert min(abs(bpm - 150 * m) for m in (0.5, 1, 2)) < 6
    # the tracked beat may start a few beats in, so compare the *phase* (offset
    # modulo one beat) against the 200 ms click position.
    beat = 60000.0 / bpm
    phase = offset % beat
    phase_err = min(abs(phase - 200.0), beat - abs(phase - 200.0))
    assert phase_err < 60.0


def test_estimate_timing_point_is_uninherited():
    tp = estimate_timing_point(_click_track(bpm=180.0))
    assert tp.uninherited is True
    assert tp.beat_length > 0
    # beat_length should correspond to a sane BPM
    bpm = 60000.0 / tp.beat_length
    assert 125 <= bpm < 250


def test_slow_song_bpm_is_not_doubled():
    """Regression for the doubled-red-line bug: a genuinely slow song (~100 BPM)
    must NOT be folded up an octave to ~200. End-to-end through librosa."""
    bpm, _ = estimate_timing(_click_track(bpm=100.0, seconds=16.0, offset_s=0.2))
    # accept the true 100 family, but it must land in the plausible band, not 200
    assert 90.0 <= bpm <= 130.0, bpm


def test_silence_falls_back():
    tp = estimate_timing_point(np.zeros(AUDIO.sample_rate * 3, dtype=np.float32))
    assert tp.uninherited is True
    assert tp.beat_length > 0  # graceful fallback, no crash
