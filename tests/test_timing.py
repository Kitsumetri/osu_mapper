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
    assert 125 <= _normalise_octave(75) < 250    # doubled
    assert 125 <= _normalise_octave(480) < 250   # halved twice
    assert _normalise_octave(0) == 0


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


def test_silence_falls_back():
    tp = estimate_timing_point(np.zeros(AUDIO.sample_rate * 3, dtype=np.float32))
    assert tp.uninherited is True
    assert tp.beat_length > 0  # graceful fallback, no crash
