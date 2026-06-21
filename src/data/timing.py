"""Estimate BPM and beat offset from audio for the generated map's timing point.

Uses ``librosa.beat.beat_track`` (no extra dependency / no model download). The
raw tempo from beat tracking is prone to octave errors (half/double) and the
single scalar can be noisy, so we:

  * refine BPM from the *median* inter-beat interval (robust to spurious beats),
  * normalise obvious octave errors into a sane musical range,
  * take the offset from the first tracked beat.

osu! timing wants ``beat_length = 60000 / bpm`` ms and an offset in ms.
"""
from __future__ import annotations

import librosa
import numpy as np

from ..config import AUDIO, AudioConfig
from ..parsing.beatmap import TimingPoint

# librosa's beat tracker is prone to octave errors (half / double tempo). We fold
# only *implausible* tempi back into a generous musical band, and leave anything
# already inside the band untouched. The band must be slightly wider than one
# octave so the fold loop always terminates (hi/lo > 2).
#
# IMPORTANT: the band's lower edge used to be 125 BPM, which silently *doubled*
# every genuinely-slow song — a real 120-BPM map folded to 240, an 89-BPM map to
# 178 (the "doubled ~240 BPM red lines on audio_* maps" report). Plenty of ranked
# maps sit in the 89-124 range, so the floor is now 89 and such tempi are kept.
OSU_BPM_LO = 89.0
OSU_BPM_HI = 205.0
OSU_START_BPM = 160.0


def _normalise_octave(bpm: float, lo: float = OSU_BPM_LO, hi: float = OSU_BPM_HI) -> float:
    """Fold half/double-tempo *errors* into ``[lo, hi)``.

    A tempo already inside the band is plausible and returned unchanged (so a real
    slow song is never doubled); only out-of-band values are octave-shifted in.
    """
    if bpm <= 0:
        return 0.0
    if lo <= bpm < hi:
        return bpm
    while bpm < lo:
        bpm *= 2
    while bpm >= hi:
        bpm /= 2
    return bpm


def estimate_timing(y: np.ndarray, cfg: AudioConfig = AUDIO,
                    hop_length: int = 512) -> tuple[float, float]:
    """Return (bpm, offset_ms) for a mono waveform ``y``."""
    onset_env = librosa.onset.onset_strength(y=y, sr=cfg.sample_rate, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=cfg.sample_rate, hop_length=hop_length,
        start_bpm=OSU_START_BPM, units="frames",
    )
    beat_times = librosa.frames_to_time(beats, sr=cfg.sample_rate, hop_length=hop_length)

    if len(beat_times) >= 2:
        ibi = float(np.median(np.diff(beat_times)))  # seconds per beat
        bpm = 60.0 / ibi if ibi > 0 else float(np.atleast_1d(tempo)[0])
        offset_ms = float(beat_times[0] * 1000.0)
    else:
        bpm = float(np.atleast_1d(tempo)[0])
        offset_ms = 0.0

    bpm = _normalise_octave(bpm)
    return round(bpm, 3), round(offset_ms, 1)


def estimate_timing_point(y: np.ndarray, cfg: AudioConfig = AUDIO) -> TimingPoint:
    """Convenience: estimate timing and return an uninherited osu! TimingPoint."""
    bpm, offset_ms = estimate_timing(y, cfg)
    if bpm <= 0:
        return TimingPoint(0, 500.0, 4, True)  # 120 BPM fallback
    beat_length = 60000.0 / bpm
    return TimingPoint(time=int(round(offset_ms)), beat_length=beat_length,
                       meter=4, uninherited=True)
