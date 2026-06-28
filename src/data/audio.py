"""Audio feature extraction (log-mel spectrogram) aligned to signal frames."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from ..config import AUDIO, AudioConfig


def load_audio(path: str | Path, cfg: AudioConfig = AUDIO) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=cfg.sample_rate, mono=True)
    return y


def log_mel(y: np.ndarray, cfg: AudioConfig = AUDIO) -> np.ndarray:
    """Return (n_mels, T) log-mel spectrogram, per-feature normalised."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # normalise roughly to ~[-1, 1]; dB range is about [-80, 0]
    logmel = (logmel + 40.0) / 40.0
    return logmel


def audio_to_mel(path: str | Path, cfg: AudioConfig = AUDIO) -> np.ndarray:
    return log_mel(load_audio(path, cfg), cfg)


def aim_intensity(y: np.ndarray, cfg: AudioConfig = AUDIO) -> float:
    """Per-song aim-intensity scalar in [0, 1] derived from the audio's rhythm.

    The diagnosis (HANDOFF §7 / lessons-learned "a passive channel can't beat the
    conditioning it shares"): v8's spacing channel regressed to the SR-average
    because it had no *new per-song* information. This supplies that information at
    the conditioning input: a single scalar capturing how rhythmically busy/percussive
    a song is, so the model can learn audio -> intensity -> spacing (more jumps *on
    the jumpy song*) instead of regressing to the SR mean. Fed into the context vector
    like ``density``.

    Definition (one line): the corpus-referenced mean of the **self-normalised onset
    envelope** (``librosa.onset.onset_strength``), squashed to [0, 1] — a busier /
    more percussive song fires more strong onsets per frame and scores higher.

    Why it is robust + loudness-stable (the round-3a audio-parity concern, see
    ``docs/versions/v9/task_audio_features.md`` §1):
      * ``onset_strength`` is built from the **derivative** of a ``power_to_db``
        mel spectrogram, so it already responds to spectral *change* (onsets), not
        absolute level — a globally louder rip barely moves it.
      * We then divide the envelope by its own peak (``ref = max``, mirroring the
        mel path's ``power_to_db(ref=np.max)`` per-file gain invariance), so the
        statistic is a *shape* of the song's own rhythm, comparable across songs of
        different loudness / mastering.
      * The aggregate is a **mean of the peak-normalised envelope** (a stable central
        statistic, not a single transient), then mapped through a fixed reference
        ``_AIM_REF`` into [0, 1] with a hard clamp, so the value is bounded and
        deterministic regardless of song length or amplitude.

    Silence / empty / non-finite audio -> 0.0 (a calm map; a safe default that also
    matches the missing-manifest-field fallback in ``conditioning.context_from_manifest``).

    CRITICAL — train/infer parity: this is the ONE function used at BOTH preprocess
    (``preprocess.py``, once per audio_id) and inference (``generate.prepare_audio``),
    each on the SAME ``load_audio(path)`` array, so there is no train/infer skew.
    """
    if y is None or y.size == 0 or not np.all(np.isfinite(y)):
        return 0.0
    # onset envelope from the SAME mel front-end as the conditioning mel (same
    # sr/n_fft/hop/n_mels), so it is frame-aligned and uses the existing librosa dep.
    env = librosa.onset.onset_strength(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
    )
    peak = float(env.max())
    if not np.isfinite(peak) or peak <= 1e-8:
        return 0.0                      # flat / silent -> no rhythmic activity
    # self-normalise by the per-song peak -> a loudness-invariant rhythm *shape*,
    # then take a robust central statistic (mean of the normalised envelope).
    score = float(np.mean(env / peak))
    # squash to [0, 1] against a fixed reference so the value is bounded + stable
    # across songs/lengths. _AIM_REF ~ a typical busy-song normalised-mean; sparse
    # songs sit well below it, dense streams approach 1.0.
    return float(min(1.0, max(0.0, score / _AIM_REF)))


# reference normalised-onset-mean that maps to aim_intensity 1.0. Calm/sparse songs
# have most frames near the envelope floor (low normalised mean); dense/percussive
# songs keep the envelope high across frames. 0.35 is a robust busy-song level for
# the peak-normalised onset_strength mean; tune against the corpus if needed (it only
# rescales the [0,1] axis — the conditioning MLP relearns the mapping at train time).
_AIM_REF = 0.35
