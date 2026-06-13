"""Encode/decode an osu! beatmap as a frame-aligned multi-channel signal.

The signal is the diffusion model's target. It has ``N_SIGNAL_CHANNELS`` rows
and ``T`` columns (one per audio frame). Channels are documented in
``src.config.SIGNAL_CHANNELS``.

Encoding choices (kept simple + decodable):
  * onset / new_combo: Gaussian bumps centred on the object start frame,
    mapped to [-1, 1] (baseline -1, peak +1).
  * slider_hold / spinner_hold: +1 between start and end frame, else -1.
  * cursor_x / cursor_y: object positions normalised to [-1, 1] and linearly
    interpolated across time so the path is smooth.
"""
from __future__ import annotations

from typing import List
import numpy as np

from ..config import AudioConfig, N_SIGNAL_CHANNELS, AUDIO
from ..parsing.beatmap import (
    Beatmap, HitObject, TYPE_CIRCLE, TYPE_SLIDER, TYPE_SPINNER, TYPE_NEW_COMBO,
    PLAYFIELD_W, PLAYFIELD_H,
)

ONSET_SIGMA_FRAMES = 1.2   # width of onset/new-combo bumps


def _gaussian_bump(signal: np.ndarray, center: float, sigma: float = ONSET_SIGMA_FRAMES):
    """Add a unit-height Gaussian centred at ``center`` (fractional frame)."""
    lo = int(np.floor(center - 4 * sigma))
    hi = int(np.ceil(center + 4 * sigma))
    lo = max(0, lo)
    hi = min(len(signal) - 1, hi)
    if hi < lo:
        return
    idx = np.arange(lo, hi + 1)
    vals = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    signal[idx] = np.maximum(signal[idx], vals)


def _norm_x(x: float) -> float:
    return float(np.clip(x / PLAYFIELD_W * 2 - 1, -1.2, 1.2))


def _norm_y(y: float) -> float:
    return float(np.clip(y / PLAYFIELD_H * 2 - 1, -1.2, 1.2))


def _denorm_x(v: float) -> int:
    return int(round((v + 1) / 2 * PLAYFIELD_W))


def _denorm_y(v: float) -> int:
    return int(round((v + 1) / 2 * PLAYFIELD_H))


def encode_beatmap(bm: Beatmap, n_frames: int, cfg: AudioConfig = AUDIO) -> np.ndarray:
    """Return a (N_SIGNAL_CHANNELS, n_frames) float32 array in [-1, 1]."""
    sig = np.zeros((N_SIGNAL_CHANNELS, n_frames), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    newcombo = np.zeros(n_frames, dtype=np.float32)
    slider = -np.ones(n_frames, dtype=np.float32)
    spinner = -np.ones(n_frames, dtype=np.float32)

    # cursor key-frames (time_frame, x, y) for interpolation
    keys_t: List[float] = []
    keys_x: List[float] = []
    keys_y: List[float] = []

    for obj in bm.hit_objects:
        f_start = cfg.time_to_frame(obj.time)
        if f_start >= n_frames:
            continue
        _gaussian_bump(onset, f_start)
        if obj.is_new_combo:
            _gaussian_bump(newcombo, f_start)

        keys_t.append(f_start)
        keys_x.append(_norm_x(obj.x))
        keys_y.append(_norm_y(obj.y))

        if obj.is_slider:
            f_end = min(n_frames - 1, cfg.time_to_frame(obj.end_time))
            a, b = int(round(f_start)), int(round(f_end))
            if b >= a:
                slider[a:b + 1] = 1.0
            # add an end key-frame at the last control point position
            if obj.curve_points:
                ex, ey = obj.curve_points[-1]
                keys_t.append(f_end)
                keys_x.append(_norm_x(ex))
                keys_y.append(_norm_y(ey))
        elif obj.is_spinner:
            f_end = min(n_frames - 1, cfg.time_to_frame(obj.end_time))
            a, b = int(round(f_start)), int(round(f_end))
            if b >= a:
                spinner[a:b + 1] = 1.0

    # build cursor channels via linear interpolation between key-frames
    if keys_t:
        order = np.argsort(keys_t)
        kt = np.array(keys_t)[order]
        kx = np.array(keys_x)[order]
        ky = np.array(keys_y)[order]
        frames = np.arange(n_frames)
        cur_x = np.interp(frames, kt, kx, left=kx[0], right=kx[-1])
        cur_y = np.interp(frames, kt, ky, left=ky[0], right=ky[-1])
    else:
        cur_x = np.zeros(n_frames)
        cur_y = np.zeros(n_frames)

    sig[0] = onset * 2 - 1
    sig[1] = slider
    sig[2] = spinner
    sig[3] = newcombo * 2 - 1
    sig[4] = cur_x
    sig[5] = cur_y
    return sig


# --- decoding -----------------------------------------------------------------
def _pick_peaks(channel: np.ndarray, threshold: float, min_gap: int) -> List[int]:
    """Local-maxima peak picker on a [-1,1] channel.

    Boundary frames are handled: an onset on the first or last frame (e.g. an
    object at time 0) is a valid peak. Out-of-range neighbours are treated as
    -inf so the endpoints can win.
    """
    peaks: List[int] = []
    n = len(channel)
    last = -10**9
    for i in range(n):
        v = channel[i]
        if v < threshold:
            continue
        left = channel[i - 1] if i > 0 else -np.inf
        right = channel[i + 1] if i < n - 1 else -np.inf
        if v >= left and v >= right:
            if i - last >= min_gap:
                peaks.append(i)
                last = i
    return peaks


def decode_signal(sig: np.ndarray, cfg: AudioConfig = AUDIO,
                  onset_threshold: float = 0.3,
                  min_gap_frames: int = 2,
                  min_spinner_frames: int = 26,
                  spinner_min_mean: float = 0.3) -> List[HitObject]:
    """Decode a generated signal back into discrete hit objects.

    Strategy: peak-pick the onset channel for object times; read cursor
    position at each onset; classify slider vs circle from the slider_hold
    channel near the onset; classify spinner from spinner_hold.
    """
    onset = sig[0]
    slider = sig[1]
    spinner = sig[2]
    newcombo = sig[3]
    cur_x = sig[4]
    cur_y = sig[5]
    n = sig.shape[1]

    objects: List[HitObject] = []

    # spinners first: contiguous runs where spinner_hold > 0. Real spinners are
    # long (>~300 ms) and strongly positive, so require a minimum length and a
    # high in-run mean to avoid turning noisy frames into fake spinners.
    spinner_mask = spinner > 0.0
    i = 0
    spinner_spans = []
    while i < n:
        if spinner_mask[i]:
            j = i
            while j < n and spinner_mask[j]:
                j += 1
            run_len = j - i
            if run_len >= min_spinner_frames and float(spinner[i:j].mean()) > spinner_min_mean:
                spinner_spans.append((i, j - 1))
            i = j
        else:
            i += 1

    peaks = _pick_peaks(onset, onset_threshold, min_gap_frames)
    for k, p in enumerate(peaks):
        # skip onsets that fall inside a spinner span
        if any(a <= p <= b for a, b in spinner_spans):
            continue
        x = _denorm_x(float(cur_x[p]))
        y = _denorm_y(float(cur_y[p]))
        x = int(np.clip(x, 0, PLAYFIELD_W))
        y = int(np.clip(y, 0, PLAYFIELD_H))
        time = int(round(cfg.frame_to_time(p)))
        is_nc = newcombo[p] > 0.0

        # an object must end before the next onset begins (osu! objects don't
        # overlap in time); also stop at the start of the next spinner span.
        next_onset = peaks[k + 1] if k + 1 < len(peaks) else n
        for a, _b in spinner_spans:
            if a > p:
                next_onset = min(next_onset, a)
                break

        # slider if slider_hold is active just after the onset
        win = slider[p:min(n, p + 4)]
        if win.size and win.max() > 0.0:
            # find slider end, clamped to before the next onset
            j = p
            while j < n and slider[j] > 0.0:
                j += 1
            end_frame = max(p + 1, j - 1)
            end_frame = min(end_frame, next_onset - 1)
            end_frame = max(end_frame, p + 1)
            end_x = _denorm_x(float(cur_x[end_frame]))
            end_y = _denorm_y(float(cur_y[end_frame]))
            end_x = int(np.clip(end_x, 0, PLAYFIELD_W))
            end_y = int(np.clip(end_y, 0, PLAYFIELD_H))
            length = float(np.hypot(end_x - x, end_y - y)) or 10.0
            typ = TYPE_SLIDER | (TYPE_NEW_COMBO if is_nc else 0)
            obj = HitObject(x=x, y=y, time=time, type=typ,
                            curve_type="L", curve_points=[(end_x, end_y)],
                            slides=1, length=length,
                            end_time=int(round(cfg.frame_to_time(end_frame))))
            objects.append(obj)
        else:
            typ = TYPE_CIRCLE | (TYPE_NEW_COMBO if is_nc else 0)
            objects.append(HitObject(x=x, y=y, time=time, type=typ, end_time=time))

    # add spinners as objects
    for a, b in spinner_spans:
        objects.append(HitObject(
            x=256, y=192, time=int(round(cfg.frame_to_time(a))),
            type=TYPE_SPINNER, end_time=int(round(cfg.frame_to_time(b)))))

    objects.sort(key=lambda o: o.time)
    return objects
