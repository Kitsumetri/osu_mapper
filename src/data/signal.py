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

import numpy as np

from ..config import AUDIO, N_SIGNAL_CHANNELS, AudioConfig
from ..parsing.beatmap import (
    PLAYFIELD_H,
    PLAYFIELD_W,
    TYPE_CIRCLE,
    TYPE_NEW_COMBO,
    TYPE_SLIDER,
    TYPE_SPINNER,
    Beatmap,
    HitObject,
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
    keys_t: list[float] = []
    keys_x: list[float] = []
    keys_y: list[float] = []

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
            # trace the slider's control points into the cursor channel over the
            # hold so the signal carries the *shape* (waves/arcs), not just the
            # endpoints. Control points are distributed evenly across the hold.
            if obj.curve_points:
                cps = obj.curve_points
                for idx, (cx, cy) in enumerate(cps, start=1):
                    ft = f_start + (f_end - f_start) * idx / len(cps)
                    keys_t.append(ft)
                    keys_x.append(_norm_x(cx))
                    keys_y.append(_norm_y(cy))
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
def _slider_path(start, cur_x, cur_y, p, end_frame, max_anchors: int = 6):
    """Build a slider curve that follows the cursor signal during the hold.

    Samples cursor positions between frame ``p`` and ``end_frame`` and returns
    ``(curve_type, control_points, length)``. A nearly-straight or very short
    path falls back to a 2-point linear slider; otherwise a Bezier through the
    sampled points gives waves / arcs / multi-direction shapes.
    """
    span = end_frame - p
    sx, sy = start
    end_x = int(np.clip(_denorm_x(float(cur_x[end_frame])), 0, PLAYFIELD_W))
    end_y = int(np.clip(_denorm_y(float(cur_y[end_frame])), 0, PLAYFIELD_H))

    if span <= 2:
        length = float(np.hypot(end_x - sx, end_y - sy)) or 10.0
        return "L", [(end_x, end_y)], length

    n_anchor = int(np.clip(span // 3, 1, max_anchors))
    frames = np.linspace(p + 1, end_frame, n_anchor).round().astype(int)
    pts: list[tuple[int, int]] = []
    prev = (sx, sy)
    length = 0.0
    for f in frames:
        px = int(np.clip(_denorm_x(float(cur_x[f])), 0, PLAYFIELD_W))
        py = int(np.clip(_denorm_y(float(cur_y[f])), 0, PLAYFIELD_H))
        if (px, py) == prev:
            continue
        length += float(np.hypot(px - prev[0], py - prev[1]))
        pts.append((px, py))
        prev = (px, py)
    if not pts:
        return "L", [(end_x, end_y)], float(np.hypot(end_x - sx, end_y - sy)) or 10.0

    # straight-line distance vs path length: if the path barely curves, keep it linear
    straight = float(np.hypot(prev[0] - sx, prev[1] - sy))
    ctype = "B" if (length > straight * 1.05 and len(pts) >= 2) else "L"
    if ctype == "L":
        pts = [pts[-1]]
        length = straight or 10.0
    return ctype, pts, max(10.0, length)


def _pick_peaks(channel: np.ndarray, threshold: float, min_gap: int) -> list[int]:
    """Local-maxima peak picker on a [-1,1] channel.

    Boundary frames are handled: an onset on the first or last frame (e.g. an
    object at time 0) is a valid peak. Out-of-range neighbours are treated as
    -inf so the endpoints can win.
    """
    peaks: list[int] = []
    n = len(channel)
    last = -10**9
    for i in range(n):
        v = channel[i]
        if v < threshold:
            continue
        left = channel[i - 1] if i > 0 else -np.inf
        right = channel[i + 1] if i < n - 1 else -np.inf
        if v >= left and v >= right and i - last >= min_gap:
            peaks.append(i)
            last = i
    return peaks


def decode_signal(sig: np.ndarray, cfg: AudioConfig = AUDIO,
                  onset_threshold: float = 0.3,
                  min_gap_frames: int = 2,
                  min_spinner_frames: int = 26,
                  spinner_min_mean: float = 0.3) -> list[HitObject]:
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

    objects: list[HitObject] = []

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
            # follow the cursor path during the hold -> a real curved slider
            ctype, pts, length = _slider_path((x, y), cur_x, cur_y, p, end_frame)
            typ = TYPE_SLIDER | (TYPE_NEW_COMBO if is_nc else 0)
            obj = HitObject(x=x, y=y, time=time, type=typ,
                            curve_type=ctype, curve_points=pts,
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
