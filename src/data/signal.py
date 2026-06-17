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

from ..config import (
    AUDIO,
    CH_CURVE,
    CH_SLIDER_ANCHORS,
    CH_SLIDES,
    CH_SV,
    N_SIGNAL_CHANNELS,
    N_SLIDER_ANCHORS,
    AudioConfig,
)
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
# decoded-slider path length within this ratio of the straight-line distance is
# treated as a *straight* (linear) slider rather than a curved bezier.
SLIDER_STRAIGHT_RATIO = 1.06
SPINNER_MERGE_MS = 800.0   # spinners within this gap are merged into one
# SV channel: encode the multiplier as log2(SV) clamped to +/-SV_LOG2_CLAMP, scaled
# to [-1, 1] (multiplicative & symmetric; 0 = SV 1.0). +/-2 covers 0.25x-4x; the rare
# 10x burst is clipped so it doesn't compress the common range.
SV_LOG2_CLAMP = 2.0


def _enc_sv(sv: float) -> float:
    """SV multiplier -> channel value in [-1, 1]."""
    lg = np.log2(max(float(sv), 1e-3))
    return float(np.clip(lg, -SV_LOG2_CLAMP, SV_LOG2_CLAMP) / SV_LOG2_CLAMP)


def _dec_sv(v: float) -> float:
    """Channel value in [-1, 1] -> SV multiplier."""
    return float(2.0 ** (float(np.clip(v, -1.0, 1.0)) * SV_LOG2_CLAMP))


# Curvature cue: a slider's sagitta (max bow off the head->tail chord, px) encoded as
# sagitta/CURVE_PX_SCALE. Decode treats a cue >= CURVE_DECODE_THRESHOLD_PX as a curve
# and bows the polygon up to CURVE_MAX_BOW_PX. Threshold tunes the curved/straight mix
# (real ranked ~38%; target 38-45%).
CURVE_PX_SCALE = 80.0
CURVE_DECODE_THRESHOLD_PX = 11.0
CURVE_MAX_BOW_PX = 130.0


def _enc_curve(sagitta_px: float) -> float:
    return float(np.clip(sagitta_px / CURVE_PX_SCALE, 0.0, 1.2))


def _dec_curve(v: float) -> float:
    return float(max(0.0, v) * CURVE_PX_SCALE)


def _polygon_sagitta(poly: list[tuple[float, float]]) -> float:
    """Max perpendicular distance (px) of a control polygon off its head->tail chord."""
    if len(poly) < 3:
        return 0.0
    (ax, ay), (bx, by) = poly[0], poly[-1]
    dx, dy = bx - ax, by - ay
    chord = float(np.hypot(dx, dy))
    if chord < 1e-6:
        return max(float(np.hypot(px - ax, py - ay)) for px, py in poly)
    return max(abs((px - ax) * dy - (py - ay) * dx) / chord for px, py in poly)


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


def _enc_slides(slides: int) -> float:
    """Repeat count -> [-1, 1]: 1->-1, 2->-1/3, 3->1/3, 4->1 (held over the span)."""
    return float(np.clip((slides - 1) / 3.0, 0.0, 1.0) * 2 - 1)


def _dec_slides(v: float) -> int:
    """Inverse of _enc_slides: span-mean -> nearest repeat count (>=1)."""
    return int(round((float(np.clip(v, -1.0, 1.0)) + 1) / 2 * 3)) + 1


def encode_beatmap(bm: Beatmap, n_frames: int, cfg: AudioConfig = AUDIO) -> np.ndarray:
    """Return a (N_SIGNAL_CHANNELS, n_frames) float32 array in [-1, 1]."""
    sig = np.zeros((N_SIGNAL_CHANNELS, n_frames), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    newcombo = np.zeros(n_frames, dtype=np.float32)
    slider = -np.ones(n_frames, dtype=np.float32)
    spinner = -np.ones(n_frames, dtype=np.float32)
    kiai = -np.ones(n_frames, dtype=np.float32)
    whistle = np.zeros(n_frames, dtype=np.float32)
    finish = np.zeros(n_frames, dtype=np.float32)
    clap = np.zeros(n_frames, dtype=np.float32)
    # v5 slider-shape channels: head-relative control-point offsets (baseline 0)
    # held over the slider span, + a repeat-count channel (baseline -1 = 1 slide).
    anchors = np.zeros((2 * N_SLIDER_ANCHORS, n_frames), dtype=np.float32)
    slides_ch = -np.ones(n_frames, dtype=np.float32)
    curve_ch = np.zeros(n_frames, dtype=np.float32)  # v7 curvature cue, baseline 0 = straight
    # v7 SV timeline: piecewise-constant multiplier from the timing points (red lines
    # reset to 1.0, green lines set their multiplier), held to the next change.
    sv_ch = np.zeros(n_frames, dtype=np.float32)  # baseline 0 = SV 1.0

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
        # hitsound accent bumps (whistle=2, finish=4, clap=8)
        if obj.hit_sound & 2:
            _gaussian_bump(whistle, f_start)
        if obj.hit_sound & 4:
            _gaussian_bump(finish, f_start)
        if obj.hit_sound & 8:
            _gaussian_bump(clap, f_start)

        keys_t.append(f_start)
        keys_x.append(_norm_x(obj.x))
        keys_y.append(_norm_y(obj.y))

        if obj.is_slider:
            f_end = min(n_frames - 1, cfg.time_to_frame(obj.end_time))
            a, b = int(round(f_start)), int(round(f_end))
            if b >= a:
                slider[a:b + 1] = 1.0
            # slider *shape* now lives in dedicated channels (not the shared cursor
            # path): RDP-simplify the control polygon to <=K anchors, store each as
            # a head-relative offset held over the span. slides -> repeat channel.
            cps = obj.curve_points or [(obj.x, obj.y)]
            simplified = _rdp([(obj.x, obj.y), *cps])[1:] or [cps[-1]]
            anchor_pts = (simplified + [simplified[-1]] * N_SLIDER_ANCHORS)[:N_SLIDER_ANCHORS]
            if b >= a:
                for i, (ax, ay) in enumerate(anchor_pts):
                    dx = float(np.clip((ax - obj.x) / PLAYFIELD_W, -1.2, 1.2))
                    dy = float(np.clip((ay - obj.y) / PLAYFIELD_H, -1.2, 1.2))
                    anchors[2 * i, a:b + 1] = dx
                    anchors[2 * i + 1, a:b + 1] = dy
                slides_ch[a:b + 1] = _enc_slides(obj.slides)
                # curvature cue = the slider's true sagitta (bow off its chord), held
                # over the span; an easy scalar the model can learn confidently.
                curve_ch[a:b + 1] = _enc_curve(_polygon_sagitta([(obj.x, obj.y), *cps]))
            # cursor end key-frame (flow into the next object) = last anchor
            ex, ey = anchor_pts[-1]
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

    # kiai box channel from the map's kiai spans
    for start_ms, end_ms in bm.kiai_spans():
        a = int(round(cfg.time_to_frame(start_ms)))
        b = int(round(cfg.time_to_frame(end_ms)))
        a, b = max(0, a), min(n_frames - 1, b)
        if b >= a:
            kiai[a:b + 1] = 1.0

    # SV timeline: walk timing points in time order (red -> SV 1.0, green -> its
    # multiplier) and hold each value until the next point.
    tps = sorted(bm.timing_points, key=lambda t: t.time)
    for i, tp in enumerate(tps):
        sv = 1.0 if tp.uninherited else tp.sv
        a = max(0, int(round(cfg.time_to_frame(tp.time))))
        b = n_frames if i + 1 >= len(tps) else int(round(cfg.time_to_frame(tps[i + 1].time)))
        b = min(max(a, b), n_frames)
        if b > a:
            sv_ch[a:b] = _enc_sv(sv)

    sig[0] = onset * 2 - 1
    sig[1] = slider
    sig[2] = spinner
    sig[3] = newcombo * 2 - 1
    sig[4] = cur_x
    sig[5] = cur_y
    sig[6] = kiai
    sig[7] = whistle * 2 - 1
    sig[8] = finish * 2 - 1
    sig[9] = clap * 2 - 1
    sig[CH_SLIDER_ANCHORS:CH_SLIDER_ANCHORS + 2 * N_SLIDER_ANCHORS] = anchors
    sig[CH_SLIDES] = slides_ch
    sig[CH_SV] = sv_ch
    sig[CH_CURVE] = curve_ch
    return sig


# --- decoding -----------------------------------------------------------------
def _rdp(points, eps: float = 18.0):
    """Ramer-Douglas-Peucker: drop control points that are within ``eps`` px of
    the line through their neighbours, so simple shapes use fewer anchors."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + bx * ay - by * ax) / norm  # point-line distance
        if d > dmax:
            dmax, idx = d, i
    if dmax <= eps:
        return [a, b]
    left = _rdp(points[:idx + 1], eps)
    right = _rdp(points[idx:], eps)
    return left[:-1] + right


def _slider_path(start, cur_x, cur_y, p, end_frame, max_anchors: int = 8):
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

    # straight-line distance vs path length: only call it a curve if the path is
    # meaningfully longer than a straight line (a low threshold turns model noise
    # into spurious wiggly Beziers).
    straight = float(np.hypot(prev[0] - sx, prev[1] - sy))
    ctype = "B" if (length > straight * 1.10 and len(pts) >= 2) else "L"
    if ctype == "L":
        pts = [pts[-1]]
        length = straight or 10.0
    else:
        # simplify the control polygon so simple shapes (waves, arcs) use few
        # anchors instead of every sampled frame (feedback: Z-waves over-pointed).
        simplified = _rdp([(sx, sy), *pts])[1:]
        if simplified:
            pts = simplified
    return ctype, pts, max(10.0, length)


def _bow_polygon(start, pts, target_px):
    """Push the apex control point perpendicular to the head->tail chord so the bow
    reaches ~``target_px`` (keeps the model's lean direction; bounds-clamped). Used to
    realise the curvature cue when the anchors collapsed to near-straight."""
    sx, sy = start
    ex, ey = pts[-1]
    cx, cy = ex - sx, ey - sy
    clen = float(np.hypot(cx, cy))
    if clen < 1e-6 or target_px <= 0:
        return pts
    ux, uy = -cy / clen, cx / clen  # unit perpendicular to the chord
    target_px = min(target_px, CURVE_MAX_BOW_PX)
    mids = pts[:-1]
    if mids:
        def _perp(q):
            return (q[0] - sx) * ux + (q[1] - sy) * uy
        ai = max(range(len(mids)), key=lambda i: abs(_perp(mids[i])))
        sign = 1.0 if _perp(mids[ai]) >= 0 else -1.0
        delta = sign * target_px - _perp(mids[ai])
        out = list(pts)
        out[ai] = (int(np.clip(mids[ai][0] + ux * delta, 0, PLAYFIELD_W)),
                   int(np.clip(mids[ai][1] + uy * delta, 0, PLAYFIELD_H)))
        return out
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    return [(int(np.clip(mx + ux * target_px, 0, PLAYFIELD_W)),
             int(np.clip(my + uy * target_px, 0, PLAYFIELD_H))), pts[-1]]


def _slider_from_anchors(start, anchor_ch, p, end_frame, curve_cue=None):
    """Build a slider curve from the v5 dedicated anchor channels.

    Each of the K anchors' offset is read as the *mean over the slider span*
    (robust to denoising noise), denormalised, and added to the head. Consecutive
    duplicate anchors are dropped (padding / collapsed shapes). ``curve_cue`` (v7, the
    intended sagitta in px) steers the straight/curved choice and bows the polygon to
    the target so curves are visible even when the anchors collapse. Returns
    ``(curve_type, control_points, length)``.
    """
    sx, sy = start
    lo = p
    hi = min(end_frame, anchor_ch.shape[1] - 1)
    if hi < lo:
        hi = lo
    prev = (sx, sy)
    pts: list[tuple[int, int]] = []
    length = 0.0
    for i in range(N_SLIDER_ANCHORS):
        dx = float(anchor_ch[2 * i, lo:hi + 1].mean())
        dy = float(anchor_ch[2 * i + 1, lo:hi + 1].mean())
        ax = int(np.clip(sx + dx * PLAYFIELD_W, 0, PLAYFIELD_W))
        ay = int(np.clip(sy + dy * PLAYFIELD_H, 0, PLAYFIELD_H))
        if (ax, ay) == prev:
            continue
        length += float(np.hypot(ax - prev[0], ay - prev[1]))
        pts.append((ax, ay))
        prev = (ax, ay)
    if not pts:  # all anchors collapsed to the head -> tiny linear slider
        return "L", [(int(np.clip(sx + 10, 0, PLAYFIELD_W)), sy)], 10.0
    # RDP-simplify: drop anchors within eps of the line through their neighbours,
    # so clustered/redundant points collapse (fixes the "imposter line" — a straight
    # slider that kept 2 bunched control points near the end).
    simplified = _rdp([(sx, sy), *pts])[1:] or [pts[-1]]
    prev = (sx, sy)
    length = 0.0
    for q in simplified:
        length += float(np.hypot(q[0] - prev[0], q[1] - prev[1]))
        prev = q
    # straight-vs-curved. v7: trust the learned curvature cue (a robust scalar) over
    # the path/chord ratio (which the anchors' collapse renders ~1.0). Pre-v7 (cue
    # None): fall back to the geometric ratio.
    straight = float(np.hypot(simplified[-1][0] - sx, simplified[-1][1] - sy))
    if curve_cue is not None:
        curved = curve_cue >= CURVE_DECODE_THRESHOLD_PX
    else:
        curved = len(simplified) >= 2 and length > straight * SLIDER_STRAIGHT_RATIO
    if not curved:
        return "L", [simplified[-1]], max(10.0, straight)
    # curved: if the anchors didn't actually bow enough, realise the cue's sagitta
    if curve_cue is not None and _polygon_sagitta([(sx, sy), *simplified]) < curve_cue:
        simplified = _bow_polygon((sx, sy), simplified, curve_cue)
        prev, length = (sx, sy), 0.0
        for q in simplified:
            length += float(np.hypot(q[0] - prev[0], q[1] - prev[1]))
            prev = q
    return "B", simplified, max(10.0, length)


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
                  spinner_min_mean: float = 0.3,
                  min_slider_frames: int = 4,
                  accent_threshold: float = 0.85) -> list[HitObject]:
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
    has_accents = sig.shape[0] >= 10
    whistle = sig[7] if has_accents else None
    finish = sig[8] if has_accents else None
    clap = sig[9] if has_accents else None
    # v5 dedicated slider channels (anchors + repeat count); fall back to the
    # cursor-traced shape for older 10-channel signals. Index-based so a 17-ch v5/v6
    # signal still decodes anchors under the v7 18-ch global.
    has_slider_ch = sig.shape[0] > CH_SLIDES
    _ar_end = CH_SLIDER_ANCHORS + 2 * N_SLIDER_ANCHORS
    anchor_ch = sig[CH_SLIDER_ANCHORS:_ar_end] if has_slider_ch else None
    slides_sig = sig[CH_SLIDES] if has_slider_ch else None
    curve_sig = sig[CH_CURVE] if sig.shape[0] > CH_CURVE else None  # v7 curvature cue
    n = sig.shape[1]

    def _hit_sound(p: int) -> int:
        # Accent channels saturate near +1 on a hitsound, baseline -1; the model
        # outputs them confidently, so only a high threshold thins them out. 0.85
        # was calibrated on real generated output to ~0.33 hitsound usage (matches
        # real maps); 0.0-0.6 all leave it ~0.52 over-firing. See RESEARCH 10.1.C.
        if not has_accents:
            return 0
        return ((2 if whistle[p] > accent_threshold else 0)
                | (4 if finish[p] > accent_threshold else 0)
                | (8 if clap[p] > accent_threshold else 0))

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
    # merge spinners that overlap or sit close together (a single intended spinner
    # split by a brief channel dip -> two overlapping spinners in-game). fb #2.
    if len(spinner_spans) > 1:
        merge_gap = int(cfg.time_to_frame(SPINNER_MERGE_MS))
        merged = [list(spinner_spans[0])]
        for a, b in spinner_spans[1:]:
            if a - merged[-1][1] <= merge_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        spinner_spans = [(a, b) for a, b in merged]

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

        # slider if slider_hold is active just after the onset, AND the hold is
        # long enough — a 1-2 frame "slider" is an unplayable ultra-fast slider,
        # so emit a circle instead.
        win = slider[p:min(n, p + 4)]
        j = p
        while j < n and slider[j] > 0.0:
            j += 1
        end_frame = max(p + 1, min(j - 1, next_onset - 1))
        hs = _hit_sound(p)
        if win.size and win.max() > 0.0 and (end_frame - p) >= min_slider_frames:
            if has_slider_ch:
                # v5: shape from dedicated anchor channels; repeat count from slides.
                # v7: curvature cue (sagitta px) steers the curve/straight decision + bow.
                cue = (_dec_curve(float(curve_sig[p:end_frame + 1].mean()))
                       if curve_sig is not None else None)
                ctype, pts, length = _slider_from_anchors((x, y), anchor_ch, p, end_frame, cue)
                sl = _dec_slides(float(slides_sig[p:end_frame + 1].mean()))
            else:
                # legacy 10-channel: follow the cursor path during the hold
                ctype, pts, length = _slider_path((x, y), cur_x, cur_y, p, end_frame)
                sl = 1
            typ = TYPE_SLIDER | (TYPE_NEW_COMBO if is_nc else 0)
            obj = HitObject(x=x, y=y, time=time, type=typ, hit_sound=hs,
                            curve_type=ctype, curve_points=pts,
                            slides=sl, length=length,
                            end_time=int(round(cfg.frame_to_time(end_frame))))
            objects.append(obj)
        else:
            typ = TYPE_CIRCLE | (TYPE_NEW_COMBO if is_nc else 0)
            objects.append(HitObject(x=x, y=y, time=time, type=typ, hit_sound=hs,
                                     end_time=time))

    # add spinners as objects
    for a, b in spinner_spans:
        objects.append(HitObject(
            x=256, y=192, time=int(round(cfg.frame_to_time(a))),
            type=TYPE_SPINNER, end_time=int(round(cfg.frame_to_time(b)))))

    objects.sort(key=lambda o: o.time)
    return objects


def decode_kiai(sig: np.ndarray, cfg: AudioConfig = AUDIO, threshold: float = 0.0,
                min_ms: float = 1500.0, merge_ms: float = 1000.0,
                max_spans: int = 3) -> list[tuple[int, int]]:
    """Decode the kiai channel into 1-3 clean (start_ms, end_ms) spans.

    Kiai is structured (a few long blocks), so we threshold the channel, drop
    short runs, merge near runs, and keep the strongest ``max_spans``.
    """
    if sig.shape[0] <= 6:
        return []
    kiai = sig[6]
    n = len(kiai)
    mask = kiai > threshold
    runs = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    if not runs:
        return []
    # merge runs separated by a small gap
    merge_frames = cfg.time_to_frame(merge_ms)
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= merge_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # keep runs of at least min_ms, then the strongest max_spans by mean activation
    min_frames = cfg.time_to_frame(min_ms)
    kept = [(a, b) for a, b in merged if (b - a) >= min_frames]
    kept.sort(key=lambda ab: float(kiai[ab[0]:ab[1] + 1].mean()), reverse=True)
    kept = sorted(kept[:max_spans])
    return [(int(round(cfg.frame_to_time(a))), int(round(cfg.frame_to_time(b))))
            for a, b in kept]


def _median_filter1d(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def _merge_smallest_sv(secs: list[list[float]]) -> list[list[float]]:
    """Drop the boundary between the two adjacent sections closest in SV (log space)."""
    best_i, best_d = 0, float("inf")
    for i in range(len(secs) - 1):
        d = abs(np.log2(max(secs[i + 1][1], 1e-3)) - np.log2(max(secs[i][1], 1e-3)))
        if d < best_d:
            best_d, best_i = d, i
    return secs[:best_i + 1] + secs[best_i + 2:]


def decode_sv(sig: np.ndarray, cfg: AudioConfig = AUDIO, min_section_s: float = 3.0,
              max_sections: int = 6, hysteresis: float = 0.2, quant: float = 0.1,
              smooth_ms: float = 700.0) -> list[tuple[int, float]]:
    """Decode the SV channel into a few *stable* (start_ms, sv) sections.

    SV is structural (a handful of green-line sections), so we smooth + quantise the
    channel and only open a new section on a change >= ``hysteresis`` that also clears
    a ``min_section_s`` minimum length, then cap the count. A wobbly channel collapses
    to stable sections instead of per-slider noise (RESEARCH 10.7 P4-A). Returns [] for
    pre-v7 signals without the SV channel.
    """
    if sig.shape[0] <= CH_SV:
        return []
    sv = np.array([_dec_sv(v) for v in sig[CH_SV]], dtype=np.float64)
    n = len(sv)
    if n == 0:
        return []
    k = max(1, int(round(cfg.time_to_frame(smooth_ms))) | 1)  # odd window
    if k > 1 and n >= k:
        sv = _median_filter1d(sv, k)
    svq = np.round(sv / quant) * quant
    min_len = max(1, int(round(min_section_s * cfg.frame_rate)))
    # hysteresis + min-length: a change only registers once min_len has elapsed, so
    # every section (bar the last) is >= min_len by construction.
    secs: list[list[float]] = [[0, float(svq[0])]]
    start = 0
    for i in range(1, n):
        if abs(svq[i] - secs[-1][1]) >= hysteresis and (i - start) >= min_len:
            secs.append([i, float(svq[i])])
            start = i
    while len(secs) > max_sections:
        secs = _merge_smallest_sv(secs)
    # emit, dropping sections identical to the previous one
    out: list[tuple[int, float]] = []
    prev: float | None = None
    for f, s in secs:
        if prev is not None and abs(s - prev) < 1e-6:
            continue
        out.append((int(round(cfg.frame_to_time(f))), round(float(s), 3)))
        prev = s
    return out
