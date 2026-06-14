"""Post-processing for generated maps. Currently: beat-snapping.

Snapping nudges generated onsets onto the estimated beat grid so the map feels
tighter against the music. It is deliberately *bounded* — only objects already
within ``max_snap_ms`` of a grid line move — so a wrong BPM estimate can't drag
everything onto a bad grid. It is *triplet-aware*: each object snaps to whichever
of the 1/4 or 1/3 subdivisions is closest (covers the ~10%+ of maps that use
triplet rhythms).
"""
from __future__ import annotations

from math import hypot

from .parsing.beatmap import PLAYFIELD_H, PLAYFIELD_W, HitObject, TimingPoint


def trim_isolated_ends(objects: list[HitObject], max_gap_ms: float = 3000.0,
                       trail_gap_ms: float | None = None) -> int:
    """Drop leading/trailing objects separated from the body by a huge silent gap.

    Fixes the "one lone note seconds after the song ends" artefact: if the last
    object starts more than ``trail_gap_ms`` after the previous one finishes, it
    is almost certainly not musical. Trailing notes are trimmed more aggressively
    than leading ones (``trail_gap_ms`` defaults below ``max_gap_ms``) because a
    lone outro note the auto-player still "hits" but a human never sees coming is
    the common play-feedback artefact (fb #7). Returns the number removed.
    """
    if len(objects) < 3:
        return 0
    if trail_gap_ms is None:
        trail_gap_ms = min(max_gap_ms, 2200.0)
    objs = sorted(objects, key=lambda o: o.time)
    removed = 0
    while len(objs) >= 2 and objs[-1].time - objs[-2].end_time > trail_gap_ms:
        objs.pop()
        removed += 1
    while len(objs) >= 2 and objs[1].time - objs[0].end_time > max_gap_ms:
        objs.pop(0)
        removed += 1
    objects[:] = objs
    return removed


def _clamp(v: float, lo: float, hi: float) -> int:
    return int(round(min(max(v, lo), hi)))


def clamp_slider_endpoints(objects: list[HitObject],
                           w: int = PLAYFIELD_W, h: int = PLAYFIELD_H) -> int:
    """Keep slider bodies inside the playfield (fb #1: sliders shooting off-screen).

    A slider's pixel ``length`` is independent of its control-point geometry: when
    ``length`` exceeds the control polygon's path length, osu! *extrapolates* the
    path beyond the last anchor along the final segment's direction. After
    ``snap_slider_ends`` stretches a slider's length to snap its duration, that
    extrapolated tail can fly far outside the 512x384 playfield.

    Two guards, both cheap: (1) clamp every anchor (head + control points) into
    the playfield; (2) cap ``length`` so any extrapolation past the last anchor
    stays in-bounds. When the length is trimmed, ``end_time`` is scaled with it
    (duration is proportional to length) so downstream gaps stay consistent.
    Returns the number of sliders adjusted.
    """
    changed = 0
    for o in objects:
        if not o.is_slider or not o.curve_points or o.length <= 0:
            continue
        before = (o.x, o.y, tuple(o.curve_points), o.length)
        o.x, o.y = _clamp(o.x, 0, w), _clamp(o.y, 0, h)
        o.curve_points = [(_clamp(cx, 0, w), _clamp(cy, 0, h)) for cx, cy in o.curve_points]

        anchors = [(o.x, o.y), *o.curve_points]
        poly_len = sum(hypot(b[0] - a[0], b[1] - a[1])
                       for a, b in zip(anchors, anchors[1:]))
        if o.length > poly_len + 0.5:
            # extrapolation distance allowed before the tail leaves the playfield
            last, prev = anchors[-1], anchors[-2]
            dx, dy = last[0] - prev[0], last[1] - prev[1]
            norm = hypot(dx, dy)
            if norm < 1e-6:
                max_extra = 0.0
            else:
                ux, uy = dx / norm, dy / norm
                bounds = [norm * 10]  # generous default if a direction is free
                if ux > 1e-6:
                    bounds.append((w - last[0]) / ux)
                elif ux < -1e-6:
                    bounds.append((0 - last[0]) / ux)
                if uy > 1e-6:
                    bounds.append((h - last[1]) / uy)
                elif uy < -1e-6:
                    bounds.append((0 - last[1]) / uy)
                max_extra = max(0.0, min(bounds))
            new_len = poly_len + min(o.length - poly_len, max_extra)
            if new_len < o.length - 0.5:
                dur = o.end_time - o.time
                o.end_time = o.time + int(round(dur * new_len / o.length))
                o.length = max(10.0, new_len)
        if (o.x, o.y, tuple(o.curve_points), o.length) != before:
            changed += 1
    return changed


def compute_breaks(objects: list[HitObject], min_gap_ms: float = 3500.0,
                   lead_out_ms: float = 200.0, lead_in_ms: float = 400.0,
                   min_len_ms: float = 800.0) -> list[tuple[int, int]]:
    """Find break periods for big silent gaps (>= ``min_gap_ms``) between objects.

    Returns ``(start_ms, end_ms)`` pairs sitting inside each gap, padded away from
    the surrounding objects (``lead_out_ms`` after the previous end, ``lead_in_ms``
    before the next start) and dropped if shorter than ``min_len_ms``. These are
    written as ``[Events]`` break periods so long gaps render as proper breaks
    rather than dead air. Note this only *marks* existing gaps; it does not make a
    dense map sparser (that is a model-side fix).
    """
    if len(objects) < 2:
        return []
    objs = sorted(objects, key=lambda o: o.time)
    breaks: list[tuple[int, int]] = []
    for a, b in zip(objs, objs[1:]):
        if b.time - a.end_time < min_gap_ms:
            continue
        start = int(round(a.end_time + lead_out_ms))
        end = int(round(b.time - lead_in_ms))
        if end - start >= min_len_ms:
            breaks.append((start, end))
    return breaks


def snap_slider_ends(objects: list[HitObject], tp: TimingPoint,
                     slider_multiplier: float, divisor: int = 4,
                     gap_frac: float = 0.88) -> int:
    """Snap slider *durations* to a clean beat-grid multiple (fixes off-rhythm
    slider ends), by adjusting each slider's pixel length.

    osu! derives slider duration from length / velocity, so generated sliders end
    at arbitrary sub-beat times. We round the duration to the nearest 1/``divisor``
    beat multiple (>=1), keeping it short enough not to overlap the next object,
    and recompute ``length`` and ``end_time`` to match. Returns sliders changed.
    """
    if tp.beat_length <= 0:
        return 0
    beat = tp.beat_length
    iv = beat / divisor
    velocity = slider_multiplier * 100.0           # px per beat (SV=1)
    objs = sorted(objects, key=lambda o: o.time)
    changed = 0
    for i, o in enumerate(objs):
        if not o.is_slider or o.length <= 0:
            continue
        dur = o.end_time - o.time
        nxt = objs[i + 1].time if i + 1 < len(objs) else o.time + 10 * beat
        gap = max(iv, nxt - o.time)
        k = max(1, round(dur / iv))
        while k > 1 and k * iv > gap * gap_frac:    # keep it inside the gap
            k -= 1
        new_dur = k * iv
        new_len = new_dur / beat * velocity / max(1, o.slides)
        if abs(new_dur - dur) >= 1:
            changed += 1
        o.length = max(10.0, new_len)
        o.end_time = int(round(o.time + new_dur))
    objects[:] = objs
    return changed


def snap_to_grid(objects: list[HitObject], tp: TimingPoint,
                 divisors: tuple[int, ...] = (4,),
                 max_snap_ms: float | None = None) -> int:
    """Snap object times in-place to the nearest beat subdivision.

    Returns the number of objects moved. Each object's ``end_time`` shifts by the
    same delta so slider/spinner durations are preserved.

    Defaults to a single 1/4 grid: that keeps gaps consistent (all multiples of
    one interval). Passing multiple ``divisors`` (e.g. ``(4, 3)`` for triplets)
    snaps each object to whichever grid is closest, which is only sensible when
    applied per *section* — mixing grids globally makes gaps irregular.
    """
    if tp.beat_length <= 0:
        return 0
    beat = tp.beat_length
    offset = tp.time
    intervals = [beat / d for d in divisors]
    if max_snap_ms is None:
        # at most ~60 ms, and never more than ~50% of the finest subdivision.
        # Loosened from 45 ms / 40% (fb #5): more circles land cleanly on the 1/4
        # grid. Still bounded so a wrong BPM estimate can't drag the whole map.
        max_snap_ms = min(60.0, 0.5 * min(intervals))

    moved = 0
    for o in objects:
        best_delta = None
        for iv in intervals:
            k = round((o.time - offset) / iv)
            grid = offset + k * iv
            delta = grid - o.time
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
        if best_delta is not None and 0 < abs(best_delta) <= max_snap_ms:
            d = int(round(best_delta))
            o.time += d
            o.end_time += d
            moved += 1
    objects.sort(key=lambda o: o.time)
    return moved
