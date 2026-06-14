"""Post-processing for generated maps. Currently: beat-snapping.

Snapping nudges generated onsets onto the estimated beat grid so the map feels
tighter against the music. It is deliberately *bounded* — only objects already
within ``max_snap_ms`` of a grid line move — so a wrong BPM estimate can't drag
everything onto a bad grid. It is *triplet-aware*: each object snaps to whichever
of the 1/4 or 1/3 subdivisions is closest (covers the ~10%+ of maps that use
triplet rhythms).
"""
from __future__ import annotations

from .parsing.beatmap import HitObject, TimingPoint


def trim_isolated_ends(objects: list[HitObject], max_gap_ms: float = 3000.0) -> int:
    """Drop leading/trailing objects separated from the body by a huge silent gap.

    Fixes the "one lone note seconds after the song ends" artefact: if the last
    object starts more than ``max_gap_ms`` after the previous one finishes, it is
    almost certainly not musical. Returns the number of objects removed.
    """
    if len(objects) < 3:
        return 0
    objs = sorted(objects, key=lambda o: o.time)
    removed = 0
    while len(objs) >= 2 and objs[-1].time - objs[-2].end_time > max_gap_ms:
        objs.pop()
        removed += 1
    while len(objs) >= 2 and objs[1].time - objs[0].end_time > max_gap_ms:
        objs.pop(0)
        removed += 1
    objects[:] = objs
    return removed


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
        # at most ~45 ms, and never more than ~40% of the finest subdivision
        max_snap_ms = min(45.0, 0.4 * min(intervals))

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
