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
