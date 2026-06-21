"""Post-processing for generated maps. Currently: beat-snapping.

Snapping nudges generated onsets onto the estimated beat grid so the map feels
tighter against the music. It is deliberately *bounded* — only objects already
within ``max_snap_ms`` of a grid line move — so a wrong BPM estimate can't drag
everything onto a bad grid. It is *triplet-aware*: each object snaps to whichever
of the 1/4 or 1/3 subdivisions is closest (covers the ~10%+ of maps that use
triplet rhythms). A snapped object is stored at the *rounded grid line*
(``round(offset + k*iv)``), which is exactly the integer-ms tick the osu! editor
draws — so the editor never flags a snapped note as "unsnapped" (see ``snap_to_grid``).
"""
from __future__ import annotations

import statistics
from math import hypot
from pathlib import Path

from .parsing.beatmap import (
    PLAYFIELD_H,
    PLAYFIELD_W,
    Beatmap,
    HitObject,
    TimingPoint,
)


def trim_isolated_ends(objects: list[HitObject], max_gap_ms: float = 3000.0,
                       trail_gap_ms: float | None = None,
                       spinner_tail_ms: float = 1200.0,
                       lead_cluster: int = 4,
                       tail_outlier_ms: float = 700.0,
                       tail_gap_mult: float = 6.0) -> int:
    """Drop leading/trailing objects separated from the body by a huge silent gap.

    Fixes the "one lone note seconds after the song ends" artefact: if the last
    object starts more than ``trail_gap_ms`` after the previous one finishes, it
    is almost certainly not musical. Trailing notes are trimmed more aggressively
    than leading ones (``trail_gap_ms`` defaults below ``max_gap_ms``) because a
    lone outro note the auto-player still "hits" but a human never sees coming is
    the common play-feedback artefact (fb #7).

    Also drops a lone trailing *circle* that lands within ``spinner_tail_ms`` after
    the final spinner: that's a phantom spin-down onset the auto-player can't hit
    (recurring fb). Only the very last object is touched, so a real circle after a
    spinner mid-map is untouched. Returns the number removed.
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
    # density-adaptive trailing trim: the model over-maps low-energy outros, leaving
    # phantom tail notes whose lead-in gap is below trail_gap_ms but a big outlier vs
    # the map's typical spacing (autoplay fails on a lone circle in the dead outro).
    gaps = [b.time - a.end_time for a, b in zip(objs, objs[1:]) if b.time - a.end_time >= 0]
    if len(gaps) >= 8:
        med = statistics.median(gaps)
        floor = max(tail_outlier_ms, tail_gap_mult * med)
        while len(objs) >= 2 and objs[-1].time - objs[-2].end_time > floor:
            objs.pop()
            removed += 1
    if (len(objs) >= 2 and objs[-1].is_circle and objs[-2].is_spinner
            and 0 <= objs[-1].time - objs[-2].end_time < spinner_tail_ms):
        objs.pop()
        removed += 1
    # leading: drop a small intro *cluster* separated from the body by a big gap
    # (fb: a stray out-of-bounds note + a downbeat note, then ~8 s of silence).
    # Find the first big gap within the first `lead_cluster` objects and drop
    # everything before it.
    for k in range(min(len(objs) - 1, lead_cluster)):
        if objs[k + 1].time - objs[k].end_time > max_gap_ms:
            del objs[:k + 1]
            removed += k + 1
            break
    objects[:] = objs
    return removed


def _clamp(v: float, lo: float, hi: float) -> int:
    # NaN/inf-safe: a non-finite coordinate (never produced by decode today, but a
    # cheap guard against a future channel / external .osu feeding garbage in)
    # collapses to the lower bound rather than raising on int(round(nan)).
    if v != v or v in (float("inf"), float("-inf")):
        return int(lo)
    return int(round(min(max(v, lo), hi)))


def clamp_objects_to_playfield(objects: list[HitObject],
                               w: int = PLAYFIELD_W, h: int = PLAYFIELD_H) -> int:
    """Clamp every object's *head* (x, y) into the playfield.

    ``clamp_slider_endpoints`` only guards slider bodies; circles/spinner heads are
    left wherever decode put them. Decode clips positions to ``[0, w] x [0, h]``
    *inclusive*, so a note can sit exactly on the edge (half off-screen). This is a
    final, cheap guard for ALL object heads — a no-op for the common case (decode
    already clipped), but it makes "every emitted object is inside the playfield" an
    explicit postprocess contract and catches anything an upstream change might push
    out. Slider control points are handled by ``clamp_slider_endpoints``. Returns the
    number of object heads moved.
    """
    changed = 0
    for o in objects:
        nx, ny = _clamp(o.x, 0, w), _clamp(o.y, 0, h)
        if (nx, ny) != (o.x, o.y):
            o.x, o.y = nx, ny
            changed += 1
    return changed


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


def snap_slider_ends(objects: list[HitObject], tps: list[TimingPoint] | TimingPoint,
                     slider_multiplier: float, divisor: int = 4,
                     gap_frac: float = 0.88) -> int:
    """Snap slider *durations* to a clean beat-grid multiple (fixes off-rhythm
    slider ends), by adjusting each slider's pixel length.

    osu! derives slider duration from ``length / (SliderMultiplier*100*SV) * beat``,
    so the pixel length needed for an on-grid duration depends on the **SV at the
    slider's time**. Pass the *full* timing-point list (with the SV/kiai green lines)
    — using SV=1 here while the green lines say otherwise is exactly what shifts every
    slider off the grid. Rounds each duration to the nearest 1/``divisor`` beat
    (>=1, kept inside the gap to the next object) and recomputes ``length``/``end_time``.
    Returns sliders changed.
    """
    tps = tps if isinstance(tps, list) else [tps]
    red = next((t for t in tps if t.uninherited and t.beat_length > 0), None)
    if red is None:
        return 0
    helper = Beatmap(path=Path("."), slider_multiplier=slider_multiplier, timing_points=tps)
    objs = sorted(objects, key=lambda o: o.time)
    changed = 0
    for i, o in enumerate(objs):
        if not o.is_slider or o.length <= 0:
            continue
        beat = helper._uninherited_at(o.time).beat_length
        sv = helper._sv_at(o.time)
        if beat <= 0 or sv <= 0:
            continue
        iv = beat / divisor
        velocity = slider_multiplier * 100.0 * sv      # px per beat AT this slider's SV
        dur = o.end_time - o.time
        nxt = objs[i + 1].time if i + 1 < len(objs) else o.time + 10 * beat
        gap = max(iv, nxt - o.time)
        k = max(1, round(dur / iv))
        while k > 1 and k * iv > gap * gap_frac:       # keep it inside the gap
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
        best_grid = None
        for iv in intervals:
            k = round((o.time - offset) / iv)
            grid = offset + k * iv
            delta = grid - o.time
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
                best_grid = grid
        if best_delta is not None and 0 < abs(best_delta) <= max_snap_ms:
            # Store the *rounded grid line* directly, not ``o.time + round(delta)``.
            # osu! stores integer-ms times and the editor's snap tick is round(grid);
            # ``round(delta)`` collapses a half-tie (delta=-0.5) to 0 via banker's
            # rounding, leaving the note one ms off the editor's tick — which the
            # editor then flags as "unsnapped" (the user has to nudge it manually).
            # ``round(grid)`` == the editor's own tick, so the note lands exactly on
            # the grid line the editor draws. The end_time shifts by the same delta
            # so slider/spinner durations are preserved.
            new_time = int(round(best_grid))
            d = new_time - o.time
            if d != 0:
                o.time = new_time
                o.end_time += d
                moved += 1
    objects.sort(key=lambda o: o.time)
    return moved


def _reflect(v: float, lo: float, hi: float) -> float:
    """Fold ``v`` into ``[lo, hi]`` by mirror reflection (triangle wave): a value past
    a wall bounces back in instead of being clamped to the wall. Clamping would
    re-compress spacing right where we are trying to expand it; reflection keeps the
    step length (and mimics how real aim bounces off the playfield edges)."""
    span = hi - lo
    if span <= 0:
        return lo
    t = (v - lo) % (2 * span)
    if t > span:
        t = 2 * span - t
    return lo + t


def respace_by_magnitude(objects: list[HitObject], magnitudes, alpha: float = 1.0,
                         w: int = PLAYFIELD_W, h: int = PLAYFIELD_H) -> int:
    """Re-space objects to target ``magnitudes`` while preserving the model's flow
    *directions* — the v8/P4-B jump-under-dispersion fix (RESEARCH §10.11).

    The model's turn angles are already ≈ real; only the step *length* (= spacing)
    is compressed toward the playfield centre by mean-regression. So we keep each
    consecutive step's direction (from the model's own positions) and set its length
    from a separately-predicted ``magnitudes[k]`` (head-to-head distance from object
    ``k-1`` to object ``k``; ``magnitudes[0]`` is ignored), walking the objects out to
    the intended spacing. Each moved object's slider control points shift with its head
    so slider shapes are preserved. New combos and spinners are **re-anchor** points —
    the walk snaps back to the model's own absolute position there, bounding drift and
    keeping the global layout. Out-of-bounds steps reflect off the walls. ``alpha`` in
    [0, 1] blends model→reconstructed (0 = unchanged, 1 = full magnitude). Returns the
    number of objects moved.
    """
    if len(objects) < 2:
        return 0
    objs = sorted(objects, key=lambda o: o.time)
    mp = [(float(o.x), float(o.y)) for o in objs]          # model (input) positions
    new = [mp[0]]
    for k in range(1, len(objs)):
        if objs[k].is_new_combo or objs[k].is_spinner or objs[k - 1].is_spinner:
            tgt = mp[k]                                    # re-anchor to model absolute
        else:
            dx, dy = mp[k][0] - mp[k - 1][0], mp[k][1] - mp[k - 1][1]
            n = hypot(dx, dy)
            if n < 1e-6:
                tgt = mp[k]                                # no direction -> leave it
            else:
                mag = max(0.0, float(magnitudes[k]))
                tgt = (_reflect(new[k - 1][0] + mag * dx / n, 0, w),
                       _reflect(new[k - 1][1] + mag * dy / n, 0, h))
        new.append((alpha * tgt[0] + (1 - alpha) * mp[k][0],
                    alpha * tgt[1] + (1 - alpha) * mp[k][1]))
    moved = 0
    for o, (nx, ny), (ox, oy) in zip(objs, new, mp):
        dx, dy = int(round(nx - ox)), int(round(ny - oy))
        if dx == 0 and dy == 0:
            continue
        o.x, o.y = int(round(nx)), int(round(ny))
        if o.curve_points:
            o.curve_points = [(cx + dx, cy + dy) for cx, cy in o.curve_points]
        moved += 1
    objects[:] = objs
    return moved
