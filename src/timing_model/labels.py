"""Ground-truth beat/downbeat/BPM/offset labels from a map's timing points.

osu! maps ship human-verified timing, so each map is a free, in-distribution training
label for the timing model (RESEARCH §10.8). This module turns a parsed beatmap's
`[TimingPoints]` into beat times, downbeat times, and a primary `(BPM, offset, meter)`.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..parsing.beatmap import Beatmap, TimingPoint


@dataclass(frozen=True)
class TimingLabel:
    bpm: float
    offset_ms: float       # anchor-downbeat time (the uninherited point's time)
    meter: int
    beat_length_ms: float


def uninherited_points(tps: list[TimingPoint]) -> list[TimingPoint]:
    """Red (uninherited) points with a real tempo, in time order."""
    return sorted((t for t in tps if t.uninherited and t.beat_length > 0),
                  key=lambda t: t.time)


def primary_timing(tps: list[TimingPoint]) -> TimingLabel | None:
    """The map's primary `(BPM, offset, meter)` — the first red point.

    This is the single-BPM target the model predicts (and what `write_osu` needs).
    Returns None if the map has no usable uninherited point.
    """
    u = uninherited_points(tps)
    if not u:
        return None
    tp = u[0]
    return TimingLabel(60000.0 / tp.beat_length, float(tp.time), max(1, tp.meter),
                       float(tp.beat_length))


def beat_grid(tps: list[TimingPoint], duration_ms: float) -> tuple[list[float], list[float]]:
    """Generate `(beat_times, downbeat_times)` (ms) across all tempo segments up to
    ``duration_ms``. Each red point starts a fresh measure (downbeat). Handles
    single- and variable-BPM maps; beats before the first red point are not emitted
    (the osu grid is anchored at the offset).
    """
    u = uninherited_points(tps)
    beats: list[float] = []
    downbeats: list[float] = []
    for i, tp in enumerate(u):
        seg_end = u[i + 1].time if i + 1 < len(u) else duration_ms
        bl, meter = tp.beat_length, max(1, tp.meter)
        k = 0
        t = float(tp.time)
        while t < seg_end - 1e-6:
            beats.append(t)
            if k % meter == 0:
                downbeats.append(t)
            k += 1
            t = tp.time + k * bl
    return beats, downbeats


def beatmap_duration_ms(bm: Beatmap) -> float:
    """Last object end (proxy for song length when the audio isn't decoded)."""
    if not bm.hit_objects:
        return 0.0
    return float(max(o.end_time for o in bm.hit_objects))
