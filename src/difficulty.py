"""Star-rating (difficulty) computation via rosu-pp (a Rust port of osu!lazer's
official difficulty calculator).

Star rating is the principled difficulty measure: osu!lazer derives it from
per-object **aim** and **speed** strain values (with time decay), combined into
a single number. Mappers' difficulty *names* ("Hard", "Insane", "Lunatic", ...)
are arbitrary and don't map cleanly to difficulty, so we bucket by SR instead.

We use ``rosu_pp_py`` for an exact, fast (Rust) match to the live SR rather than
re-implementing the (large, frequently-revised) algorithm.
"""
from __future__ import annotations

from pathlib import Path

try:
    import rosu_pp_py as _rosu
except Exception:  # pragma: no cover - optional dep
    _rosu = None

# Official osu!standard star-rating difficulty bands (the song-select spectrum).
SR_BANDS = [
    (0.0, 2.0, "Easy"),
    (2.0, 2.7, "Normal"),
    (2.7, 4.0, "Hard"),
    (4.0, 5.3, "Insane"),
    (5.3, 6.5, "Expert"),
    (6.5, 99.0, "Expert+"),
]
SR_BUCKET_ORDER = [name for *_ , name in SR_BANDS]


def sr_bucket(sr: float) -> str:
    for lo, hi, name in SR_BANDS:
        if lo <= sr < hi:
            return name
    return "Expert+"


def star_rating(path: str | Path) -> float | None:
    """osu!standard star rating for a .osu file, or None if it can't be computed."""
    if _rosu is None:
        return None
    try:
        bm = _rosu.Beatmap(path=str(path))
        if str(bm.mode) not in ("GameMode.Osu", "Osu"):
            return None
        return float(_rosu.Difficulty().calculate(bm).stars)
    except Exception:
        return None
