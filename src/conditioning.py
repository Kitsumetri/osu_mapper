"""Difficulty context vector for conditioning the diffusion model.

The denoiser is told the difficulty of the map it should produce via a small
normalised vector ``c = [SR, AR, OD, HP, CS, density]``. Values are scaled to
roughly [0, 1] so the conditioning MLP sees a well-behaved input. Used
identically at train time (from each map's true values) and inference time
(from the user's target). See RESEARCH.md §9.1.
"""
from __future__ import annotations

CONTEXT_FIELDS = ["sr", "ar", "od", "hp", "cs", "density"]
CONTEXT_DIM = len(CONTEXT_FIELDS)

# per-field scale so each lands ~[0, 1]
_SCALE = {"sr": 10.0, "ar": 10.0, "od": 10.0, "hp": 10.0, "cs": 7.0, "density": 12.0}


def context_vector(sr: float, ar: float, od: float, hp: float, cs: float,
                   density: float) -> list[float]:
    """Return the normalised difficulty context vector (length CONTEXT_DIM)."""
    raw = {"sr": sr, "ar": ar, "od": od, "hp": hp, "cs": cs, "density": density}
    return [min(1.5, max(0.0, raw[f] / _SCALE[f])) for f in CONTEXT_FIELDS]


def target_context(sr: float, ar: float | None = None, od: float | None = None,
                   hp: float | None = None, cs: float | None = None,
                   density: float | None = None) -> list[float]:
    """Build an inference context from a target star rating.

    AR/OD/HP/CS/density default to rough functions of SR (matching the corpus
    trends in RESEARCH.md §8) when not given explicitly.
    """
    ar = ar if ar is not None else min(10.0, 4.0 + 0.7 * sr)
    od = od if od is not None else min(10.0, 3.0 + 0.8 * sr)
    hp = hp if hp is not None else 5.0
    cs = cs if cs is not None else 4.0
    density = density if density is not None else max(0.8, 0.8 * sr)
    return context_vector(sr, ar, od, hp, cs, density)


def context_from_manifest(item: dict) -> list[float]:
    """Build the context vector from a manifest entry (defaults if missing)."""
    dur = max(1e-6, item.get("duration_s", 1.0))
    density = item.get("n_objects", 0) / dur
    return context_vector(
        sr=item.get("star_rating", 0.0), ar=item.get("ar", 5.0),
        od=item.get("od", 5.0), hp=item.get("hp", 5.0),
        cs=item.get("cs", 4.0), density=density,
    )
