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


def target_settings(sr: float) -> dict[str, float]:
    """Raw (un-normalised) difficulty settings a target SR maps to (rough corpus
    trends, RESEARCH.md §8). Single source of truth: ``target_context`` conditions
    the model on these, and ``generate`` writes the same values into the ``.osu``
    so the file's AR/OD/CS/HP match what the model was asked to produce."""
    return {
        # AR tuned to modern player expectations (AR9 ~ median): SR<=3 -> 8.0-8.5,
        # SR<=5 -> 8.5-9.0, SR>5 -> 9.0-10. Also matches real ranked AR (8-10) better
        # than the old 4+0.7*sr, which conditioned on unrealistically low AR.
        "ar": min(10.0, 7.75 + 0.25 * sr),
        "od": min(10.0, 4.0 + 0.7 * sr),
        "hp": 5.0,
        "cs": 4.0,
        "density": max(0.8, 0.8 * sr),
    }


def target_context(sr: float, ar: float | None = None, od: float | None = None,
                   hp: float | None = None, cs: float | None = None,
                   density: float | None = None) -> list[float]:
    """Build an inference context from a target star rating.

    AR/OD/HP/CS/density default to ``target_settings(sr)`` when not given explicitly.
    """
    s = target_settings(sr)
    ar = ar if ar is not None else s["ar"]
    od = od if od is not None else s["od"]
    hp = hp if hp is not None else s["hp"]
    cs = cs if cs is not None else s["cs"]
    density = density if density is not None else s["density"]
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
