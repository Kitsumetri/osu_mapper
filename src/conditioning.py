"""Difficulty context vector for conditioning the diffusion model.

The denoiser is told the difficulty of the map it should produce via a small
normalised vector ``c = [SR, AR, OD, HP, CS, density, aim]``. Values are scaled
to roughly [0, 1] so the conditioning MLP sees a well-behaved input. Used
identically at train time (from each map's true values) and inference time
(from the user's target). See RESEARCH.md §9.1.

``aim`` (v9, index 6, appended last so 0-5 keep their meaning / older checkpoints):
a per-song **aim-intensity** in [0, 1] derived from the AUDIO (onset energy, via
``data.audio.aim_intensity``). It is the missing per-song lever the v8 spacing
channel lacked (lessons-learned: "a passive channel can't beat the conditioning it
shares") — it carries NEW per-song information so the model can learn
audio -> intensity -> spacing instead of regressing to the SR-average. Its TRAIN
value comes from the audio (stored on each manifest item, like the mel); its
INFERENCE value is computed from the input audio (overridable via ``--aim-intensity``),
mirroring how ``density`` is handled but sourced from audio rather than SR.
"""
from __future__ import annotations

CONTEXT_FIELDS = ["sr", "ar", "od", "hp", "cs", "density", "aim"]
CONTEXT_DIM = len(CONTEXT_FIELDS)

# per-field scale so each lands ~[0, 1]. ``aim`` is already in [0, 1] (scale 1.0).
_SCALE = {"sr": 10.0, "ar": 10.0, "od": 10.0, "hp": 10.0, "cs": 7.0,
          "density": 12.0, "aim": 1.0}


def context_vector(sr: float, ar: float, od: float, hp: float, cs: float,
                   density: float, aim: float = 0.0) -> list[float]:
    """Return the normalised difficulty context vector (length CONTEXT_DIM).

    ``aim`` (per-song audio aim-intensity, already ~[0, 1]) defaults to 0.0 so old
    call sites that pass only the 6 difficulty fields stay valid (the model then
    sees the neutral baseline for this slot)."""
    raw = {"sr": sr, "ar": ar, "od": od, "hp": hp, "cs": cs,
           "density": density, "aim": aim}
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
                   density: float | None = None,
                   aim_intensity: float | None = None) -> list[float]:
    """Build an inference context from a target star rating.

    AR/OD/HP/CS/density default to ``target_settings(sr)`` when not given explicitly.
    ``aim_intensity`` is the per-song audio aim-intensity (``data.audio.aim_intensity``
    on the input song); ``None`` -> 0.0 baseline (back-compat: callers that don't pass
    it, e.g. the spacing-channel probe, behave exactly as before). At real inference
    ``generate`` passes the audio-derived value, with the ``--aim-intensity`` CLI as an
    explicit override (mirrors ``density``).
    """
    s = target_settings(sr)
    overrides = {"ar": ar, "od": od, "hp": hp, "cs": cs, "density": density}
    vals = {k: (s[k] if v is None else v) for k, v in overrides.items()}
    return context_vector(sr, aim=(0.0 if aim_intensity is None else aim_intensity),
                          **vals)


def context_from_manifest(item: dict) -> list[float]:
    """Build the context vector from a manifest entry (defaults if missing).

    ``aim_intensity`` is the per-audio scalar written by ``preprocess`` (shared by a
    song's difficulties, like the mel); old datasets processed before this field
    existed fall back to 0.0 so they still train without re-preprocessing — the model
    simply sees the neutral baseline for the slot (the new conditioning only takes
    effect once the USER reprocesses to populate it)."""
    dur = max(1e-6, item.get("duration_s", 1.0))
    density = item.get("n_objects", 0) / dur
    return context_vector(
        sr=item.get("star_rating", 0.0), ar=item.get("ar", 5.0),
        od=item.get("od", 5.0), hp=item.get("hp", 5.0),
        cs=item.get("cs", 4.0), density=density,
        aim=item.get("aim_intensity", 0.0),
    )
