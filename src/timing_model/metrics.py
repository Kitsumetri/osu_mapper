"""Correctness metrics for the timing model (RESEARCH §10.8).

Two families:
- generic beat tracking: `f_measure` (±tol match) — a native, dependency-free version of
  the standard MIR metric (use `mir_eval.beat` for CMLt/AMLt during a full benchmark).
- osu-specific exact-match: `bpm_offset_metrics` + `grid_drift` — the numbers that decide
  whether a predicted `(BPM, offset)` is usable in-game (we have ground truth from maps).
"""
from __future__ import annotations

# tempo ratios that look "metrically plausible" but are wrong for osu (octave/triple errors)
OCTAVE_RATIOS = (2.0, 0.5, 3.0, 1 / 3, 1.5, 2 / 3, 4.0, 0.25)


def f_measure(pred: list[float], ref: list[float], tol_ms: float = 70.0) -> dict:
    """Beat F-measure: a predicted beat is correct if within ``tol_ms`` of a ref beat
    (greedy one-to-one match on sorted times). Returns precision/recall/f."""
    pred_s = sorted(pred)
    ref_s = sorted(ref)
    used = [False] * len(ref_s)
    tp = 0
    j = 0
    for p in pred_s:
        # advance to the first ref within reach
        while j < len(ref_s) and ref_s[j] < p - tol_ms:
            j += 1
        k = j
        while k < len(ref_s) and ref_s[k] <= p + tol_ms:
            if not used[k]:
                used[k] = True
                tp += 1
                break
            k += 1
    precision = tp / len(pred_s) if pred_s else 0.0
    recall = tp / len(ref_s) if ref_s else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f": round(f, 4)}


def _phase_err_ms(pred_offset: float, ref_offset: float, beat_length: float) -> float:
    """Offset error modulo one beat (the grid repeats, so only the phase matters)."""
    d = (pred_offset - ref_offset) % beat_length
    return min(d, beat_length - d)


def bpm_offset_metrics(pred_bpm: float, pred_offset: float,
                       ref_bpm: float, ref_offset: float, bpm_tol: float = 0.1) -> dict:
    """osu exact-match for a predicted (BPM, offset) vs ground truth.

    - ``bpm_err`` / ``exact_bpm`` (within ``bpm_tol``); ``octave`` flags a plausible-but-wrong
      tempo multiple (2x, 1/2, 3x, ...).
    - ``offset_err_ms`` = phase error mod one beat; bucket excellent<5 / good<10 / playable<20.
    - ``exact`` = right tempo AND a playable offset.
    """
    bl = 60000.0 / ref_bpm
    bpm_err = abs(pred_bpm - ref_bpm)
    exact_bpm = bpm_err <= bpm_tol
    octave = (not exact_bpm) and any(
        abs(pred_bpm - ref_bpm * r) <= max(bpm_tol, 0.02 * ref_bpm * r) for r in OCTAVE_RATIOS)
    off_err = _phase_err_ms(pred_offset, ref_offset, bl)
    bucket = ("excellent" if off_err < 5 else "good" if off_err < 10
              else "playable" if off_err < 20 else "off")
    return {"bpm_err": round(bpm_err, 3), "exact_bpm": exact_bpm, "octave": octave,
            "offset_err_ms": round(off_err, 2), "offset_bucket": bucket,
            "exact": bool(exact_bpm and off_err < 20)}


def grid_drift(pred_bpm: float, pred_offset: float, ref_beats: list[float]) -> dict:
    """How far the predicted grid drifts from the true beats over the whole song
    (compounds BPM + offset error). Returns max/mean nearest-line distance (ms)."""
    bl = 60000.0 / pred_bpm
    errs = []
    for b in ref_beats:
        d = (b - pred_offset) % bl
        errs.append(min(d, bl - d))
    if not errs:
        return {"max_ms": 0.0, "mean_ms": 0.0}
    return {"max_ms": round(max(errs), 2), "mean_ms": round(sum(errs) / len(errs), 2)}


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-song `bpm_offset_metrics` dicts into benchmark rates."""
    n = len(rows)
    if not n:
        return {}
    return {
        "n": n,
        "exact_bpm_rate": round(sum(r["exact_bpm"] for r in rows) / n, 3),
        "octave_rate": round(sum(r["octave"] for r in rows) / n, 3),
        "offset_excellent_rate": round(sum(r["offset_bucket"] == "excellent" for r in rows) / n, 3),
        "offset_playable_rate": round(
            sum(r["offset_bucket"] in ("excellent", "good", "playable") for r in rows) / n, 3),
        "exact_rate": round(sum(r["exact"] for r in rows) / n, 3),
    }
