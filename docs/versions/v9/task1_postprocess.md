# Task 1 — Postprocess grid-snapping & edge-case audit

**Question (user, research point #1):** "Does postprocess ensure that circles/sliders land
on 1/4 directly? In the osu! editor I can shift notes more carefully to 1/4 spacing — maybe
a rounding error? Check postprocess for edge cases (notes out of a map, out of the playfield,
not in direct spacing)."

**TL;DR.** There *was* a real rounding bug, and it is exactly the user's complaint. `snap_to_grid`
stored `time + round(delta)` instead of `round(grid)`; on half-tie deltas (Python banker's rounding
sends `-0.5 -> 0`) the move collapsed to zero and the note stayed **1 ms off the editor's tick**,
which the osu! editor then flags as "unsnapped" → the user has to nudge it. On the real generated
samples this hit **1.8 %–52.9 %** of objects depending on BPM/offset. Fixed by storing the rounded
grid line directly; re-measuring drops the editor-unsnapped rate to **0 %** on every sample.
The other half of "notes look off 1/4" is **legitimate 1/8 and 1/6 placement** (a model/decode
choice, not a postprocess bug) — quantified below, left for the orchestrator to judge.

---

## Method

Two probes (hermetic, against the *output* `.osu` files in `artifacts/generated/`, which have
already been through the v8 postprocess pipeline, so they show the residual users actually see):

1. **off-subdivision** — for each object, distance (ms) to the nearest of {1/4, 1/8, 1/6}, using
   the file's first red timing point.
2. **editor-unsnapped** — does the object's integer time equal the editor's integer tick
   `round(offset + k*beat/d)` for *some* divisor d ∈ {1,2,3,4,6,8,12,16}? If no divisor matches,
   the editor draws the note as unsnapped (the thing the user manually nudges).

I also re-ran the **fixed** `snap_to_grid` over the existing outputs to confirm the recovery.

---

## Item-by-item

### 1. Integer-ms rounding of float grid lines — **REAL BUG (this is the user's complaint).** Severity: medium-high.

The orchestrator suspected the integer-ms storage was the issue but guessed the *formula was at
worst 0.5 ms off the true subdivision*. That part is fine and unavoidable — osu! stores integer
ms, so any subdivision is at best 0.5 ms from the float line, and the editor tolerates that
(it rounds its own ticks to int the same way). The actual bug is subtler:

`snap_to_grid` computed `delta = grid - o.time`, then `o.time += int(round(delta))`. When `delta`
is a half-tie (e.g. `-0.5`), `round(-0.5) == 0` (banker's rounding), so **the note is not moved at
all** and remains on the *wrong* side of the rounded grid line. Worked example (240 BPM, offset 22):

```
t = 1585 ; iv = beat/4 = 62.5 ; k = round((1585-22)/62.5) = 25
grid = 22 + 25*62.5 = 1584.5
delta = 1584.5 - 1585 = -0.5  ->  int(round(-0.5)) = 0   (note stays at 1585)
editor's own tick = round(1584.5) = 1584
=> note (1585) != editor tick (1584)  -> editor shows it UNSNAPPED.
```

**The fix:** store `int(round(grid))` directly (== the editor's tick) when the object is within the
snap bound. Proven equivalent to the editor's tick by construction. Slider/spinner `end_time` shifts
by the same realized delta so durations are preserved.

**Measured editor-unsnapped rate, before fix vs after re-snapping the existing outputs:**

| file | BPM | n | before | after |
|------|-----|---|--------|-------|
| `audio_sr5.osu` | 240 | 971 | 17 (1.8 %) | **0** |
| `audio_sr7.osu` | 240 | 1448 | 102 (7.0 %) | **0** |
| `audio_sr9.osu` | 240 | 1912 | 162 (8.5 %) | **0** |
| `audio_sr6.osu` | 205 | 1411 | **747 (52.9 %)** | **0** |
| `[Nightcore]_sr5.osu` | 204 | 1092 | 0 | 0 |
| `Casket of Star_sr5.5.osu` | 190 | 1150 | 0 | 0 |

The rate depends on BPM × offset: when the 1/4 (or 1/8/1/6) tick frequently lands near an `x.5`
ms value, the half-tie path triggers a lot (`audio_sr6` is the worst case at >half the map). Maps
whose ticks happen to land near integers (Nightcore, Casket) were already clean. This BPM-dependence
is exactly why the user saw it on some songs and not others.

### 2. The bounded snap (`max_snap_ms = min(60, 0.5*min(intervals))`) — **CORRECT AS-IS, not the cause.**

The orchestrator suspected objects further off-grid than the bound stay floating off 1/4. Measured:
**0.0 %** of objects in *every* sample are off **all** of {1/4, 1/8, 1/6} by more than 1 ms (worst
residual across all files: 0.50 ms = pure integer rounding). So in practice the model already
emits onsets within the bound of *some* subdivision, and the bound never leaves a note floating.
The bound is doing its protective job (a wrong BPM estimate can't drag the map) without side effects.
**Recommendation: keep it.** No change made.

### 3. Divisor choice {1/4, 1/8, 1/6} — **LEGITIMATE, but a real quality signal. Not a postprocess bug.** Severity: low/model-side.

The remaining "off 1/4" notes are **exactly the notes the snapper placed on the 1/8 or 1/6 grid** —
they are on a clean grid line, just not the 1/4 one, so the editor's *default 1/4 view* shows them
between ticks. Breakdown (after the current pipeline):

| file | on 1/8 | on 1/6 | "off 1/4" total |
|------|--------|--------|-----------------|
| `Casket of Star_sr5.5` | 0.1 % | 0.3 % | 0.3 % |
| `audio_sr5` | 4.2 % | 2.7 % | 6.9 % |
| `audio_sr9` | 3.1 % | 10.9 % | 14.0 % |
| `[Nightcore]_sr5` | 36.4 % | 16.1 % | 42.2 % |

This is **not** a rounding error and **not** something postprocess should "fix" by force-snapping to
1/4 — that would wreck genuine 1/8 streams and 1/6 triplets (and `generate.py` deliberately passes
`(4,8,6)` for that reason). **But** an experienced mapper is right to be suspicious: 16 % on 1/6 for
a song that is plainly straight 1/4/1/8 (Nightcore) is the model placing onsets the snapper then
honors on the wrong sub-grid — i.e. the model's onset timing is noisy, and the 1/6 grid is "catching"
notes that should be 1/8. This is a **model/decode** concern (onset-channel precision, or dropping
1/6 from the default divisor set on songs with no detected triplets), to weigh in v9. I did **not**
change the divisor set — doing so blindly would regress real triplet maps. Flagging for the
orchestrator: consider a per-song divisor decision (detect whether the map uses triplets before
allowing 1/6), or tightening onset precision so fewer notes land near a 1/6 line by accident.

### 4. Out of the playfield — **circles cannot go OOB via postprocess; added an explicit guard anyway.** Severity: low.

- `decode_signal` clips circle/slider-head positions to `[0,512]×[0,384]` *inclusive* (signal.py
  ~638). `respace_by_magnitude` moves heads via `_reflect` (verified: returns a value in `[lo,hi]`
  for all inputs over 100 k random samples) and then blends two in-bounds points, so a circle head
  **can never leave the playfield** through respace. `snap_to_grid` only touches time. The
  orchestrator's worry that respace pushes a *circle* OOB is **unfounded.**
- Slider *bodies/anchors* are the real OOB risk and are already handled by `clamp_slider_endpoints`
  (confirmed: a slider whose anchors get shifted OOB by respace is pulled back in).
- The only residual: a decode-clamped note sitting *exactly* on the edge (0 or 512) is technically
  valid but visually half-off-screen. That's a decode/model placement choice, not a postprocess bug.
- **Change made (cheap insurance, makes the contract explicit):** added
  `clamp_objects_to_playfield(objects)` — clamps *every* object head (circles + spinner + slider
  heads) into the playfield — and wired it into `generate._one_pass` after `clamp_slider_endpoints`.
  It is a no-op on today's pipeline (decode already clips) but guarantees "every emitted head is in
  bounds" against future channel/respace changes.

### 5. Out of a map (time bounds) — **acceptable, no crash; no clamp needed.** Severity: none.

`snap_to_grid` can push a note at the very start to a slightly-negative or zero time (e.g. `t=8`,
offset 0 → 0). osu! tolerates negative/lead-in times and times past the audio end (it simply doesn't
render frames past the end). `trim_isolated_ends` already drops huge-gap leading/trailing outliers.
No object is dropped or corrupted at the boundary. **No change made** — adding a hard `[0, audio_len]`
clamp would risk deleting a legitimate first/last note for no real benefit.

### 6. Degenerate sliders & timing — **already robust (one theoretical NaN crash fixed).** Severity: low.

Tested zero/one anchor, `length<=0`, `inf` length, SV=0, `beat_length<=0`, no red line, single
object, empty list, slider-after-spinner — **none crash** and all guards in `snap_slider_ends` /
`clamp_slider_endpoints` / `compute_breaks` behave. One gap: a **NaN control point** crashed
`clamp_slider_endpoints` at `int(round(nan))`. Decode never produces NaN today (anchor channels are
denormalized + int-clipped, verified with extreme inputs), so this is theoretical — but I made
`_clamp` NaN/inf-safe (collapses a non-finite coord to the lower bound) as a cheap defensive guard.

---

## Changes made

- **`src/postprocess.py`**
  - `snap_to_grid`: store `int(round(grid))` (the editor's tick) instead of `o.time + round(delta)`;
    shift `end_time` by the realized delta; don't count a no-op move. **(the main fix)**
  - `_clamp`: NaN/inf-safe.
  - new `clamp_objects_to_playfield(objects)`: clamp every object head into the playfield.
  - module + function docstrings updated.
- **`src/generate.py`**: import + call `clamp_objects_to_playfield` after `clamp_slider_endpoints`
  in `_one_pass`.
- **`tests/test_postprocess_grid.py`** (new, 8 tests): half-tie lands on editor tick; sweep of
  notes all equal the editor tick; slider duration preserved across the half-tie; clean-BPM no-op;
  no double-count; playfield guard pulls in / no-ops; NaN-safe clamp.

## Test / ruff status

`uv run --extra dev pytest` — **157 passed** (+8 new here; the rest of the delta from the 142
baseline is pre-existing uncommitted test files already in the tree, e.g. `tests/test_reward.py`,
which are not part of this task and were left untouched). `uv run --extra dev ruff check .` — clean.

## Open questions for the orchestrator

1. **1/6 over-firing (item 3).** Nightcore-style straight songs get ~16 % of notes on 1/6 — almost
   certainly the model's onset noise being caught by the 1/6 grid, not real triplets. Worth a v9
   per-song divisor decision (detect triplets before enabling 1/6) or tighter onset precision?
   This is model/decode-side; I deliberately did not touch the divisor set.
2. **Existing artifacts are stale.** The `.osu` files in `artifacts/generated/` were produced by the
   old snap; they still carry the 1-ms offsets. Regenerate (or the orchestrator can ship as-is — the
   offsets are ≤1 ms and only matter when editing, not playing).
3. **240 BPM red lines** on the `audio_*` maps look like a possibly-doubled BPM estimate (timing
   comes from `--timing-from` ref or the estimator). Out of scope for this task but flagging it —
   if the true BPM is 120 mapped at 1/8, the grid story is unchanged but it affects readability.
