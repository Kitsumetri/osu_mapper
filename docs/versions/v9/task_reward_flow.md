# v9 — rhythm≫flow reweight + objective playability penalty + brute-force audit

*STATIC / frozen — v9 task report.*

Three changes to the ranked-map reward (`src/eval/reward.py`), the metric set
(`src/metrics.py`), and the gold-measurement tool (`src/eval/measure_reward.py`).
The flat-top anti-reward-hacking core and the family-balanced structure are kept
intact end to end. **All edits are hermetic-tested only; the USER must still run
`measure_reward --all` (command at the bottom) on the real corpus to confirm gold
maps still score high — synthetic tests cannot calibrate against real maps.**

> **Recalibration (2026-06-24, audit-driven).** The first brute-force audit
> (`n=87176`, all ranked/loved) found the playability penalty firing on **real**
> gold maps — `playability` averaged **0.76** (min 0.52), dragging reward
> 0.95 → 0.72 — because three of its four "defects" (`unintended_stack`,
> `slider_overlap`, `degenerate_anchor`) are **intentional osu! patterns** (stacks,
> overlaps, red-anchor corners). Fix: those three became **distributional band
> metrics** (`stack_ratio`, `slider_overlap_ratio`, plus the existing
> `slider_anchor_spread_px`), and the penalty is now **velocity-only** with the
> ceiling recalibrated `4.0 → 10.0 px/ms`. `measure_reward` gained a **`--gold`**
> filter (calibrate on the single-BPM training subset) and per-defect reporting.
> Sections B & C below describe this **final** state; the reweight (A) is unchanged.

---

## A. Reweight — rhythm ≫ flow, heavier 1/4 grid-snap

Rationale: **wrong rhythm = unplayable; wrong flow = merely stylistic.** So rhythm
is the heaviest family and flow the lightest *pattern* family, and within rhythm
the 1/4 grid-snap metric is up-weighted so snap dominates the rhythm score.

| family | OLD `W_F` | NEW `W_F` |
|--------|----------:|----------:|
| **rhythm** | 1.0 | **2.0** |
| spacing_aim | 1.0 | 1.0 |
| **flow** | 1.0 | **0.6** |
| slider_shape | 1.0 | 1.0 |
| accents | 0.4 | 0.4 |

Within `rhythm` (within-family weights): `on_quarter_grid_ratio` **2.0 → 3.0**,
`density_per_s` 1.5 (unchanged). So grid-snap is the single largest *effective*
metric weight in the whole reward, and tanking grid-snap costs more `quality` than
tanking any flow metric (asserted in `test_reward_flow.py`).

Family shares of `quality` (family weight / total 5.0): rhythm **40%**,
spacing_aim 20%, slider_shape 20%, flow **12%**, accents 8%. (Within-family
weights only set a metric's relative importance *inside* its family — the share is
set by the family weight, independent of metric count, by design.)

**Anti-hacking unchanged:** each metric sub-score is still the flat-topped tent
(`_band_score`): 1.0 anywhere inside the gold p10–p90 band, linear falloff outside,
no overshoot gradient. Reweighting only changes the *outer* family averages.

### Single-BPM caveat (carried, amplified by the reweight)

`on_quarter_grid_ratio` measures gap-to-nearest-1/4 against a **single** BPM
(`bm.bpm` = first uninherited timing point). On a **variable-BPM** ranked map the
later sections' gaps don't line up with that one BPM, so the metric under-counts →
under-measures those maps. Up-weighting it (now the dominant reward metric)
**amplifies the bias against variable-BPM maps in gold-calibration / real-map
validation** — expect such maps in the `measure_reward --all` low-reward tail.
**It is SAFE for GENERATED maps**, which are single-BPM by construction (one
uninherited line), so the metric is exact for everything the model produces and
everything best-of-N / RL ranks. The audit tail is the place to confirm the
variable-BPM hit is acceptable on real maps before baking the reward into training.

---

## B. Measure flow more carefully — new metrics + objective defect penalty

Two kinds of signal, handled differently.

### (i) Distributional traits → band metrics in `compute_metrics` (need a gold band)

Style varies and these are **intentional patterns at a ranked-typical rate**, so
each gets a per-SR-bucket band (registered in `src/corpus_stats.py` `KEYS`;
**band-less until the USER refreshes gold stats** — `reward.py` ignores band-less
metrics via `_metric_band_score` returning `None`, so adding them now is safe). A
ranked-typical amount scores 1.0 on the flat-topped tent; only a far-out-of-band
amount costs — and it is symmetric (too *few* stacks/overlaps also drifts below the
band), which can't be farmed by piling them on.

- **`stream_spacing_cv`** (flow, within-weight 1.0) — *stream-spacing regularity.*
  CV `std/mean` of circle→circle spacing of stream pairs (`nb ≤ 0.30` beat, `d ≤
  120` px). Clean even stream → CV ≈ 0; jittered → high. 0.0 when no stream.
- **`slider_anchor_spread_px`** (slider_shape, within-weight 0.75) — *sane anchor
  placement.* Mean over sliders of `slider_anchor_min_gap(o)` (smallest px gap
  between consecutive control points), capped at 200 px. Low ⇒ bunched/degenerate
  polygons (incl. red-anchor corners, which are *intended* — hence a band, not a
  penalty). 0.0 with no anchored sliders.
- **`stack_ratio`** (flow, within-weight 0.5) — *stacking rate.* Fraction of
  consecutive pairs on the same spot (head-distance `≤ STACK_RADIUS_PX ≈ 3 px`,
  positive gap). Stacking is a normal pattern; a band rewards a ranked-typical
  amount. *(Was the `unintended_stack` defect — recalibrated to a band.)*
- **`slider_overlap_ratio`** (slider_shape, within-weight 0.5) — *overlap rate.*
  Fraction of sliders whose body polyline (`slider_polyline`, shape-agnostic) passes
  within a circle radius `r = 54.4 − 4.48·CS` px of a **non-adjacent** object's head
  (immediate neighbours skipped = normal follow-through). Overlap is a stylistic
  device; a band rewards a ranked-typical amount. *(Was the `slider_overlap`
  defect — recalibrated to a band.)*

### (ii) Objective DEFECT → bounded playability penalty in `reward.py` (no band)

**Only one** thing is a true defect no ranked map of any style does and that has no
distribution to sit inside: a cursor move **physically impossible** at the map's
rhythm — the decode-glitch teleport failure mode of a GENERATED map. (`stack` /
`overlap` / `degenerate_anchor` were *removed* from the penalty in the
recalibration — see the banner — and are the band metrics above.) The penalty is
the `DEFECT_WEIGHTS`-weighted mean of the rate(s) below, a fraction in [0, 1];
`playability = 1 - penalty`.

| defect (weight) | rate definition | osu! basis |
|-----------------|-----------------|------------|
| **unhittable_jump** (1.0) | frac. of consecutive pairs with required cursor velocity `dist/Δt > 10.0 px/ms` (`UNHITTABLE_PX_PER_MS`) | even the hardest ranked 1/4 jump-bursts top out ~6–7 px/ms; 10 clears real aim while still catching teleport glitches (full playfield ~600 px in < 60 ms). **TUNE to the gold p99** via `measure_reward --gold`'s per-defect rate (want ~0 on gold). The earlier 4.0 clipped real aim maps (the audit's tech/jump tail). |

**Fold into the reward (principled, bounded, anti-hackable):**

```
blended = (1 - sr_weight)·quality + sr_weight·sr_closeness   # the existing convex blend
reward  = blended · playability                              # NEW multiplicative fold
```

- **Multiplicative**, applied *after* the quality/SR blend: a clean map
  (`playability = 1`) is unchanged (full back-compat — `reward_from_metrics`
  defaults `playability=1.0`); defects can only pull the reward DOWN.
- **Bounded** in [0, 1]: product of two [0, 1] numbers; `playability` is clamped.
- **Anti-hackable**: a defect can never *raise* the reward above the
  band-membership ceiling, so there is nothing to farm — same stance as the
  flat-top bands. `RewardBreakdown` gains `playability` + `defects` for auditing.

---

## C. Brute-force tooling — `measure_reward --all`, parallel, low-reward tail

`src/eval/measure_reward.py`:

- **`--all` (or `--limit 0`)** scores EVERY gold map (no cap); `sample_gold_paths`
  returns all candidates when `limit ≤ 0`.
- **`--workers N`** parallelises scoring with `ProcessPoolExecutor`, mirroring
  `corpus_stats.collect` (the rosu SR call dominates; embarrassingly parallel;
  aggregation is order-independent so the result is identical to serial).
- **`--bottom-n N`** (default 50) dumps the N lowest-reward maps — `path`,
  `reward`, `bucket`, `worst_family` (lowest `family_breakdown`), `playability`,
  `defects` — to the `--json` report and prints the worst 25.
- **`--gold`** restricts the scan to the preprocess gold subset (std + single-BPM +
  kiai + ≥10% hitsounds + 1 ≤ SR ≤ 10) via `_is_gold`, so calibration matches the
  model's **training distribution** and sidesteps the variable-BPM grid artifact.
  Reports `n_filtered` (skipped non-gold). The worker parses once and reuses the
  map (`reward_from_osu(..., bm=...)`).
- **per-defect report** — mean of each `defects` rate over scored maps (want ~0 on
  gold). This is the **velocity-threshold calibration signal**: if `unhittable_jump`
  isn't ~0 on gold, raise `UNHITTABLE_PX_PER_MS`.

### EXACT command the USER should run to brute-force-validate

```
# single-BPM GOLD subset (matches training; avoids the variable-BPM grid artifact):
uv run python -m src.eval.measure_reward --all --gold --workers 10 \
    --songs "C:/osu!/Songs" --db "C:/osu!/osu!.db" \
    --ref-stats artifacts/reference_stats.json \
    --json artifacts/reward_audit_gold.json --bottom-n 50
```

(Drop `--gold` to score all ranked/loved, as the first audit did.)

**Acceptance:** with the recalibration, `playability` mean should be **≈ 1.0** (the
0.76 → ~1.0 fix — gold maps have ~no impossible-velocity pairs; per-defect
`unhittable_jump` ≈ 0) so reward ≈ quality·blend (~0.9+), gold maps score high/tight
per bucket + family (rhythm now dominant — watch it doesn't sag), and the bottom-N
tail is *expected* cases (variable-BPM maps even under `--gold` should be gone;
genuinely odd maps remain), not whole healthy styles. Then the **gold-stats refresh**
(`uv run python -m src.corpus_stats --songs "C:/osu!/Songs"
--out artifacts/reference_stats.json --workers 10`) gives the four band metrics
(`stream_spacing_cv`, `slider_anchor_spread_px`, `stack_ratio`,
`slider_overlap_ratio`) their p10/p90 bands; re-run `--all --gold` to confirm.

---

## Files touched

- `src/metrics.py` — `circle_radius_px`, `STACK_RADIUS_PX`, `slider_anchor_min_gap`,
  `slider_polyline`; band metrics `stream_spacing_cv`, `slider_anchor_spread_px`,
  and (recalibration) `stack_ratio_of` + `slider_overlap_ratio_of` → `stack_ratio`,
  `slider_overlap_ratio`.
- `src/eval/reward.py` — `FAMILIES` reweight (+ `stack_ratio` in flow,
  `slider_overlap_ratio` in slider_shape); `playability_penalty` now **velocity-only**
  (`UNHITTABLE_PX_PER_MS` 4.0 → 10.0; the stack/overlap/anchor defect helpers +
  constants removed); `RewardBreakdown.playability`/`.defects`; multiplicative fold
  in `reward_from_metrics`; `reward_from_osu` parses once + accepts a reuse `bm=`.
- `src/corpus_stats.py` — registered all four band metrics in `KEYS`.
- `src/eval/measure_reward.py` — `--all`/`--limit 0`, `--workers`, `--bottom-n`,
  parallel `measure`, low-reward tail; **`--gold`** filter (`_is_gold`) + `n_filtered`
  + per-defect mean reporting.
- `tests/test_flow_metrics.py`, `tests/test_reward_flow.py` — hermetic tests
  (updated for the recalibration: band metrics + velocity-only penalty).
