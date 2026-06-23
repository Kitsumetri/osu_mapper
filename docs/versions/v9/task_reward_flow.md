# v9 — rhythm≫flow reweight + objective playability penalty + brute-force audit

*STATIC / frozen — v9 task report.*

Three changes to the ranked-map reward (`src/eval/reward.py`), the metric set
(`src/metrics.py`), and the gold-measurement tool (`src/eval/measure_reward.py`).
The flat-top anti-reward-hacking core and the family-balanced structure are kept
intact end to end. **All edits are hermetic-tested only; the USER must still run
`measure_reward --all` (command at the bottom) on the real corpus to confirm gold
maps still score high — synthetic tests cannot calibrate against real maps.**

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

### (i) Distributional traits → NEW metrics in `compute_metrics` (need a gold band)

Style varies, so these get a per-SR-bucket band (registered in
`src/corpus_stats.py` `KEYS`; **band-less until the USER refreshes gold stats** —
`reward.py` ignores band-less metrics via `_metric_band_score` returning `None`,
so adding them now is safe).

- **`stream_spacing_cv`** (flow family, within-weight 1.0) — *stream-spacing
  regularity.* Coefficient of variation `std / mean` of the circle→circle spacing
  of the pairs that are detected as a stream (`nb ≤ 0.30` beat, `d ≤ 120` px). A
  clean, even stream has near-constant spacing → CV ≈ 0; a messy/jittered stream
  → high CV. 0.0 when there is no stream.
- **`slider_anchor_spread_px`** (slider_shape family, within-weight 0.75) — *sane
  anchor placement.* Mean over sliders of `slider_anchor_min_gap(o)` = the smallest
  px gap between consecutive control points (head + anchors), capped at 200 px so a
  few huge sliders don't dominate. Low ⇒ anchors bunched (kinky/degenerate
  polygons); real ranked sliders space anchors sanely. 0.0 with no anchored sliders.

### (ii) Objective DEFECTS → bounded playability penalty in `reward.py` (no band)

A ranked map of *any* style has ~zero of these, so they need no distribution.
Each is a **fraction of affected objects in [0, 1]**; the penalty is the
`DEFECT_WEIGHTS`-weighted mean of the four rates (so itself in [0, 1]).
`playability = 1 - penalty`.

osu! domain math (playfield 512×384; circle radius `r = 54.4 - 4.48·CS` px via
`metrics.circle_radius_px`; stack radius ≈ 3 px):

| defect (weight) | rate definition | osu! basis |
|-----------------|-----------------|------------|
| **unhittable_jump** (1.0) | frac. of consecutive pairs with required cursor velocity `dist/Δt > 4.0 px/ms` | `UNHITTABLE_PX_PER_MS`; sustained human snap cap ~3–4 px/ms — generous, only true defects fire |
| **unintended_stack** (1.0) | frac. of pairs with head-distance `≤ 3 px` (`STACK_RADIUS_PX`) **and** time gap `> 10 ms` (`STACK_GAP_MS`) | a deliberate stack is a near-instant gap; a surprise stack lands a 2nd note under the 1st with a playable gap (cursor has nowhere to go) |
| **slider_overlap** (0.6) | frac. of sliders whose body polyline (`slider_polyline`, shape-agnostic) passes within `r` px of a **non-adjacent** object's head | adjacent follow-through is normal; tangling a far object reads as an overlapping blob |
| **degenerate_anchor** (0.6) | frac. of sliders with `slider_anchor_min_gap < 4 px` (`DEGENERATE_ANCHOR_PX`) | anchors clustered within a few px decode to a spike / zero-length segment |

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

### EXACT command the USER should run to brute-force-validate

```
uv run python -m src.eval.measure_reward --all --workers 10 \
    --songs "C:/osu!/Songs" --db "C:/osu!/osu!.db" \
    --ref-stats artifacts/reference_stats.json \
    --json artifacts/reward_audit.json --bottom-n 50
```

**Acceptance:** gold maps should still score high/tight per SR bucket and per
family (rhythm now dominant — watch it doesn't sag), `playability` mean ≈ 1.0
(real ranked maps have ~no objective defects), and the bottom-N tail should be
dominated by *expected* cases — variable-BPM maps (the grid-snap caveat above) and
genuinely odd maps — not whole healthy styles. After the **gold-stats refresh**
(`uv run python -m src.corpus_stats --songs "C:/osu!/Songs"
--out artifacts/reference_stats.json --workers 10`) the two new metrics get bands
and join `flow` / `slider_shape`; re-run `--all` to confirm they're well-calibrated.

---

## Files touched

- `src/metrics.py` — `circle_radius_px`, `STACK_RADIUS_PX`, `slider_anchor_min_gap`,
  `slider_polyline`; new metrics `stream_spacing_cv`, `slider_anchor_spread_px`.
- `src/eval/reward.py` — `FAMILIES` reweight; `playability_penalty` + helpers +
  constants; `RewardBreakdown.playability`/`.defects`; multiplicative fold in
  `reward_from_metrics`; `reward_from_osu` parses the beatmap once.
- `src/corpus_stats.py` — registered the two new metrics in `KEYS`.
- `src/eval/measure_reward.py` — `--all`/`--limit 0`, `--workers`, `--bottom-n`,
  parallel `measure`, low-reward tail in the report/JSON.
- `tests/test_flow_metrics.py`, `tests/test_reward_flow.py` — new hermetic tests.
