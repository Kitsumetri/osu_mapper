# v9 — A general, pattern-balanced "ranked-map" reward

**Status:** implemented. `src/eval/reward.py` is now family-balanced; a reusable
gold-measurement tool ships at `src/eval/measure_reward.py` (+ tests). Public API
unchanged (`reward_from_osu`, `reward_from_metrics`, `RewardBreakdown` — now with a
`family_breakdown` field). All tests green, ruff clean.

This builds directly on `task3_rl_alignment.md` (part A) and keeps every property
that doc argued for (band membership, flat top, convex SR blend, schema-robustness).
The only change is **how the per-metric band scores are aggregated into `quality`.**

---

## 0. The request

> "As I understand rewards are more for 'jump-maps', but I need [a] general reward
> for any map and patterns. Research and implement how to deal with it. Also try to
> measure gold-data maps with [the] implemented reward."

A *general* reward must give a well-made map of **any** style — jump, stream,
tech/slider, balanced — a high score, and only penalise maps that fall outside what
real ranked maps of that SR look like. The old reward was biased toward jump/spacing
character. This doc diagnoses that bias, fixes it by grouping metrics into **pattern
families**, and verifies the fix against real gold maps.

---

## 1. Diagnosis — where the jump bias came from

The old `quality` was a **flat weighted mean over 14 metrics**. The band membership
itself is *not* jump-biased — it is SR-bucketed, so "a real Insane map's jump_ratio"
is already the target, not "max jumps". **The bias was entirely in the weights**, via
two mechanisms:

### 1a. The spacing/aim family had 3 metrics; slider-shape's signal was tiny

The spacing/aim character is described by **three** metrics
(`mean_spacing_px`, `std_spacing_px`, `jump_ratio`), each weighted 1.5. Slider shape
is described by far fewer high-weight metrics (`slider_ratio` 0.75,
`curved_slider_ratio` 1.0). Because the mean was flat over *metrics*, the family with
more metrics simply got more votes. Effective share of `quality` (weight / total):

| family | metrics | OLD share | NEW share |
|--------|---------|----------:|----------:|
| rhythm | on_quarter_grid, density | 24.1% | 22.7% |
| **spacing_aim** | mean_spacing, std_spacing, jump | **31.0%** | 22.7% |
| flow | stream, turn_angle, reversal | 22.4% | 22.7% |
| **slider_shape** | slider, curved_slider, sv_changes, new_combo | **19.0%** | 22.7% |
| accents | kiai, hitsound | 3.4% | 9.1% |

`spacing_aim` was the single biggest family at 31% — **1.64×** the slider-shape
family. Each spacing metric individually carried 10.3% of `quality`; `slider_ratio`
carried only 5.2%.

### 1b. The bias is confirmed by the user's real best-of-N run

`artifacts/generated/audio_bon_sr5.osu.bon.json`: **every** candidate scored
`slider_ratio = 0.0` (slider structure completely off the real band) yet the
spacing/jump/stream metrics were mostly `1.0`. The winner (cand 6) still scored
`quality = 0.852` under the old reward — three perfect spacing 1.0s out-voting the
broken sliders. A map that is "good jumps, broken sliders" was being scored as nearly
ranked. That is the precise failure the user reported: the reward over-rewards jump
character and under-weights slider shape / structure.

(The sr8 winner shows the dual case — there every family was in-band, so old≈new.
The bias only bites when a family is *off*-band and a bigger family masks it.)

---

## 2. The fix — score `quality` over pattern families, not raw metrics

Metrics are grouped into five families by what they describe about play. `quality`
is now a weighted mean **over families**; within each family it is a weighted mean
over that family's metrics. So a family's contribution is set by its **family
weight**, independent of how many metrics it contains — the within-family weights
only set relative importance *inside* the family.

```
family_score(F) = Σ_{m∈F} w_m · band(m)  /  Σ_{m∈F} w_m      # within-family mean
quality         = Σ_F  W_F · family_score(F) / Σ_F W_F        # cross-family mean
```

Both means run only over metrics/families present in BOTH the map and the ref bucket
(schema-robust, exactly as before).

### Families and weights (`FAMILIES` in `reward.py`)

| family | W_F | metrics (within-weights) | rationale |
|--------|----:|--------------------------|-----------|
| **rhythm** | 1.0 | on_quarter_grid (2.0), density (1.5) | WHEN you click — grid-snap is the strongest ranked discriminator |
| **spacing_aim** | 1.0 | mean_spacing (1.0), std_spacing (1.0), jump (1.0) | HOW FAR the cursor travels — the v8/v9 crux, but no longer out-votes the rest |
| **flow** | 1.0 | stream (1.5), turn_angle (1.0), reversal (0.75) | HOW the cursor moves — **streams live here**, so a stream map is judged as flow, not lumped against jumps |
| **slider_shape** | 1.0 | slider (1.0), curved_slider (1.0), sv_changes (0.75), new_combo (0.5) | slider & combo structure — the family the old reward drowned out |
| **accents** | 0.4 | kiai (0.5), hitsound (0.5) | cosmetic; handled by a separate v9 head — nudges, never decides |

The four **pattern** families are equal-weight (22.7% each); accents is deliberately
low (9.1%). `n_objects`, `bpm`, `duration_s` remain excluded (scene facts, not
quality — and rewarding `n_objects` invites spam).

### Why this makes the reward *general* (and not jump-biased)

- A great **jump** map scores high: spacing_aim in-band, and (because the band is the
  real Insane jump distribution) the other families are also where ranked jump maps
  sit. Nothing changed for it.
- A great **stream** map scores high: stream_ratio is a full member of the *flow*
  family (1.5 within-weight), so streaminess is rewarded on its own axis rather than
  being penalised for "not enough jumps". The SR bucket already encodes that an
  Expert+ stream map has high stream_ratio and modest jumps.
- A great **tech/slider** map scores high: slider_shape is now a full family (22.7%,
  was 19% and easily masked), so curvy sliders + SV structure carry real weight.
- A map that is **good in one family but broken in another** (the bon sr5 winner) no
  longer hides the breakage: each family is ~a quarter of `quality`, so wrecking one
  costs ~a quarter, not the ~5% a single masked metric used to cost.

Crucially, "general" does **not** mean "reward jumps and streams and sliders all at
once". It means: judge each family against *what real ranked maps of this SR look
like*, and let a map be excellent by being a textbook example of its style. Because
the targets are the real per-SR-bucket bands, a stream map is not asked to also be a
jump map — it is asked to look like ranked stream maps, which it does.

---

## 3. Anti-reward-hacking — preserved end to end

Every guard from `task3_rl_alignment.md` §A.4 still holds; the family aggregation
only changes the *outer* average, not the per-metric scoring:

- **Flat-topped band (the core guard).** Each metric's sub-score is `1.0` anywhere
  inside the real `[p10, p90]` and falls off only *outside*. There is **no gradient
  for going more extreme** — the `--spacing-scale`-played-worse lesson. A family
  score is a mean of flat-topped tents, so a family also has no overshoot gradient;
  `quality` is a mean of those, so neither does `quality`. The optimum is "land in
  the real distribution", never "be the most jumpy map possible".
- **Cannot farm one family.** With four equal pattern families, maximising spacing
  past the band cannot raise `quality` (flat top caps spacing at 1.0) **and** cannot
  compensate for a tanked slider_shape (each family is capped at 1.0 and weighted
  equally) — strictly *harder* to hack than the old reward, where 3 big spacing
  metrics could prop up a broken family. `test_family_balance_slider_not_masked_by_spacing`
  and `test_reward_hacking_overshoot_not_better_than_ranked` assert exactly this.
- **SR-closeness via rosu** (a broken/unparseable map mis-rates → `sr_close = 0`)
  and the **convex blend** (a momentary SR miss doesn't zero a good map) are
  unchanged. Held-out **in-game play feedback** remains the final gate.

---

## 4. Measuring gold maps — `src/eval/measure_reward.py`

A reusable calibration tool: it samples K real **ranked** std maps (via
`data/osu_db.py` ranked status; falls back to a random real-map sample if `osu!.db`
is absent), scores each at its **own rosu SR** as the target (so `sr_closeness ≈ 1.0`
by construction and `quality` is the thing under test), and reports the reward
distribution overall, per SR bucket, and per family, plus per-metric averages.

```
uv run python -m src.eval.measure_reward --limit 500 --json artifacts/eval/gold_reward_new.json
```

**Acceptance:** real gold maps should score HIGH and TIGHT across every SR bucket and
every family. A family that scores low on gold maps means the reward is mis-weighted
or a metric is mis-calibrated.

### Result (K = 500 ranked maps, seed 1, ref n = 94639)

```
reward       mean 0.9530  median 0.9762  p10 0.8663  p90 1.0000
quality      mean 0.9277  median 0.9634  p10 0.7944  p90 1.0000
sr_closeness mean 1.0000  (own-SR target -> ~1.0 by construction)

per SR bucket (reward mean / quality mean):
  Easy 0.979/0.967   Normal 0.960/0.939   Hard 0.948/0.920
  Insane 0.954/0.930  Expert 0.947/0.919   Expert+ 0.949/0.922

per family (mean band-membership over gold maps):
  rhythm 0.929   spacing_aim 0.931   flow 0.935   slider_shape 0.907   accents 0.951
```

Gold maps score **0.95 reward / 0.93 quality**, tight (median 0.976), high in every
bucket and every family — the calibrated "this is a ranked map" behaviour we want,
and crucially **balanced across families** (no family is systematically penalised, so
no map *style* is). Slider_shape (0.907) is the lowest family, mostly from
`slider_ratio` 0.899 and `sv_changes_per_min` 0.911 — real ranked maps have a wide
slider-ratio spread, so the band is wide and well-populated; this is healthy, not a
mis-calibration.

### Before / after on the same gold sample

Reweighting barely moves **gold** maps (they already satisfy every family, so the
cross-family vs flat mean is nearly identical) — which is the *point*: a calibrated
reward shouldn't penalise good maps under either scheme. The fix shows up on
**off-band (generated)** maps, where a big family used to mask a broken one:

| sample | OLD quality | NEW quality |
|--------|------------:|------------:|
| gold maps (n=400) — mean | 0.9248 | 0.9258 |
| gold maps — p10 | 0.7838 | 0.7939 |
| **bon sr5 winner** (broken sliders, perfect spacing) | **0.852** | **0.805** |

The bon sr5 winner — perfect spacing, `slider_ratio = 0.0` — drops the most, because
the slider_shape family now pulls its full equal weight instead of being out-voted by
three spacing 1.0s. That is the bias being corrected: a "good jumps, broken sliders"
map is no longer scored as nearly ranked.

---

## 5. Known caveat (carried over, accounted for)

`on_quarter_grid_ratio` assumes a **single BPM** and under-measures *variable-BPM*
maps. It averages 0.928 over the gold sample (gold maps are mostly single-BPM, so the
hit is bounded) and — importantly — it is **one metric inside one family** (rhythm,
22.7% × within-family share ≈ 13% of `quality`), so a single variable-BPM map cannot
be tanked by it the way the old reward let a single metric swing things. The
measurement tool reports it per-metric so the caveat stays visible. It does **not**
affect best-of-N (generated maps are single-BPM by construction).

---

## 6. Summary

- **Biggest bias fixed:** the spacing/aim family was 31% of `quality` (1.64× slider
  shape) because the flat mean let its 3 metrics out-vote everything; a "good jumps,
  broken sliders" map scored ~0.85. Now the four pattern families are equal-weight
  (22.7% each), so slider shape, flow/streams, rhythm and spacing all count the same.
- **New family weights:** rhythm / spacing_aim / flow / slider_shape = 1.0 each;
  accents = 0.4.
- **Gold-map reward:** **before ≈ after on gold (0.925 → 0.926 quality)** — as it
  should be — while the off-band bon sr5 winner correctly drops 0.852 → 0.805.
  K=500 gold acceptance: **reward mean 0.953, median 0.976**, balanced across all SR
  buckets and all families.
- **Anti-hacking preserved:** flat-topped bands at every level (metric → family →
  quality), so there is no overshoot gradient and one family cannot prop up another —
  strictly harder to game than the old flat mean.
- **API back-compatible:** `reward_from_osu` / `reward_from_metrics` / `RewardBreakdown`
  unchanged; `RewardBreakdown` gains a `family_breakdown` dict (best-of-N and other
  callers keep working).
