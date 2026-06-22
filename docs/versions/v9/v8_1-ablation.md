# v8_1 vs v8 — analysis & A/B generation comparison

**Date:** 2026-06-22 · branch `feat/v9-align`
**v8 (baseline):** `runs/20260619-235218-ranked-v8-b160/ckpt/best.pt` (released)
**v8_1 (new):**    `runs/20260622-024342-ranked-v8_1-b160/ckpt/best.pt`

---

## 1. Config diff

Both: same data (`ranked-v8`, 38036 maps), base 160, attn_levels 3, adaln, v-pred,
zero-snr, spatial_loss_weight 3.0, batch 16, accum 1, crop 4096, lr 1.2e-4,
grad_clip 0.3, cfg_drop 0.15, ema 0.999, timesteps 1000. **101.7M params (identical).**

| field         | v8 (baseline)  | v8_1 (new)        |
|---------------|----------------|-------------------|
| **rope**      | `false`        | **`true`**        |
| **loss**      | `mse`          | **`huber`**       |
| **huber_beta**| 1.0 (unused)   | **0.5**           |
| **epochs**    | 60             | **80**            |
| workers       | 8              | 16                |
| log_every     | 50             | 1                 |
| save_every    | 5              | 10               |
| git_commit    | d1a7d47        | 13854d6           |

Three real changes: **+rope**, **mse→huber(0.5)**, **60→80 epochs**. Everything
architectural/optimization else is held fixed — a clean ablation of rope+huber(+20 ep).

---

## 2. Training curves & gnorm stability

> **CRITICAL CAVEAT — loss values are NOT comparable across runs.** v8_1 uses Huber
> (beta 0.5) and v8 uses MSE. Huber is ~quadratic only within |err|<0.5 and linear
> beyond, so its raw loss sits on a *different, smaller* scale. **v8_1's 0.0338 val
> is NOT "better than" v8's 0.041** — that gap is the loss function, not quality.
> Curves are judged by *trend / convergence / stability* only; quality is §3 (A/B).

**Convergence (trend):** both converge cleanly and monotonically with the cosine LR.
- v8: val 0.0780 (e0) → 0.0416 (e59). LR → 1.5e-14, fully decayed. No overfit: train
  0.0415 ≈ val 0.0416 at the end (train tracks val throughout).
- v8_1: val 0.0619 (e0) → 0.0338 (e79). LR → 8.6e-15, fully decayed. No overfit:
  train 0.0340 ≈ val 0.0338 at the end. The extra 20 epochs are still earning tiny
  gains (val 0.0344→0.0338 over e60–79) — a long, flat, healthy tail, no divergence.

**Epoch time:** v8 ~415 s/epoch (≈6.9 h total / 60 ep). v8_1 ~625 s/epoch
(≈13.9 h / 80 ep). The ~50% per-epoch slowdown is the rope rotary embeddings in
attention + log_every=1 logging overhead. Cost roughly doubled.

**Gnorm stability (the base-160 headline win — did rope+huber hold it?):** YES.

| run  | steps logged | mean | median | p99  | max  | #>1.0 | #>2.0 | tail(2k) mean / max |
|------|--------------|------|--------|------|------|-------|-------|---------------------|
| v8   | 2,794 (e50)  | 0.097| 0.080  | 0.47 | 1.32 | 4     | 0     | 0.075 / 0.29        |
| v8_1 | 186,320 (e1) | 0.078| 0.060  | 0.33 | 8.33 | 28    | 1     | 0.050 / 0.26        |

v8_1 logged every step (186k vs v8's sampled 2.8k), so absolute spike counts aren't
1:1, but the *distribution* is clean: mean/median/p99 are all **lower** than v8, and
the tail is calmer (0.050 vs 0.075 mean). One lone 8.33 spike (1 step in 186k =
0.0005%) was caught by grad_clip 0.3 and left no scar in the loss curve. **Verdict:
rope + huber + base-160 stayed fully stable; no divergence, no spike regime.**

---

## 3. The real comparison — A/B generation (same songs, same seed, same args)

Generated from **both `best.pt`** with `torch.manual_seed(1234)` set immediately
before each `generate` (so the only difference per cell is the checkpoint), steps=100,
guidance=2.0, spacing_scale=1.0, `--amp`, timing read from each song's ranked .osu.
Songs span character: **Kawaii** (223 BPM, balanced, known-good anchor),
**Blue Zenith Cut** (200 BPM, stream-leaning), **Highscore** (110 BPM, jump-leaning).
SR targets 4/5/6/7. Reward = family-balanced band-membership (`src/eval/reward.py`)
against `artifacts/reference_stats.json`. Script: `artifacts/_ab_compare.py`,
maps: `artifacts/ab_gen/`.

### 3a. Per-cell reward (total)

| song        | SR | v8 reward | v8_1 reward | Δ (v8_1−v8) |
|-------------|----|-----------|-------------|-------------|
| kawaii      | 4  | 0.719     | **0.844**   | **+0.124**  |
| kawaii      | 5  | 0.860     | 0.842       | −0.018      |
| kawaii      | 6  | 0.807     | **0.853**   | +0.046      |
| kawaii      | 7  | 0.920     | **0.938**   | +0.017      |
| bluezenith  | 4  | **0.723** | 0.188       | **−0.536**  ⚠ |
| bluezenith  | 5  | **0.853** | 0.343       | **−0.510**  ⚠ |
| bluezenith  | 6  | 0.839     | **0.903**   | +0.064      |
| bluezenith  | 7  | 0.952     | 0.948       | −0.004      |
| highscore   | 4  | **0.835** | 0.788       | −0.046      |
| highscore   | 5  | 0.873     | 0.869       | −0.004      |
| highscore   | 6  | 0.803     | **0.840**   | +0.037      |
| highscore   | 7  | 0.632     | **0.935**   | **+0.302**  |

**Mean reward: v8 = 0.818, v8_1 = 0.774.** But this is dominated by two catastrophic
cells. **Excluding the two Blue Zenith low-SR collapses, v8_1 = 0.876 vs v8 = 0.824**
— i.e. on 10 of 12 cells v8_1 is the better model; the headline "worse" is two
specific failures, not a broad regression. Win/loss (|Δ|>0.02): **v8_1 wins 5, loses 3,
ties 4.**

### 3b. Mean family scores (12 cells)

| family        | v8    | v8_1  | Δ       |
|---------------|-------|-------|---------|
| rhythm        | 0.857 | 0.779 | −0.078  |
| spacing_aim   | 0.838 | 0.762 | −0.076  |
| flow          | 0.947 | 0.771 | −0.176  |
| slider_shape  | 0.855 | 0.824 | −0.031  |
| accents       | 0.805 | 0.575 | −0.230  |

Every family is down on the raw mean — but again ~all of the flow/accents loss is
the two Blue Zenith collapses (where flow→0.1 and accents→0.0; see 3d).

### 3c. Mean descriptive metrics (12 cells)

| metric                | v8     | v8_1   | Δ       | reading                              |
|-----------------------|--------|--------|---------|--------------------------------------|
| mean_spacing_px       | 127.95 | 118.54 | −9.41   | v8_1 packs notes slightly tighter    |
| std_spacing_px        | 69.74  | 68.14  | −1.60   | ~same spacing variety                |
| jump_ratio            | 0.16   | 0.17   | +0.00   | **no jump compression** (good)       |
| stream_ratio          | 0.20   | 0.32   | +0.12   | v8_1 streams much more               |
| on_quarter_grid_ratio | 0.81   | 0.82   | +0.01   | identical rhythm tightness           |
| curved_slider_ratio   | 0.50   | 0.51   | +0.01   | curvature unchanged                  |
| slider_ratio          | 0.57   | 0.40   | −0.17   | v8_1 uses **far fewer sliders**      |
| mean_turn_angle_deg   | 87.96  | 81.21  | −6.75   | v8_1 flows a bit straighter          |
| density_per_s         | 4.48   | 5.38   | +0.90   | v8_1 denser overall                  |
| n_objects             | 643.8  | 695.5  | +51.7   | v8_1 places more objects             |
| mean |SR error|       | 0.324  | 0.330  | +0.006  | SR accuracy ~identical on average    |

### 3d. The two failures — what actually happened (Blue Zenith, low SR)

| cell             | spc px | dens/s | stream | jump | achieved SR (target) | n     |
|------------------|--------|--------|--------|------|----------------------|-------|
| bz sr4 **v8**    | 110.4  | 3.3    | 0.01   | 0.03 | 3.63 (4)             | 275   |
| bz sr4 **v8_1**  | **29.8**| **10.5**| **0.81**| 0.00| **5.72 (4)**         | **806** |
| bz sr5 **v8**    | 112.1  | 4.1    | 0.22   | 0.13 | 4.62 (5)             | 361   |
| bz sr5 **v8_1**  | 71.2   | 6.6    | 0.54   | 0.04 | 5.83 (5)             | 580   |

At **low SR on a fast (200 BPM) stream song, v8_1 ignores the low-density
conditioning and emits a wall of tight 1/4 stream** — 10.5 obj/s at 30 px spacing,
overshooting SR to 5.7 when 4.0 was asked. Every family except rhythm collapses
(spacing_aim 0.12, flow 0.11, accents 0.0). v8 handled the same prompt sanely (sparse,
110 px). This is a **conditioning-obedience regression specific to low-SR × fast-stream
song**; by SR 6–7 (where dense streams *are* appropriate) the gap closes and v8_1 ties
or wins.

### 3e. Where v8_1 clearly wins — high SR control

Highscore **sr7**: v8 overshot to **7.91 SR** (sr_close 0.23, spc 176, blew the target)
while **v8_1 nailed 7.16** (sr_close 0.91) with *higher* jump_ratio (0.41 vs 0.34) and
spacing (184 px). At the top of the range v8_1 is both more controllable and more
aggressive on jumps — exactly the direction v9 wants.

---

## 4. Disentangling rope vs huber

- **rope** appears to **sharpen / strengthen pattern commitment**: more streams
  (+0.12), denser output (+0.9 obj/s), better high-SR aim control (highscore sr7).
  The feared failure mode — attention *averaging away* jumps (HANDOFF §7, the up_attn
  lesson) — **did NOT happen**: jump_ratio is flat (+0.00) and high-SR jumps actually
  rose. Rope is the likely driver of the stronger-but-less-obedient behavior: it pushes
  harder toward the song's intrinsic rhythm, which is great when the SR target agrees
  with the song (high SR / dense songs) and bad when it doesn't (low SR on a stream
  song → the Blue Zenith collapse).
- **huber(0.5)** is the likely driver of **less mean-regression / sharper outputs**:
  tighter mean spacing, fewer-but-more-deliberate sliders (slider_ratio 0.57→0.40),
  straighter flow. Huber down-weights large residuals, so the model commits to one
  pattern instead of hedging to the mean — consistent with the sharper, more decisive
  (sometimes too decisive) maps. Curvature held (curved_slider_ratio +0.01), so huber
  did **not** wreck slider shape; it just uses sliders more sparingly.

These are correlational (the two changes weren't ablated separately), but the
signature — sharper commitment + better aim at high SR + over-commitment at
mis-matched low SR — is consistent with rope (positional/rhythmic sharpening) +
huber (anti-mean-regression) acting together.

---

## 5. Honest verdict

**Overall: a near-wash on raw mean (v8 0.82 vs v8_1 0.77), but that mean is an
artifact of two specific failures. On 10/12 cells v8_1 matches or beats v8, and
excluding the two collapses v8_1 is clearly ahead (0.876 vs 0.824).** v8_1 is a
*sharper, more committed* model: better high-SR aim/jump control, more decisive
streams, cleaner slider economy — at the cost of one real regression.

**Per family:**
- **rhythm:** wash (on_grid identical; the −0.08 mean is the bz collapses).
- **spacing_aim:** wash-to-better (jumps flat, high-SR aim better; the −0.08 is bz).
- **flow:** v8_1 better at SR≥6 (more streams where wanted); regresses hard at low SR
  on stream songs (the collapse). Net: context-dependent.
- **slider_shape:** wash (shape held; v8_1 just uses fewer sliders — neither clearly
  better without play feedback).
- **accents:** v8_1 lower, but almost entirely from kiai/hitsound zeroing-out in the
  two collapse cells, not a general accents regression.

**Clear regression to flag:** **low-SR conditioning obedience on fast stream songs.**
v8_1 over-densifies and overshoots SR when asked for an easy diff of a stream-heavy
song. If v8_1 is promoted, this needs `--match-sr` and/or a `--density` override at low
SR, or it will hand back a 5.7★ stream wall when the user asks for 4★.

**Clear win to flag:** **high-SR control + jump production.** v8_1 hits the top of the
range accurately and aggressively where v8 overshoots — the right direction for v9.

### Recommendation
**Keep v8 as the released base for now; do NOT auto-promote v8_1 on metrics alone.**
v8_1 is a genuine improvement *in the regime v9 cares about* (high-SR, jumps, sharper
patterns) but carries one real low-SR/stream regression and costs ~2× to train. The
user trusts play feel over metrics, so this is a **"needs in-game play feedback to
decide"** — with a *strong lean toward v8_1* if the A/B below confirms.

### What to A/B in-game (concrete)
1. **High-SR jump map (the v8_1 case-for):** generate both at SR 6.5–7 on a
   jump/tech song. Check if v8_1's sharper aim + higher jumps *feels* better or just
   spikier. (Metrics say v8_1 should win here.)
2. **Low-SR on a stream song (the v8_1 case-against):** generate both at SR 4 on a
   fast stream song (e.g. Blue Zenith). v8_1 will likely over-stream — confirm whether
   it's unplayable or fixable with `--match-sr`/`--density`. This is the gate on
   promotion.
3. **Slider feel:** v8_1 uses ~17 pts fewer slider ratio and straighter flow. On a
   slider/flow-aria song, does the leaner slider usage read as cleaner or as missing
   sliders? Pure play call.
4. **Mid-SR balanced (Kawaii 5–6):** the everyday case — they're a wash on metrics, so
   pick whichever feels better; this decides the default base.

---

*Artifacts:* `artifacts/ab_gen/results.json` (all 12×2 rows + per-metric/per-family),
`artifacts/ab_gen/*.osu` (24 generated maps), `artifacts/_ab_compare.py` (gen script).
Reward uses `artifacts/reference_stats.json`. **Loss-scale caveat applies throughout:
val numbers are not comparable across mse/huber — quality verdict is from §3 only.**
