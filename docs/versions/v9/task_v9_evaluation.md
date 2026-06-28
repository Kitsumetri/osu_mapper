# v9 — model evaluation: the `aim` lever is density, NOT the jump fix

*STATIC / frozen — v9 task report (2026-06-28).*

The v9 model (`runs/20260627-005105-ranked-v9`, `epoch_80.pt`) = v8 recipe + rope +
huber(β0.5) + the audit's encode fixes + per-song `aim` conditioning, 80 ep on the
encode-fixed gold data (`ranked-v9`). Training was healthy: train loss 0.034 / val 0.0345
on the leakage-free held-out-song split, and **`val_reward` rose 0.66 → 0.77** over the run
(the reward-in-val probe — the first real map-quality trend *during* training, vs diffusion
loss). This report is the **evaluation**: does the per-song `aim` conditioning — the reason
v9 exists — actually fix the jump under-production?

## TL;DR — NO. `aim` is a density/stream lever; the jumps come from elsewhere.

The per-song `aim` conditioning does NOT supply controllable jumps. It is a real, strong
lever — but it controls **density/streams**, and it is **anti-correlated with jumps**. v9's
good in-game jumps come from **rope+huber + the cleaner encode-fixed data + best-of-N
selection**, not the `aim` mechanism. **The jump fix is now RWR / best-of-N distillation,
not conditioning.**

## The experiment (aim sweep)

Same song + SR, sweep `--aim-intensity` low vs high, 3 samples each, measure the spatial
(spacing/jump) and rhythmic (density/stream) axes.

**Blue Zenith (Cut Ver), SR 6, best.pt — 3-sample averages:**

| aim | density_per_s | mean_spacing_px | jump_ratio | stream_ratio |
|----:|--------------:|----------------:|-----------:|-------------:|
| 0.0 | 5.00 | **115.4** | **0.165** | 0.324 |
| 1.0 | 6.00 | **95.3** | **0.108** | 0.434 |

`aim↑` → **more density + more streams, but TIGHTER spacing + FEWER jumps.** The ranges
separate cleanly (real signal). `epoch_80.pt` showed the same direction but noisier (one
outlier flipped its average) → the aim response is weak relative to per-sample variance and
not robust across checkpoints. On Kawaii (a non-jump song) aim barely moved anything.

## Why (the durable lesson)

`data.audio.aim_intensity` = the mean **onset strength** — a *rhythmic-density* signal.
Density trades off against spacing (more notes/sec in fixed time → tighter → streams, not
jumps). And **jump-vs-stream for the same song is a mapper's STYLE choice, not encoded in
the audio** — so no audio-onset feature can supply "jump intent." This generalises the v8
finding (the passive spacing channel regressed to the SR-average): **spacing under-production
is not an information problem the model can be *told* about — it is an under-dispersion /
style problem.**

## What this changes

- **`aim` ships as a useful density/style knob, not a jump fix.** `--aim-intensity` low
  (~0.1) = sparser + bigger spacing (jumpier output); high (~0.9) = denser + streamier.
  Exposed in `main.py infer` (threaded through both the single-gen and best-of-N paths).
- **The jump lever is now RWR / best-of-N distillation** (roadmap #1): the reward already
  prefers ranked-shaped spacing; fine-tune v9 toward best-of-N's high-reward self-generations
  to commit to the spacing/jump tail. Conditioning is OFF the jump path.
- **Checkpoint:** `epoch_80.pt` (peak `val_reward`) made the in-game maps the USER liked;
  `best.pt` (val-loss) is the auto-default and slightly different feel. They are close.

## In-game (USER — the real judge)

best-of-8 winners on Blue Zenith scored reward **0.967 (SR 6) / 0.938 (SR 7)** — near-gold,
SR on target, best-of-8 lifting ~+0.13 over the candidate mean. **USER played them: "jumps
good, streams good"** → v9 is the best model so far. So while the `aim` *mechanism* didn't
pan out, the v9 *model* is a clear win.
