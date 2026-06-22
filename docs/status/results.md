# Results — training-run history & generated-map quality

**Purpose:** the run-by-run training history (loss, eval tables, play feedback), appended every train.
| **DYNAMIC** (append a section per run/version). Metrics via `src/metrics.py`.

**Current release: v8** (`runs/20260619-235218-ranked-v8-b160/ckpt/best.pt`, 21-ch, base-160).
Per-version *design* lives in [docs/versions/](../versions/README.md); this file is the run log.

## v8 (P4-B) — base-160 full train (TRAINED 2026-06-20)

`runs/20260619-235218-ranked-v8-b160/ckpt/best.pt` (val **0.0412**, 21-ch, **base 160 / 101.7M**),
v8 recipe `--objective v --zero-snr --compile --spatial-loss-weight 3`, 60 ep on `ranked-v8` (38k),
~7 h (~400 s/ep). Design: [versions/v8.md](../versions/v8.md).

**WIN — base-160 is unblocked.** Trained clean through the **e12–21 divergence zone** that killed
v2/v3 (ε-pred): val 0.078→0.041 monotonic, gnorm stable/decreasing. v-pred + zero-SNR is the unblock;
`--compile` works at base 160. **Bigger models are now trainable** — a durable capability win
independent of the spacing outcome.

**MISS — the spacing channel does NOT auto-adapt per-song.** On the Happppy jump song (real: spacing
173.6 / jump 0.418), `eval_spacing_channel` shows the channel predicts only ~120–127 px (ratio
1.03–1.04 over the cursor) — it regressed to the **SR-average**, same as the cursor. **Root cause:** the
channel shares the cursor's audio+SR conditioning, so it has no extra info about whether a song is
jump-heavy — both → `E[spacing | audio, SR]`. (The curve cue worked because decode *forces* a bow;
respace faithfully reproduces the channel's compressed magnitude.)

**PARTIAL WIN — `--spacing-scale` is a usable manual dial (no retrain).** Amplifying past 1.0 uniformly
scales spacing — a per-song jump knob v7.5 never had (Happppy @ sr6.5):
| spacing-scale | mean_spacing | jump_ratio |
|---|---|---|
| 0 (raw) | 117.7 | 0.116 |
| 1.0 (faithful) | 129.9 | 0.144 |
| 2.0 | 146.5 | 0.197 |
| 2.5 | 187.9 | 0.297 |
(real 173.6 / 0.418.) No regression — curves 0.369 (≈ real 0.38), SV intact. Packaged `[AI-v8 s2.0]`
(146/0.20). Caveats: the dial is **global** (over-spaces calm songs) and uniform scaling can't make
real's bimodal stream+jump structure (std 130). `--match-sr` backfires on jump songs; condition a high
`--sr` directly instead.

**Verdict:** net **+** over v7.5 (stable bigger base + a working spacing dial + no regression), but it
did **not** break the jump ceiling *automatically*. The real per-song fix is **conditioning on an
audio-inferred aim-intensity** — see [versions/v9.md](../versions/v9.md).

**Play feedback (2026-06-20): `--spacing-scale` HURTS in-game → use `--spacing-scale 0`.** The respace
lifts the spacing *metrics* (jump_ratio) but the relocated objects read/play worse (flow + readability
suffer when positions are walked/reflected), so in-game it's a net negative. **The respace is
effectively shelved; generate raw (scale 0).** v8's real value is the **base-160 stability + no
regression** (SR conditioning already gives meaningful per-song/per-SR spacing variation: Kano sr5→sr7
raw spacing 123→169, jump 0.09→0.30). The automatic per-song jump fix remains v9.

## v8 (P4-B) — draft trains: pipeline debug + base-160 probe (2026-06-19, drafts only)

Two 2-epoch drafts on `ranked-v8` (**38,036 maps** — the +13k library reprocess), v8 recipe
`--objective v --zero-snr --spatial-loss-weight 3` (no `--compile`). **Drafts validate the pipeline +
mechanism *direction*, NOT quality.**

**Pipeline ✅** both trained clean (no crash/divergence), 21-ch, loss-weighting active, SV + curves
intact.

**Mechanism ✅ (the v8 bet, via `eval_spacing_channel`):** the spacing *channel* predicts a larger
magnitude than the collapsed cursor *positions*; ratio (channel ÷ cursor mean spacing) grows with SR:
| SR | base-128 ratio | base-160 ratio | base-160 channel_sp px |
|---|---|---|---|
| 3 | 1.04 | 1.13 | 116.6 |
| 4 | 1.11 | 1.16 | 129.7 |
| 5 | 1.14 | 1.16 | 152.0 |
| 6 | 1.12 | 1.14 | 154.1 |
The channel mean-regresses *above* the positions even undertrained → respace has a real gap to exploit.

**base-160 — promising.**
| | base-128 draft | base-160 draft |
|---|---|---|
| params | 66.1M | 101.7M |
| val_loss e0 / e1 | 0.0768 / 0.0655 | 0.0724 / **0.0619** |
| grad-norm | (logged later) | **stable/decreasing**, 1.0 (warmup) → 0.1–0.7 |
| s/epoch (no compile) | 571 / 514 | 1444 / 1065 (steady ~1065) |
| memory | fits | fits (batch 16, ~7 GB) |
base-160 fits, has a lower loss, a stable grad-norm, and a stronger spacing channel — the first
base-160 that didn't fall over. The prior divergences (v2 @e21, v3 @e12) were ε-pred. **CAVEAT: a
2-epoch draft CANNOT confirm long-run stability** (the documented divergences hit e12–21) — confirmed
by the full train above.

## v8 (P4-B) — decode reconstruction de-risk (2026-06-19, no train yet)

Validated the **decode half** — `postprocess.respace_by_magnitude` — on real v7.5 output before
spending a train. Keeps each step's model *direction* and sets its *length* from a target magnitude,
re-anchoring at new combos/spinners and reflecting off the walls.

| map | spacing px | jump_ratio | turn° | in-bounds (head+body) |
|---|---|---|---|---|
| sr5 v7.5 input | 121.6 | 0.103 | 90.2 | ✓ |
| sr5 respace α=0 | 121.6 | 0.103 | 90.2 | ✓ (perfect no-op) |
| sr5 respace α=1 (+clamp) | **151.0** | **0.288** | 96.3 | ✓ |
| sr6 v7.5 input | 137.9 | 0.194 | 106.0 | ✓ |
| sr6 respace α=1 (+clamp) | **160.4** | **0.374** | 105.3 | ✓ |

**Validated:** spacing expands controllably via the `alpha` blend (0 = unchanged), jump_ratio more than
doubles into real-jump territory (ref 0.39), turn-angle preserved, everything in-bounds. **Conclusion:**
the reconstruction is sound; the open question is whether the model learns honest per-song magnitudes —
which the train tested (it did not, see above).

## v7.5 — attention dropped + red points (TRAINED 2026-06-17, best before v8)

`runs/20260617-223917-ranked-v75/ckpt/best.pt` (val 0.0473 v-scale). 20-ch `ranked-v75` gold
(+`corner` cue), `--objective v --zero-snr` **(no rope/up-attn)** `--compile`, 60 ep, clean. **~203
s/epoch (~3.4× faster than v7-full's 685 s).** 66.1 M params. Design: [versions/v7.md](../versions/v7.md).

**Phase-1 A/B — the attention ablation worked:**
| measure | real | v7-vpred | v7-full | **v7.5** |
|---|---|---|---|---|
| jump_ratio | 0.207 | 0.145 | 0.048 | **0.131** |
| mean spacing px | 133.6 | 129.6 | 102.8 | **123.9** |
| std spacing px | 77.4 | 67.5 | 57.2 | **67.3** |
| turn_deg | 88.6 | — | 76.8 | **89.8** |
| visible-curve % | 37.4 | 12.1 | 29.3 | **28.4** |
| SV changes/map | 10 | 0 | 5 | **4** |

**Confirmed: up-path attention was killing jumps** (self-attention averages → compresses spatial
variance). Dropping it recovered jump_ratio 0.048→0.131, mean-spacing and turn-angle back to ≈ real —
**while keeping** SV (90% non-trivial, ~4 stable sections) and curves (28% visible). **Red corners
generate but under-produced (2% vs real ~13%)** — decode knob `CORNER_DECODE_THRESHOLD` tunable.
Packaged `[AI-v75]` (SR4.9, 451 obj).

**Play feedback (2026-06-18) — best model so far:** rhythm 7/10; hitsounds 4/10 (unstable); sliders
6/10 (red-corner + short/long SV sliders work — new wins); patterns 6/10 (jumps sometimes good/nasty,
streams clearly better). Bug: the last circle sits in the dead outro ("Happppy") → trailing-trim
decode fix.

## v7 full — SV + curve channels + attention (TRAINED 2026-06-17)

`runs/20260617-083444-ranked-v7/ckpt/best.pt` (epoch ~56, **val 0.0457** v-scale). 19-ch `ranked-v7`
gold, base 128, `--objective v --zero-snr --rope --up-attn --grad-checkpoint`, 60 ep, clean (~690
s/epoch = ~2× v6 from up-attention), 71.6 M params.

| measure | real | v6 | v7-vpred | **v7-full** |
|---|---|---|---|---|
| SV non-trivial % | 83.9 | 0 | 0 | **100** ✅ |
| SV changes/map | 10 | 0 | 0 | **5** ✅ (target ~6-8) |
| visible-curve % (sagitta≥10) | 37.4 | 13.4 | 12.1 | **29.3** ✅ (target 38-45) |
| mean spacing px | 133.6 | 119.7 | 129.6 | **102.8** ✗ |
| jump_ratio | 0.207 | 0.119 | 0.145 | **0.048** ✗✗ |
| std spacing px | 77.4 | 66.2 | 67.5 | **57.2** ✗ |
| turn_deg | 88.6 | 85.1 | — | **76.8** ✗ |

**Mixed.** ✅ SV channel works + curvature jumped 13→29%. ✗ spacing/jumps REGRESSED hard. Prime
suspect: the attention add (rope/up-attn) — bundling P2+P3+P4 lost attribution. Packaged `[AI-v7full]`.
Led to v7.5 (ablate attention).

## v7 Phase 2 — v-prediction + zero-terminal-SNR (TRAINED 2026-06-17)

`runs/20260617-001225-v7-vpred/ckpt/best.pt` (epoch 59, **val 0.0507** — v-loss scale, ~100× eps, NOT
comparable to v6's 0.003). Same gold-v6 data + adaLN, base 128, 60 ep, `--objective v --zero-snr`.
Clean (~329 s/epoch).

| measure | real | v6 | v7-vpred |
|---|---|---|---|
| mean spacing px | 133.5 | 119.7 | **129.6** |
| jump_ratio | 0.205 | 0.119 | **0.145** |
| std spacing px | 77.3 | 66.2 | 67.5 |
| stream_ratio | 0.149 | 0.101 | 0.088 |
| visible-curve % (sagitta≥10) | 38.1 | 13.4 | 12.1 |
| median sagitta px | 4.8 | 0.0 | 0.0 |

**Partial win:** v-pred closed ~70% of the mean-spacing gap and ~28% of the jump gap, but **did NOT
improve spacing variety, streams, or slider curvature** → P4 flow/curvature channels justified.
Packaged `[AI-v7]`.

## v6 — adaLN-zero + gold data (TRAINED 2026-06-16)

`runs/20260616-013932-ranked-v6/ckpt/best.pt` (epoch 59, **val 0.00314**). 17-ch, **adaLN-zero**
(`--adaln` default on) on **gold data** `ranked-v6` (**25,073 maps**), base 128 / crop 4096 /
attn_levels 3 / batch 16 / 60 epochs, flip aug. Clean (~330 s/epoch); 66.1 M params. Design:
[versions/v6.md](../versions/v6.md).

**Eval — SR sweep (Headphone Actor):**
```
 target  got SR   dens  strm  jump  grid   bez  kiai    hs  in-range
    2.0    2.82   2.45  0.00  0.02  0.75  0.09  0.21  0.24  14/19
    3.0    3.29   2.97  0.01  0.04  0.79  0.13  0.11  0.28  16/19
    4.0    3.59   3.02  0.01  0.03  0.79  0.08  0.00  0.27  17/19
    5.0    5.24   4.18  0.08  0.07  0.78  0.18  0.10  0.33  16/19
    6.0    6.52   5.19  0.18  0.12  0.75  0.16  0.14  0.44  17/19
```
SR monotonic ✓. bez 0.75–0.79 (curved sliders solid). **One flag: kiai 0.00 at SR4.** Packaged
`[AI-v6]`. A/B vs v5: v6 SR calibration tighter to target low/mid; static metrics roughly a wash.

## v5 — slider-shape + reverse sliders (DONE 2026-06-15)

`runs/20260614-224107-ranked-v5/ckpt/best.pt` (epoch 55, val 0.00330). 17-channel (K=3 slider-anchor
`dx/dy` + `slides`), ranked-v5 data, base 128 / crop 4096 / attn_levels 3 / batch 16 / 60 epochs, flip
aug. Clean (~295 s/epoch). Design: [versions/v5.md](../versions/v5.md).

**The two target complaints are fixed** (Headphone Actor @ SR 5.36, 533 obj):
- **Curved sliders: 100% bezier** (170/170, 3 control points) — was ~20–40%.
- **Reverse sliders: 27%** (slides {1:124, 2:40, 3:5, 4:1}) — was **0** (impossible in 10-ch). *Note:
  above real ~8% — may over-produce.*
- AR/OD now match the conditioned SR (C-2 fix). Caveats: every slider is a 3-point curve; no spinners;
  rhythm regression from v4b still open at this point.

**Play feedback (2026-06-15):** kiai 9/10, reverse sliders good, streams way better. Fixes (no
retrain): realistic AR (`7.75+0.25·sr`), straight-vs-curved mix (~77/23), package_map difficulty. Also
fixed: `package_map` was overriding generated AR/OD with the original's. Next dataset: `preprocess
--gold`.

## v4b — ranked train (2026-06-14)

`runs/20260614-151630-ranked-full/ckpt/last.pt`, **epoch 48, val_loss 0.00486** (below the v4
release's 0.0077). Ranked-only data (~23.6k), more context (`--crop 4096 --attn-levels 3`), h/v flip
aug, train/val split. Stopped near convergence. Design: [versions/v4.md](../versions/v4.md).

**Eval:** SR monotonic; **17–19/19 in-range** (beats v4's 16–17); hitsounds ≈ real. SR offset persists
→ `--match-sr`. Packaged `[AI-v4b]` (5.92★, 520 obj).

**Play feedback (v4b vs v4):** ✅ Kiai much better; ✅ no dead trailing note; ✅ jumps/patterns/streams
much better (validates ranked + context + flip aug). ⚠️ Rhythm REGRESSED (0.5–2 s pauses; off-¼ →
1/6/1/8). ➖ No spinners; curve sliders still low.

## v4 — full curated library (previous release)

`runs/20260614-110223-std-v4-full` — 31,270 curated maps (≤12★), base 128, **epoch 15, loss 0.0077**
(killed by an OS sleep at e16, undertrained but strong). SR monotonic, 16–17/19 in-range. Superseded by
v4b. The decode/post-process wins shipped on v4 are listed in [versions/v4.md](../versions/v4.md).

## Earlier versions (v1–v3) — summary

| ver | data | model | loss | takeaway |
|-----|------|-------|------|----------|
| v1 | 601 | base 96, no attn, DDPM | 0.011 | pipeline works; too dense, straight sliders, loose rhythm (0.70 on-grid) |
| v2 | 3004 | base 160 (97M), QK-norm attn, bf16 | 0.0075 | much closer to real; low jumps, few curves, no SV |
| v3 draft | 1504 | base 128 + difficulty cond + CFG | 0.0097 | **conditioning steers difficulty** ✓ |
| v3 heavy | 6001 | base 128 + cond | 0.0056 | SR near-calibrated 3–5★; curved sliders + kiai + hitsounds all generate |

Durable lessons from these (base-160+bf16 diverges, DDIM not strided DDPM, curved-slider encoder fix)
live in [knowledge/lessons-learned.md](../knowledge/lessons-learned.md). Per-version detail:
[versions/v1-v3.md](../versions/v1-v3.md).
