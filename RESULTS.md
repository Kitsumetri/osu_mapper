# Results

Training-run history + generated-map quality (metrics via `src/metrics.py`).
**Current release: v5** (`runs/20260614-224107-ranked-v5/ckpt/best.pt`, 17-ch).
**v6 + v7-Phase2 trained; v7 work in progress (RESEARCH §10.7).**

## v8 (P4-B) — base-160 full train (TRAINED 2026-06-20)

`runs/20260619-235218-ranked-v8-b160/ckpt/best.pt` (val **0.0412**, 21-ch, **base 160 / 101.7M**),
v8 recipe `--objective v --zero-snr --compile --spatial-loss-weight 3`, 60 ep on `ranked-v8` (38k),
~7 h (~400 s/ep).

**WIN — base-160 is unblocked.** Trained clean through the **e12–21 divergence zone** that killed
v2/v3 (ε-pred): val 0.078→0.041 monotonic, gnorm stable/decreasing. v-pred + zero-SNR is the unblock
(§7); `--compile` works at base 160. **Bigger models are now trainable** — a durable capability win
independent of the spacing outcome.

**MISS — the spacing channel does NOT auto-adapt per-song.** On the Happppy jump song (real: spacing
173.6 / jump 0.418), `eval_spacing_channel` shows the channel predicts only ~120–127 px (ratio
1.03–1.04 over the cursor) — it regressed to the **SR-average**, same as the cursor. The v8 thesis
("a magnitude scalar mean-regresses to the correct *larger* magnitude") only half held: it regresses
to the SR-average, **not** the per-song extreme. **Root cause:** the channel shares the cursor's
audio+SR conditioning, so it has no extra info about whether a song is jump-heavy — both →
`E[spacing | audio, SR]`. (The curve cue worked because decode *forces* a bow; respace faithfully
reproduces the channel's compressed magnitude, so it can't add what the channel didn't learn.)

**PARTIAL WIN — `--spacing-scale` is a usable manual dial (no retrain).** Amplifying past 1.0
uniformly scales spacing — a per-song jump knob v7.5 never had (Happppy @ sr6.5):
| spacing-scale | mean_spacing | jump_ratio |
|---|---|---|
| 0 (raw) | 117.7 | 0.116 |
| 1.0 (faithful) | 129.9 | 0.144 |
| 2.0 | 146.5 | 0.197 |
| 2.5 | 187.9 | 0.297 |
(real 173.6 / 0.418.) No regression — curves 0.369 (≈ real 0.38), SV intact. Packaged `[AI-v8 s2.0]`
(146/0.20). Caveats: the dial is **global** (over-spaces calm songs) and uniform scaling can't make
real's bimodal stream+jump structure (std 130). `--match-sr` backfires on jump songs (the model
under-produces SR → it lands on a sparse low-SR map); condition a high `--sr` directly instead.

**Verdict:** net **+** over v7.5 (stable bigger base + a working spacing dial + no regression), but
it did **not** break the jump ceiling *automatically*. The real per-song fix is **conditioning on an
audio-inferred aim-intensity** (the §10.7-P5 density idea, on the spacing axis), not a passive
channel — see RESEARCH §10.11. Play-test `[AI-v8 s2.0]` to decide vs v7.5.

**Play feedback (2026-06-20): `--spacing-scale` HURTS in-game → use `--spacing-scale 0`.** The
respace lifts the spacing *metrics* (jump_ratio) but the relocated objects read/play worse (flow +
readability suffer when positions are walked/reflected), so in-game it's a net negative. **The
respace is effectively shelved; generate raw (scale 0).** v8's real value is the **base-160
stability + no regression** (and SR conditioning already gives meaningful per-song/per-SR spacing
variation: e.g. Kano sr5→sr7 raw spacing 123→169, jump 0.09→0.30). The automatic per-song jump fix
remains v9 (audio-inferred aim-intensity conditioning), not a decode lever.

## v8 (P4-B) — draft trains: pipeline debug + base-160 probe (2026-06-19, drafts only)

Two 2-epoch drafts on `ranked-v8` (**38,036 maps** — the +13k library reprocess), v8 recipe
`--objective v --zero-snr --spatial-loss-weight 3` (no `--compile`). **Drafts validate the
pipeline + mechanism *direction*, NOT quality (2 epochs = heavily undertrained).**

**Pipeline ✅** both trained clean (no crash/divergence), 21-ch, loss-weighting active, SV +
curves intact (no regression from adding the spacing channel).

**Mechanism ✅ (the v8 bet, via `eval_spacing_channel`):** the spacing *channel* predicts a larger
magnitude than the collapsed cursor *positions*; ratio (channel ÷ cursor mean spacing) grows with SR:
| SR | base-128 ratio | base-160 ratio | base-160 channel_sp px |
|---|---|---|---|
| 3 | 1.04 | 1.13 | 116.6 |
| 4 | 1.11 | 1.16 | 129.7 |
| 5 | 1.14 | 1.16 | 152.0 |
| 6 | 1.12 | 1.14 | 154.1 |
The channel mean-regresses *above* the positions even undertrained → respace has a real gap to
exploit (and base-160's extra capacity widens it). Net respace lift on a full map is still modest
at 2 ep (small gap + new-combo re-anchoring); judge it on the full train.

**base-160 — promising (the user's hypothesis: v-pred stability unblocks the bigger base).**
| | base-128 draft | base-160 draft |
|---|---|---|
| params | 66.1M | 101.7M |
| val_loss e0 / e1 | 0.0768 / 0.0655 | 0.0724 / **0.0619** |
| grad-norm | (logged later) | **stable/decreasing**, 1.0 (warmup) → 0.1–0.7 |
| s/epoch (no compile) | 571 / 514 | 1444 / 1065 (steady ~1065) |
| memory | fits | fits (batch 16, ~7 GB) |
base-160 fits, has a **lower loss**, a **stable grad-norm** (no divergence signature), and a
**stronger spacing channel** — the first base-160 that didn't fall over. The prior divergences
(v2 @e21, v3 @e12) were ε-pred; **v-pred + zero-SNR appears to be the unblock** (§7). **CAVEAT: the
documented divergences hit epoch 12–21 — a 2-epoch draft CANNOT confirm long-run stability.** Next
(USER): a full base-160 train, `--compile` (~6–8 h), watching gnorm/val through e15–25; if gnorm
climbs, fall back to per-channel standardisation (§11 5.2) or lr 1.0e-4.

## v8 (P4-B) — decode reconstruction de-risk (2026-06-19, no train yet)

Before spending a train on the spacing-magnitude channel (RESEARCH §10.11), validated the
**decode half** — `postprocess.respace_by_magnitude` — on real v7.5 output (`aiv75_sr5/sr6.osu`).
It keeps each step's model *direction* (turn angles are already ≈ real) and sets its *length* from a
target magnitude (here synthesised as `scale × v7.5's own spacing` toward the Happppy 167 px ref),
re-anchoring at new combos/spinners and reflecting off the walls.

| map | spacing px | jump_ratio | turn° | in-bounds (head+body) |
|---|---|---|---|---|
| sr5 v7.5 input | 121.6 | 0.103 | 90.2 | ✓ |
| sr5 respace α=0 | 121.6 | 0.103 | 90.2 | ✓ (perfect no-op) |
| sr5 respace α=1 (+clamp) | **151.0** | **0.288** | 96.3 | ✓ |
| sr6 v7.5 input | 137.9 | 0.194 | 106.0 | ✓ |
| sr6 respace α=1 (+clamp) | **160.4** | **0.374** | 105.3 | ✓ |

**Validated:** spacing expands controllably via the `alpha` blend (0 = unchanged), jump_ratio more
than doubles into real-jump territory (ref 0.39), turn-angle is preserved, and after the existing
`clamp_slider_endpoints` everything is in-bounds (heads via wall-reflection; slider bodies via the
clamp). α=1 lands ~151/160 vs the 167 target — the new-combo re-anchoring trades a little expansion
for bounded drift (raise the magnitude scale to compensate). **Conclusion:** the reconstruction is
sound; the only open question is whether the model learns honest per-song magnitudes — which the
train tests. Greenlights the v8 channel build.

## v7.5 — attention dropped + red points (TRAINED 2026-06-17, best so far)

`runs/20260617-223917-ranked-v75/ckpt/best.pt` (val 0.0473 v-scale). 20-ch `ranked-v75`
gold (+`corner` cue), `--objective v --zero-snr` **(no rope/up-attn)** `--compile`, 60 ep,
clean. **~203 s/epoch (~3.4× faster than v7-full's 685 s** — dropping up-attn + `--compile`
working now). 66.1 M params.

**Phase-1 A/B — the attention ablation worked:**
| measure | real | v7-vpred | v7-full | **v7.5** |
|---|---|---|---|---|
| jump_ratio | 0.207 | 0.145 | 0.048 | **0.131** |
| mean spacing px | 133.6 | 129.6 | 102.8 | **123.9** |
| std spacing px | 77.4 | 67.5 | 57.2 | **67.3** |
| turn_deg | 88.6 | — | 76.8 | **89.8** |
| visible-curve % | 37.4 | 12.1 | 29.3 | **28.4** |
| SV changes/map | 10 | 0 | 5 | **4** |

**Confirmed: up-path attention was killing jumps** (self-attention averages → compresses
spatial variance). Dropping it recovered jump_ratio 0.048→0.131 (~v-pred level), mean-spacing
and turn-angle back to ≈ real — **while keeping** the SV channel (90% non-trivial, ~4 stable
sections) and curves (28% visible). So v7.5 = v-pred's jumps + v7's SV + curves = best combo yet.
**Red corners generate but under-produced (2% vs real ~13%)** — decode knob `CORNER_DECODE_THRESHOLD`
(tunable, no retrain), to calibrate after play-test. Packaged `[AI-v75]` (SR4.9, 451 obj).

**Play feedback (in-game, 2026-06-18) — best model so far:** rhythm 7/10 (still some gaps);
hitsounds 4/10 (unstable song-to-song); sliders 6/10 (still many straight lines, but red-corner
sliders work + short/long SV sliders appear — both new wins); patterns 6/10 (jumps sometimes
good/sometimes nasty, streams clearly better — the SV-snap fix helped a lot). Bug: on "Happppy
song" the **last circle sits in the dead outro** (audio 318.85s, real content ends ~316.9s, gen
placed one at 318.82s) → autobot fails there → needs a stronger trailing trim (decode). Action
plan in RESEARCH §10.7 P5 / HANDOFF.

## v7 full — SV + curve channels + attention (TRAINED 2026-06-17)

`runs/20260617-083444-ranked-v7/ckpt/best.pt` (epoch ~56, **val 0.0457** v-scale). 19-ch
`ranked-v7` gold data, base 128, `--objective v --zero-snr --rope --up-attn --grad-checkpoint`,
60 ep, clean (no divergence, ~690 s/epoch = ~2× v6 from up-attention), 71.6 M params.

**Phase-1 A/B** (`analyze_phase1.py`, real vs v6 vs v7-vpred vs v7-full; 10-map sweeps):
| measure | real | v6 | v7-vpred | **v7-full** |
|---|---|---|---|---|
| SV non-trivial % | 83.9 | 0 | 0 | **100** ✅ |
| SV changes/map | 10 | 0 | 0 | **5** ✅ (target ~6-8) |
| visible-curve % (sagitta≥10) | 37.4 | 13.4 | 12.1 | **29.3** ✅ (target 38-45) |
| mean spacing px | 133.6 | 119.7 | 129.6 | **102.8** ✗ |
| jump_ratio | 0.207 | 0.119 | 0.145 | **0.048** ✗✗ |
| std spacing px | 77.4 | 66.2 | 67.5 | **57.2** ✗ |
| turn_deg | 88.6 | 85.1 | — | **76.8** ✗ |

**Mixed result.** ✅ The **SV channel works** (0→5 sections/map, 100% non-trivial, sensible
0.35–1.4× range — the stability-first decode landed in target) and **curvature jumped** 13→29%
(toward 38-45; `CURVE_DECODE_THRESHOLD_PX` can push higher, decode-only). ✗ But **spacing/jumps
REGRESSED hard** vs v7-vpred (jump 0.145→0.048, mean-spacing 129.6→102.8, turn 88→77) — the
bundled SV+curve+attention cost the pattern dispersion v-pred had gained. Spacing is object
*positions* (not affected by the SV/curve decode), so this is model-side. Prime suspect: the
**attention add (rope/up-attn)** — it was demoted by the flow-angle finding (attention ≠ the
bottleneck) and turn-angle dropping points to over-clustering. Bundling P2+P3+P4 lost attribution,
as flagged. Packaged `[AI-v7full]` (SR4.8, 500 obj) for play-test. **Next: play-test feel, then
likely ablate attention** (retrain v7 with `--objective v --zero-snr` only, no rope/up-attn) to
confirm and recover jumps while keeping SV/curve.

## v7 Phase 2 — v-prediction + zero-terminal-SNR (TRAINED 2026-06-17)

`runs/20260617-001225-v7-vpred/ckpt/best.pt` (epoch 59, **val 0.0507** — v-loss scale,
~100x eps, NOT comparable to v6's 0.003). Same gold-v6 data + adaLN, base 128, 60 ep,
`--objective v --zero-snr`. Clean convergence, no divergence (~329 s/epoch) — confirms the
objective swap is stable and unblocks future base-160 scaling.

**Phase-1 metric A/B** (`analyze_phase1.py`, real 397 vs v6 vs v7; 10-map sweeps):
| measure | real | v6 | v7-vpred |
|---|---|---|---|
| mean spacing px | 133.5 | 119.7 | **129.6** |
| jump_ratio | 0.205 | 0.119 | **0.145** |
| std spacing px | 77.3 | 66.2 | 67.5 |
| stream_ratio | 0.149 | 0.101 | 0.088 |
| visible-curve % (sagitta≥10) | 38.1 | 13.4 | 12.1 |
| median sagitta px | 4.8 | 0.0 | 0.0 |

**Partial win:** v-pred closed ~70% of the mean-spacing gap and ~28% of the jump gap
(bigger, more confident movements = the under-dispersion mechanism), but **did NOT improve
spacing variety (std), streams, or slider curvature** (median slider still dead-straight).
`guidance_rescale 0.7` didn't help → keep 0. Conclusion: objective fixes average *magnitude*,
not *variety*/curvature → **P4 flow (B) + curvature-cue (C) channels now justified** (were
conditional on P2). Packaged `[AI-v7]` (SR 5.18, 450 obj, 2-min set) for play test.

## v6 — adaLN-zero + gold data (TRAINED 2026-06-16, awaiting play test)

`runs/20260616-013932-ranked-v6/ckpt/best.pt` (epoch 59, **val 0.00314**). 17-ch,
**adaLN-zero conditioning** (DiT per-block scale/shift/gate, `--adaln` default on) on
**gold data** `data/processed/ranked-v6` (**25,073 maps**: ranked + 100% kiai + single-BPM
+ hitsounds≥10% + 1<SR<10), base 128 / crop 4096 / attn_levels 3 / batch 16 / 60 epochs,
flip aug. Clean convergence, no divergence (~330 s/epoch); 66.1 M params.

**Eval — SR sweep (Headphone Actor, vs `reference_stats.json`):**
```
 target  got SR   dens  strm  jump  grid   bez  kiai    hs  in-range
    2.0    2.82   2.45  0.00  0.02  0.75  0.09  0.21  0.24  14/19
    3.0    3.29   2.97  0.01  0.04  0.79  0.13  0.11  0.28  16/19
    4.0    3.59   3.02  0.01  0.03  0.79  0.08  0.00  0.27  17/19
    5.0    5.24   4.18  0.08  0.07  0.78  0.18  0.10  0.33  16/19
    6.0    6.52   5.19  0.18  0.12  0.75  0.16  0.14  0.44  17/19
```
SR monotonic ✓. bez 0.75–0.79 (curved sliders solid). hitsounds 0.24→0.44 scale with SR
(real ~0.33). **One flag: kiai 0.00 at SR4** (others 0.10–0.21) — watch in-game given
gold is 100% kiai. Packaged `[AI-v6]` (target SR5, `--match-sr`→4.90, `--timing-from`
Collab Expert, 926 obj, CS4/AR9).

**A/B vs v5** (same audio/eval, v5 = `20260614-224107-ranked-v5/ckpt/best.pt`):
```
            target:  2.0   3.0   4.0   5.0   6.0
got SR   v6 (adaLN) 2.82  3.29  3.59  5.24  6.52   in-range 14/16/17/16/17
         v5         3.01  3.46  4.29  5.79  6.87   in-range 13/16/13/16/18
```
Sense-check: v6 SR calibration is **tighter to target** in the low/mid range (v5 over-shoots
everywhere; v6 close except an SR4 dip), and **more consistent in-range** mid-curve (SR4
17 vs 13). v5 has marginally more uniform kiai across SRs (0.15–0.23 vs v6's 0.00–0.21 with
the SR4 outlier) and slightly higher hitsounds. **Static metrics are roughly a wash** — the
adaLN difficulty-control + gold-data kiai/hitsound consistency wins are the kind that show
up in play, not corpus stats. Promotion to release pending in-game feedback.

## v5 — slider-shape + reverse sliders (DONE 2026-06-15)

`runs/20260614-224107-ranked-v5/ckpt/best.pt` (epoch 55, val 0.00330). 17-channel
representation (K=3 dedicated slider-anchor `dx/dy` + `slides` channel) on the same
ranked-v5 data, base 128 / crop 4096 / attn_levels 3 / batch 16 / 60 epochs, flip
aug. Clean convergence, no divergence (~295 s/epoch). (val isn't comparable to the
10-ch loss — the extra structured channels lower the average MSE.)

**The two target complaints are fixed** (Headphone Actor @ SR 5.36, 533 obj):
- **Curved sliders: 100% bezier** (170/170, 3 control points) — was ~20–40% before,
  rest straight lines. The dedicated anchor channels work.
- **Reverse sliders: 46/170 = 27%** (slides {1:124, 2:40, 3:5, 4:1}) — was **0**
  (impossible in the 10-ch representation). *Note: 27% is higher than real maps'
  ~8% — the model may over-produce reverses; watch in-game.*
- AR/OD now written to match the conditioned SR (AR7.5/OD7.0 at SR5 — C-2 fix).
- Caveats: **every** slider is a 3-point curve (model always fills K=3 anchors → no
  straight sliders; could look over-wavy — needs in-game eyeball); no spinners;
  rhythm regression from v4b (off-¼ → 1/6·1/8) still open (§10.4). Packaged `[AI-v5]`.

**Also fixed during this eval:** `package_map` was overriding the generated map's
AR/OD/HP/CS with the *original* beatmap's — so all prior `[AI-*]` in-game tests ran
at the original's (often harder) AR. Now it keeps the generated settings.

**Play feedback (in-game, 2026-06-15):** kiai 9/10, reverse sliders good, streams way
better. Fixes applied (no retrain): realistic AR (`7.75+0.25·sr` → AR9 median),
straight-vs-curved slider mix (~77/23), package_map difficulty. Still open → §10.5:
rhythm (off-¼ + gaps, task #8), hitsounds below ranked level, slider-velocity support
(task #9), pattern quality. Next dataset: `preprocess --gold` (task #12).

## v4b — ranked train (current; v4 branch merged to main) — 2026-06-14

`runs/20260614-151630-ranked-full/ckpt/last.pt`, **epoch 48, val_loss 0.00486** (well
below the v4 release's 0.0077). Ranked-only data (`osu!.db` filter → ~23.6k ranked/
approved/loved maps), **more context** (`--crop 4096 --attn-levels 3`), **h/v flip
augmentation**, train/val split. Stopped by the user near convergence (~0.0050).

**Eval (SR sweep, Headphone Actor):** SR monotonic ✓; **17–19/19 metrics in-range**
(beats the v4 release's 16–17); hitsounds 0.23–0.31 ≈ real 0.33. SR offset persists
(target 6→5.69, 5→4.43) → use `--match-sr`. Packaged `[AI-v4b]` (5.92★, 520 obj).

**Play feedback (v4b vs v4 — in-game):**
- ✅ **Kiai much better** — 2 sections start at near-perfect timing (v4 lagged ~10–12 s);
  minor: ends 1–3 s early.
- ✅ **No dead trailing note**; ✅ **hitsounds slightly better**.
- ✅ **Jumps / patterns / streams much better** (streams still "feel bad" but clearly
  improved) — **validates ranked data + context + flip aug**.
- ⚠️ **Rhythm REGRESSED vs v4**: strange 0.5–2 s pauses; some notes off the ¼ grid
  (look 1/6 or 1/8). **NEW top decode issue** → RESEARCH §10.4.
- ➖ **No spinners** generated; ➖ **curve sliders still low** (the slider-representation
  gap the v5 17-ch channels target).

## v4 — full curated library (previous release)

`runs/20260614-110223-std-v4-full` — 31,270 curated maps (≤12★), base 128, **epoch 15,
loss 0.0077** (run killed by an OS sleep at e16, undertrained but strong). SR monotonic,
16–17/19 metrics in-range. Superseded by v4b (ranked data fixes the junk that ≤12★
curation missed). The **decode/post-process wins** shipped on v4 and still in the code:

- `clamp_slider_endpoints` — caps slider length so osu! extrapolation can't shoot tails
  off the playfield (0/116 off-field after `snap_slider_ends`).
- `decode_signal(accent_threshold=0.85)` — accent channels saturate near +1; 0.85 → ~0.33
  hitsound usage (matches real; 0.0–0.6 all stay ~0.52).
- `trim_isolated_ends` — asymmetric trailing trim (2.2 s) + drops a lone circle after the
  final spinner (phantom spin-down note).
- `snap_to_grid` loosened 45→60 ms / 40→50 % (fb #5) — *suspected contributor to the v4b
  rhythm regression; revisit (§10.4)*.
- `compute_breaks` + `write_osu(breaks=)` — `[Events]` breaks for gaps ≥3.5 s (cosmetic;
  marks existing gaps only).

## Earlier versions (v1–v3) — summary

| ver | data | model | loss | takeaway |
|-----|------|-------|------|----------|
| v1 | 601 | base 96, no attn, DDPM | 0.011 | pipeline works end-to-end; too dense, straight sliders, loose rhythm (0.70 on-grid) |
| v2 | 3004 | base 160 (97M), QK-norm attn, bf16 | 0.0075 | much closer to real (density/streams/mix in-range); low jumps, few curves, no SV |
| v3 draft | 1504 | base 128 + difficulty cond + CFG | 0.0097 | **conditioning steers difficulty** ✓ (density/streams scale with target SR) |
| v3 heavy | 6001 | base 128 + cond | 0.0056 | SR near-calibrated 3–5★; curved sliders + kiai + hitsounds all generate |

Durable lessons from these (base 160+bf16 diverges, DDIM not strided DDPM, curved-slider
encoder fix, etc.) live in **HANDOFF §7**.
