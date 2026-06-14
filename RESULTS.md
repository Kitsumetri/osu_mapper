# Results

Status of training runs and generated-map quality. Metrics use `src/metrics.py`.
**Current release: v4** (`runs/20260614-110223-std-v4-full/ckpt/best.pt`).

## v4 full-data (current release)

Trained on the **entire curated library** (31,270 std maps ≤12★, parallel
preprocess) at base 128 / batch 32. The run was **killed by an OS/sleep event at
epoch 16** (no traceback, 0 procs — not a code bug); `best.pt` = **epoch 15,
loss 0.0077**, which is already a strong model.

SR sweep (`evaluate.py`): conditioning monotonic; metric-realism **improved over
v3** (16–17/19 metrics in the real p10–p90 band; streams now match real, e.g.
SR6 stream 0.34 ≈ real 0.31). SR calibration is looser than v3-heavy2 (only 15
epochs) — `--match-sr` corrects it at inference (target 5→4.86, 6→6.02 in 2–3
iters). Curated data (≤12★) removes the joke-map outliers.

Packaged samples: `[AI-v4]` (4.86★), `[AI-v4-6star]` (6.02★).

**Play feedback (v4)** — better than v3, ≈ v3b. Open items (all in
RESEARCH §10 / HANDOFF): slider endpoints sometimes off-playfield; **no
reverse sliders** (representation gap); kiai #1 short/mis-timed (alignment);
streams sometimes poor; some onsets slightly off-¼ (beyond snap tolerance); curve
shape sometimes bad; trailing unhittable last note recurs; no breaks (too dense).

**To finish v4 properly**: resume/retrain to ~50 epochs (add `--resume`; the run
died undertrained) for tighter SR calibration + cleaner curves.

### v4 decode/post-process wins (2026-06-14, no retrain)

Cheap play-feedback fixes on the same `best.pt` (validated on a real generation,
Thaehan - Kawaii @ 4.5★, 100 DDIM steps):

- **Slider tails clamped to playfield** (fb #1) — `clamp_slider_endpoints` caps a
  slider's pixel length so osu!'s extrapolation past the last anchor can't shoot
  the tail off-screen (anchors clamped too; `end_time` scaled with length).
  Result: **0 / 116 off-field slider tails** (sliders previously escaped the
  512x384 field after `snap_slider_ends` stretched their length).
- **Hitsound usage calibrated** (10.1.C) — accent channels saturate near +1, so
  the old threshold 0 over-fired (~0.52). Swept on real output: threshold **0.85
  -> 0.33 hitsound fraction** (matches real ~0.33); 0.0-0.6 all stay ~0.52. New
  `decode_signal(accent_threshold=0.85)` default.
- **Trailing trim tightened** (fb #7) — `trim_isolated_ends` now trims trailing
  lone notes at a smaller gap (2.2 s) than leading ones (3.0 s), catching outro
  notes the old 3 s threshold missed.
- **Onset snap loosened** (fb #5) — `snap_to_grid` bound raised 45 ms/40% ->
  60 ms/50% of the subdivision, so more circles land cleanly on 1/4 (still bounded
  so a wrong BPM can't drag the whole map).
- **`[Events]` breaks** (10.1.D-iii) — `compute_breaks` writes break periods for
  gaps >=3.5 s. *Cosmetic / honest limit*: it only marks gaps that already exist;
  the dense v4 model leaves none on busy songs (0 breaks on Kawaii), so this helps
  intros/outros and sparse maps, not the "too dense" root cause (that's model-side
  — density conditioning, 10.1.D-i/ii).

## v1 baseline (complete)

First full run — establishes that the pipeline works end-to-end.

- **Data**: 601 osu!standard difficulties.
- **Model**: 1D conditional U-Net, base 96 (~14M params), no attention, DDPM.
- **Run**: 240 epochs, batch 8, crop 2048, AMP fp16, RTX 4070 Ti (~48 min).
- **Loss** (ε-prediction MSE): 0.35 → **0.011**.

Generation (DDIM 100 steps, ~0.5 s) on a held-out song vs the real Expert diff:

| metric | generated (v1) | real Expert |
|--------|---------------:|------------:|
| objects | 1726 | 962 |
| density / s | 6.9 | 4.0 |
| circle / slider / spinner ratio | .56 / .44 / .00 | .66 / .34 / .00 |
| bezier-slider ratio | 0.00 | 0.18 |
| stream ratio | 0.50 | 0.16 |
| on-¼-grid ratio | 0.70 | 0.997 |
| est. timing | 198.8 BPM | 192 BPM |

Valid, playable `.osu` that re-parses cleanly. Reads the rhythm, but feels loose
(low on-grid), is too dense/stream-heavy, and had only straight sliders.

## v3 heavy (complete) — best model

Full v3 on the large dataset. **base 128** (base 160 + bf16 diverged twice).

- **Data**: 6001 difficulties (10-channel signals + star rating).
- **Model**: base 128 + QK-norm attention + difficulty conditioning + CFG.
- **Run**: 100 epochs, batch 14, crop 3072, ~46 s/epoch (~75 min); loss → **0.0056**,
  no divergence (lr 1.2e-4, grad-clip 0.3).

SR sweep (`evaluate.py`, guidance 2.0) — achieved SR now tracks the target
closely (the draft's +1.5–2★ offset largely self-corrected with full data):

| target SR | achieved | density | real density | metrics in p10–p90 |
|----------:|---------:|--------:|-------------:|-------------------:|
| 2.0 | 2.50 | 1.77 | 1.7 | 13/19 |
| 3.0 | 2.78 | 2.47 | 2.7 | **17/19** |
| 4.0 | 3.69 | 2.96 | 3.6 | 15/19 |
| 5.0 | 5.07 | 3.79 | 4.4 | 16/19 |
| 6.0 | 6.86 | 5.32 | 5.8 | 15/19 |

Conditioning is monotonic and near-calibrated for 3–5★; density/streams track
real maps; curved sliders, kiai (1–3 spans), and hitsounds all generate.
Hitsound usage ~0.35–0.58 (real ~0.33 — slightly high, was 0.67 in the draft).
Packaged sample: `[AI-v3]` folder (SR 5.07, 947 objects, 0 overlaps, 3 kiai).

Remaining: SR slightly drifts at the extremes (2, 6); streams a touch low;
hitsounds a touch high. Next: v4 (RESEARCH §10) — style conditioning,
slider-shape channels, multi-section timing.

## v3 draft (complete) — conditioning works

Proof-of-concept for the v3 representation (10 channels: + kiai + whistle/finish/
clap) and **difficulty conditioning** (SR context vector + classifier-free
guidance).

- **Data**: 1504 difficulties (10-channel signals + star rating in manifest).
- **Model**: base 128 + attention + conditioning (ctx_dim 6, CFG drop 0.15).
- **Run**: 60 epochs, ~8 s/epoch (~8 min); loss → **0.0097**, no divergence.

Same song generated at two target star ratings (guidance 2.5):

| target SR | density/s | stream | jump | objects | real trend |
|-----------|----------:|-------:|-----:|--------:|------------|
| **3.0**   | 3.34 | 0.13 | 0.04 | 834  | Hard ≈ 2.7 / 0.08 |
| **6.0**   | 6.40 | 0.45 | 0.10 | 1600 | Expert ≈ 4.4 / 0.21 |

**Conditioning clearly steers difficulty** in the right direction (density and
streams scale with SR, matching the reference trends — absolute density runs a
bit high, expected to tighten with the full dataset). Kiai (3 spans) and
hitsounds both generate. This validated launching the heavy v3 run.

## v2 scaled (complete)

Rebuilt on the bug-fixes + bigger model + more data + new features:

- **Data**: 3004 difficulties / 888 audios (deduped, manifest-indexed).
- **Model**: base 160, **97.4M params**, self-attention (QK-norm) at coarse
  levels, bf16, EMA, cosine LR + warmup.
- **Run**: 120 epochs, batch 12, crop 3072, ~32 s/epoch (~64 min).
- **Loss**: 0.44 → **0.0075** (no divergence; the QK-norm fix held).
- **Features since v1**: DDIM, slider time-overlap fix, curved Bezier sliders,
  beat-snapping, realistic difficulty defaults, estimated timing.

Generation scored against the 12k-map reference (Section 8 of `RESEARCH.md`);
v2 auto-lands in the **Hard** density bucket. `z` = std-devs from the real mean.

| metric | v1 | **v2** | real (Hard) | v2 z | in-range |
|--------|---:|------:|-----------:|----:|:--------:|
| density / s | 6.9 | **3.36** | 3.74 | −0.89 | ✓ |
| circle / slider ratio | .56/.44 | **.69/.31** | .56/.44 | ±1.0 | ✓ |
| stream_ratio | 0.50 | **0.13** | 0.14 | −0.11 | ✓ |
| on-¼-grid | 0.70 | **0.81** | 0.92 | −0.50 | ✓ |
| reversal_ratio | .18 | **.17** | .20 | −0.33 | ✓ |
| jump_ratio | .13 | **0.06** | 0.28 | −1.30 | ✗ |
| std_spacing_px | 73 | **57** | 83 | −1.67 | ✗ |
| bezier_slider_ratio | .00 | **0.016** | 0.15 | −0.88 | ✗ |
| sv_changes_per_min | 0 | **0** | 15.5 | −0.53 | ✗ |

**v2 is much closer to real maps than v1**: density, stream ratio, and
circle/slider mix went from clearly-wrong to in-range, and on-grid improved.

Remaining gaps (next work):
- **Too smooth / conservative**: low `jump_ratio` and `std_spacing_px` — the
  model averages toward safe spacing. Needs flow/DS-aware modelling (§3.A).
- **Few curved sliders**: v2 trained on data preprocessed *before* the
  curved-slider encoder fix, so it never saw curved holds. Re-preprocess +
  retrain to lift `bezier_slider_ratio` toward ~0.15.
- **No SV variety**: structural (single timing point); needs multi-section
  timing / inherited points.

## Honest assessment

Works: fully functional audio → `.osu`; realistic density and circle/slider/
spinner mix; full-playfield cursor use; valid output; now with curved sliders.

Open gaps (see `README.md` roadmap + `RESEARCH.md`):

- **Rhythm**: onsets aren't beat-snapped (~0.70 on-grid vs ~0.99 real).
- **Timing accuracy**: BPM estimate exact only ~28%, which also skews the grid.
- **Controllability**: no difficulty/style conditioning yet (one fixed tier).
- **Scale**: 3004 of 31k+ available difficulties.

## Reproduce

```bash
python main.py preprocess --songs "C:/osu!/Songs" --out data/processed/std-v1 --limit 3000
python main.py train    --data data/processed/std-v1 --tag std-v1-base160 \
    --epochs 120 --batch 12 --crop 3072 --base 160
python main.py generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --out out.osu --steps 100
python -m src.metrics   --osu out.osu --ref some_real_map.osu     # compare
```
