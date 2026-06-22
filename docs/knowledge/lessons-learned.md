# Hard-won lessons (don't re-learn)

**Purpose:** the durable, paid-for lessons of the project — training stability, the diffusion
objective, the under-dispersion root cause, decode/timing gotchas, sampling perf, and Windows/ops
quirks. A new agent should read this before changing the model, the objective, or the decode. |
**STATIC** (each line was learned the hard way; append only when something new is proven).

Where a lesson came from a specific version, the version doc has the full story
([docs/versions/](../versions/README.md)).

## Training stability & objective

- **base-128 stable; base-160 + bf16 DIVERGES under ε-prediction** (lr 1.2e-4, clip 0.3). The old
  divergences (v2 @e21, v3 @e12) were ε-pred. base-160 is *stability*-blocked, **not** memory-blocked
  (train peak ~5.3/12 GB; activations ~80%, weights ~5% → **fp8/fp4 weight quant is the wrong lever**;
  grad-checkpointing is the memory lever). **The unblock: v-prediction + zero-terminal-SNR** —
  base-160 v8 trained a clean 60 epochs through the e12–21 zone (val 0.078→0.041 monotonic, gnorm
  stable). Per-channel target standardisation is the other principled scaling candidate.
- **v-loss ≈ 100× ε-loss** — never compare a v-pred run's loss to an ε-pred run's; judge by trend.
- **DDIM, not strided DDPM.** Naively skipping steps while reusing the per-step DDPM coefficients
  under-denoises into noise-like maps. Use the DDIM sampler (correct over any step subsequence).
- **QK-norm attention is load-bearing** — never remove the QK-normalisation, the learnable temperature,
  or the zero-init output projection. Plain dot-product attention diverged under bf16; this is the fix.

## Under-dispersion — the central quality problem

- **Patterns + straight sliders are ONE root cause: under-dispersion from ε/v-MSE.** The model
  regresses spatial outputs toward the conditional mean → compressed spacing *and* collinear (straight)
  slider anchors. **Flow angles are already ≈ real → attention is NOT the bottleneck.**
- **`--up-attn` actively HURTS.** Self-attention is a weighted *average* across time → it smooths the
  spatial channels and compresses their variance. Up-path attention sits right before the output, so it
  hits final positions directly: v7-full's jumps collapsed (jump_ratio 0.145→0.048). Leave `--up-attn`
  off. (Down+mid attention from v5/v6 is fine.)
- **The model regresses to the AVERAGE map for the conditioned SR.** Per-song *extremes* (deathstream,
  jump-spam, heavy-curve) are all under-produced. Density has a `--density` lever; jumps/corners don't.
- **A passive channel can't beat the conditioning it shares.** v8's spacing-magnitude channel
  regressed to the SR-average because it shared the cursor's audio+SR conditioning — no extra per-song
  info. The lever must be **new information at the input** (per-song conditioning) or **an objective
  that samples extremes** (RL). This is the diagnosed primary v9 fix.
- **Encode dense positive scalars, not sparse binaries.** The `curve` cue worked (13→28% visible) — a
  non-negative scalar whose mean is a useful "typical bow". The `corner` cue under-fired (~2% vs 13%) —
  a *rare binary* whose mean = base-rate, then thresholded to ≈0. Corollary: **don't "count-scale" a
  binary** — a lower encoded value clears the decode threshold *less* often → *worse* under-fire (the
  rejected corner count-re-encode).
- **`--spacing-scale` HURTS in-game → use 0.** Respacing lifts the spacing *metrics* but the relocated
  objects read/play worse (flow + readability), so it's a net negative. The respace is shelved;
  generate raw (scale 0). SR conditioning already gives meaningful per-song/per-SR spacing variation.
- **`--match-sr` backfires on jump songs** (the model under-produces SR → it lands on a sparse low-SR
  map); condition a high `--sr` directly instead.

## Decode / timing gotchas

- **Slider duration comes from length.** osu! derives a slider's duration from `length / velocity`, so
  the writer clamps each slider's length to fit the gap before the next object, and slider ends are
  beat-snapped (else ~half land off the ¼-grid).
- **SV is structural, not per-slider.** Real maps have few coarse SV sections (≈ kiai-tied), not
  per-slider noise. Learn it as a channel; decode quantises to ~6–8 stable sections. **SV-aware slider
  snapping is required** — slider length depends on the SV at its time, else every slider drifts off
  the grid. (The early per-slider-geometric SV approach was "terrible".)
- **Red points = doubled control points** (sharp corners); ~13% of real sliders. `write_osu` writes
  `curve_points` verbatim → doubling them = corners. Detect them *before* RDP (RDP collapses
  duplicates → destroys corners).
- **Snap to the editor's tick, not `time + round(delta)`.** Banker's rounding of a half-tie delta
  (`round(±0.5)→0`) left a note 1 ms off the editor's own integer tick → editor flags it unsnapped
  (BPM-dependent, was 1.8–52.9% of objects). Store `int(round(grid))` directly. (v9 fix.)
- **Octave-fold band must be a no-op in-band.** `_normalise_octave` folding `[125,250)` doubled every
  sub-125 tempo (120→240) — the reported "doubled 240 BPM red lines". Now `[89,205)`, no-op inside the
  band. Only hit novel songs generated without `--timing-from`.
- **`write_osu` must not mutate caller objects.** `_clamp_slider_lengths` clamped `.length` in place →
  multi-difficulty packaging (best-of-N / multi-SR `infer`) progressively over-clamped. Clamp on
  `dataclasses.replace` copies.
- **Novel-song timing is ~28% phase-exact** (librosa). Use `--timing-from <ref.osu>` for known songs
  (always exact). A dedicated beat-tracker model is the planned fix.

## Sampling performance (2026-06-20)

- **Batched CFG** (one batch-2 forward, default on) ~2× sampling speed, bit-identical to the
  two-forward path. **But** batch-2 ~doubles peak activations → OOMs marathon songs at base-160; use
  `--no-batch-cfg` for those.
- **`--amp` (bf16) for long songs.** fp32 inference does NOT get the fused flash kernel → materialises
  the O(T²) attention matrix → an 8-min song (~42k frames) at base-160 OOMs (~14 GB). bf16 halves it →
  fits (~11.5 GB, still slow ~12 s/step). Real long-song fix = chunked/windowed sampling (future).
- **`--compile` works on Windows** (triton-windows + MSVC). flash-attn-from-source is not worth it —
  SDPA already uses the fused flash kernel *in bf16*.
- **Achieved SR overshoots on dense songs** (cond sr7.5 → 9.61 on the ICDD marathon).

## Ops / environment

- **My background processes get reaped (~40–50 min idle)** → the USER runs the long (5–6 h) trains. I
  do eval, codegen, decode work, analysis, and short drafts; the USER pushes + PRs + runs GPU trains.
- **`.gitignore` must root-anchor `/data/ /runs/ /artifacts/`** (else they'd match `src/data/`).
- **Windows console cp1251 → ASCII in prints.** Dataset mel cache = module-level `lru_cache` (Windows
  spawn). Don't re-preprocess existing data.
- **Tests are hermetic** (synthetic fixtures, no GPU/dataset) — keep them green + ruff clean after
  every change.
- **Trust the mapper's in-game play feedback over the metrics.** Metrics can rise while play gets worse
  (the `--spacing-scale` lesson) — every alignment/RL phase has a play-feedback acceptance gate.
