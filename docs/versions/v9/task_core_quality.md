# v9 — Core-component quality audit & test hardening

Task: rank the core pipeline components by quality impact, audit the highest-impact
ones for correctness bugs, fix real defects minimally + add regression tests, and add
targeted invariant/round-trip tests. Hermetic only. (User request #4.)

Result: **2 real bugs found and fixed** (1 medium = corrupted novel-song timing; 1
low-medium = caller-object mutation on write), **+15 new hermetic tests**. The rest of
the audited core is solid. `pytest` 203 passed, `ruff` clean.

## Component quality ranking (by how badly a bug corrupts output)

| # | Component | Verdict | Notes |
|---|-----------|---------|-------|
| 1 | `src/data/signal.py` (encode/decode) | **Solid** | Round-trips (times, positions, slider/circle/spinner type, anchors+curve+corner, SV sections, kiai, spacing) all hold. Found edge-case behaviours worth locking, no bugs. |
| 2 | `src/model/diffusion.py` | **Solid** | v↔x0/eps inversions exact (~1e-7); zero-SNR schedule finite everywhere incl. `posterior_var`/`sqrt_recip_alphas` at terminal (`acp_T=0`, `beta_T=1`); min-SNR-γ weights match diffusers; batched CFG == two-forward bit-identical (already tested); guidance_rescale runs finite. |
| 3 | `src/parsing/beatmap.py` (parse/write) | **1 bug fixed** | `write_osu` mutated caller objects in place. |
| 4 | `src/conditioning.py` | **Solid** | CONTEXT_DIM single-source-of-truth consistent with UNet wiring; target_settings in legal osu ranges. |
| 5 | `src/data/timing.py` | **1 bug fixed (the reported doubled-BPM)** | `_normalise_octave` doubled every slow song. |

## Bugs found + fixed

### BUG 1 (MEDIUM) — `_normalise_octave` doubled genuinely-slow songs → ~240 BPM red lines
`src/data/timing.py`. The octave-fold band was `[125, 250)`. Any tempo below 125 BPM —
a *very* common range for ranked maps — was multiplied by 2 to force it into the band:
a real 120-BPM song became **240**, 100 → 200, 89 → 178. librosa itself tracks these
tempi correctly (verified: median-IBI gives 89.1 for a true-90 click); the corruption
was entirely in the post-fold. This is exactly the open report in HANDOFF §0 / §6
("possible doubled 240 BPM red lines on `audio_*` maps"). It only bites novel songs
generated *without* `--timing-from` (a reference overrides the estimate), which is why
it showed up specifically on `audio_*` outputs.

**Fix:** lowered the band to `[89, 205)` and made the fold a no-op for tempi already
inside the band (only out-of-band octave *errors* are shifted). The band is wider than
one octave so the loop still terminates; output is idempotent. 120 → 120 now; 60 → 120,
240 → 120, 440 → 110 still fold correctly.

**Regression tests:** `test_normalise_octave_keeps_plausible_slow_bpm` (locks 120≠240
and the whole 90–200 band), `test_normalise_octave_idempotent_and_in_band`,
`test_slow_song_bpm_is_not_doubled` (end-to-end through librosa on a 100-BPM click).
Updated `test_normalise_octave_folds_into_range` to the new band. (Confirmed all three
fail on the old code.)

### BUG 2 (LOW-MEDIUM) — `write_osu` mutated the caller's HitObjects in place
`src/parsing/beatmap.py`. `_clamp_slider_lengths` shortened each over-long slider's
`.length` *on the caller's objects*. Consequences: (a) writing the same object list to
two files (multi-difficulty packaging) progressively over-clamped — the second file's
sliders were shorter than the first; (b) any metrics/analysis computed on the objects
after a write saw corrupted lengths. Writes were not idempotent.

**Fix:** clamp on `dataclasses.replace` copies; the caller's objects are never touched.
Output is unchanged for a single write and now identical across repeated writes.

**Regression tests:** `test_write_does_not_mutate_caller_objects` (asserts `.length`
preserved + double-write byte-identical), `test_written_slider_no_overlap_is_independent_of_caller_state`.

## Invariant / round-trip tests added (no bug, lock current behaviour)

- **signal.py**: peak-picker plateau collapse under the decode default `min_gap>=2`
  (model can emit flat-topped onsets → no duplicate 1-frame-apart notes); boundary
  onsets at frame 0 / last; spinner start+end time round-trip; **SV stacked red+green
  at the same time → green wins** (matches osu! ordering); all-zero-distance spacing
  channel → `decode_spacing` safely returns `[]`.
- **diffusion.py**: zero-SNR schedule finite everywhere (incl. posterior_var ≥ 0);
  q_sample ↔ posterior coefficient consistency; guidance_rescale finite on both
  batch_cfg paths.
- **conditioning.py**: target_settings AR/OD/CS/HP within legal osu ranges across SR;
  CONTEXT_DIM consistent with a UNet built at `ctx_dim=CONTEXT_DIM`.

## Audited and judged correct (no change)

- signal channel indexing (CH_SV=17, CH_CURVE=18, CH_CORNER=19, CH_SPACING=20) matches
  config and all decode reads; pre-v7/v8 signals correctly return `[]` from
  decode_sv/decode_spacing via the `sig.shape[0] <= CH_*` guards.
- `_recover_stream_gaps` gating (only fills holes with a real sub-threshold bump).
- SV/curve/spacing enc↔dec scalar round-trips and clamps.
- new_combo, hitsound-accent, kiai (incl. green-line-started kiai) round-trips.
- UNet `ctx_drop` → null embedding equivalence (the batched-CFG load-bearing claim).

## Defects flagged but NOT fixed (out of scope / by design)

- **Hitsound recall vs. peak-frame offset (by design, not a bug):** `_hit_sound` reads
  accent channels at the picked peak frame; if the peak lands 1 frame off the Gaussian
  bump centre, the accent value drops below the high `accent_threshold` (0.85) and the
  hitsound is dropped. This is the intended thinning (HANDOFF §6 lists hitsounds as a
  known-weak P1 with a planned rule-based head), so left as-is.
- **`decode_kiai` / `decode_sv` use a mix of literal `6` and `CH_*` guards** — currently
  correct (CH_KIAI=6), purely cosmetic; not touched to avoid churn in another agent's
  decode-threshold tuning surface.

## Verification
`uv run --extra dev pytest` → 203 passed. `uv run --extra dev ruff check .` → clean.
