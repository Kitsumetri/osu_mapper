# External audit findings (2026-06-14) + core-component audit

**Purpose:** the findings of the external code/math audit (a separate auditor read every `src/` file
and re-derived the diffusion math) plus the v9 core-component quality audit — what was fixed, what was
deferred, and what was judged correct. | **STATIC** (a frozen record of audits; new audits append a
section).

## Full code audit (2026-06-28) — 3 parallel opus agents (~39 findings)

Three agents split `src/` into ~equal thirds (data/representation · model/sampling · eval/metrics) and read
every file; the main agent verified each finding skeptically before acting. Headline: the diffusion math, the
flat-top reward, `sample_logprob`, the grid membership, and SR buckets all checked out; no stale references to
the removed playability defects remained.

**Fixed — encode (needed a `ranked-v9` reprocess), `bafc46d`:**
- **A#3** reverse-slider cursor flow-OUT keyed to the slider tail unconditionally; an even-slide
  (single-reverse) slider ends gameplay at the HEAD → use the gameplay end (parity-aware). +test.
- **A#6** timing points sorted by time only → a green (sets SV) listed before a red (resets SV→1) at the same
  offset let the red win in `_sv_at` (last-wins). Tie-break uninherited-first. Feeds `slider_duration` →
  `end_time` → the encode. +test.

**Fixed — correctness / perf (`e4f8fbb`):** **C#1** `curved_slider_ratio` was the reward's heaviest slider
metric but missing from `corpus_stats.KEYS` → no gold band → silently dropped from EVERY reward (added). **C#3**
`reward_from_osu` gained `achieved_sr=` reuse so the `--all` scan stops running rosu SR twice. **C#5** the
velocity defect measured from the slider head, not its end (now `end_time` + parity). **C#6** `slider_overlap`
got an exact bbox prefilter (O(sliders×objects) → ~O(sliders×neighbours)). **B#2** `guidance_rescale` rescaled
x0 but reused the un-rescaled eps (re-derive). **B#5** the val-reward probe sampled at `aim=0` (now uses the
manifest aim). **B#6** latent zero-SNR `sigma` NaN clamp. **A#1** empty-val-set guard, **A#2** mel-cache
aliasing copy, **B#4** hardened the dead `p_sample`.

**Fixed — guards + stale docs (`6d90b64`):** crash guards (`evaluate`, `package_map`, `_is_gold` min-objects);
stale "band-less until refresh" comments (the v9 band metrics ARE banded post-refresh); `sr_closeness` docstring
(0.37→0.5); the early-abort "winner identical" claim softened (the cheap quality-only proxy omits SR-closeness →
near-identical, not a hard guarantee).

**Rejected (skeptical):** snap `(4,8,6)` (play-validated; doc-only); the `aim_intensity` 2nd-STFT "fix" (would
change a load-bearing feature's values); octave-fold widening (a real estimator ambiguity, not a bug); the
match-sr difficulty-name (cosmetic). Agent suggestions to *delete* `p_sample`/`sqrt_recip_alphas` were overruled
(referenced by the RL plan / a finiteness test) — hardened instead.

## External audit (2026-06-14)

The auditor confirmed **the diffusion math is all correct**. Defects were at the encode/decode/writer
boundary + config hygiene.

### Fixed (commits `44f8a80`, `6444f3e` on `feat/v5-slider-style`)
- **C-1** spec-correct slider `edgeSounds`/`edgeSets`/`hitSample` in `write_osu` (was a malformed,
  shifted field that lazer could reject).
- **C-2** `generate` writes AR/OD/HP/CS from `conditioning.target_settings(sr)` instead of a hardcoded
  AR8/OD7 — file difficulty (and the rosu SR read-back) now match what the model was conditioned on.
- **C-3** a slider that lost its curve points is rewritten as a circle (not type&2 with no path).
- **S-8** short-song mel pad uses −1.0 (silence) not 0.0 (≈−40 dB).
- **S-16** `corpus_stats` parses + mode-filters before the expensive rosu call.
- **S-6** `AttnBlock1d` asserts `ch % heads == 0`.
- **S-9** `item_id` gets a source-path hash (no silent npz overwrite).
- **S-3** removed dead `skip_chs`; **S-1** noted `p_sample` reference-only; checkpoints store
  `sig_channels`.
- Deleted dead `src/utils/` + unused `h5py`/`pyyaml`/`colorlog`; synced `requirements.txt`.
- Fixed TECH_REPORT §9 (had listed the *diverged* base-160/0.5 config as "current" → now base-128/0.3),
  the v5 decode section, and the README channel count.

### Deferred (tasks created / future work)
- **Per-slider SV** (5.1) — emit an inherited timing point per slider so geometry length *and*
  model-intended duration are both honoured. *(Superseded by the learned SV channel, v7.)*
- **Per-channel target standardisation** (5.2) — train on `(x−μ_c)/σ_c` to zero-centre the
  −1-baseline channels; a suspected contributor to the base-160 bf16 divergence. *(Still a candidate.)*
- **Zero-terminal-SNR β + v-prediction** (5.3) — the principled fix to unblock base ≥160.
  *(Shipped v7 — and it did unblock base-160 at v8.)*
- **Batched CFG** (5.4) — one concatenated forward instead of two → ~2× faster sampling, identical
  output. *(Done, `diffusion.ddim_sample`, 2026-06-20; memory tradeoff: ~2× peak activations → OOMs
  marathon songs at base-160, so `--no-batch-cfg` keeps the low-memory path.)*
- **Attention on the up-path** (S-5) / fuse the top skip (S-4) — *(up-path attention was tried at v7
  and HURTS — see [lessons-learned.md](lessons-learned.md); do not enable.)*

### Minor (left as-is, cosmetic/negligible)
`package_map` re-parse drops `[Events]` breaks (S-17); `_validate` last-batch over-weighting (S-14);
`snap_slider_ends` SV=1 (S-11, no bug for single-timing generated maps; subsumed by the SV channel);
`compute_breaks` ordering (S-13); slider 1-frame demotion (S-7, rare).

### Investigations (need runs/data)
base-160 divergence root cause (U-4, grad-norm trace — largely resolved by v-pred); long-song
attention-length transfer (U-5); `osu!.db` size-prefix robustness (U-1, fine for the current client).
Moot after fixes: lazer slider-field tolerance (U-2 → C-1), short-song pad frequency (U-6 → S-8).

## Core-component quality audit (v9, 2026-06-22, commit `07f8912`)

Ranked the core pipeline by quality impact, audited the highest-impact components, fixed real defects
minimally + added regression tests (+15 hermetic tests). Full report:
[versions/v9/task_core_quality.md](../versions/v9/task_core_quality.md).

| # | Component | Verdict |
|---|-----------|---------|
| 1 | `src/data/signal.py` (encode/decode) | **Solid** — all round-trips hold |
| 2 | `src/model/diffusion.py` | **Solid** — v↔x0/eps exact, zero-SNR finite, min-SNR-γ correct, batched CFG bit-identical |
| 3 | `src/parsing/beatmap.py` | **1 bug fixed** (`write_osu` in-place mutation) |
| 4 | `src/conditioning.py` | **Solid** |
| 5 | `src/data/timing.py` | **1 bug fixed** (the reported doubled-BPM) |

**Two bugs fixed** (both also recorded in [lessons-learned.md](lessons-learned.md)):
- **MEDIUM — `_normalise_octave` doubled slow songs.** Octave-fold band `[125,250)` forced every
  sub-125 tempo up an octave (120→240) — the reported "doubled 240 BPM red lines on `audio_*`". Now
  `[89,205)` with a no-op in-band. Only hit novel songs generated without `--timing-from`.
- **LOW-MED — `write_osu` mutated caller objects in place.** `_clamp_slider_lengths` shortened
  sliders' `.length` on the caller's objects → multi-difficulty packaging progressively over-clamped.
  Now clamps on `dataclasses.replace` copies.

**Flagged but not fixed (by design):** hitsound recall vs peak-frame offset (the intended thinning at
`accent_threshold` 0.85; the rule-based hitsound head is the planned fix); `decode_kiai`/`decode_sv`
mixing literal `6` and `CH_*` guards (currently correct, cosmetic).
