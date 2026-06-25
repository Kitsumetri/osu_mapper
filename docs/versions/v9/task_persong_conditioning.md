# v9 — per-song aim-intensity conditioning (the diagnosed primary fix)

*STATIC / frozen — v9 task report.*

The diagnosed root cause (HANDOFF §7 / [v8.md](../v8.md) / lessons-learned
"a passive channel can't beat the conditioning it shares"): v8's spacing channel
regressed to the **SR-average** because it shared the cursor's conditioning (audio
mel + SR) and carried **no new per-song information**. The model produces "the
average map for this SR", under-producing per-song extremes (jumps / streams), and
`--spacing-scale` (a passive decode lever) HURTS in-game. The missing lever is **NEW
information at the input**: a per-song aim-intensity target derived from the AUDIO,
fed into the conditioning context like `--density`. That makes "more jumps *on the
jumpy song*" expressible — the model can learn `audio -> intensity -> spacing`
instead of regressing to the SR mean.

This task adds that input (code + tests only). **No training / GPU / full preprocess
here — the USER runs the reprocess + train (commands at the bottom).** The new
conditioning takes effect only after that retrain.

---

## A. The feature — one shared function (`data.audio.aim_intensity`)

**Definition (one line):** the corpus-referenced **mean of the self-normalised onset
envelope** (`librosa.onset.onset_strength`), squashed to **[0, 1]** — a busier / more
percussive song fires more strong onsets per frame and scores higher.

Computation (`aim_intensity(y, cfg=AUDIO) -> float`):
1. `env = librosa.onset.onset_strength(y, …)` using the **same** mel front-end as the
   conditioning mel (same `sr / n_fft / hop / n_mels / fmin / fmax` from `AudioConfig`),
   so it reuses the existing librosa dependency and is frame-aligned to the signal.
2. `peak = env.max()`; flat/silent (`peak ≤ 1e-8`) → return `0.0`.
3. `score = mean(env / peak)` — divide the envelope by its **own peak** (a per-song
   gain invariance), then take a robust **central** statistic (the mean, not a single
   transient).
4. `return clamp(score / _AIM_REF, 0, 1)`, with `_AIM_REF = 0.35` a fixed reference
   normalised-onset-mean that maps a typical busy song to ~1.0.

**Why it is robust + loudness-stable** (the round-3a audio-parity concern,
[task_audio_features.md](task_audio_features.md) §1):
- `onset_strength` is built from the **derivative** of a `power_to_db` mel
  spectrogram → it responds to spectral **change** (onsets), not absolute level, so a
  globally louder rip barely moves it.
- Dividing by the per-song peak mirrors the mel path's `power_to_db(ref=np.max)`
  per-file gain invariance → the statistic is the **shape** of the song's own rhythm,
  comparable across songs of different loudness / mastering. (Verified hermetically:
  a 2× gain changes the value by < 0.05 — `test_loudness_invariance`.)
- The aggregate is a **mean**, not a max, so a single clipped transient can't pin it;
  the final `clamp(·/_AIM_REF)` keeps it **bounded + deterministic** regardless of
  song length or amplitude.
- Silence / empty / non-finite → `0.0` (a calm map; the same value as the
  missing-manifest-field default, so the two fallbacks agree).

**Per-song, not per-window** (design decision). The diagnosis is *per-song* ("the
jumpy song" vs "the calm song"), and the value is a per-audio scalar shared by all of
a song's difficulties — exactly like the mel. A single robust scalar is the simplest
thing that supplies the missing per-song axis; a per-window intensity is deferred (it
would need a new conditioning *channel* — a tensor-shape change and a much larger
blast radius, like the audio-features rungs — not a context-vector slot).

**Which librosa feature** (design decision). `onset.onset_strength` (spectral-flux
onset envelope) over a raw spectral-flux or RMS energy: it is purpose-built for
**onset / percussive activity**, which is what aim-intensity tracks, and it is the
same family the decode onset channel uses, so "busy rhythm" is measured consistently.

### Parity guarantee (the key property — no train/infer skew)

The **same** `aim_intensity` function runs at BOTH ends on the **same decoded array**:
- **Preprocess** (`data/preprocess.py`): `y = load_audio(path)` → `log_mel(y)` and
  `aim_intensity(y)` (decoded once, reused; the value is stored per audio in the
  manifest).
- **Inference** (`generate.prepare_audio`): `y = load_audio(path)` → `log_mel(y)` and
  `aim_intensity(y)` (the value is attached to `PreparedAudio.aim`).

Both call sites import the one function and feed it the identical `load_audio(path)`
output, so the train value and the inference value are bitwise-identical for the same
file. `test_aim_intensity.py::test_preprocess_inference_parity` asserts exactly this
(`prepare_audio(path).aim == aim_intensity(load_audio(path))`).

---

## B. The context-dim change (exact old → new + ordering)

`CONTEXT_DIM`: **6 → 7**. The new field is **appended last** so indices 0–5 keep their
meaning (old checkpoints / call sites unaffected):

| idx | 0 | 1 | 2 | 3 | 4 | 5 | **6** |
|-----|----|----|----|----|----|---------|-------|
| field | sr | ar | od | hp | cs | density | **aim** |
| scale (`_SCALE`) | 10 | 10 | 10 | 10 | 7 | 12 | **1.0** |

`aim` is already in `[0, 1]`, so its scale is `1.0` (then clamped to `[0, 1.5]` like
every slot). Wiring:
- **`context_vector(…, aim=0.0)`** — new trailing arg, defaults to `0.0` so the
  6-field call sites stay valid.
- **`context_from_manifest(item)`** (TRAIN side) — reads `item["aim_intensity"]`,
  **defaults to `0.0`** when absent (old datasets).
- **`target_context(sr, …, aim_intensity=None)`** (INFERENCE side) — `None` → `0.0`
  baseline; the audio-derived value (or `--aim-intensity` override) is passed here.

### Train-value vs inference-value flow

- **TRAIN value comes from the AUDIO, stored in the manifest.** `preprocess` computes
  `aim_intensity(y)` per audio and writes `aim_intensity` onto each item;
  `dataset.__getitem__` → `context_from_manifest(it)` reads it into the ctx the U-Net
  trains on. (Per-audio scalar, shared across a song's difficulties — like the mel.)
- **INFERENCE value comes from the INPUT audio, overridable.** `prepare_audio`
  computes `aim_intensity(y)` from the song being mapped and stores it on
  `PreparedAudio.aim`; `generate._one_pass` passes it into `target_context(…,
  aim_intensity=aim_eff)`. The `--aim-intensity` CLI overrides it (`aim_eff =
  override if override is not None else audio_value`) — mirroring `--density`: the
  default is audio-derived, and the user can push (higher = jumpier) or dampen.

---

## C. Backward-compat (load-bearing — verified)

Bumping `CONTEXT_DIM` changes the U-Net input width (`ctx_mlp.0` is
`Linear(ctx_dim, t_dim)`), so a pre-v9 checkpoint's `ctx_mlp.0.weight` is `[t_dim, 6]`
and rebuilding it at 7 would crash `load_state_dict`. Fix in `generate.load_model`:
the context width is now read from the **checkpoint's own weights** —
`weights["ctx_mlp.0.weight"].shape[1]` (authoritative), else `cargs["ctx_dim"]`, else
`CONTEXT_DIM` (a conditioned ckpt with `cfg_drop`), else `0` (unconditioned). So:
- **old v8 / v8_1 checkpoints (ctx_dim 6) still load + generate** at width 6 — the new
  aim slot simply isn't part of their context. (`train.py` also now **saves
  `ctx_dim`** so future ckpts carry it explicitly, like `sig_channels`.)
- **new data missing `aim_intensity`** → `context_from_manifest` defaults the slot to
  `0.0` (trains without re-preprocessing; sees the neutral baseline).

Verified hermetically — `test_load_model_uses_checkpoint_ctx_dim[6]` saves a pre-v9
6-ctx checkpoint (no `ctx_dim` key, exactly like the real v8 ckpts), loads it under
the new `CONTEXT_DIM=7`, asserts it loads at `ctx_dim=6` and samples; `[7]` does the
same for a v9 ckpt. **The new conditioning only takes effect after the USER retrains.**

---

## D. Retrain implication + the EXACT USER commands

`CONTEXT_DIM` 6 → 7 changes the model input, and the train value must come from the
manifest, so the USER must (1) **reprocess** the dataset to add the `aim_intensity`
field, then (2) **train from scratch** (resuming a 6-ctx checkpoint is impossible —
different `ctx_mlp` shape). Mirroring the v8 recipe in HANDOFF §4:

```bash
# 1. reprocess gold data -> 21-ch + per-song aim_intensity in the manifest
uv run python -m src.data.preprocess --songs "C:/osu!/Songs" \
    --out data/processed/ranked-v9 --gold --workers 10
#    (verify: a manifest row now has an "aim_intensity" in [0,1])

# 2. train from scratch (v8 recipe; NO --resume — ctx width changed). v-pred+zero-SNR
#    unblocks base-160; NO --rope/--up-attn per the lessons (they're a separate A/B).
uv run python -m src.train --data data/processed/ranked-v9 --tag ranked-v9 \
    --base 160 --crop 4096 --attn-levels 3 --batch 16 --epochs 60 --save-every 5 \
    --augment true --val-frac 0.10 --workers 8 \
    --objective v --zero-snr --compile --spatial-loss-weight 3

# 3. generate: aim-intensity is auto-computed from the song; override to push/dampen
uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 5 6 7
uv run python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt \
    --sr 6 --timing-from ref.osu --aim-intensity 0.85 --out out.osu   # push jumps
```

(`--val-frac 0.10` + the held-out-song split from round-3a; watch `val_reward`.)
After the retrain, the natural next step is the planned **RWR / best-of-N distillation
on the conditioned model** ([v9.md](../v9.md) step 5) to commit to the per-song tail.

---

## Files touched

- `src/data/audio.py` — `aim_intensity(y, cfg=AUDIO) -> float` (the ONE shared
  function) + `_AIM_REF`.
- `src/conditioning.py` — `CONTEXT_FIELDS` += `"aim"` (`CONTEXT_DIM` 6 → 7);
  `context_vector(…, aim=0.0)`; `context_from_manifest` reads `aim_intensity`
  (default 0.0); `target_context(…, aim_intensity=None)`.
- `src/data/preprocess.py` — decode once via `load_audio` + `log_mel`; compute
  `aim_intensity(y)` per audio; store `aim_intensity` on each manifest item.
- `src/generate.py` — `PreparedAudio` gains `aim` (defaulted, back-compat);
  `prepare_audio` computes it; `generate(…, aim_override=None)` resolves audio-default
  vs override and feeds `target_context`; `--aim-intensity` CLI; **`load_model` reads
  the ctx width from the checkpoint's own weights** (the backward-compat fix).
- `src/train.py` — checkpoint now also saves `ctx_dim` (future-proofs the width).
- `tests/test_aim_intensity.py` — NEW: parity, bounded range, determinism,
  loudness-invariance, busy>calm monotonicity, silence→0, context-dim layout +
  ordering, `context_from_manifest` graceful default, the `target_context` /
  `--aim-intensity` override, and the load_model backward-compat (6-ctx + 7-ctx). 16
  tests, ruff clean.
