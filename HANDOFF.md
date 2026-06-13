# Agent handoff — osu_mapper

Full working context for a fresh agent session. Read this first, then
`README.md` (usage), `TECH_REPORT.md` (the math), `RESEARCH.md` (design + plans),
`RESULTS.md` (run history), `STORAGE.md` (data/run layout).

## 1. What this project is

Train a **conditional diffusion model** that generates osu!standard beatmaps from
raw audio. A beatmap is encoded as a frame-aligned multi-channel **signal**; a 1D
U-Net denoises it conditioned on the audio's mel spectrogram (and, in v3, a
difficulty context vector). The signal is decoded back into hit objects and
written to a playable `.osu`.

The user (`Kitsumetri`, GitHub) is an osu! player + ML dev on a Windows/RTX 4070
Ti box. Songs library at `C:\osu!\Songs` (~31.8k `.osu`, ~8k sets). They test
generated maps in-game and give concrete play feedback.

## 2. Pipeline & architecture

```
audio.mp3 ─► log-mel (64×T) ──────────────┐
                                          ├─► 1D U-Net (DDIM, QK-norm attn) ─► signal (10×T) ─► decode ─► .osu
   noise (10×T) ──────────────────────────┘     ▲ cond: mel + difficulty ctx [SR,AR,OD,HP,CS,density]
```

- **Signal = 10 channels** (`src/config.py`): onset, slider_hold, spinner_hold,
  new_combo, cursor_x, cursor_y, kiai_hold, whistle, finish, clap. ~86 fps
  (sr 22050, hop 256). Encode/decode in `src/data/signal.py`.
- **Diffusion**: DDPM (1000 steps, linear β), ε-prediction. `src/model/diffusion.py`
  (full DDPM `p_sample` + accelerated `ddim_sample` with CFG). Math in TECH_REPORT.
- **U-Net** (`src/model/unet.py`): base-channel × (1,2,4,8), FiLM timestep
  embedding, **QK-norm self-attention** at coarse levels (the fix for bf16
  divergence — do NOT remove the QK-norm / learned temperature / zero-init proj).
- **Conditioning** (`src/conditioning.py`): difficulty vector added to the time
  embedding; **classifier-free guidance** (train drops it 15%, sample guides).
- **Star rating** (`src/difficulty.py`): exact via `rosu-pp-py` (don't reimplement).

## 3. Repo map (where to find things)

| Path | What |
|------|------|
| `src/config.py` | audio + signal channel config, frame↔time |
| `src/conditioning.py` | difficulty context vector + `target_context` |
| `src/difficulty.py` | star rating (rosu-pp) + SR bands |
| `src/parsing/beatmap.py` | `.osu` parser + `write_osu` (bitflags, sliders, kiai, hitsounds) |
| `src/data/signal.py` | `encode_beatmap`, `decode_signal`, `decode_kiai`, `_slider_path` |
| `src/data/audio.py` | log-mel |
| `src/data/timing.py` | BPM/offset estimate (librosa) |
| `src/data/preprocess.py` | crawl library → deduped mels + items + manifest (+SR) |
| `src/data/dataset.py` | manifest dataset → (signal, mel, ctx) |
| `src/model/{unet,diffusion}.py` | denoiser + DDPM/DDIM/CFG |
| `src/train.py` | training loop (bf16, EMA, cosine LR, CFG dropout, runs/ logging) |
| `src/generate.py` | audio→.osu (DDIM+CFG, `--sr`, `--guidance`, `--match-sr`) |
| `src/postprocess.py` | beat-snap, trim dangling ends |
| `src/metrics.py` | pattern metrics + `--ref-stats` z-scoring |
| `src/corpus_stats.py` | reference distributions over the library (by SR) |
| `src/evaluate.py` | SR-sweep eval (achieved vs target + metrics) |
| `src/package_map.py` | build a playable Songs folder from a generated map |
| `tests/` | 66 hermetic tests (no dataset/GPU) |
| `runs/<id>/` | training runs (config.json, metrics.csv, ckpt/) — gitignored |
| `data/processed/<tag>/` | preprocessed datasets — gitignored |
| `artifacts/` | generated maps, reference_stats.json — gitignored |

## 4. How to run

```bash
pytest                                   # 66 tests, ~6 s
ruff check . && ruff format .
python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/std-v3-full --limit 6000
python -m src.train --data data/processed/std-v3-full --tag std-v3-heavy --base 160 --crop 3072 --epochs 150
python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --sr 4.5 --guidance 2 --match-sr --out out.osu
python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --srs 2,3,4,5,6
python -m src.metrics --osu out.osu --ref-stats artifacts/reference_stats.json
python -m src.corpus_stats --songs "C:/osu!/Songs"   # rebuild reference_stats.json (all maps)
```

## 5. Current state (2026-06-14)

- **Heavy v3 training RUNNING**: `runs/20260614-015114-std-v3-heavy` — base 160,
  97.5M params, 6001 maps, 150 epochs, ~62–66 s/epoch (≈2.7 h), loss healthy
  (~0.012 @ epoch 11). A persistent monitor watches for divergence/completion.
- **Reference** `artifacts/reference_stats.json`: 31,362 maps, bucketed by SR,
  includes kiai/hitsound metrics.
- **v3 draft** (1500 maps) already proved conditioning works: target SR 2/4/6 →
  monotonic achieved SR; kiai + hitsounds generate.
- Git: `feat/diffusion-pipeline`, ~8 commits unpushed (**I cannot push — the user
  pushes**; HTTPS needs their browser auth, their SSH key isn't on GitHub).

## 6. When the heavy run finishes — do this

1. Check `runs/.../metrics.csv` (no divergence; best loss).
2. `python -m src.evaluate --audio <a real audio.mp3> --ckpt runs/.../ckpt/best.pt --srs 2,3,4,5,6`
   — confirm achieved SR tracks target; note the **SR offset**.
3. **Validate `--match-sr`** at runtime (deferred during training to keep GPU free).
4. **Hitsounds are over-applied** (draft 0.67 vs real 0.33): raise the accent
   decode threshold in `signal.decode_signal._hit_sound` (currently `> 0`; try a
   positive threshold) — verify against `reference_stats.json`.
5. Package a sample (`src/package_map.py`, prefix `[AI-v3]`) for the user to test.
6. Write v3 heavy results into `RESULTS.md` + memory.

## 7. Known issues / next (priority order)

- **SR offset**: model generates ~1.5–2★ harder than requested (draft). `--match-sr`
  corrects via feedback; after eval, bake the offset into `target_context` (§10.2).
- **Hitsound over-application** (see §6.4 above).
- **Sliders**: curved now (good) but shape rides on the noisy cursor channel;
  proper slider control-point channels = a v4 item (RESEARCH §10.4).
- **v4 features** (need a re-preprocess+retrain, so batch): style/mapper
  conditioning (§10.1), slider-shape channels (§10.4). Inference-only (no
  retrain): multi-section BPM timing (§10.3), SR calibration baking (§10.2).

## 8. Hard-won lessons (don't re-learn these)

- **DDIM, not strided ancestral DDPM** — strided ancestral under-denoises to mush.
- **bf16 + attention diverges** without QK-norm + learned temperature + zero-init
  output proj. Even *with* QK-norm, **base 160 + bf16 diverged twice** (v2 @ e21,
  v3 @ e12) — a sudden loss spike (0.012 → 0.5 → ~1.0, stuck). **base 128 is the
  proven-stable size** (the v3 draft ran 60 epochs clean); it's now the default,
  with LR 1.2e-4 / grad-clip 0.3. If scaling to 160+, lower LR further (≤8e-5).
  Watch the loss curve; `best.pt` survives a late divergence (it's the pre-spike
  epoch). Divergence monitor must match `avg_loss 0.[3-9]` (stdout uses spaces,
  not `loss: 0.9`).
- **Slider duration comes from pixel length / SV**, not your stored end-time —
  `write_osu` clamps slider length to avoid time overlap (~19% overlapped before).
- **Frame grid, not ms** — keeps audio↔map aligned.
- **`.gitignore` must root-anchor** `/data/ /runs/ /artifacts/` — an unanchored
  `data/` once hid the whole `src/data/` package from git.
- **Star rating = the difficulty axis**; mappers' diff *names* are arbitrary.
- Windows console is cp1251 — avoid μ/σ/non-latin in `print` (use ASCII).
- The dataset mel cache must be a **module-level** `lru_cache` (per-instance
  lru_cache isn't picklable for num_workers>0 on Windows spawn).

## 9. Conventions

- Tests hermetic (synthetic fixtures, no GPU/dataset). Keep them green; ruff clean.
- Commit locally with descriptive messages (Co-Authored-By Claude). **Never push.**
- Heavy artifacts gitignored; runs are self-contained under `runs/<id>/`.
- Memory lives in `…/memory/` (MEMORY.md index + per-fact files). Update it.
