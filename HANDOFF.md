# Agent handoff — osu_mapper

Full working context for a fresh agent. Read this first, then `README.md` (usage +
data/run layout), `TECH_REPORT.md` (math), `RESEARCH.md` (design + v5 plans),
`RESULTS.md` (run history).

## 1. What this is

Conditional **diffusion model** that generates osu!standard beatmaps from raw audio.
A beatmap → frame-aligned multi-channel **signal**; a 1D U-Net denoises it conditioned
on the audio's mel + a difficulty context vector; the signal decodes to a playable `.osu`.

User (`Kitsumetri`, GitHub): osu! player + ML dev, Windows / RTX 4070 Ti, Songs at
`C:\osu!\Songs` (~31.8k `.osu`). They test generated maps in-game and give play feedback.

## 2. Pipeline & architecture

```
audio.mp3 ─► log-mel (64×T) ──────┐
                                  ├─► 1D U-Net (DDIM, QK-norm attn) ─► signal (C×T) ─► decode ─► .osu
   noise (C×T) ──────────────────┘     ▲ cond: mel + difficulty ctx [SR,AR,OD,HP,CS,density]
```

- **Signal channels** (`src/config.py`): v4 = **10** (onset, slider_hold, spinner_hold,
  new_combo, cursor_x/y, kiai_hold, whistle, finish, clap). **v5 = 17** (+6 slider-anchor
  dx/dy + 1 slides). ~86 fps (sr 22050, hop 256). Encode/decode in `src/data/signal.py`.
- **Diffusion**: DDPM (1000 steps, linear β), ε-prediction; DDIM sampler + CFG (`diffusion.py`).
- **U-Net** (`unet.py`): base × (1,2,4,8), FiLM timestep emb, **QK-norm self-attention** at
  the `attn_levels` coarsest levels — do NOT remove QK-norm / learned temperature / zero-init
  proj (the bf16-divergence fix, §7).
- **Conditioning** (`conditioning.py`): difficulty vector added to the time emb; CFG (train
  drops it 15%, sample guides).
- **Star rating** (`difficulty.py`): exact via `rosu-pp-py` (don't reimplement).

## 3. Repo map

| Path | What |
|------|------|
| `src/config.py` | audio + signal channel config (10→17), frame↔time |
| `src/conditioning.py` | difficulty context vector + `target_context` |
| `src/difficulty.py` | star rating (rosu-pp) + SR bands |
| `src/parsing/beatmap.py` | `.osu` parser + `write_osu` (bitflags, sliders, kiai, hitsounds, breaks) |
| `src/data/signal.py` | `encode_beatmap`, `decode_signal` (+ v5 slider anchors/slides), `decode_kiai` |
| `src/data/osu_db.py` | parse `osu!.db` → ranked status; `ranked_osu_paths()` |
| `src/data/{audio,timing,preprocess,dataset}.py` | log-mel / BPM est / crawl→manifest / torch Dataset (+flip aug) |
| `src/model/{unet,diffusion}.py` | denoiser + DDPM/DDIM/CFG |
| `src/train.py` | training loop (bf16, EMA, cosine LR, CFG drop, val split, resume, `--compile`) |
| `src/generate.py` | audio→.osu (DDIM+CFG, `--sr`, `--guidance`, `--match-sr`) |
| `src/postprocess.py` | beat-snap, slider clamp, trim, `[Events]` breaks |
| `src/{metrics,corpus_stats,evaluate,package_map}.py` | metrics / reference dists / SR-sweep eval / package a Songs folder |
| `tests/` | 94 hermetic tests (no dataset/GPU) |
| `runs/<id>/`, `data/processed/<tag>/`, `artifacts/` | gitignored heavy outputs |

## 4. How to run (uv env: `uv run …` or activate `.venv`)

```bash
uv run pytest                            # 94 hermetic tests
uv run ruff check .
uv run python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/<tag> --ranked-only --workers 10
uv run python -m src.train --data data/processed/<tag> --tag <t> --base 128 --crop 4096 --attn-levels 3 --batch 16 --epochs 60 --save-every 5
#   base 128 is the proven-stable size — base 160 + bf16 DIVERGES (§7). Resume: --resume runs/<id>/ckpt/last.pt
uv run python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/last.pt --sr 5 --match-sr --out out.osu
uv run python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/last.pt --srs 2,3,4,5,6 --ref-stats artifacts/reference_stats.json
```

## 5. Current state (2026-06-14)

- **Released = v4b** (ranked model `runs/20260614-151630-ranked-full/ckpt/last.pt`, **10-ch**,
  epoch 48, val 0.00486). Trained on **ranked-only** data (osu!.db filter, ~23.6k maps) with
  crop 4096 + attn_levels 3 + flip augmentation. Eval: SR monotonic, **17–19/19 metrics
  in-range**. **v4 branch merged to `main`.** Packaged `[AI-v4b]`. In-game feedback →
  RESULTS.md / RESEARCH §10.4.
- **Active branch `feat/v5-slider-style`** (off v4): the **17-channel slider representation**
  (dedicated anchor dx/dy + `slides` channels; design in RESEARCH §10.3). Code DONE; dataset
  `data/processed/ranked-v5` DONE (23,626 items, 17-ch); **fresh train not yet run** (GPU-gated).
- **Env**: `uv` venv, **torch 2.11.0+cu128**. `--compile` flag wired but Windows-blocked
  (needs MSVC + triton-windows; §6). `artifacts/reference_stats.json` = 31,362-map reference.
- **Git**: I cannot push — the user pushes. Commit locally with descriptive messages.

## 6. Open queue (next work, on `feat/v5-slider-style`)

1. **Rhythm fix** — NEW top issue from v4b feedback (decode-side, no retrain): some notes
   off the ¼ grid (look 1/6·1/8) + strange 0.5–2 s pauses. A/B snap tolerance (60→45 ms) +
   add 1/6, 1/8 divisors; probe density/`onset_threshold`. Test on the v4b 10-ch ckpt (v4
   code is in `main`). RESEARCH §10.4.
2. **v5 fresh train** on `ranked-v5` (17-ch; GPU-gated) → eval + package `[AI-v5-sliders]`;
   checks whether the anchor channels fix curved + reverse sliders.
3. **adaLN-zero** conditioning (impl + A/B vs plain v5) — the contained DiT upgrade, RESEARCH §10.2.
4. **SR-offset bake** into `target_context` (§10.1.B); **density/break** control is model-side (§10.1.D).
5. **`torch.compile`** — ready behind `--compile`, blocked here (no MSVC/triton-windows); use on Linux/cloud.

## 7. Hard-won lessons (don't re-learn these)

- **DDIM, not strided ancestral DDPM** — strided ancestral under-denoises to mush.
- **bf16 + attention diverges** without QK-norm + learned temperature + zero-init output
  proj. Even *with* QK-norm, **base 160 + bf16 diverged twice** (sudden loss spike
  0.012→0.5→~1.0, stuck). **base 128 is the proven-stable size** (LR 1.2e-4, grad-clip 0.3);
  if scaling to 160+, lower LR (≤8e-5). `best.pt`/`last.pt` survive a late spike (pre-spike
  epoch). Divergence monitor must match `avg_loss 0.[3-9]` (stdout uses spaces).
- **Slider duration = pixel length / SV**, not the stored end-time — `write_osu` clamps slider
  length to avoid time overlap (~19% overlapped before).
- **Frame grid, not ms** — keeps audio↔map aligned.
- **`.gitignore` must root-anchor** `/data/ /runs/ /artifacts/` — an unanchored `data/` once
  hid the whole `src/data/` package from git.
- **Star rating = the difficulty axis**; mappers' diff *names* are arbitrary.
- Windows console is cp1251 — avoid μ/σ/non-latin in `print` (use ASCII).
- Dataset mel cache must be a **module-level** `lru_cache` (per-instance isn't picklable for
  `num_workers>0` on Windows spawn).
- **My background processes get reaped** when the session goes idle (~observed) — long trains
  must be run by the user in their own terminal; I handle eval/codegen that fits a turn.

## 8. Conventions

- Tests hermetic (synthetic fixtures, no GPU/dataset). Keep them green; ruff clean.
- `uv` env (`uv run …`). Commit locally, descriptive messages (Co-Authored-By Claude). **Never push.**
- Heavy artifacts gitignored; runs self-contained under `runs/<id>/` (keep `last.pt`, prune milestones).
- Memory lives in `…/memory/` (MEMORY.md index + per-fact files). Update it.
