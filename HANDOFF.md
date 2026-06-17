# Agent handoff — osu_mapper

**This file is the entry point for a fresh agent** — read it first, then `README.md`
(usage + data/run layout), `TECH_REPORT.md` (math), `RESEARCH.md` (design + roadmap),
`RESULTS.md` (run history). These docs are the source of truth.

Work autonomously: write code, run it (`uv run …`), fix errors, iterate. Be honest about
quality — state metrics, don't oversell. **You cannot push**; commit locally with
descriptive messages (`Co-Authored-By: Claude`), the user pushes + PRs to main. Trust the
user's (`Kitsumetri`) in-game play/mapping feedback. **Next concrete step:** train the
**v7 Phase 2** model (`--objective v --zero-snr`, base 128) and eval vs the Phase-1
pattern baselines (RESEARCH §10.7); v6 is the current candidate awaiting that comparison.

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
  dx/dy + 1 slides). **v7 = 19** (+1 `sv` timeline `CH_SV`, +1 `curve` cue `CH_CURVE`). ~86 fps (sr 22050,
  hop 256). Encode/decode in `src/data/signal.py`; channel checks index-based so old ckpts load.
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
| `src/generate.py` | audio→.osu (DDIM+CFG, `--sr`/`--guidance`/`--match-sr`/`--match-iter`/`--timing-from`); `load_model`+`prepare_audio` helpers let an SR sweep reuse one checkpoint/audio load |
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
#   v7 Phase 2 objective: add `--objective v --zero-snr` (v-loss is O(1), ~100x eps; LR/clip may want retuning)
uv run python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/last.pt --sr 5 --match-sr --out out.osu
uv run python analyze_phase1.py   # real-vs-generated curvature/spacing/SV probe (track v7 progress)
uv run python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/last.pt --srs 2,3,4,5,6 --ref-stats artifacts/reference_stats.json
```

## 5. Current state (2026-06-16)

- **Released = v5** (`runs/20260614-224107-ranked-v5/ckpt/best.pt`, **17-ch**, epoch 55,
  val 0.0033). 17-channel slider representation (anchor dx/dy + `slides`) on ranked-v5 data,
  crop 4096 / attn_levels 3 / flip aug. **Curved sliders + reverse sliders work.** Many
  decode fixes shipped from two play-test rounds (RESEARCH §10.5): rhythm snap to {1/4,1/8,1/6},
  slider RDP (real lines vs imposter-curves), realistic AR (`7.75+0.25·sr`), intro-cluster trim,
  spinner merge, `generate --timing-from <ref.osu>` (exact BPM/offset), package_map keeps
  generated difficulty. In-game: "way better, fixes helped, kiai generates." Branch
  `feat/v5-slider-style` — **user will push + PR to main.**
- **v6 TRAINED (2026-06-16)** on branch `feat/v6-sv-adaln`: `runs/20260616-013932-ranked-v6/
  ckpt/best.pt`, epoch 59, **val 0.00314**, clean (no divergence, ~330 s/epoch, 66.1 M params).
  v6 = **adaLN-zero** (`--adaln` default on) + **gold data** (`data/processed/ranked-v6`,
  25,073 maps, 100% kiai + single-BPM + hitsounds≥10%, SR 1.1–10, 17-ch). Eval: SR monotonic,
  in-range 14–17/19, curved sliders solid; vs v5 metrics ~a wash but SR calibration tighter
  (RESULTS v6). Packaged `[AI-v6]` (926 obj). **Per-slider SV reverted** (structural, §10.6.A).
- **v7 ACTIVE = "patterns"** (RESEARCH §10.7). v6 play feedback: patterns are now the #1
  issue (beginner jumps/streams), + persistent weak hitsounds, fluctuating kiai, too-straight
  sliders. **Phase 1 finding:** patterns + straight-sliders = one root cause, **under-dispersion
  from ε-MSE** (flow angles already ≈ real → attention is *not* the bottleneck). **Phase 2 DONE**
  (v-pred + zero-terminal-SNR; trained `runs/20260617-001225-v7-vpred`, partial win — spacing/
  jumps toward real, but variety/streams/curvature flat → P4 B+C justified). **Phase 4-A+C DONE
  (code):** SV channel (`CH_SV`, `decode_sv`→stable green lines) + curvature cue (`CH_CURVE`,
  decode bows to the cue → visible curves, target 38-45%). 17→**19 ch**. 17-ch ckpts still load
  (cue/SV dormant). **P4-B (flow/Δpos) HELD** pending `[AI-v7]` play-test (low-confidence aux).
  **P3 attention upgrade DONE (code, 6459ee7):** `--rope --up-attn --grad-checkpoint` (backward-
  compatible). v7-draft memory probe: baseline 5.30 GB, +rope+up_attn 9.83, +grad_ckpt 5.02;
  full-res attn4 OOMs (not viable). **Next: reprocess `gold-v7` (19-ch) + train** with
  `--objective v --zero-snr --rope --up-attn` (user). Track with `analyze_phase1.py --ckpt …`.
- **Env**: `uv` venv, **torch 2.11.0+cu128**. `--compile` wired but Windows-blocked (no MSVC).
  `data/processed/ranked-v5` (17-ch) on disk; `artifacts/reference_stats.json` = 31,362-map ref.
- **Git**: I cannot push — the user pushes. Commit locally with descriptive messages.

## 6. v6 batch — the next retrain (branch `feat/v6-sv-adaln`; design RESEARCH §10.6)

One re-preprocess + fresh train bundling the model-side wins that v5's decode fixes can't reach:
1. **Gold data** — `preprocess --gold` (ranked + kiai + single-BPM + hitsounds≥10% + 1<SR<10;
   code DONE, manifest fields added). User refreshed `osu!.db` + added maps. *Re-preprocess
   with the v6 encoding (below) when the channels are final.*
2. ❌ **Slider velocity (SV)** — decode-side per-slider attempt REVERTED (too many SV sections;
   SV is structural like kiai, not per-slider geometry). Proper SV = learned/structural, future
   v6+ (RESEARCH §10.6.A).
3. ✅ **adaLN-zero conditioning** — DONE (`--adaln`, default on). DiT per-block scale/shift/gate.
4. ✅ **Gold data** — `preprocess --gold` (17-ch) → `ranked-v6` DONE. **Fresh train** (adaLN)
   DONE 2026-06-16 (`runs/20260616-013932-ranked-v6`, val 0.00314). Eval + `[AI-v6]` packaged
   (RESULTS v6). *Remaining: in-game play test → promote v6 to release if it beats v5.*

Carry-over / parallel: SR-offset bake (§10.1.B), density conditioning for the 0.6–0.8 s gaps,
a real timing model for novel songs ("super timing", §10.2), flow/pattern modelling (§10.2).

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
