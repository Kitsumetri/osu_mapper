# Agent handoff — osu_mapper

Full working context for a fresh agent session. Read this first, then
`README.md` (usage), `TECH_REPORT.md` (the math), `RESEARCH.md` (design + plans),
`RESULTS.md` (run history). Data/run layout is in `README.md` ("Data & run layout").

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
| `tests/` | 86 hermetic tests (no dataset/GPU) |
| `runs/<id>/` | training runs (config.json, metrics.csv, ckpt/) — gitignored |
| `data/processed/<tag>/` | preprocessed datasets — gitignored |
| `artifacts/` | generated maps, reference_stats.json — gitignored |

## 4. How to run

```bash
pytest                                   # 86 tests, ~6 s
ruff check . && ruff format .
python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/std-v3-full --limit 6000
python -m src.train --data data/processed/std-v3-full --tag std-v3-heavy --base 160 --crop 3072 --epochs 150
python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --sr 4.5 --guidance 2 --match-sr --out out.osu
python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --srs 2,3,4,5,6
python -m src.metrics --osu out.osu --ref-stats artifacts/reference_stats.json
python -m src.corpus_stats --songs "C:/osu!/Songs"   # rebuild reference_stats.json (all maps)
```

## 4b. NEXT SESSION — v5 ranked-data + context experiment (prep DONE, train SCHEDULED)

User direction (2026-06-14): train on **ranked maps only** (quality), give the
model **more context** (attention/crop), then draft→debug→full train. They asked
to start the heavy work **~1 h later** (machine in use) — a wakeup is scheduled.

**Prep committed this session (no training run yet):**
- `src/data/osu_db.py` — parses `osu!.db` (validated: consumes the real 32 MB DB
  to the exact byte). `ranked_osu_paths(songs, db)` → **23,825 ranked/approved/
  loved std maps on disk** (vs the v4 set which included 4.3k graveyard + 1.2k
  unsubmitted). 3 hermetic tests (synthetic DB).
- `preprocess --ranked-only [--osu-db PATH]` — joins osu!.db, keeps only
  ranked/approved/loved. Default db path `<songs>/../osu!.db`.
- `train --resume runs/<id>/ckpt/last.pt` — saves optimizer+gstep+best, appends
  metrics, continues the LR schedule in place. **Crash-resilient** (machine slept
  mid-train twice). Old checkpoints lack opt state (resume still works, opt warm-starts).
- `UNet1d(attn_levels=N)` + `train --attn-levels N` — self-attention at the N
  deepest levels (default 2; **3 = finer-resolution pattern context**). `generate`
  reads it from ckpt args (old ckpts → 2, back-compat verified).
- Decode: `trim_isolated_ends` now drops a lone trailing **circle right after the
  final spinner** (phantom spin-down onset, recurring fb "auto can't hit it").

**Execution steps (run after the wakeup fires):**
1. Draft preprocess (subset): `python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/ranked-draft --ranked-only --limit 2500 --workers 16`
2. Draft train (~12 ep, more context): `python -m src.train --data data/processed/ranked-draft --tag ranked-draft --base 128 --crop 4096 --attn-levels 3 --batch 16 --epochs 12 --save-every 4` — **watch divergence** (`avg_loss 0.[3-9]`) and VRAM (drop `--batch` to 12 / `--crop` to 3072 if OOM). Generate a sample; eyeball streams/sliders.
3. If stable → full ranked preprocess: same cmd, `--out data/processed/ranked-full`, no `--limit`.
4. Full train: `... --data data/processed/ranked-full --tag ranked-full --base 128 --crop 4096 --attn-levels 3 --batch 16 --epochs 60 --save-every 5` (use `--resume .../last.pt` after any sleep/crash).
5. Eval (`evaluate.py` SR sweep), package `[AI-v5]`, update RESULTS + memory + §5.

**Design notes:** hop length is deliberately NOT increased — coarser hop = worse
stream timing precision (the opposite of the fix). More context comes from larger
`--crop` + `attn_levels`. base stays 128 (160+bf16 diverges, §8). The straight-
slider issue (60-80% are lines) is mostly a **representation** gap — dedicated
slider-shape channels (RESEARCH §10.1.F); ranked data + more training helps only
partially. That's the next re-preprocess batch, not this one.

## 5. Current state (2026-06-14)

- **Released v4 model = `runs/20260614-110223-std-v4-full/ckpt/best.pt`** (base 128,
  31,270 curated maps, epoch 15, loss 0.0077; undertrained but strong). Difficulty
  conditioning + CFG + `--match-sr`. All v4 decode/post-process wins shipped
  (slider clamp, hitsound 0.85, trailing-spinner trim, looser snap, `[Events]`
  breaks). See RESULTS.md for full history.
- **RUNNING: the final v4/v5 "ranked" train** — trained on **ranked-only data**
  (`data/processed/ranked-full`, ~23.8k ranked/approved/loved maps via the
  `osu!.db` filter) with **more context** (`--crop 4096 --attn-levels 3`) + **h/v
  flip augmentation** + train/val split + `train.log`. A 12-epoch draft on the
  ranked subset was clean (loss 0.53→0.022, no divergence, fits VRAM). See §4b.
- **Reference** `artifacts/reference_stats.json`: 31,362 maps, bucketed by SR.
- **Git**: v3 merged to `main`; active branch `feat/v4-fulldata`. **I cannot push
  — the user pushes**. Commit locally with descriptive messages.

## 5b. Open TODOs (the live task list is session-scoped — these are the durable copy)

### Blocked queue (2026-06-14) — waiting on GPU / the live training to finish
These can't run while the 10-ch ranked train holds the GPU + venv:
1. **Eval + package the 10-ch ranked model** ("final v4") once training ends — on
   `feat/v4-fulldata`: best-val epoch, SR sweep, package `[AI-v5]`, write RESULTS (§6).
2. **Prune `ranked-full` milestone ckpts** (epoch_N.pt, ~8 GB) after picking best.
3. **Update torch** 2.6→2.11+/cu128 (venv is DLL-locked by the run): bump pyproject,
   `uv lock --upgrade` + `uv sync` + `pytest` + generate smoke. (librosa, no torchaudio.)
4. **Test + wire `torch.compile`** (GPU-free): verify Windows/Triton works; if so add a
   `--compile` flag handling the `_orig_mod.` state_dict prefix (else resume/generate break).
5. **v5 fresh train** on `data/processed/ranked-v5` (17-ch) on `feat/v5-slider-style`
   (code DONE, commit 272c736; preprocess running) — then eval/package `[AI-v5-sliders]`.
6. **adaLN-zero** conditioning (impl + fresh train; A/B vs plain v5) — RESEARCH §10.2.
7. **SR-offset bake** into `target_context` (needs a finished model's SR sweep; §10.1.B).

### Standing
1. **Ranked train** (the final v4/v5 run) — in progress; see §4b + §6.
2. **Cheap post-train wins** (no retrain, decode/postprocess):
   - ✅ **DONE (2026-06-14)** — clamp slider tails to the playfield
     (`postprocess.clamp_slider_endpoints`, fb #1); tighten trailing trim
     (`trim_isolated_ends` trailing 2.2 s < leading 3.0 s, fb #7); loosen onset
     snap (`snap_to_grid` 60 ms/50 %, fb #5); hitsound-threshold tune
     (`decode_signal(accent_threshold=0.85)` → 0.33 usage, §10.1.C); `[Events]`
     breaks (`compute_breaks` + `write_osu(breaks=…)`, §10.1.D-iii). See
     RESULTS.md "v4 decode/post-process wins". *Not yet validated in-game by the
     user — package a fresh sample for them to test.*
   - **Still open**: SR-offset bake into `target_context` (§10.1.B, needs an
     `evaluate.py` sweep); real density/break control is **model-side** (§10.1.D-i/ii)
     — the `[Events]` writer only marks existing gaps, it doesn't make a dense map
     sparser (dense songs still produce 0 breaks).
3. **v5 slider-shape + style batch** (CHOSEN 2026-06-14; branch `feat/v5-slider-style`
   off `feat/v4-fulldata`) — **full design in RESEARCH §10.3**: dedicated K=3
   slider-anchor "hold-box" channels + `slides` (reverse-slider) channel + coarse
   style-class conditioning; channels 10→17; fresh train. Workflow: analysis (done)
   → doc/memory sync (done) → implement code + hermetic tests (no GPU) → gate on the
   running ranked run's eval → re-preprocess `ranked-v5` → fresh train.
4. **v5**: flow/distance-snap pattern modelling, multi-section BPM timing, learned
   kiai/break segmentation (RESEARCH §10.2). **Arch (2026 survey)**: keep MHSA+QK-norm
   + U-Net long skips; try **adaLN-zero** conditioning (contained DiT win). **Perf**:
   FlashAttention-2 build only worth it *if* v5 goes DiT/attention-heavy (FA3 is
   Hopper-only; our Ada GPU + Windows wheel already uses fused cuDNN SDPA, so the
   gain is single-digit % for the conv U-Net). See RESEARCH §10.2.
5. Optional: parallelise `corpus_stats` like `preprocess` if re-run often.

## 6. When the ranked train finishes — do this

1. Check `runs/<id>/metrics.csv` (now has `val_loss`) + `train.log` — no divergence
   (watch `avg_loss 0.[3-9]`); note best val loss. If killed by sleep, resume:
   `python -m src.train --resume runs/<id>/ckpt/last.pt` (continues in place).
2. `python -m src.evaluate --audio <real audio.mp3> --ckpt runs/<id>/ckpt/best.pt --srs 2,3,4,5,6`
   — confirm SR tracks target; compare metrics to the v4 release (expect cleaner
   patterns/streams from ranked data + flip aug + more context).
3. Package a sample (`src/package_map.py`, prefix `[AI-v5]`) for the user to test.
4. Write results into `RESULTS.md` + memory; update §5 above.
5. Then the v5 representation batch (RESEARCH §10.1.E/F): slider-shape + repeat
   channels (the real fix for straight/reverse sliders) + style/mapper conditioning.
   See [[reference-mapperatorinator]] for the validated approach + cheap-vs-heavy
   triage.

## 7. Known issues / next — see RESEARCH §10 for the full v4/v5 plan

From play feedback (mostly model/conditioning, not decode):
- **No breaks** — model is too dense (no gaps >4 s). Density/break control (v4
  §10.1.D): condition density harder, or suppress onsets in low-mel-energy
  sections, or write `[Events]` breaks for big gaps.
- **Kiai lags the drop ~10–12 s, ~1/3 coverage** — channel alignment; more data +
  downbeat-snap kiai edges (v4).
- **Odd circle placement in spots** — pattern quality; flow/DS modelling (v5 §10.2).
- **Streams slightly low, SR drift at extremes** — undertrained tails → more data (v4 §10.1.A).
- **Hitsounds slightly high** (~0.5 vs real 0.33) → raise accent decode threshold (cheap, v4 §10.1.C).
- **SR offset** — `--match-sr` corrects at inference; bake into `target_context` after eval (§10.1.B).

v4 representation items (batch into one re-preprocess+retrain): **style/mapper
conditioning** (§10.1.E), **slider-shape channels** (§10.1.F).

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
