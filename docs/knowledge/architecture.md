# Architecture & repo map

**Purpose:** what the system is, the end-to-end pipeline, the signal/channel layout at a glance,
the repo map, and how to run it. | **STATIC** (slow-changing; channel counts/paths update only on a
representation or refactor change).

## What this is

A conditional **diffusion model** that generates osu!standard beatmaps from raw audio. A beatmap is
turned into a frame-aligned multi-channel **signal**; a 1D U-Net denoises that signal conditioned on
the audio mel + a 6-D difficulty vector; the signal is decoded back to a playable `.osu`.

Target hardware: Windows / RTX 4070 Ti 12 GB; osu! Songs at `C:\osu!\Songs`.

## Pipeline

```
audio.mp3 ─► log-mel (64×T) ─────┐
                                 ├─► 1D U-Net (DDIM, QK-norm attn, adaLN) ─► signal (C×T) ─► decode ─► .osu
   noise (C×T) ─────────────────┘     ▲ cond: mel + difficulty ctx [SR,AR,OD,HP,CS,density]
```

The full math is in [diffusion-math.md](diffusion-math.md); the signal encode/decode spec is in
[signal-encoding.md](signal-encoding.md).

## Core components (one line each)

- **Signal channels** (`src/config.py`): v4=10, v5=17 (+6 slider-anchor dx/dy, +slides), v7=19
  (+`sv`, +`curve`), v7.5=20 (+`corner`), **v8=21** (+`spacing`). ~86 fps. Channel checks are
  index-based AND `generate.load_model` builds from the ckpt's own `sig_channels`, so **old ckpts
  still load**.
- **Diffusion** (`src/model/diffusion.py`): DDPM 1000 steps; **ε- or v-prediction** (`--objective`,
  v7+ uses `v`) + optional **zero-terminal-SNR**; DDIM sampler + CFG (+rescale); `loss_weight` =
  Min-SNR-γ.
- **U-Net** (`src/model/unet.py`): base 128 × (1,2,4,8), **adaLN-zero**, **QK-norm self-attention**
  (do NOT remove QK-norm/temperature/zero-init proj — the bf16-divergence fix), optional
  `--rope`/`--up-attn`, `--grad-checkpoint`. NB: `--up-attn` HURTS (averages → kills jump dispersion)
  — leave off (see [lessons-learned.md](lessons-learned.md)).
- **Star rating** (`src/difficulty.py`): exact via `rosu-pp-py` (don't reimplement).

## Repo map

| Path | What |
|------|------|
| `src/config.py` | signal channel config (10→21: `CH_SV` 17, `CH_CURVE` 18, `CH_CORNER` 19, `CH_SPACING` 20), frame↔time |
| `src/conditioning.py` | difficulty context vector + `target_context` (density overridable) |
| `src/difficulty.py` | star rating (rosu-pp) + SR bands |
| `src/parsing/beatmap.py` | `.osu` parser + `write_osu` (sliders, kiai, hitsounds, SV green lines) |
| `src/data/signal.py` | `encode_beatmap`, `decode_signal` (anchors, curve bow, red corners), `decode_kiai`, `decode_sv`, `_recover_stream_gaps` |
| `src/data/osu_db.py` | parse `osu!.db` → ranked status; `ranked_osu_paths()` |
| `src/data/{audio,timing,preprocess,dataset}.py` | mel / BPM est / crawl→manifest / torch Dataset (flip aug, channel-pad) |
| `src/model/{unet,diffusion}.py` | denoiser (adaLN, QK-norm, RoPE, up-attn, grad-ckpt) + DDPM/DDIM/CFG (ε/v, zero-SNR, min-snr) |
| `src/train.py` | training loop; flags `--objective/--zero-snr/--rope/--up-attn/--grad-checkpoint/--compile/--loss/--min-snr-gamma` |
| `src/generate.py` | audio→.osu; `--sr/--match-sr/--match-iter/--timing-from/--guidance/--guidance-rescale/--density/--onset-threshold`; `load_model`+`prepare_audio` reuse |
| `src/postprocess.py` | beat-snap, SV-aware slider snap, slider clamp, `trim_isolated_ends` (adaptive tail trim), breaks, `respace_by_magnitude`, `clamp_objects_to_playfield` |
| `src/{metrics,corpus_stats,evaluate,package_map}.py` | metrics (`curved_slider_ratio` etc.) / ref dists / SR-sweep eval / package a Songs folder |
| `src/eval/reward.py`, `src/eval/measure_reward.py` | "ranked-map" reward (family-balanced) + gold-data calibration tool |
| `src/best_of_n.py` | sample N, reward-rank, keep best (`<out>.bon.json` audit) |
| `src/rl/sample_logprob.py` | DDPO/DPOK per-step log-prob prototype (not wired into training) |
| `src/timing_model/` | **separate package** (BPM/offset beat tracker): `labels.py` + `metrics.py`. CPU foundation done; model/train pending |
| `src/eval/analyze_phase1.py` | real-vs-generated probe (curvature/spacing/flow/SV) — track per-version progress |
| `src/eval/eval_spacing_channel.py` | v8 spacing-channel probe (channel vs cursor spacing ratio) |
| `src/run_inference.py` | friendly `infer` pipeline: generate → stats → auto-package (one beatmapset folder); supports `--best-of-n` |
| `tests/` | hermetic tests (no dataset/GPU); 203 at last count |
| `runs/<id>/`, `data/processed/<tag>/`, `artifacts/` | git-ignored heavy outputs |

## How to run (`uv run …`)

```bash
uv run --extra dev pytest          # hermetic tests
uv run --extra dev ruff check .
# preprocess gold data -> 21-ch (--gold = ranked+kiai+single-BPM+hitsounds>=10%+1<SR<10)
uv run python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/<tag> --gold --workers 10
# train (USER runs; base 160 STABLE with v-pred+zero-snr; gnorm logged). v8 recipe:
uv run python -m src.train --data data/processed/<tag> --tag <t> --base 160 --crop 4096 \
    --attn-levels 3 --batch 16 --epochs 60 --save-every 5 --augment true --val-frac 0.02 --workers 8 \
    --objective v --zero-snr --compile --spatial-loss-weight 3   # NO --rope/--up-attn (they hurt jumps)
#   v-loss is O(0.05) (~100x eps) -> NOT comparable to v6's 0.003; judge by trend. Resume: --resume <ckpt>
# USER-FRIENDLY entry point (generate + stats + auto-package, multi-SR, auto bf16/low-mem for long songs):
uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 5 6 7   # [--ckpt auto, --best-of-n N, --no-package, --match-sr, --amp]
# lower-level:
uv run python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --sr 5 --match-sr --timing-from ref.osu --out out.osu
#   long song? add --amp (bf16, fixes OOM) [+ --no-batch-cfg]; speed: batched-CFG on by default, --compile opt-in
#   stream/jump levers: --density <n> (raise for streams), --onset-threshold, --guidance 3-4 (more extreme)
uv run python -m src.eval.analyze_phase1 --ckpt runs/<id>/ckpt/best.pt --label <name>   # real-vs-gen probe
uv run python -m src.package_map --generated out.osu --original ref.osu --prefix "[AI-v75]"
```

## Conventions

- Tests hermetic (synthetic, no GPU/dataset). Keep green + ruff clean after every change.
- `uv` env. Commit locally (descriptive, `Co-Authored-By: Claude`). **Never push** — the user pushes.
- Runs self-contained under `runs/<id>/`; prune milestone ckpts to best+last after picking best.
- Heavy data git-ignored. `.gitignore` must root-anchor `/data/ /runs/ /artifacts/`.
- Windows console cp1251 → ASCII in prints. Dataset mel cache = module-level lru_cache (Windows spawn).
