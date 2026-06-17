# osu_mapper — audio → osu! beatmap generation

> **Continuing this project? Start with [`HANDOFF.md`](HANDOFF.md).**

A diffusion-based pipeline that learns to generate **osu!standard** beatmaps
from raw audio, trained on a local osu! Songs library.

## Approach

Inspired by [osu-dreamer](https://github.com/jaswon/osu-dreamer): instead of
predicting a discrete token list, a beatmap is represented as a **frame-aligned
multi-channel signal** at the audio spectrogram's frame rate, and a **1D
conditional DDPM U-Net** (with self-attention at the coarse levels for
long-range structure) denoises that signal conditioned on the mel spectrogram.
The generated signal is decoded back into discrete hit objects — including
curved Bezier sliders read from the cursor path — and written to a valid `.osu`
file.

```
audio.mp3 ──► log-mel (64×T) ──┐
                               ├─►  1D U-Net (DDIM denoise)  ──►  signal (C×T)  ──► decode ──► .osu
   noise (C×T) ────────────────┘     ▲ conditioned on mel + difficulty (SR/AR/OD/HP/CS/density)
```
(C = 19 signal channels on v7, 17 on v5, 10 on v4 — see below.)

### Signal representation (`src/data/signal.py`)

At ~86 frames/sec (sr 22050, hop 256 → 11.6 ms/frame) each beatmap becomes a
`(C, T)` array in `[-1, 1]`. **v4 = 10 channels**: onset, slider_hold,
spinner_hold, new_combo, cursor_x/y, kiai_hold, whistle/finish/clap. **v5 = 17**:
adds 6 slider-anchor `dx/dy` channels (control-point offsets held over the slider
span) + a `slides` channel, so slider shape and reverse sliders are first-class
rather than read off the noisy cursor path (see [`TECH_REPORT.md`](TECH_REPORT.md) §3.2).
**v7 = 19**: adds an `sv` slider-velocity timeline and a per-slider `curve` cue.
Channel checks are index-based (and the loader uses each checkpoint's own channel
count), so older 17-ch checkpoints still load under the 19-ch build.

Difficulty is supplied as an **input context vector** `[SR, AR, OD, HP, CS,
density]` (conditioning, not a channel).

Encode/decode is near-lossless on real maps (100% onset timing recall, exact
circle/slider/spinner counts on round-trip).

For the full mathematical model — the diffusion process, training objective,
DDIM sampling, the U-Net + QK-norm attention, and the decode math — see
[`TECH_REPORT.md`](TECH_REPORT.md).

## Install

With [uv](https://docs.astral.sh/uv/) (recommended) — the CUDA torch index is
already wired up in `pyproject.toml`:

```bash
uv sync --extra dev      # creates .venv and installs everything (incl. cu128 torch)
uv run pytest            # run anything inside the env with `uv run ...`
```

Or with pip:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. preprocess ranked maps into a deduped, manifest-indexed dataset
uv run python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/ranked --ranked-only --workers 10

# 2. train (logs + checkpoints under runs/<id>/). base 128 is the stable default —
#    base 160 + bf16 diverges (see gotchas). --resume runs/<id>/ckpt/last.pt to continue.
uv run python -m src.train --data data/processed/ranked --tag mymodel \
    --base 128 --crop 4096 --attn-levels 3 --batch 16 --epochs 60 --save-every 5

# 3. generate a .osu at a target star rating (DDIM + CFG, EMA, kiai + hitsounds, snapped)
#    --match-sr iterates to hit the SR exactly; --timing-from <ref.osu> uses exact BPM/offset
uv run python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/last.pt \
    --out generated.osu --sr 5 --match-sr --guidance 2.0
```

Training features: bf16, EMA, cosine LR + warmup, grad accumulation, self-attention
U-Net, flip augmentation, train/val split, per-run logging, `--resume`. See "Data &
run layout" below.

> **Why train from scratch (not a pretrained model)?** The diffusion target is a
> bespoke multi-channel beatmap "signal" with no pretrained equivalent on HF/torch.
> A pretrained *audio* encoder (e.g. for the mel conditioning) could help later,
> but the denoiser itself must be trained for this representation.

## Project layout

```
src/
  config.py              audio/signal config (10->17 channels) + frame<->time helpers
  conditioning.py        difficulty context vector [SR,AR,OD,HP,CS,density] + target_settings
  difficulty.py          exact star rating via rosu-pp + SR bands
  metrics.py             pattern/quality metrics (density, stream/jump, on-grid, ...)
  corpus_stats.py        reference distributions over the real library
  evaluate.py            SR-sweep eval vs reference stats
  postprocess.py         beat/slider snap, slider clamp, trim, [Events] breaks
  package_map.py         build a playable osu! Songs folder from a generated map
  parsing/beatmap.py     robust .osu parser + writer (bitflags, sliders, spinners, kiai)
  data/
    audio.py             log-mel spectrogram extraction
    signal.py            beatmap <-> signal encode/decode (+ Bezier-slider decoder)
    timing.py            BPM + beat-offset estimation (librosa)
    osu_db.py            parse osu!.db -> ranked status / gold-filter inputs
    preprocess.py        library crawler -> deduped mels + items + manifest.json
    dataset.py           manifest-indexed, mel-deduped torch Dataset (+ flip aug)
  model/
    unet.py              1D conditional U-Net (adaLN, QK-norm attn, optional RoPE/up-attn)
    diffusion.py         Gaussian DDPM schedule, DDPM + DDIM + CFG sampling
  train.py               training loop (bf16, EMA, cosine LR, runs/ logging)
  generate.py            audio -> .osu inference (DDIM+CFG, EMA, --match-sr, --timing-from)
tests/                   116 hermetic pytest tests (no dataset/GPU needed)
main.py                  CLI dispatcher (preprocess | train | generate)
```

### Data & run layout

Everything heavy is git-ignored (root-anchored `/data/`, `/runs/`, `/artifacts/`);
only code + small configs are tracked.

```
data/processed/<tag>/          # one preprocessing run
  mels/<audio_id>.npy          # log-mel per *audio file* (deduped across difficulties)
  items/<item_id>.npz          # signal per difficulty
  manifest.json                # index: every item + metadata (creator, cs/ar/od/hp,
                               #   bpm, has_kiai, star_rating, frames, ...)
runs/<run_id>/                 # one training run (run_id = YYYYmmdd-HHMMSS-tag)
  config.json                  # full args + git commit + dataset size + n_params
  metrics.csv                  # epoch, avg_loss, val_loss, lr, sec
  train.log                    # full stdout/stderr transcript (prints, warnings)
  ckpt/{last,best,epoch_N}.pt  # {model, ema, opt, gstep, best, args, epoch, git_commit}
artifacts/                     # exported, shareable outputs (generated/packaged maps)
```

- **Mel dedup**: a song's many difficulties share one audio, so the mel is stored
  once per `audio_id` (~4–5× fewer big arrays at full-library scale).
- **Manifest** lets training filter/sample and compute stats without opening every
  shard. **`best.pt`** keeps the lowest val loss (EMA weights are used at inference).
- Reproduce a run from `runs/<id>/config.json` (pins git commit + args).

## Development

```bash
uv run pytest          # 116 hermetic tests
uv run ruff check .    # lint
```

The test suite is hermetic — it builds tiny synthetic `.osu`/`.npz` fixtures in
a temp dir and never touches the real Songs library or a GPU. Lint/format rules
live in `pyproject.toml` (`E,F,I,UP,B,SIM`, 100-col).

## Notes & gotchas (learned while building)

- **Sampler matters.** Naive *strided* ancestral DDPM under-denoises and yields
  dense, noise-like maps. Use full DDPM or, for speed, the **DDIM** sampler in
  `diffusion.py` (correct over a step subsequence).
- **Frame grid, not milliseconds.** Everything is aligned on the audio frame
  grid to avoid timing drift between audio and hit objects.
- **Slider duration comes from length, not your end-time.** osu! derives a
  slider's duration from `pixel_length / slider_velocity`, so `write_osu` clamps
  each slider's length to fit the gap before the next object (otherwise ~19% of
  generated objects overlapped in time).
- **bf16 + attention can diverge.** Fixed plain dot-product attention with
  **QK-normalisation + learnable temperature + zero-init output projection** — but
  even then **base 160 diverged twice** (sudden loss spike, ~e12–21). **base 128
  is the stable default** (lr 1.2e-4, grad-clip 0.3); `best.pt` survives a late
  spike. Watch the loss; monitor on `avg_loss 0.[3-9]`.
- **Slider ends must be snapped too.** Beat-snap only moves onsets; a slider's
  duration (= length/velocity) lands off-grid → `postprocess.snap_slider_ends`
  snaps it (55%→0% off-grid). `package_map` must keep the *generated*
  SliderMultiplier or it rescales durations and un-snaps them.
- The original prototype's `uninherited = bool(str)` bug made every timing point
  "uninherited"; fixed and covered by a regression test.
- An unanchored `data/` gitignore rule once hid the whole `src/data/` package —
  artifact ignores are now anchored to the repo root (`/data/`, `/runs/`, ...).

## Roadmap / TODO

**Shipped:** robust `.osu` parser/writer; near-lossless signal encode/decode;
deduped manifest preprocessing; conditional diffusion U-Net (base 128, QK-norm
attn, bf16, EMA) + DDPM/DDIM/CFG; end-to-end `audio → .osu` + packaging; difficulty
conditioning + adaLN-zero; kiai + hitsound channels; curved + reverse sliders;
beat/slider snap; eval harness (`metrics`/`corpus_stats`/`evaluate` + `--match-sr`);
ranked-only/gold data (osu!.db filter) + flip aug; v-prediction + zero-terminal-SNR;
learned SV channel + curvature cue; RoPE / up-path attention / grad-checkpointing.
**Releases: v5** (17-ch). v6 (adaLN + gold) and v7-Phase2 (v-pred) trained; **v7**
("patterns": 19-ch SV+curve + attention) code done, reprocess+train pending.

**Next** — see `RESEARCH.md §10.7` (full plan), `RESULTS.md` (history), `HANDOFF.md §6`
(queue): the v7 reprocess+train + play-test; then flow/Δpos channels (P4-B),
kiai segmentation head, hitsound musicality, and a BPM/offset model for novel songs.

## Prior art / credits

- [jaswon/osu-dreamer](https://github.com/jaswon/osu-dreamer) — signal + diffusion approach
- [OliBomby/osu-diffusion](https://github.com/OliBomby/osu-diffusion),
  [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) — DiT-style coordinate diffusion
- [gyataro/osuT5](https://github.com/gyataro/osuT5) — seq2seq event tokens
- [kotritrona/osumapper](https://github.com/kotritrona/osumapper) — earlier TF approach
- [osu! file format wiki](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
