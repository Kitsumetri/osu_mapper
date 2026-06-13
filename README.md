# osu_mapper — audio → osu! beatmap generation

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
                               ├─►  1D U-Net (DDIM denoise)  ──►  signal (6×T)  ──► decode ──► .osu
   noise (6×T) ────────────────┘            ▲ conditioned on mel
```

### Signal representation (`src/data/signal.py`)

At ~86 frames/sec (sr 22050, hop 256 → 11.6 ms/frame) each beatmap becomes a
`(6, T)` array in `[-1, 1]`:

| ch | name         | encoding |
|----|--------------|----------|
| 0  | onset        | Gaussian bump at each circle/slider-head |
| 1  | slider_hold  | +1 during slider body, else -1 |
| 2  | spinner_hold | +1 during spinner, else -1 |
| 3  | new_combo    | Gaussian bump at new-combo objects |
| 4  | cursor_x     | object x normalised to [-1,1], interpolated |
| 5  | cursor_y     | object y normalised to [-1,1], interpolated |

Encode/decode is near-lossless on real maps (100% onset timing recall, exact
circle/slider/spinner counts on round-trip).

For the full mathematical model — the diffusion process, training objective,
DDIM sampling, the U-Net + QK-norm attention, and the decode math — see
[`TECH_REPORT.md`](TECH_REPORT.md).

## Install

With [uv](https://docs.astral.sh/uv/) (recommended) — the CUDA torch index is
already wired up in `pyproject.toml`:

```bash
uv sync --extra dev      # creates .venv and installs everything (incl. cu124 torch)
uv run pytest            # run anything inside the env with `uv run ...`
```

Or with pip:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. preprocess into a deduped, manifest-indexed dataset (see STORAGE.md)
python main.py preprocess --songs "C:/osu!/Songs" --out data/processed/std-v1 --limit 3000

# 2. train the diffusion model (logs + checkpoints under runs/<id>/)
python main.py train --data data/processed/std-v1 --tag std-v1-base160 \
    --epochs 120 --batch 12 --crop 3072 --base 160

# 3. generate a .osu from any audio file (DDIM, ~100 steps, uses EMA weights)
python main.py generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --out generated.osu
```

Each stage is also runnable directly, e.g. `python -m src.train ...`. Training
features: bf16, EMA, cosine LR + warmup, gradient accumulation, self-attention
U-Net. See `STORAGE.md` for the data/runs/artifacts layout.

> **Why train from scratch (not a pretrained model)?** The diffusion target is a
> bespoke 6-channel beatmap "signal" with no pretrained equivalent on HF/torch.
> A pretrained *audio* encoder (e.g. for the mel conditioning) could help later,
> but the denoiser itself must be trained for this representation.

## Project layout

```
src/
  config.py              audio/signal config + frame<->time helpers
  metrics.py             pattern/quality metrics (density, stream/jump, on-grid, ...)
  parsing/beatmap.py     robust .osu parser + writer (bitflags, sliders, spinners, kiai)
  data/
    audio.py             log-mel spectrogram extraction
    signal.py            beatmap <-> signal encode/decode (+ Bezier-slider decoder)
    timing.py            BPM + beat-offset estimation (librosa)
    preprocess.py        library crawler -> deduped mels + items + manifest.json
    dataset.py           manifest-indexed, mel-deduped torch Dataset
  model/
    unet.py              1D conditional U-Net (FiLM time emb + QK-norm attention)
    diffusion.py         Gaussian DDPM schedule, DDPM + DDIM sampling
  train.py               training loop (bf16, EMA, cosine LR, runs/ logging)
  generate.py            audio -> .osu inference (DDIM, EMA, estimated timing)
  package_map.py         build a playable osu! Songs folder from a generated map
tests/                   hermetic pytest suite (no dataset/GPU needed)
main.py                  CLI dispatcher (preprocess | train | generate)
```

## Development

```bash
pytest                 # 45 hermetic tests, ~6 s
ruff check .           # lint
ruff format .          # format
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
- **bf16 + attention can diverge.** Plain dot-product attention blew up the
  scaled run at epoch 21; fixed with **QK-normalisation + a learnable temperature
  + zero-init output projection**.
- The original prototype's `uninherited = bool(str)` bug made every timing point
  "uninherited"; fixed and covered by a regression test.
- An unanchored `data/` gitignore rule once hid the whole `src/data/` package —
  artifact ignores are now anchored to the repo root (`/data/`, `/runs/`, ...).

## Roadmap / TODO

Done:

- [x] Robust `.osu` parser + writer (type bitflags, slider timing, spinners, kiai)
- [x] Beatmap ↔ signal encode/decode (near-lossless round-trip)
- [x] Deduped, manifest-indexed preprocessing (mel-per-audio + per-difficulty
      signal + metadata for difficulty/style/kiai conditioning)
- [x] Conditional diffusion U-Net + DDPM/DDIM sampling
- [x] End-to-end `audio → .osu` generation + playable-folder packaging
- [x] Hermetic pytest suite (45 tests) + ruff + uv
- [x] Timing estimation v1 (`data/timing.py`: BPM + offset via librosa)
- [x] Curved Bezier sliders (encode slider shape into the cursor signal, decode
      a multi-point Bezier from the cursor path)
- [x] Eval metrics (`src/metrics.py`: density, stream/jump, spacing, on-grid)
- [x] Scaled, stable training: 97M-param attention U-Net, bf16, EMA, cosine LR,
      `runs/` logging; QK-norm fixes attention divergence

Done (cont.):

- [x] Rhythm snapping v1 (`postprocess.py`: bounded 1/4-grid beat-snap;
      on-¼-grid 0.70 → 0.82 on the sample). Triplet/per-section snapping still
      open.

Next (roughly in priority order):

- [ ] **Difficulty conditioning**: condition on AR/OD/HP/CS + density so one
      model targets a chosen difficulty (manifest already stores these).
- [ ] **Style / mapper conditioning**: condition on `Creator` or a derived style
      class (farm-aim / stream / tech / alt) — see `RESEARCH.md §5`.
- [ ] **Kiai channel**: 7th signal channel `kiai_hold` to ramp density in
      choruses (`RESEARCH.md §7`).
- [ ] **Variable-BPM / multi-section timing** on output (26% of maps); downbeat
      tracking for better timing accuracy (`RESEARCH.md §6`).
- [ ] **Scale data** to the full library (31k+ difficulties).
- [ ] **Hitsounds** (whistle/finish/clap accent channel) — lowest priority.

See `RESEARCH.md` for the detailed plan behind each item.

## Prior art / credits

- [jaswon/osu-dreamer](https://github.com/jaswon/osu-dreamer) — signal + diffusion approach
- [OliBomby/osu-diffusion](https://github.com/OliBomby/osu-diffusion),
  [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) — DiT-style coordinate diffusion
- [gyataro/osuT5](https://github.com/gyataro/osuT5) — seq2seq event tokens
- [kotritrona/osumapper](https://github.com/kotritrona/osumapper) — earlier TF approach
- [osu! file format wiki](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
