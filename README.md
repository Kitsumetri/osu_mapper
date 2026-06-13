# osu_mapper — audio → osu! beatmap generation

A diffusion-based pipeline that learns to generate **osu!standard** beatmaps
from raw audio, trained on a local osu! Songs library.

## Approach

Inspired by [osu-dreamer](https://github.com/jaswon/osu-dreamer): instead of
predicting a discrete token list, a beatmap is represented as a **frame-aligned
multi-channel signal** at the audio spectrogram's frame rate, and a **1D
conditional DDPM U-Net** denoises that signal conditioned on the mel
spectrogram. The generated signal is decoded back into discrete hit objects and
written to a valid `.osu` file.

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
# 1. preprocess a subset of the library into aligned (mel, signal) shards
python main.py preprocess --songs "C:/osu!/Songs" --out data/processed --limit 600

# 2. train the diffusion model
python main.py train --data data/processed --epochs 240 --batch 8 --crop 2048 --base 96

# 3. generate a .osu from any audio file (DDIM, ~100 steps)
python main.py generate --audio song.mp3 --ckpt checkpoints/model_last.pt --out generated.osu
```

Each stage is also runnable directly, e.g. `python -m src.train ...`.

## Project layout

```
src/
  config.py              audio/signal config + frame<->time helpers
  parsing/beatmap.py     robust .osu parser + writer (bitflags, sliders, spinners)
  data/
    audio.py             log-mel spectrogram extraction
    signal.py            beatmap <-> signal encode/decode (+ peak-pick decoder)
    preprocess.py        library crawler -> .npz shards
    dataset.py           random-crop torch Dataset
  model/
    unet.py              1D conditional U-Net (FiLM time embedding)
    diffusion.py         Gaussian DDPM schedule, DDPM + DDIM sampling
  train.py               training loop (AMP, checkpointing)
  generate.py            audio -> .osu inference
tests/                   hermetic pytest suite (no dataset/GPU needed)
main.py                  CLI dispatcher
```

## Development

```bash
pytest                 # 33 hermetic tests, ~5 s
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
- The original prototype's `uninherited = bool(str)` bug made every timing point
  "uninherited"; fixed and covered by a regression test.

## Roadmap / TODO

- [x] Robust `.osu` parser + writer (type bitflags, slider timing, spinners)
- [x] Beatmap ↔ signal encode/decode (near-lossless round-trip)
- [x] Audio feature pipeline + `.npz` shard preprocessing
- [x] Conditional diffusion U-Net + DDPM/DDIM sampling
- [x] End-to-end `audio → .osu` generation
- [x] Hermetic pytest suite
- [ ] **Scale data**: preprocess the full library (31k+ difficulties); dedupe
      shared audio across difficulties to cut storage.
- [ ] **Difficulty/length conditioning**: condition on star rating / CS / AR so
      generation is controllable.
- [ ] **Timing extraction**: estimate BPM + offset from audio so generated maps
      get real `[TimingPoints]` instead of a placeholder.
- [ ] **Slider shapes**: model curve control points (currently linear 2-point
      sliders on decode).
- [ ] **Rhythm snapping**: quantise generated onsets to beat subdivisions.
- [ ] **Eval metrics**: onset F1 vs ground truth, density/spacing histograms.
- [ ] EMA weights + cosine LR schedule for higher-quality samples.

## Prior art / credits

- [jaswon/osu-dreamer](https://github.com/jaswon/osu-dreamer) — signal + diffusion approach
- [OliBomby/osu-diffusion](https://github.com/OliBomby/osu-diffusion),
  [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) — DiT-style coordinate diffusion
- [gyataro/osuT5](https://github.com/gyataro/osuT5) — seq2seq event tokens
- [kotritrona/osumapper](https://github.com/kotritrona/osumapper) — earlier TF approach
- [osu! file format wiki](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
