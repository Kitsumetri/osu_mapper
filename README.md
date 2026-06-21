# osu_mapper — generate osu! beatmaps from audio

A diffusion model that turns **raw audio into playable osu!standard beatmaps**. Point it at a
song, pick a star rating, and it writes a `.osu` you can drop straight into osu! and play.

## ▶ Showcase

[![osu_mapper showcase](https://img.youtube.com/vi/iMc9QJ8uyQM/hqdefault.jpg)](https://youtu.be/iMc9QJ8uyQM)

*(click to watch generated maps played in-game — https://youtu.be/iMc9QJ8uyQM)*

## How it works

A beatmap is represented as a **frame-aligned multi-channel signal** at the audio's spectrogram
frame rate, and a **1D conditional diffusion U-Net** (DDIM sampling, QK-norm attention) denoises
that signal conditioned on the mel spectrogram + a difficulty vector. The signal is decoded back
into hit objects — circles, curved sliders, spinners, hitsounds, kiai, slider-velocity — and
written to a valid `.osu`. (The full math is in [`TECH_REPORT.md`](TECH_REPORT.md).)

```
audio.mp3 ──► log-mel (64×T) ──┐
                               ├─►  1D U-Net (DDIM denoise)  ──►  signal (21×T)  ──► decode ──► .osu
   noise (21×T) ───────────────┘     ▲ conditioned on mel + difficulty [SR,AR,OD,HP,CS,density]
```

The current model (**v8**, base-160) generates rhythm, jumps, streams, curved + reverse sliders,
red-corner sliders, slider-velocity sections, kiai and hitsounds. See [`RESULTS.md`](RESULTS.md)
for the quality history.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended — the CUDA torch index is wired in):

```bash
uv sync                  # creates .venv and installs everything (incl. CUDA torch)
```

Or with pip:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

You also need an osu! install with a **Songs** library (the model is trained on, and generates
into, your local Songs folder).

## Generate a map

One command generates the map, prints its stats, and drops it into your Songs folder as a playable
difficulty:

```bash
uv run python main.py infer \
    --audio "C:/osu!/Songs/123 Artist - Song/audio.mp3" \
    --reference "C:/osu!/Songs/123 Artist - Song/Artist - Song (Mapper) [Insane].osu" \
    --sr 5 6 7
```

- `--reference` is any existing `.osu` for that song — it gives **exact timing** (BPM/offset) and
  lets the map be packaged with the right audio + background. Recommended; without it the map uses
  estimated timing and is just written to disk.
- `--sr` takes one or more star ratings; all of them land as difficulties in **one** beatmapset
  folder. Open osu! and press F5 to see them.
- Handy flags: `--match-sr` (iterate to hit the exact rating), `--density` (push streams),
  `--no-package` (just write the `.osu`), `--amp` (faster, lower memory). `--help` lists them all.

A pre-trained checkpoint is auto-discovered from `runs/`; pass `--ckpt path/to/best.pt` to choose one.

## Train your own model

```bash
# 1. preprocess your ranked maps into a deduped, manifest-indexed dataset
uv run python main.py preprocess --songs "C:/osu!/Songs" --out data/processed/ranked --gold --workers 10

# 2. train (logs + checkpoints under runs/<id>/; --resume runs/<id>/ckpt/last.pt to continue)
uv run python main.py train --data data/processed/ranked --tag mymodel \
    --base 160 --crop 4096 --attn-levels 3 --batch 16 --epochs 60 \
    --objective v --zero-snr --compile --spatial-loss-weight 3

# 3. generate with your checkpoint
uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 5 --ckpt runs/<id>/ckpt/best.pt
```

Training uses bf16 + EMA + cosine LR, flip augmentation, a train/val split, and per-run logging.
`best.pt` keeps the lowest validation loss (EMA weights are used at inference).

## Project layout

```
main.py                  CLI entrypoint: preprocess | train | infer | generate
src/
  config.py              signal channel layout (21 ch) + frame<->time helpers
  conditioning.py        difficulty context vector [SR,AR,OD,HP,CS,density]
  difficulty.py          exact star rating (rosu-pp)
  metrics.py             pattern/quality metrics (spacing, stream/jump, curves, ...)
  postprocess.py         beat/slider snap, slider clamp, trim, breaks, jump respace
  package_map.py         build a playable osu! Songs folder (beatmapset) from generated maps
  run_inference.py       the friendly `infer` pipeline (generate -> stats -> package)
  parsing/beatmap.py     .osu parser + writer
  data/                  audio mel, signal encode/decode, timing, osu!.db, preprocess, dataset
  model/                 unet.py (1D conditional U-Net) + diffusion.py (DDPM/DDIM/CFG)
  train.py / generate.py training loop / low-level inference
  eval/                  analysis probes (real-vs-generated, spacing channel)
tests/                   142 hermetic tests (no dataset/GPU needed)
```

Everything heavy is git-ignored (`/data/`, `/runs/`, `/artifacts/`); only code + small configs
are tracked. A run is reproducible from `runs/<id>/config.json` (pins git commit + args).

## Notes & gotchas (learned while building)

- **Sampler matters.** Naive strided DDPM under-denoises into noise-like maps; use the **DDIM**
  sampler (correct over a step subsequence).
- **Frame grid, not milliseconds.** Everything is aligned on the audio frame grid to avoid drift.
- **Slider duration comes from length.** osu! derives a slider's duration from `length / velocity`,
  so the writer clamps each slider's length to fit the gap before the next object, and slider ends
  are beat-snapped (else ~half land off-grid).
- **bf16 + attention can diverge.** Fixed with QK-normalised attention + zero-init projection;
  base-160 needed **v-prediction + zero-terminal-SNR** to train stably (earlier ε-prediction runs
  diverged around epoch 12–21).
- **Long songs** (8-min marathons) materialise a big attention matrix — pass `--amp` (bf16) so they
  fit in VRAM.

## Development

```bash
uv run pytest          # 142 hermetic tests (synthetic fixtures, no GPU/library needed)
uv run ruff check .    # lint (rules in pyproject.toml, 100-col)
```

## Credits / prior art

- [jaswon/osu-dreamer](https://github.com/jaswon/osu-dreamer) — the signal + diffusion approach
- [OliBomby/osu-diffusion](https://github.com/OliBomby/osu-diffusion) ·
  [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) — DiT-style coordinate diffusion
- [gyataro/osuT5](https://github.com/gyataro/osuT5) · [kotritrona/osumapper](https://github.com/kotritrona/osumapper)
- [osu! file format wiki](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format))
