# Results

Status of training runs and generated-map quality. Metrics use `src/metrics.py`.

## v1 baseline (complete)

First full run — establishes that the pipeline works end-to-end.

- **Data**: 601 osu!standard difficulties.
- **Model**: 1D conditional U-Net, base 96 (~14M params), no attention, DDPM.
- **Run**: 240 epochs, batch 8, crop 2048, AMP fp16, RTX 4070 Ti (~48 min).
- **Loss** (ε-prediction MSE): 0.35 → **0.011**.

Generation (DDIM 100 steps, ~0.5 s) on a held-out song vs the real Expert diff:

| metric | generated (v1) | real Expert |
|--------|---------------:|------------:|
| objects | 1726 | 962 |
| density / s | 6.9 | 4.0 |
| circle / slider / spinner ratio | .56 / .44 / .00 | .66 / .34 / .00 |
| bezier-slider ratio | 0.00 | 0.18 |
| stream ratio | 0.50 | 0.16 |
| on-¼-grid ratio | 0.70 | 0.997 |
| est. timing | 198.8 BPM | 192 BPM |

Valid, playable `.osu` that re-parses cleanly. Reads the rhythm, but feels loose
(low on-grid), is too dense/stream-heavy, and had only straight sliders.

## v2 scaled (in progress)

Rebuilt on the bug-fixes + bigger model + more data + new features:

- **Data**: 3004 difficulties / 888 audios (deduped, manifest-indexed).
- **Model**: base 160, **97.4M params**, self-attention (QK-norm) at coarse
  levels, bf16, EMA, cosine LR + warmup.
- **Run**: 120 epochs, batch 12, crop 3072, ~32 s/epoch.
- **Features since v1**: DDIM (correct sampler), slider time-overlap fix, curved
  Bezier sliders, realistic difficulty defaults, estimated timing.

> Numbers to be filled in when the run completes (loss curve in
> `runs/<id>/metrics.csv`, generated-vs-real metrics via `src/metrics.py`).

## Honest assessment

Works: fully functional audio → `.osu`; realistic density and circle/slider/
spinner mix; full-playfield cursor use; valid output; now with curved sliders.

Open gaps (see `README.md` roadmap + `RESEARCH.md`):

- **Rhythm**: onsets aren't beat-snapped (~0.70 on-grid vs ~0.99 real).
- **Timing accuracy**: BPM estimate exact only ~28%, which also skews the grid.
- **Controllability**: no difficulty/style conditioning yet (one fixed tier).
- **Scale**: 3004 of 31k+ available difficulties.

## Reproduce

```bash
python main.py preprocess --songs "C:/osu!/Songs" --out data/processed/std-v1 --limit 3000
python main.py train    --data data/processed/std-v1 --tag std-v1-base160 \
    --epochs 120 --batch 12 --crop 3072 --base 160
python main.py generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --out out.osu --steps 100
python -m src.metrics   --osu out.osu --ref some_real_map.osu     # compare
```
