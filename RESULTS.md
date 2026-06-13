# Results

First full training run + end-to-end generation on the local osu! library.

## Training

- **Data**: 601 osu!standard difficulties (preprocessed to aligned mel/signal
  `.npz` shards from the local Songs library).
- **Model**: 1D conditional U-Net, base 96 (~14M params), DDPM (1000 steps).
- **Run**: 240 epochs, batch 8, crop 2048 frames (~24 s), AMP, RTX 4070 Ti.
  ~12 s/epoch (~48 min total).
- **Loss** (ε-prediction MSE): 0.35 → **0.011**.

## Generation

Sampling with **DDIM (100 steps, ~0.5 s)** on a held-out song, decoded to hit
objects, timing estimated from audio:

| metric | value |
|--------|-------|
| objects | 1726 over 250 s (**6.9 / s**, Insane-level) |
| circles / sliders / spinners | 961 / 764 / 1 |
| new-combos | 621 (~1 per 2.8 objects) |
| cursor coverage | full playfield (0–512 × 0–384) |
| inter-onset interval | median 82 ms (p10 70, p90 313) |
| estimated timing | 198.8 BPM @ 139 ms (true song: 192 BPM) |
| onsets on ¼-beat grid | 34% |

The output is a **valid, playable-format `.osu`** that re-parses cleanly.

## Honest assessment

What works: the pipeline is fully functional audio → `.osu`; the model learned
realistic object **density** and a sensible **circle/slider/spinner mix** and
uses the whole playfield.

What needs work (see README roadmap):
- **Rhythm**: onsets don't snap to a beat grid (34%), so the map feels loose —
  needs rhythm-aware training and/or post-hoc quantisation.
- **Timing accuracy**: the BPM estimate (~28% exact) skews the grid further.
- **Slider shapes**: decoded as linear 2-point sliders only.
- **Scale**: trained on 601 of 31k+ available difficulties.

## Reproduce

```bash
python main.py train    --data data/processed --epochs 240 --batch 8 --crop 2048 --base 96
python main.py generate --audio song.mp3 --ckpt checkpoints/model_e240.pt --out out.osu --steps 100
```
