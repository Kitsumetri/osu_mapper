# Storage layout

A clean, reproducible scheme for data, model checkpoints, and logs. Everything
heavy is git-ignored; only code + small configs are tracked.

```
osu_mapper/
├── data/
│   └── processed/<dataset_tag>/        # one preprocessing run
│       ├── mels/<audio_id>.npy         # log-mel per *audio file* (deduped)
│       ├── items/<item_id>.npz         # signal + meta per difficulty
│       └── manifest.json               # index: every item + its metadata
├── runs/<run_id>/                      # one training run (run_id = YYYYmmdd-HHMMSS-tag)
│   ├── config.json                     # full args + git commit + dataset_tag + n_params
│   ├── metrics.csv                     # epoch, avg_loss, lr, sec
│   └── ckpt/
│       ├── last.pt                     # rolling latest
│       ├── best.pt                     # lowest avg_loss so far
│       └── epoch_NNN.pt                # milestones (every --save-every)
└── artifacts/                          # exported, shareable outputs
    └── <name>/ (generated .osu, packaged maps)
```

## Why

- **Mel dedup**: a song has many difficulties sharing one audio. Storing the mel
  once per `audio_id` instead of per difficulty cuts the dominant cost (~4–5×
  fewer big arrays for full-library scale).
- **Manifest**: `manifest.json` lets training filter/sample (by difficulty,
  density, creator, BPM, kiai) and compute stats **without** opening every shard.
  Each entry records `item_id, audio_id, creator, title, version, n_objects,
  cs/ar/od/hp, slider_multiplier, bpm, n_timing_points, has_kiai, duration_s,
  frames`.
- **`runs/<run_id>/`**: self-contained, comparable experiments. `config.json`
  pins the git commit + `dataset_tag` so any run is reproducible. Only `last.pt`,
  `best.pt`, and milestone checkpoints are kept — no more 1.5 GB of
  every-20-epoch checkpoints.
- **`artifacts/`**: human-facing outputs separate from training state.

## Conventions

- `dataset_tag` e.g. `std-v1`, `std-full`. `run_id` e.g.
  `20260613-205222-std-v1-base160` (timestamp + `--tag`).
- Checkpoints store `{model, ema, args, epoch, git_commit}` (the EMA weights are
  what `generate.py` prefers at inference).
- Nothing under `data/`, `runs/`, `artifacts/` is committed (see `.gitignore`,
  where these are root-anchored so they don't match `src/data/`).
- Reproduce a run: read `runs/<id>/config.json` → same dataset_tag + args.
