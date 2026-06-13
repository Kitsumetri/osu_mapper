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
│   ├── config.json                     # full args + git commit + dataset_tag
│   ├── train.log                       # stdout/metrics
│   ├── metrics.csv                     # epoch,loss,lr,...
│   └── ckpt/
│       ├── last.pt                     # rolling latest (+ EMA, optimizer)
│       └── best.pt / epoch_NNN.pt      # milestones only
└── artifacts/                          # exported, shareable outputs
    └── <name>/ (generated .osu, packaged maps)
```

## Why

- **Mel dedup**: a song has many difficulties sharing one audio. Storing the mel
  once per `audio_id` instead of per difficulty cuts the dominant cost (~4–5×
  fewer big arrays for full-library scale).
- **Manifest**: `manifest.json` lets training filter/sample (by difficulty,
  star-ish density, creator, BPM) and compute stats **without** opening every
  shard. Each entry records `item_id, audio_id, creator, version, n_objects,
  cs/ar/od/hp, bpm, n_timing_points, has_kiai, duration_s`.
- **`runs/<run_id>/`**: self-contained, comparable experiments. `config.json`
  pins the git commit + `dataset_tag` so any run is reproducible. Only `last.pt`
  + milestones are kept — no more 1.5 GB of every-20-epoch checkpoints.
- **`artifacts/`**: human-facing outputs separate from training state.

## Conventions

- `dataset_tag` e.g. `std-v1`, `std-full`. `run_id` e.g. `20260613-2council-base160`.
- Checkpoints store `{model, ema, optimizer, args, epoch, git_commit}`.
- Nothing under `data/`, `runs/`, `artifacts/` is committed (see `.gitignore`).
- Reproduce a run: read `runs/<id>/config.json` → same dataset_tag + args.
