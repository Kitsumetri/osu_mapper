# Reference corpus statistics (per-SR-bucket gold distribution)

**Purpose:** the per-star-rating-bucket distribution of pattern metrics over the real ranked library —
the "what ranked maps look like" gold used by `metrics.score_against_reference` and the v9 reward, plus
how to read it and how to regenerate it. | **MOSTLY STATIC** (the *interpretation* is durable; the
*numbers/`n`* refresh when the library or parser changes — the live file is
`artifacts/reference_stats.json`, the source of truth, this doc is a snapshot + guide).

Computed with `src/corpus_stats.py`, bucketed by **star rating** (rosu-pp, `difficulty.py`). Mappers'
difficulty *names* are arbitrary, so SR is the principled axis. Score a generated map against its SR
bucket with `python -m src.metrics --osu gen.osu --ref-stats artifacts/reference_stats.json` (computes
SR, picks the bucket, reports z-score + in-p10–p90 flag).

## Snapshot (n=31,362; ~Jun 14) — means ± std

This is the v8-era table; the current file is **n=94,639** (3× the library, refreshed in v9 — see
"Regenerating" below). Cells are `mean +/- std`.

| metric | Easy (n=1861) | Normal (n=4387) | Hard (n=5732) | Insane (n=7640) | Expert (n=7181) | Expert+ (n=4561) |
|---|---|---|---|---|---|---|
| `star_rating` | 1.794 ± 0.158 | 2.336 ± 0.183 | 3.445 ± 0.337 | 4.728 ± 0.354 | 5.848 ± 0.334 | 8.181 ± 7.95 |
| `density_per_s` | 1.125 ± 0.252 | 1.72 ± 0.352 | 2.66 ± 0.498 | 3.608 ± 0.675 | 4.424 ± 0.897 | 5.764 ± 3.317 |
| `circle_ratio` | 0.347 ± 0.104 | 0.393 ± 0.104 | 0.443 ± 0.12 | 0.56 ± 0.13 | 0.612 ± 0.123 | 0.669 ± 0.146 |
| `slider_ratio` | 0.64 ± 0.104 | 0.599 ± 0.104 | 0.553 ± 0.12 | 0.438 ± 0.13 | 0.386 ± 0.123 | 0.326 ± 0.143 |
| `bezier_slider_ratio` | 0.202 ± 0.213 | 0.152 ± 0.177 | 0.14 ± 0.172 | 0.145 ± 0.163 | 0.167 ± 0.159 | 0.192 ± 0.17 |
| `new_combo_ratio` | 0.288 ± 0.064 | 0.237 ± 0.058 | 0.236 ± 0.057 | 0.254 ± 0.106 | 0.257 ± 0.077 | 0.257 ± 0.095 |
| `mean_spacing_px` | 145.9 ± 24.4 | 130.5 ± 20.3 | 123.3 ± 20.2 | 140.0 ± 31.5 | 152.0 ± 39.1 | 154.6 ± 54.9 |
| `std_spacing_px` | 62.9 ± 11.8 | 62.5 ± 10.8 | 67.7 ± 11.3 | 79.2 ± 13.6 | 88.7 ± 14.5 | 97.0 ± 20.8 |
| `stream_ratio` | 0.003 ± 0.022 | 0.009 ± 0.038 | 0.081 ± 0.097 | 0.15 ± 0.13 | 0.212 ± 0.16 | 0.308 ± 0.22 |
| `jump_ratio` | 0.195 ± 0.12 | 0.133 ± 0.09 | 0.129 ± 0.08 | 0.239 ± 0.138 | 0.322 ± 0.161 | 0.343 ± 0.196 |
| `on_quarter_grid_ratio` | 0.944 ± 0.187 | 0.943 ± 0.187 | 0.938 ± 0.187 | 0.926 ± 0.195 | 0.917 ± 0.213 | 0.851 ± 0.277 |
| `mean_turn_angle_deg` | 81.8 ± 11.1 | 75.1 ± 10.4 | 82.8 ± 13.8 | 101.8 ± 18.5 | 103.5 ± 19.9 | 96.8 ± 24.3 |
| `reversal_ratio` | 0.074 ± 0.054 | 0.046 ± 0.042 | 0.077 ± 0.068 | 0.188 ± 0.094 | 0.218 ± 0.092 | 0.214 ± 0.111 |
| `sv_changes_per_min` | 0.867 ± 4.294 | 1.946 ± 9.202 | 6.704 ± 18.88 | 14.118 ± 30.668 | 19.579 ± 23.369 | 24.651 ± 39.639 |

(Bucket split: Easy 1861, Normal 4387, Hard 5732, Insane 7640, Expert 7181, Expert+ 4561 = 31,362.
Raw per-bucket mean/std/p10/p90 for every metric is in `artifacts/reference_stats.json`. The Expert+
`star_rating` std is inflated by a few extreme-SR outliers / non-ranked joke maps.)

## What this tells us (targets for the generator)

- **Everything scales monotonically with star rating** — density, circle ratio, streams, jumps,
  spacing spread, turn angle, SV changes. Strong evidence that **SR is a good single conditioning
  axis**: the model can interpolate difficulty along it.
- **Circle↔slider trade-off**: harder = more circles (Easy 0.35 → Expert+ 0.67), sliders inversely.
- **Streams scale hard** (Easy 0.003 → Expert+ 0.31); **jumps dip mid (Hard 0.13) then climb** to 0.34
  at Expert+ — Hard maps lean on sliders/rhythm, top diffs on aim.
- **On-¼-grid ~0.85–0.94** everywhere — real maps are tight but not perfect (triplets, 1/8, sub-beat).
  A bounded beat-snap target is ~0.9, not 1.0.
- **`sv_changes_per_min` 0.9 → 25** — SV variety strongly marks difficulty; a single-timing-point
  output (0) was the biggest systematic gap.
- **`bezier_slider_ratio` ~0.14–0.20** across buckets — a concrete target for the curved-slider work.

## Regenerating (and the v9 refresh)

`corpus_stats` is parallelized (`--workers`, default `cpu_count-1`; serial ≡ parallel):

```bash
uv run python -m src.corpus_stats --songs "C:/osu!/Songs" --out artifacts/reference_stats.json
```

The reward / `score_against_reference` read the JSON **at call time** and renormalise over whatever
metrics are present, so they degrade gracefully across schema/`n` changes.

**v9 refresh (n=31,362 → 94,639).** The old file was systematically stale (older parser): every bucket
showed `hitsound_ratio` **+16…28%**, `kiai_ratio` **−9…−16%**, `mean_spacing_px` **−2…−6%**,
`sv_changes_per_min` **+12…72%**, `reversal_ratio` **−7…−30%**. The bands are now reliable. (Expert+
SR-mean 8.18→7.68 = library composition, not a metric change.) Old file preserved at
`artifacts/reference_stats_v8_old.json`. Detail: [versions/v9/task2_data_stats.md](../versions/v9/task2_data_stats.md).

**Caveat:** `on_quarter_grid_ratio` (and the reward via it) assumes a **single BPM** — it
under-measures variable-BPM maps. Does not affect best-of-N (generated maps are single-BPM by
construction; gold training data is single-BPM-filtered) — but note it before scoring arbitrary real
maps with the reward.
