# task_autopackage — Best-of-N auto-packaging in `infer`

**v9 feature implemented in:** `src/run_inference.py`, `src/best_of_n.py`, `main.py`

## Problem

`main.py bestofn` scored and selected the best of N candidates but wrote raw `.osu` files
with no auto-packaging. The friendly `infer` path packaged maps but could only generate a
single candidate per SR. Users had to choose between quality selection and packaging — two
separate commands, two separate model loads.

## Solution

Added `--best-of-n N` (and supporting flags) to `main.py infer` / `src/run_inference.py`.
When `N > 1`, each SR runs reward-ranked best-of-N via the `best_of_n()` library function
(passing `loaded=`, `prepared=` so the model and audio are loaded exactly once for the whole
run). The winner replaces what a normal single `generate()` call would produce, then flows
through the identical `_print_stats` + `package_set` path — all SRs land in one beatmapset
folder, exactly like normal `infer`.

## Final CLI

### Everyday flow (recommended)

```
# Single SR, normal generation + package
uv run python main.py infer \
    --audio song.mp3 \
    --reference ref.osu \
    --sr 5

# Multiple SRs, reward-ranked, auto-packaged into ONE Songs folder
uv run python main.py infer \
    --audio song.mp3 \
    --reference ref.osu \
    --sr 4 5 6 \
    --best-of-n 8

# Same but keep all N candidate .osu files on disk
uv run python main.py infer \
    --audio song.mp3 \
    --reference ref.osu \
    --sr 5 \
    --best-of-n 12 \
    --keep-candidates
```

### Debug / no-package path (unchanged)

```
uv run python main.py bestofn \
    --audio song.mp3 \
    --sr 5 6 \
    --n 8 \
    --keep-candidates
```

`main.py bestofn` is kept for debugging (inspect all N candidates, no packaging overhead).
For production use, `infer --best-of-n N` is the blessed entrypoint.

## New flags on `infer`

| Flag | Default | Description |
|---|---|---|
| `--best-of-n N` | `1` | Candidates per SR. `1` = unchanged single-sample behaviour. |
| `--ref-stats PATH` | `artifacts/reference_stats.json` | Reward reference stats (only required when `N > 1`). Build with `uv run python -m src.corpus_stats`. |
| `--bon-seed INT` | `0` | RNG seed for candidate variety. |
| `--keep-candidates` | off | Keep all N `.osu` candidate files alongside the winner. |

## What gets printed

For each SR in best-of-N mode, after the per-candidate `[bon] cand XX` lines from
`best_of_n()`, `infer` prints a lift line:

```
  [bon] SR 5: best R=0.7231 / mean R=0.6104 (lift +0.1127) in 143s  n=8
```

After all SRs a summary table is printed:

```
================================================================
  BEST-OF-8 SUMMARY
================================================================
  SR 4*   best R=0.6812 / mean R=0.5990 (lift +0.0822)
  SR 5*   best R=0.7231 / mean R=0.6104 (lift +0.1127)
  SR 6*   best R=0.7055 / mean R=0.6211 (lift +0.0844)
```

## Audit trail

Each SR winner gets a `<winner>.bon.json` sidecar (written by `best_of_n()`) containing
the full per-candidate reward breakdown. In single-sample mode no `.bon.json` is written
(unchanged behaviour).

## Design decisions

- `best_of_n()` is called as a library (not subprocess); `loaded=` and `prepared=` are
  threaded in so the model and audio are loaded once even across multiple SRs.
- `main.py bestofn` is retained as the debug/no-package path. Its module docstring now
  prominently points users at `infer --best-of-n N`.
- `_load_ref_stats_for_infer` in `run_inference.py` is a thin wrapper around the same
  JSON-loading logic as `best_of_n._load_ref_stats`, kept separate to give `infer`-specific
  error messaging and to avoid a circular import.
- Ref-stats are loaded before any model load (fail-fast on a missing file).
- `_print_bon_summary` is a pure helper in `run_inference.py` (no I/O side effects beyond
  `print`) so it can be tested hermetically.
