# v9 task 2 — refresh the corpus/data stats

**Status:** parallelized + diagnosed; **the full refresh is the user's to run** (the
serial full run was stopped mid-way). Backup of the old stats is in place.

## The problem
`artifacts/reference_stats.json` (the per-SR-bucket mean/std/p10/p90 of every
metric over real ranked maps — the "what ranked maps look like" gold used by
`metrics.score_against_reference` and the v9 reward) was **n_maps=31362, dated
~Jun 14**, computed with an older parser/metrics lib and an older snapshot of
`C:/osu!/Songs`. Both have changed since → the bands are miscalibrated.

## What was done
1. **Backup:** old stats copied to `artifacts/reference_stats_v8_old.json` (preserved
   for the old-vs-new diff). `reference_stats.json` itself is **still the old file** —
   the full regeneration did not complete.
2. **Parallelized `corpus_stats`** (commit `113ab26`). The scan was single-threaded
   and the rosu SR call per map dominates; it now distributes per-file work over a
   `ProcessPoolExecutor` (`--workers`, default `cpu_count-1`; `--workers 1` = serial).
   Aggregation is order-independent (`_summary` sorts), so the parallel result is
   identical to the serial one — proven by 5 hermetic equivalence tests
   (`tests/test_corpus_stats.py`). This makes the full refresh cheap to run.
3. **Probes** (current code, small samples): `artifacts/_stats_probe.json` (300 maps),
   `artifacts/_stats_probe1k.json` (1000 maps).

## Preliminary finding — the stats DID shift (1k-probe vs old 31k corpus)
Most large single-bucket %s are **sampling noise** (Easy has n=33; the Expert+
`star_rating −9%` is just the 1k sample under-representing the 8★ tail — rosu SR
itself is unchanged). But three shifts are **systematic — same direction in every
bucket**, i.e. a real *code/parser* change, not library growth:

| metric | shift (every bucket) | read |
|--------|----------------------|------|
| `hitsound_ratio` | **+14% … +37%** (always +) | parser now counts more hitsounds |
| `kiai_ratio` | **−9% … −27%** (always −) | kiai detection reads less coverage |
| `mean_spacing_px` | **−2% … −7%** (−3…−10 px) | modest, consistent downward shift |

`sv_changes_per_min` also rose in the higher buckets (+9…+48%, noisier).
Per-bucket old→new means (1k probe) for the key metrics:

```
                       Easy    Normal   Hard   Insane  Expert  Expert+   (old->new mean)
mean_spacing_px      146->136 131->121 123->119 140->137 152->147 155->150
jump_ratio          .195->.144 .133->.109 .129->.116 .239->.234 .322->.309 .343->.329
stream_ratio        .003->.001 .009->.015 .081->.099 .150->.185 .212->.226 .308->.309
on_quarter_grid     .944->.918 .943->.895 .938->.932 .926->.935 .917->.924 .851->.842
hitsound_ratio      .272->.330 .303->.366 .290->.339 .333->.395 .347->.396 .278->.380
kiai_ratio          .242->.220 .233->.200 .238->.212 .240->.203 .237->.193 .270->.196
sv_changes/min       .87->.73  1.95->2.87 6.70->6.23 14.1->15.4 19.6->21.7 24.7->26.9
```
(1k-sample p10/p90 are too noisy to trust per-bucket — the **full run** is needed
for the reward's band edges.)

## Why it matters for the v9 reward (task 3)
The reward (`src/eval/reward.py`) scores band-membership against these p10/p90 bands.
- `mean_spacing_px` is weighted **1.5** and shifted ~3–7% → worth refreshing before
  the reward is used in anger.
- `hitsound_ratio` / `kiai_ratio` shifted most but are down-weighted **0.25** (and
  hitsounds/kiai get dedicated v9 heads), so lower-stakes — but still refresh.
- The reward reads `reference_stats.json` **at call time** and renormalises over
  whatever metrics are present, so it degrades gracefully; it just wants current numbers.

## Action for the user (the full refresh — now cheap)
```bash
# full library, parallel (default cpu_count-1 workers); overwrites the gold stats
uv run python -m src.corpus_stats --songs "C:/osu!/Songs" --out artifacts/reference_stats.json
# or pin worker count:  --workers 12
```
The old file is safe at `artifacts/reference_stats_v8_old.json` for an exact
old-vs-new diff afterwards (re-run the comparison cell the orchestrator used, or just
eyeball the three systematic metrics above on the full numbers).

**Note:** `rglob` over ~95k files plus eager future-submission means a tiny `--limit`
run still pays the listing/submit overhead; the parallelism pays off on the **full**
run (the compute, not the listing, is what scales across cores).
