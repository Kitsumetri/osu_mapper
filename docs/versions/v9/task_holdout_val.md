# v9 — leakage-free held-out val split + reward-in-validation

STATIC/frozen — v9 task report.

**Status:** implemented. New `src/data/val_split.py` (stdlib-only) + `tests/test_val_split.py`
(9 hermetic tests); `src/train.py` rewired to the grouped split and given a gated
reward-in-val probe. Ruff clean on all touched files.

---

## 0. The two bugs

1. **Audio leakage in the val split.** The old `train.py` split held out a random slice
   of **manifest items (difficulties)** via `torch.randperm`. But one mel is stored *per
   audio file* (`mels/<audio_id>.npy`) and **shared by every difficulty of a song**
   (`dataset.py:60`). So a song's difficulties scattered across train and val → the *same
   audio the model trained on* showed up in validation → `val_loss` was optimistically low
   and not a real held-out signal.

2. **Duplicate audio across mappers** (user refinement). The SAME song (title/artist) is
   frequently mapped by *different people*, sometimes with a slightly different audio file
   (length differs 1–3 s, or a different encoding) → different bytes → **different
   `audio_id`**. Deduping by `audio_id` alone is therefore insufficient: a song still leaks
   across the split via a second mapper's copy.

3. **Validation never sampled a map.** `_validate()` only averaged the diffusion loss
   (MSE/huber) on *noised real* signals — it never generated a map or computed the
   ranked-map reward, so it couldn't see sampling-quality regressions.

## 1. The song-identity key + how leakage is fixed (`src/data/val_split.py`)

`song_key(item)` normalises `title` (lowercase, strip, drop punctuation/symbols, collapse
whitespace) and prepends a normalised `artist` **when the manifest has it**. The current
`preprocess.py` stores `title` but **not** `artist` (confirmed — the parser reads artist
but the manifest row omits it), so in practice the key is normalised-title; the code uses
artist automatically if a future manifest adds it. Empty title → a per-item fallback key
(never collides). So `"FREEDOM DiVE!!"` and `"freedom  dive"` map to one key — a song
mapped by two mappers collapses to a single group regardless of mapper or audio bytes.

`grouped_split(items, val_frac, seed)` then:
- builds a tiny **union-find** over two label namespaces: song-keys and `audio_id`s. Each
  item unions its song-key node with its `audio_id` node, so the transitive closure pulls
  together *every* difficulty that shares a title OR an audio file — including the chain
  "two mappers, two audio files, same title".
- shuffles the **groups** (not items) deterministically by `seed` and assigns whole groups
  to val until ≈`val_frac` of items are held out.

Guarantee (asserted by `assert_no_leakage`, used in train.py and tests): **no song-identity
key and no `audio_id` is shared between train and val.** The split is by whole songs, so
no audio the model trained on can appear in validation.

## 2. Static (frozen) val set — identical across configs/runs

The user wanted the val set fixed across every config/run. `write_static_split()` (and the
CLI) compute the grouped split once and freeze the **held-out item_ids** to
`data/processed/<tag>/val_split.json` (with frac/seed/counts for provenance):

```
uv run python -m src.data.val_split --data data/processed/<tag> --frac 0.10 --seed 1234
```

`train.py` calls `resolve_split()`, which **loads the frozen file if present** (intersected
with the current manifest so a stale id from a reprocess is ignored), else falls back to a
fresh `grouped_split` with `--val-seed` (default 1234). Either way the no-leakage invariant
is re-asserted. The val view is the non-augmented dataset, as before.

## 3. Reward-in-validation (cheap + gated)

A new `reward_probe` closure in `train()`: every `--val-reward-every` epochs it samples a
map on a **small fixed subset of held-out songs** (one difficulty per `audio_id`, capped at
`--val-reward-songs`), decodes **in-memory**, and logs the **mean reward**.

- **Reuses the real machinery, no reimplementation:** wraps the live (EMA) model in a
  `generate.LoadedModel`, builds a `generate.PreparedAudio` straight from the held-out mel
  on disk (`mels/<audio_id>.npy`) + a `TimingPoint` from the manifest `bpm`, and calls
  `generate.generate(...)` to write a `.osu`, then scores it with
  `eval.reward.reward_from_osu` against the corpus `--ref-stats`.
- **Cheap & gated:** off by default (`--val-reward-every 0`); few songs (`--val-reward-songs`,
  default 4); fewer DDIM steps (`--val-reward-steps`, default 50). A single bad sample is
  caught and skipped — it never kills training.
- **Lazy imports** (`generate`, `eval.reward` imported *inside* the closure) so a missing
  rosu / ref-stats only disables the probe, and so train.py's import doesn't depend on the
  reward stack.

`metrics.csv` gains a **`val_reward`** column. On a *resumed* run whose `metrics.csv`
predates the column, the header is detected and the old 5-column layout is preserved so
appended rows stay aligned (graceful header handling).

## 4. New / changed flags (`src/train.py`)

| flag | default | meaning |
|------|--------:|---------|
| `--val-frac` | **0.10** (was 0.02) | fraction of **songs** held out (group-aware) |
| `--val-seed` | 1234 | seed for the grouped split (ignored if a static `val_split.json` exists) |
| `--val-reward-every` | 0 (off) | sample + log mean reward every N epochs |
| `--val-reward-songs` | 4 | held-out songs sampled per probe (keep small) |
| `--val-reward-steps` | 50 | DDIM steps for probe samples (cheaper) |
| `--ref-stats` | `artifacts/reference_stats.json` | corpus stats for the reward |

## 5. Tests (`tests/test_val_split.py`, hermetic, imports only `val_split`)

Synthetic manifest with deliberate traps (shared-audio difficulties; FREEDOM DiVE mapped by
two mappers with different `audio_id` + cosmetic title diff; a Blue Zenith chain across two
audio files + two mappers). Covers: **zero song-key AND zero audio_id overlap**;
duplicate-audio-across-mappers always grouped together; key normalisation; static-split
round-trip + reproducibility across seeds; static overrides frac/seed; stale-id handling;
fallback without a static file; `val_frac=0` and bounds. The reward-in-val aggregation is
tested with a **stubbed** reward callable (no `eval.reward` / `metrics` import).

## 6. Caveats / decisions

- **Title-only key in practice** (no artist in the current manifest). Worst case: two
  genuinely different songs that share a title get grouped together — *more* conservative
  (slightly larger held-out group), never leakier. If a reprocess adds `artist` to the
  manifest, the key uses it automatically.
- **Whole-group holdout makes `val_frac` approximate** (it assigns whole songs until it
  reaches the target). With ~38k–95k maps this is a negligible deviation from 10%.
- **Probe timing** uses the manifest `bpm` (offset 0). For metrics that depend only on
  inter-onset structure this is fine; it is a quality *trend* signal across epochs, not a
  packaging-grade map. The probe samples from the **EMA** weights (what gets shipped).
- **Static set vs reprocess.** A frozen `val_split.json` is intersected with the live
  manifest, so it degrades gracefully after a reprocess (stale ids dropped); regenerate it
  with the CLI if the dataset changes materially.
- **Existing `runs/*/val_split.json`?** None today — the file is new and lives under the
  git-ignored `data/processed/<tag>/`.
