# v9 — early-abort sampling on the reward trajectory (best-of-N efficiency gate)

*STATIC / frozen — v9 task report.*

Best-of-N samples `N` candidates per (song, SR) and reward-ranks them. Many
candidates are clearly doomed well before the last denoising step. **Early-abort**
watches the LATE DDIM steps with a *cheap* quality proxy and aborts a candidate whose
quality trajectory is heading to the bottom, so best-of-N spends its compute on the
viable candidates. This is an **efficiency gate, NOT steering**: `ddim_sample` is
`eta=0` deterministic (we can only abort, never resample) and the reward is
non-differentiable (no gradient). It is **opt-in and default-OFF** — existing best-of-N
behaviour is byte-for-byte unchanged unless you pass `early_abort=True`.

This is round-3 candidate #4 (`docs/status/roadmap.md`), and the one abort-only piece
that the policy-gradient deep-dive (`task_rl_policy_gradient.md`) left on the table:
there is no rollout, no log-prob, no new sampler — just a per-step exit gate.

## The abort rule (monitored window · proxy · threshold)

- **Monitored window — late steps only.** The monitor is armed only past
  `ea_monitor_frac` of the reverse process (default **0.70** → last ~30% of steps).
  Early `x0` is blurry and its decoded quality is meaningless, so we never abort on it.
- **Cheap proxy — `quality_score` band-membership ONLY, decoded in memory.** At a
  monitored step we take the (clamped) predicted clean signal `x0`, run the existing
  `decode_signal(sig)` → wrap the objects in a throwaway in-memory `Beatmap` (one
  uninherited timing point so `compute_metrics` has a BPM) → `compute_metrics` →
  `quality_score(metrics, ref_stats, bucket)`. **No rosu star rating** (it dominates
  per-step cost — aborting cheaply is the whole point) and **no file I/O**. A
  signal that decodes to <2 objects scores 0.0 ("going nowhere"). (`best_of_n.cheap_quality`.)
- **Threshold — step-relative AND an absolute floor.** As candidates *complete*, we
  cache the best proxy quality seen at each monitored step index `k` (`best_by_step`,
  populated only from COMPLETED candidates so "best-so-far at this step" is
  well-defined). A candidate is aborted at step `k` iff **both**: (i) its proxy
  quality trails the best completed candidate at `k` by more than `ea_margin`
  (default **0.15**) — the user's "stop if the reward curve approaches the lowest
  values"; AND (ii) it is below an absolute floor `ea_abs_floor` (default **0.55**) —
  a second safety so a merely-behind-but-respectable candidate is never thrown away.
  The rule is **disarmed until `ea_min_candidates` (default 2) have completed**, so the
  cache is trustworthy and the first candidate(s) are never aborted blindly.
- **Err strongly toward NOT aborting.** A false abort discards a possible winner —
  far worse than wasting one sample. Requiring both the relative drop and the absolute
  floor, plus the late window and the arming delay, makes the gate conservative by
  design. Note the first completed candidate always wins-or-loses on its own merits:
  the very first candidate is never armed (empty cache) and so **always completes** —
  which is why `best_of_n` always has ≥1 completed candidate (its empty-`completed`
  `SystemExit` guard is purely defensive and unreachable under the normal path).

## Winner-identical guarantee (the critical correctness property)

> The selected winner is **identical with and without early-abort.**

Two layers enforce it:

1. **We only ever abort would-be losers.** A candidate is aborted only when it both
   trails the best completed candidate by a margin and sits below the absolute floor —
   i.e. it could not have won. Aborted candidates are **excluded from the ranking**
   (and never even pay the expensive rosu reward), so removing them cannot change the
   `argmax`.
2. **A candidate that COMPLETES is byte-for-byte identical to a plain `generate`.**
   The monitor is an exit gate that **cannot mutate `x0`** (in `diffusion.py` it is
   called *after* the `x0` clamp and only its truthy return breaks the loop). With
   `monitor=None` the sampling path is unchanged line-for-line. best-of-N injects the
   monitor into the *production* `generate` pass by temporarily wrapping
   `loaded.diff.ddim_sample` for one call (then restoring it) — so we never edit
   `generate.py`, yet a non-aborted candidate runs the exact same code and produces the
   exact same bytes.

## Implementation

- **`src/model/diffusion.py`** — `ddim_sample(..., monitor=None)`. After the `x0`
  clamp, if a monitor is set it is called as `monitor(k_index, frac_done, x0)` where
  `k_index` is the remaining-step index (counts down to 0) and `frac_done ∈ [0,1]` is
  progress through the reverse process (0 = noisiest first step, ~1 = last). A truthy
  return breaks the loop and returns the current `x0`. The default `monitor=None` path
  is **byte-for-byte unchanged**; CFG / batched / amp / progress logic untouched. The
  return type is always the tensor (the caller learns of an abort from the monitor
  object it owns), so existing callers are unaffected.
- **`src/best_of_n.py`** — `cheap_quality(...)` (the in-memory decode→quality proxy,
  with injectable decode/metrics/quality hooks for hermetic tests); `EarlyAbortMonitor`
  (per-candidate closure over `ref_stats`, `bucket = sr_bucket(target_sr)`, the song
  BPM, and the shared `best_by_step` cache; records `aborted` / `abort_k` /
  `last_quality` / per-step qualities); `_patched_monitor` (the wrap/restore of
  `ddim_sample` so the unmodified `generate` runs with the monitor). `best_of_n` gains
  `early_abort=False` + `ea_monitor_frac` / `ea_margin` / `ea_abs_floor` /
  `ea_min_candidates`, tracks `n_aborted` + `steps_saved` (= remaining steps skipped at
  each abort), excludes aborted candidates from ranking, and surfaces all of it in the
  `<out>.bon.json` audit (`early_abort`, `n_aborted`, `n_completed`, `steps_saved`,
  `steps_per_candidate`, `aborted{idx→abort_k,proxy_q}`). The `bestofn` CLI gets
  matching `--early-abort` + `--ea-*` flags (default OFF).
- **Not edited:** `generate.py`, `conditioning.py`, `metrics.py`, `eval/reward.py`,
  `data/*` — all imported, none modified (the monitor reuses their public functions).

## Tests — `tests/test_early_abort.py` (15, hermetic; no GPU/rosu/network/audio)

- `ddim_sample(monitor=None)` is byte-identical whether the arg is omitted or `None`,
  and a never-aborting monitor cannot move the result (it is an exit gate, not a
  steerer); an aborting monitor breaks early and returns the partial `x0`.
- The cheap proxy returns 0.0 for <2 objects, returns the injected quality otherwise,
  and runs end-to-end through the REAL `decode_signal` + `compute_metrics` on a
  synthetic signal (in-memory, no rosu, no file I/O).
- The step-relative rule + absolute floor + late-window + arming each tested in
  isolation: no abort before the late window; no abort until `min_candidates` complete;
  no abort when above the floor even if trailing; no abort when within the margin even
  if below the floor; abort only when below the floor AND trailing by the margin; a
  good candidate never aborts across all late steps.
- **Headline:** with a deterministic fake sampler, `best_of_n`'s selected winner —
  index, reward, AND output bytes — is **identical with and without early-abort**,
  while `n_aborted > 0` / `steps_saved > 0` and the aborted candidates are exactly the
  low-reward bottom-feeders (never the winner). Plus: the worst-possible first
  candidate still completes (never aborted blindly).

All 15 pass; `src/best_of_n.py`, `src/model/diffusion.py`, `tests/test_early_abort.py`
are ruff-clean. The neighbouring `test_best_of_n.py` / `test_model.py` / `test_reward.py`
/ `test_metrics.py` still pass (the new param is optional and default-OFF).

## GPU validation — RAN (after the ctx-truncation backward-compat fix)

The original blocker (the in-progress per-song conditioning bumped the context 6→7, which
the released 6-dim v8/v8_1 ctx_mlp rejected) was fixed during integration: `generate._one_pass`
now truncates the context to the loaded model's `ctx_dim`, and because the `aim` slot is
appended LAST, a pre-v9 6-dim checkpoint drops exactly it (`generate.py`; locked by
`tests/test_aim_intensity.py::test_generate_ctx_truncates_to_old_checkpoint_width`). With
that, the bounded validation ran on the v8 release ckpt + the Kawaii song:

- **N=6 @ SR 5, both ways — winner IDENTICAL: cand 01, R=0.8196** (the critical correctness
  property, now confirmed on a real model, not just the deterministic fake). **0/6 aborted,
  0 DDIM steps saved** — every candidate's cheap proxy quality (~0.66–0.69) stayed above the
  `ea_abs_floor` (0.55), so the conservative gate correctly never fired (no false aborts).
- **N=8 @ SR 7 — 0/8 aborted, and it shows WHY:** v8's candidate *quality* is uniformly high
  (band-membership 0.96–1.00); the real variance is in **SR-closeness** (e.g. one candidate
  `sr_close` 0.299 vs another 0.997), which the **quality-only cheap proxy deliberately
  excludes** (rosu SR is the expensive term we abort to avoid). So on this model the doomed
  candidates are doomed by SR — invisible to the proxy — and nothing aborts.

**Finding (honest):** early-abort is correct, safe, and default-off, but its practical payoff
on the current v8 model + reward is **~zero**, because candidate variance lives in the SR term
the cheap proxy omits. It will save compute only when candidate *quality* varies (a weaker /
earlier model, or songs where decode quality collapses) — or if a cheap SR proxy is found to
fold into the gate (none currently; rosu SR is the cost we're avoiding). Kept opt-in / default-off.
