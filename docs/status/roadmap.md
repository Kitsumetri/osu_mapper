# Roadmap & tasks TODO

**Purpose:** the live priority-ordered task list — what's done, what's next, and the available decode
levers. | **DYNAMIC** (edit as tasks complete / priorities shift). The narrative "current state" lives
in the entry-point handoff (`HANDOFF.md` at repo root); per-version design is in
[docs/versions/](../versions/README.md).

## Done

- **P0** slider-mix probe · **P1** pattern analysis · **P2** v-pred + zero-SNR · **P3** attention
  (tried, reverted — up-attn HURTS) · **P4-A** SV channel · **P4-C** curvature cue · **v7.5** red
  corners · P0 decode batch (trailing trim, curve calib, dropped-1/4 recovery) · timing-model CPU
  foundation · loss flags (huber/min-snr) + density/onset levers wired.
- **v8 RELEASED** — base-160 unblocked + spacing channel + spatial-loss-weight (P4-B partial; see
  [versions/v8.md](../versions/v8.md)). **Perf:** batched CFG, `--amp`, `--compile`, tqdm. **Refactor:**
  code → `src/`, `main.py infer` entry.
- **v9 round 1** ([versions/v9.md](../versions/v9.md)): snap-bug FIXED (`bc3c80a`); `corpus_stats`
  parallelized (`113ab26`); stats refreshed (n=94639); best-of-N DONE (`2263443`).
- **v9 round 2** (all DONE, 203 tests): `infer --best-of-n N` autopackage (`1bcacd0`); general 5-family
  reward (`61535cd`, gold 0.953); RL/policy-gradient verdict + log-prob prototype (`5be40f7`,
  `src/rl/`); core audit — 2 bugs fixed (doubled-BPM, write_osu mutation) +15 tests (`07f8912`).
- **v8_1 ablation** (rope+huber+80ep) trained by USER + A/B'd vs v8 — wins 10/12 cells, sharper
  high-SR/jumps, one low-SR over-streaming regression; **pending in-game A/B** to decide promotion
  ([versions/v9/v8_1-ablation.md](../versions/v9/v8_1-ablation.md)).

## Priority-ordered task list

| # | task | evidence | priority | cost |
|---|------|----------|----------|------|
| **v8_1 promotion — USER A/B** | decide v8_1 (rope+huber+80ep) vs v8 by in-game play, not metrics. A/B: (1) SR 6.5–7 jump/tech song (v8_1's case-for: sharper aim, higher jumps); (2) **SR 4 on a fast stream song = the gate** — is v8_1's over-streaming unplayable, or fixable with `--match-sr`/`--density`?; (3) slider feel (v8_1 uses fewer); (4) Kawaii 5–6 default. | [versions/v9/v8_1-ablation.md](../versions/v9/v8_1-ablation.md) (wins 10/12 cells) | **P1 (USER, play)** | gen + play |
| **Best-of-N — USER run** | run a real `python main.py infer --audio song.mp3 --reference ref.osu --sr 5 6 7 --best-of-n 8` on a jump song (reward now general/family-balanced); judge the winner in-game. If best-of-N alone satisfies, RL may be unnecessary. | harness+reward done (`2263443`,`1bcacd0`,`61535cd`) | **P1 (USER, no train)** | N× sampling GPU-min |
| **v9 per-song conditioning** | condition target spacing / aim-intensity inferred from audio (onset-energy / spectral-flux), fed like `--density` — the *real* per-song jump fix. v8's spacing channel shares the cursor's audio+SR conditioning → both regress to the SR-average; a passive channel can't beat its own conditioning. **The diagnosed primary fix: do this BEFORE RL — RL alone pushes spacing globally (the `--spacing-scale` failure).** | Happppy: v8 channel ~122 vs real 173 px | **P1 (top)** | reprocess + train |
| **RWR / Diffusion-DPO align** | post-train the *conditioned* model toward the reward (best-of-N distill / RWR, then DPO if it plateaus). Gated on play feedback; DRaFT/reward-guidance blocked (non-diff reward). | [versions/v9/task3_rl_alignment.md](../versions/v9/task3_rl_alignment.md) | P2 (after conditioning) | short/moderate train (USER) |
| Hitsounds | rule-based placement from beat-phase + per-band audio onsets (claps backbeat, finish downbeat/cymbal). | hitsounds 4/10, unstable | P1 (parallel) | ~1 day, no big train |
| Kiai head | small supervised mel→kiai 1D-conv head; use its deterministic output at decode. | kiai unstable song-to-song | P1 (parallel) | ~1 day + small head train |
| Density cond. | condition on a per-song density inferred from audio onset-rate (not the SR default); also fixes intro-empty. | stream-shy; rhythm gaps; intro-empty | P2 | reprocess + train |
| 1/6 over-firing | per-song divisor decision (detect triplets before enabling 1/6) or tighter onset precision — ~16% of straight-song gaps land on 1/6 (onset noise, not real triplets). | [versions/v9/task1_postprocess.md](../versions/v9/task1_postprocess.md) | P2 (model/decode) | decode/model work |
| Timing model | benchmark beat_this/BeatNet/librosa vs corpus ground-truth ([versions/v8.md](../versions/v8.md) §timing), then bespoke if needed. | novel-song timing (~28% exact) | P2 | medium, GPU/libs |
| Loss A/B | `--loss huber` and `--min-snr-gamma 5` are wired; A/B vs mse for sharper/less-mean output. | under-dispersion | P2 | rides a train |

**Decode levers already available (no retrain):** `--density` (streams), `--onset-threshold`,
`--guidance`/`--guidance-rescale`, curve/corner thresholds in `signal.py`. Note `--spacing-scale` is
shelved (hurts in-game → use 0); see [knowledge/lessons-learned.md](../knowledge/lessons-learned.md).

## v9 round 3 — reward / val / audio batch (2026-06-23)

Five user ideas. **Round 3a (#1 tooling, #2, #3, #5) landed via three sub-agents on 2026-06-23**
(committed `d32fa0d`/`13027cd`/`fefdb48`). Detailed task reports under [versions/v9/](../versions/v9/):
[task_holdout_val](../versions/v9/task_holdout_val.md), [task_reward_flow](../versions/v9/task_reward_flow.md),
[task_audio_features](../versions/v9/task_audio_features.md). **Two real bugs surfaced + fixed:** the
val-split audio leakage (#2) and the inference mel pad/precision skew (#5 rung 0).

**Round 3a-fix (2026-06-24, audit-driven).** The USER ran the brute-force audit (#1, `n=87176`); it found
the playability penalty firing on real gold maps (`playability` 0.76, dragging reward 0.95→0.72) because
stacks / overlaps / red-anchor corners are *intentional*. **Recalibrated:** those three are now
distributional **band metrics** (`stack_ratio`, `slider_overlap_ratio`, + existing `slider_anchor_spread_px`);
the penalty is **velocity-only** (`UNHITTABLE_PX_PER_MS` 4.0→10.0); `measure_reward` gained a **`--gold`**
filter + per-defect reporting. The follow-up `--gold` audit (`n=38140`) **confirmed it: reward 0.97,
playability 1.0, `unhittable_jump` rate 0.0.** Its low tail then exposed the grid metric being **1/4-only**
(burst/triplet maps mis-scored, and it's 40% of the reward) → **broadened `on_quarter_grid_ratio` to credit
the {1/4, 1/8, 1/6} grid** (requires a `corpus_stats` refresh to recalibrate its band). 242 tests green,
ruff clean. See [task_reward_flow](../versions/v9/task_reward_flow.md).

**Round 3b (2026-06-24, 2 parallel agents).** (1) **Per-song aim-intensity conditioning** — the diagnosed
primary jump-fix — implemented ([task_persong_conditioning](../versions/v9/task_persong_conditioning.md)):
a loudness-stable audio onset-intensity scalar (`data.audio.aim_intensity`, train↔infer parity), appended
to the context vector (`CONTEXT_DIM` 6→7, `aim` last) + `--aim-intensity` override; backward-compat so
v8/v8_1 still load (`load_model` reads ctx width from the ckpt) AND **generate** (the integration fix:
`_one_pass` truncates ctx to the model's `ctx_dim`). (2) **Early-abort sampling (#4)** — implemented +
GPU-validated ([task_early_abort](../versions/v9/task_early_abort.md)): opt-in, default-off, winner-identical
(confirmed on the v8 ckpt: cand 01 R=0.8196 both ways). **Finding: 0 savings on v8** — its candidate variance
is in SR-closeness, which the quality-only cheap proxy omits; it'll only pay off when candidate *quality*
varies. 274 tests green, ruff clean.

| # | candidate | status + what landed | next |
|---|-----------|----------------------|------|
| 1 | **Brute-force reward over ALL maps** | ✅ **DONE + VALIDATED** — `measure_reward.py` (`--all`/`--workers`/`--bottom-n`/`--gold`/per-defect). Audit-1 (`n=87176`) drove the 3a-fix; audit-2 (`--gold`) confirmed playability 1.0; after the `corpus_stats` refresh, audit-3 (`n=38140`) **confirmed the grid lift: `on_quarter_grid_ratio` 0.98→0.999, rhythm 0.977, reward 0.970, all families 0.94–0.98, the 4 new band metrics 0.92–0.95 (well-calibrated).** The low tail is now genuine stylistic outliers (off-grid gimmicks, unusual-spacing diffs, tech sliders), play 1.0 throughout — healthy discrimination, not a bug. | **Reward track CLOSED.** Use it as a selector (best-of-N), not a maximization target. |
| 2 | **Held-out-SONG val + reward-in-val** | ✅ **implemented** (agent A). Fixed the **leakage bug** (split was by difficulty; mels shared per `audio_id` → audio leaked into val). New `src/data/val_split.py`: union-find group split over song-key (normalized title, +artist if a future manifest adds it) ∪ `audio_id` → whole songs held out; reproducible static `val_split.json`; default `--val-frac` 0.02→0.10. Reward-in-val: gated `--val-reward-every` samples a few held-out songs, decodes, logs mean reward (new `val_reward` CSV col). | USER: freeze a static split + train with it; watch the `val_reward` trend (reward, not MSE, tracks the objective). |
| 3 | **Reward: rhythm ≫ flow + flow done right** | ✅ **implemented + recalibrated** (agent C, then 3a-fix). Family weights rhythm **2.0** / spacing_aim 1.0 / slider_shape 1.0 / flow **0.6** / accents 0.4 (rhythm 40%, flow 12% of quality); `on_quarter_grid_ratio` within-weight 2.0→3.0. **Four distributional band metrics** `stream_spacing_cv`, `slider_anchor_spread_px`, `stack_ratio`, `slider_overlap_ratio` (band-less until a gold-stats refresh; reward ignores band-less). Playability penalty is now **velocity-only** (`UNHITTABLE_PX_PER_MS` 4.0→10.0) — stacks/overlaps/anchors were moved to bands because they are intentional patterns (the 3a-fix). Flat-top anti-hacking + family balance preserved. **Caveat:** `on_quarter_grid_ratio` single-BPM (safe for generated maps; `--gold` avoids it on real maps). | USER: re-run #1 `--gold` to confirm; then rerun `corpus_stats` (heavy) to give the 4 band metrics bands. |
| 4 | **Early-abort sampling on reward trajectory** | ✅ **DONE + GPU-validated** (round 3b). Opt-in `monitor` hook in `ddim_sample` → decode partial x0_hat at late steps → quality-only proxy → abort doomed best-of-N candidates (step-relative threshold + 0.55 floor, default OFF). Winner-identical (proven hermetically + on the v8 ckpt). **But 0 savings on v8** (variance is in SR-closeness, not the quality the proxy sees). | Revisit only if a future model's candidate *quality* varies, or a cheap SR proxy turns up. Low priority. |
| 5 | **Audio input: richer / cleaner features** | ✅ **researched** (agent B) — recommendation note, NOT implemented (any change past value-tweaks = full retrain, since the mel *is* the U-Net input width). Ladder: **(0)** loudness-norm (pyloudnorm / EBU R128) + per-channel mel norm + silence trim + a train↔infer **mel-parity test** (fixes the float16-vs-float32 + `-1.0`-vs-`0.0` pad skew; closes the "same song, different loudness/rip" gap); **(a)** librosa-HPSS percussive mel channel (cheap, rhythm-aligned → best shot at 1/6 over-firing); **(b)** Demucs drums-stem mel; **(c)** frozen MERT/EnCodec embedding @86 fps (big; MERT is CC-BY-NC = release hazard). **Honesty flags:** ~28% novel-song timing is a BPM/offset problem (not audio features); 1/6 over-firing is the one issue a percussion band can actually move. | USER: fold rung 0 into the next planned retrain. |
