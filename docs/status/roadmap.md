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
