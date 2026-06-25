# Docs index — navigation map

**Purpose:** the map of every doc — its one-line purpose and whether it is **STATIC** (knowledge base,
rarely changes) or **DYNAMIC** (changes often). A fresh agent reads this and jumps to the small file it
needs. Each file is small + single-topic with a clear H1 and a one-line static/dynamic header.

**Entry point:** a fresh agent reads **`HANDOFF.md`** (repo root, git-ignored — the live current-state
mirror) first, then follows it into this tree.

---

## How the docs are organised

- **`docs/knowledge/`** — STATIC knowledge base (architecture, the math, the signal spec, osu! mapping
  patterns, hard-won lessons, audit findings, references). Changes only on a real
  representation/architecture change.
- **`docs/versions/`** — frozen per-version design + outcome (v1–v8 done; v9 active). One small file per
  version. The detailed v9 task reports live under `docs/versions/v9/`.
- **`docs/status/`** — DYNAMIC: the roadmap/TODO and the run-history results. (The narrative
  current-state handoff is `HANDOFF.md` at repo root, git-ignored.)

---

## STATIC — knowledge base (`docs/knowledge/`)

| doc | purpose |
|-----|---------|
| [knowledge/architecture.md](knowledge/architecture.md) | what the system is, the end-to-end pipeline, repo map, how to run |
| [knowledge/diffusion-math.md](knowledge/diffusion-math.md) | the math: DDPM forward/reverse + objective, DDIM, U-Net/adaLN/QK-norm attention, CFG, optimisation |
| [knowledge/signal-encoding.md](knowledge/signal-encoding.md) | the 21-channel signal spec + encode + decode (peak-pick, sliders, SV/curve/corner/spacing, timing, beat-snap) |
| [knowledge/mapping-patterns.md](knowledge/mapping-patterns.md) | osu! domain knowledge: pattern vocabulary, slider shapes, style language, timing/difficulty/mode facts, kiai/hitsounds, the conditioning design |
| [knowledge/corpus-stats.md](knowledge/corpus-stats.md) | per-SR-bucket gold distribution (the reward/eval target) + how to read & regenerate it |
| [knowledge/lessons-learned.md](knowledge/lessons-learned.md) | the paid-for lessons: stability, under-dispersion, decode/timing gotchas, perf, ops |
| [knowledge/audit-findings.md](knowledge/audit-findings.md) | the external audit (2026-06-14) + the v9 core-component audit — fixed / deferred / correct |
| [knowledge/references.md](knowledge/references.md) | external links & papers (osu! docs, diffusion/ML, RL, prior-art repos) |

## STATIC/frozen — version history (`docs/versions/`)

| doc | purpose |
|-----|---------|
| [versions/README.md](versions/README.md) | the version index (table of all versions) |
| [versions/v1-v3.md](versions/v1-v3.md) | pipeline → scale → difficulty conditioning + CFG (10-ch) |
| [versions/v4.md](versions/v4.md) | ranked-only data + context + flip aug + decode/post wins |
| [versions/v5.md](versions/v5.md) | slider-anchor channels + reverse sliders (17-ch); style cond deferred |
| [versions/v6.md](versions/v6.md) | adaLN-zero conditioning + gold data |
| [versions/v7.md](versions/v7.md) | v-pred + zero-SNR + SV/curve/corner; the up-attn lesson; v7.5 (best pre-v8) |
| [versions/v8.md](versions/v8.md) | spacing channel + spatial loss weight; base-160 unblocked (RELEASED); + the timing-model design |
| [versions/v9.md](versions/v9.md) | **active**: alignment — postprocess fix, reward, best-of-N, per-song conditioning plan |
| [versions/v9/](versions/v9/) | v9 detailed task reports (postprocess, data-stats, RL alignment, autopackage, general reward, RL policy-gradient, core-quality; **round 3a:** [holdout-val](versions/v9/task_holdout_val.md), [reward-flow](versions/v9/task_reward_flow.md), [audio-features](versions/v9/task_audio_features.md); **round 3b:** [per-song conditioning](versions/v9/task_persong_conditioning.md), [early-abort sampling](versions/v9/task_early_abort.md)) + [v8_1-ablation](versions/v9/v8_1-ablation.md) (rope+huber A/B) |

## DYNAMIC — status (`docs/status/` + repo root)

| doc | purpose | static/dynamic |
|-----|---------|----------------|
| `HANDOFF.md` (repo root, **git-ignored**) | **THE ENTRY POINT** — live current-state mirror; read first | DYNAMIC |
| [status/roadmap.md](status/roadmap.md) | priority-ordered task list + available decode levers | DYNAMIC |
| [status/results.md](status/results.md) | run-by-run training history (loss, eval, play feedback) — appended per train | DYNAMIC |

## Root docs (user-facing)

| doc | purpose |
|-----|---------|
| `README.md` (repo root) | user-facing: install, generate a map, train your own, project layout |

---

## Where the old monolith docs went

The old root `RESEARCH.md`, `TECH_REPORT.md`, and `RESULTS.md` were split into the tree above:

- **`TECH_REPORT.md`** (the math) → [knowledge/diffusion-math.md](knowledge/diffusion-math.md) (model
  half) + [knowledge/signal-encoding.md](knowledge/signal-encoding.md) (data/decode half).
- **`RESULTS.md`** (run history) → [status/results.md](status/results.md).
- **`RESEARCH.md`** (design + roadmap) split by topic:
  - §1–2 vocabulary/sliders, §5 community style, §6 timing/difficulty/modes, §7 kiai/hitsounds, §9
    conditioning design → [knowledge/mapping-patterns.md](knowledge/mapping-patterns.md).
  - §3 encoding changes, §4 decode → [knowledge/signal-encoding.md](knowledge/signal-encoding.md).
  - §8 reference distributions → [knowledge/corpus-stats.md](knowledge/corpus-stats.md).
  - §11 audit follow-ups → [knowledge/audit-findings.md](knowledge/audit-findings.md).
  - §References → [knowledge/references.md](knowledge/references.md).
  - §10.0–10.12 per-version roadmap drafts → the matching [versions/](versions/README.md) files
    (v1-v3 / v4 / v5 / v6 / v7 / v8 / v9).
  - the open roadmap/TODO → [status/roadmap.md](status/roadmap.md).
