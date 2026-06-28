# Version history — frozen per-version design + outcome

**Purpose:** one small file per model version, each with the design (what changed and why) and the
outcome (what the train + play-test showed). The frozen versions (v1–v8) are **STATIC** records; the
**active** version's design lives here too but is the one that changes — see the roadmap for what's
next. | mostly **STATIC** (frozen history); the current version's file is the live one.

These split the old `RESEARCH.md` §10.x roadmap drafts + the per-version outcomes from the old
`RESULTS.md`. The full run-by-run training log (loss numbers, eval tables, play feedback) is in
[status/results.md](../status/results.md).

| version | file | channels | status | one-line |
|---------|------|----------|--------|----------|
| v1–v3 | [v1-v3.md](v1-v3.md) | 10 | frozen | pipeline → scale → difficulty conditioning + CFG |
| v4 / v4b | [v4.md](v4.md) | 10 | frozen | ranked-only data + crop 4096 + flip aug + decode/post wins |
| v5 | [v5.md](v5.md) | 17 | frozen | slider-anchor channels + reverse sliders; style conditioning deferred |
| v6 | [v6.md](v6.md) | 17 | frozen | adaLN-zero conditioning + gold data |
| v7 / v7.5 | [v7.md](v7.md) | 19 / 20 | frozen | v-pred + zero-SNR + SV/curve/corner; up-attn lesson |
| v8 | [v8.md](v8.md) | 21 | **RELEASED** | base-160 unblocked + spacing channel + spatial loss weight |
| v9 | [v9.md](v9.md) | 21 (→ TBD) | **IN PROGRESS** | alignment: postprocess fix, reward, best-of-N, per-song conditioning |

The **v9 round-1/round-2 task reports** (the detailed source documents behind v9) live in
[v9/](v9/): postprocess, data-stats, RL alignment, autopackage, general reward, RL policy-gradient,
core-quality.

For the static knowledge these versions build on, see [docs/knowledge/](../knowledge/) (architecture,
diffusion math, signal encoding, mapping patterns, lessons, audit findings).
