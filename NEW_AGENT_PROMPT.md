# Continue osu_mapper — fresh-agent prompt

You are continuing **osu_mapper**: a from-scratch ML project that trains a
**conditional diffusion model to generate osu!standard beatmaps from raw audio**.
Work autonomously — write code, run it, fix errors, iterate. Be honest about
quality (state metrics, don't oversell). The user (`Kitsumetri`) is an osu! player
+ ML dev and an experienced **mapper** — trust their play/mapping feedback.

## Read first (repo root), in order
1. **`HANDOFF.md`** — architecture, repo map, how-to-run, current state, open queue,
   hard-won lessons, conventions. The source of truth.
2. `RESEARCH.md` — design + roadmap (§10.x per-version; §10.6 = v6; §11 = audit).
3. `RESULTS.md` — run history + play feedback.
4. `TECH_REPORT.md` — the math, if needed.

## Environment / constraints
- Windows, **RTX 4070 Ti (12 GB)**, 20 cores, `uv` venv, **torch 2.11.0+cu128**.
  Run everything via **`uv run …`** (e.g. `uv run python -m src.train …`,
  `uv run --extra dev python -m pytest`). Songs at `C:\osu!\Songs`, `osu!.db` is current.
- **You cannot push.** Commit locally (descriptive msgs, `Co-Authored-By: Claude`).
  The **user pushes + PRs to main**. v4 and v5 are already merged to main.
- **Your background processes get reaped when the session idles (~40–50 min).** Long
  trains (5–6 h) are **run by the user in their own terminal**; you handle eval,
  codegen, decode work, and short drafts (<~20 min) that fit a turn. Don't re-launch
  a long train as your own background task.
- **Don't re-preprocess data that already exists** — check `data/processed/` first.
- Keep the **81→now hermetic tests** green + **ruff clean** after every change.

## Current state (2026-06-16)
- **Released = v5** (`runs/20260614-224107-ranked-v5/ckpt/best.pt`, 17-ch, epoch 55,
  val 0.0033). 17-channel slider representation (anchor dx/dy + `slides`); curved +
  reverse sliders work. Two play-test rounds of decode fixes shipped (RESEARCH §10.5):
  rhythm snap to {1/4,1/8,1/6}, slider RDP (real lines vs imposter-curves), realistic
  AR (`7.75+0.25·sr`), intro-cluster trim, spinner merge, `--timing-from`, package_map
  keeps generated difficulty. Branch `feat/v5-slider-style`, **merged to main**.
- **v6 TRAINED (branch `feat/v6-sv-adaln`)**: `runs/20260616-013932-ranked-v6/ckpt/best.pt`,
  epoch 59, **val 0.00314**, clean. v6 = **adaLN-zero** (`--adaln` default on) + **gold data**
  (`data/processed/ranked-v6`, **25,073 maps**, 100% kiai + single-BPM + hitsounds≥10%, SR
  1.1–10, 17-ch). Eval done (SR monotonic, in-range 14–17/19; vs v5 ~a wash, SR calibration
  tighter — RESULTS v6). `[AI-v6]` packaged. **Per-slider SV was tried and REVERTED** (SV is
  structural like kiai, not per-slider — see §10.6.A).
- **THE NEXT STEP: in-game play test of `[AI-v6]`** → promote v6 to release if it beats v5
  (kiai consistency, hitsounds, difficulty control via adaLN). Then **structural SV** (§10.6.A).

## How to run
```bash
uv run --extra dev python -m pytest        # hermetic tests (~6 s)
uv run --extra dev ruff check .
# v6 full train (USER runs in their terminal; --adaln defaults True):
uv run python -m src.train --data data/processed/ranked-v6 --tag ranked-v6 --base 128 --crop 4096 --attn-levels 3 --batch 16 --epochs 60 --save-every 5 --augment true --val-frac 0.02 --workers 8
#   resume after interrupt: --resume runs/<id>/ckpt/last.pt
uv run python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --sr 5 --match-sr --timing-from "ref.osu" --out out.osu
uv run python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --srs 2,3,4,5,6 --ref-stats artifacts/reference_stats.json
uv run python -m src.package_map --generated out.osu --original "ref.osu" --prefix "[AI-v6]"
uv run python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/<tag> --gold --workers 10  # gold = ranked+kiai+single-BPM+hitsounds>=10%+1<SR<10
```

## Immediate TODO (do in order; full list in HANDOFF §6 + the task list)
1. ✅ **v6 train done** (2026-06-16): eval (SR sweep), `[AI-v6]` packaged, RESULTS + memory +
   A/B vs v5 written, milestone ckpts pruned. **Remaining: in-game play test → promote to
   release if it beats v5.** Watch the SR4 kiai=0.00 eval outlier in-game.
2. **Structural SV** (the right way this time): SV is a *few coarse sections* tied to
   song structure (slow→low SV, drop→≥1, mostly ~1, rare 1–6 s fast burst), NOT
   per-slider. Options: learn it via an SV channel, or a mel-energy/kiai heuristic.
   **Analyze real SV maps' section count/diversity first.** RESEARCH §10.6.A.
3. **Model-side residue** (from v5 feedback, RESEARCH §10.5): 0.6–0.8 s density gaps
   (density conditioning); novel-song timing ("super timing" — infer 20× + average);
   pattern quality / flow modelling (§10.2).
4. **Deferred/queued** (RESEARCH §11): per-channel target standardisation (#10, also a
   base-160 divergence-fix candidate, pairs with zero-terminal-SNR + v-pred), batched
   CFG (#11), SR-offset bake (#7). `torch.compile` wired (`--compile`) but Windows-blocked
   (no MSVC/triton-windows) — for Linux/cloud.

## Hard-won lessons (don't re-learn — full list HANDOFF §7)
- **base 128 is the proven-stable size**; base 160 + bf16 **diverges**. DDIM not strided
  DDPM. QK-norm attention is load-bearing (the bf16 fix). Frame grid, not ms. SV is
  **structural** not geometric. Slider shape now lives in dedicated anchor channels (v5).
- Adopt techniques (not the whole stack) from the sibling repo `workspace/Mapperatorinator`
  (memory `reference-mapperatorinator`).
