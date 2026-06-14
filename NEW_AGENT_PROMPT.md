# New-agent kickoff prompt

Paste this to start a fresh session (Opus 4.8 High recommended). It is
self-contained; the first action is to read the repo docs for full detail.

---

You are continuing **osu_mapper**, a from-scratch ML project that trains a
**conditional diffusion model to generate osu!standard beatmaps from raw audio**.
The previous agent built it through a working **v4** release and handed off due to
context limits. Work autonomously: write code, run it, fix errors, iterate. Be
honest about quality (state metrics, don't oversell).

**First, read these (in the repo root), in order:**
1. `HANDOFF.md` — full working context: architecture, repo map, how-to-run,
   current state, open TODOs (§5b), per-feedback issues (§7), hard-won lessons
   (§8), conventions (§9). **This is the source of truth.**
2. `RESEARCH.md §10` — the prioritised **v4/v5 roadmap**.
3. `RESULTS.md` — run history + the v4 release status.
4. `TECH_REPORT.md` — the math, if you need it.

**Current state (2026-06-14):**
- **Release model**: `runs/20260614-110223-std-v4-full/ckpt/best.pt` (base 128,
  10-channel signal, difficulty conditioning + CFG, trained on the full 31k-map
  curated library; epoch 15 — the run was killed by an OS/sleep event at e16, so
  it's slightly undertrained but strong: 16–17/19 metrics in the real range).
- Pipeline works end-to-end: `audio → .osu` with difficulty control (`--sr` +
  `--match-sr`), kiai, hitsounds, curved (RDP) sliders, beat-snapped onsets +
  slider ends. 81 hermetic tests pass, ruff clean. v3 is merged to `main`;
  active branch `feat/v4-fulldata`.

**Environment/constraints:**
- Windows, RTX 4070 Ti (12 GB), 20 CPU cores, Python 3.12, `C:\osu!\Songs`.
- **You cannot push** — the user pushes (interactive auth). Commit locally,
  descriptive messages, `Co-Authored-By: Claude`. Branch off `main` for new work.
- The user's machine has slept mid-training twice — keep checkpoints frequent;
  consider adding a `--resume` flag (train.py has none yet).
- Heavy artifacts (`data/`, `runs/`, `artifacts/`) are git-ignored.

**Your immediate options (pick with the user, or default to the first that adds
most value):**
1. **Finish the v4 model**: add `--resume` to `src/train.py`, continue
   `best.pt` from epoch 15 → ~50 epochs for tighter SR calibration + cleaner
   curves. (Highest value; the model is undertrained.)
2. **Cheap decode wins from v4 play feedback** (no retrain): clamp slider
   endpoints to the playfield; fix the trailing unhittable note; loosen onset
   beat-snap; tune the hitsound threshold; write `[Events]` breaks / density
   control. (See HANDOFF §5b item 2.)
3. **v4 representation batch** (one re-preprocess + retrain): **reverse sliders**
   (`slides` encoding), dedicated slider-shape channels, style/mapper
   conditioning. (RESEARCH §10.1.E/F.)

**How to run** (also in README):
```
pytest ; ruff check .
python -m src.generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --sr 5 --match-sr --out out.osu
python -m src.evaluate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --srs 2,3,4,5,6
python -m src.train --data data/processed/std-v3-all --tag <t>   # base 128 default; do NOT use base 160 (diverges)
python -m src.data.preprocess --songs "C:/osu!/Songs" --out data/processed/<tag> --workers 16
```

Start by reading `HANDOFF.md`, confirm the release model loads and tests pass,
then discuss the plan with the user.
