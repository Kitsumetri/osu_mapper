# Results

Training-run history + generated-map quality (metrics via `src/metrics.py`).
**Current release: v4b** (`runs/20260614-151630-ranked-full/ckpt/last.pt`, 10-ch).

## v4b — ranked train (current; v4 branch merged to main) — 2026-06-14

`runs/20260614-151630-ranked-full/ckpt/last.pt`, **epoch 48, val_loss 0.00486** (well
below the v4 release's 0.0077). Ranked-only data (`osu!.db` filter → ~23.6k ranked/
approved/loved maps), **more context** (`--crop 4096 --attn-levels 3`), **h/v flip
augmentation**, train/val split. Stopped by the user near convergence (~0.0050).

**Eval (SR sweep, Headphone Actor):** SR monotonic ✓; **17–19/19 metrics in-range**
(beats the v4 release's 16–17); hitsounds 0.23–0.31 ≈ real 0.33. SR offset persists
(target 6→5.69, 5→4.43) → use `--match-sr`. Packaged `[AI-v4b]` (5.92★, 520 obj).

**Play feedback (v4b vs v4 — in-game):**
- ✅ **Kiai much better** — 2 sections start at near-perfect timing (v4 lagged ~10–12 s);
  minor: ends 1–3 s early.
- ✅ **No dead trailing note**; ✅ **hitsounds slightly better**.
- ✅ **Jumps / patterns / streams much better** (streams still "feel bad" but clearly
  improved) — **validates ranked data + context + flip aug**.
- ⚠️ **Rhythm REGRESSED vs v4**: strange 0.5–2 s pauses; some notes off the ¼ grid
  (look 1/6 or 1/8). **NEW top decode issue** → RESEARCH §10.4.
- ➖ **No spinners** generated; ➖ **curve sliders still low** (the slider-representation
  gap the v5 17-ch channels target).

## v4 — full curated library (previous release)

`runs/20260614-110223-std-v4-full` — 31,270 curated maps (≤12★), base 128, **epoch 15,
loss 0.0077** (run killed by an OS sleep at e16, undertrained but strong). SR monotonic,
16–17/19 metrics in-range. Superseded by v4b (ranked data fixes the junk that ≤12★
curation missed). The **decode/post-process wins** shipped on v4 and still in the code:

- `clamp_slider_endpoints` — caps slider length so osu! extrapolation can't shoot tails
  off the playfield (0/116 off-field after `snap_slider_ends`).
- `decode_signal(accent_threshold=0.85)` — accent channels saturate near +1; 0.85 → ~0.33
  hitsound usage (matches real; 0.0–0.6 all stay ~0.52).
- `trim_isolated_ends` — asymmetric trailing trim (2.2 s) + drops a lone circle after the
  final spinner (phantom spin-down note).
- `snap_to_grid` loosened 45→60 ms / 40→50 % (fb #5) — *suspected contributor to the v4b
  rhythm regression; revisit (§10.4)*.
- `compute_breaks` + `write_osu(breaks=)` — `[Events]` breaks for gaps ≥3.5 s (cosmetic;
  marks existing gaps only).

## Earlier versions (v1–v3) — summary

| ver | data | model | loss | takeaway |
|-----|------|-------|------|----------|
| v1 | 601 | base 96, no attn, DDPM | 0.011 | pipeline works end-to-end; too dense, straight sliders, loose rhythm (0.70 on-grid) |
| v2 | 3004 | base 160 (97M), QK-norm attn, bf16 | 0.0075 | much closer to real (density/streams/mix in-range); low jumps, few curves, no SV |
| v3 draft | 1504 | base 128 + difficulty cond + CFG | 0.0097 | **conditioning steers difficulty** ✓ (density/streams scale with target SR) |
| v3 heavy | 6001 | base 128 + cond | 0.0056 | SR near-calibrated 3–5★; curved sliders + kiai + hitsounds all generate |

Durable lessons from these (base 160+bf16 diverges, DDIM not strided DDPM, curved-slider
encoder fix, etc.) live in **HANDOFF §7**.
