# v9 task 3 — RL / post-train alignment to a "ranked-map" reward

**Status:** design (research point #3). Minimal code: a hermetic reward prototype
at `src/eval/reward.py` (+ `tests/test_reward.py`, 7 tests, green + ruff-clean).
**The RL loop itself is NOT built** — this doc is the plan for it.

**Scope.** Two halves, per the task:
- **A.** A concrete, computable reward `R(generated_map, target_SR)` that says
  "this looks like a ranked map", built on the existing machinery
  (`metrics.compute_metrics` + per-SR-bucket reference distributions + rosu SR).
- **B.** A survey of RL / post-train methods for aligning our (v-pred, mostly
  non-differentiable-reward) diffusion model, with a cheapest-first phased plan
  for one RTX 4070 Ti (12 GB), and an honest call on **RL vs the planned v9
  per-song conditioning**.

**The crux this must engage with** (HANDOFF §7, RESEARCH §10.11 Outcome): the
model **mean-regresses to the SR-average map**; per-song extremes (jump-spam,
deathstream, heavy-curve) are under-produced. v8's spacing channel half-failed
because *"a passive channel can't beat the conditioning it shares — the lever
must be NEW information at the input, or an objective that samples extremes
rather than regressing to the mean."* RL is the second of those two levers.

---

## A. The "is this a ranked map?" reward

### A.1 What we already have (reward primitives — do not rebuild)
- `metrics.compute_metrics(bm)` → the descriptive pattern vector (spacing,
  jump/stream ratio, grid-snap, flow turn-angle, curvature, SV, kiai, hitsounds).
- `corpus_stats` → `artifacts/reference_stats.json`: per-SR-bucket
  `{mean, std, p10, p90}` for **every** metric over ~31k real ranked maps. This
  is the gold "what ranked maps look like" distribution. (Numbers are being
  refreshed; the reward is written against the **schema**, not specific values.)
- `metrics.score_against_reference` → already z-scores a map's vector vs a bucket
  and flags p10–p90 membership. A ready-made reward primitive.
- `difficulty.star_rating` (rosu-pp) → exact SR, and `sr_bucket` for the band.

### A.2 Design principle: **band membership, not z-maximisation**
The single most important constraint comes from a lesson already paid for in
play feedback: **maximising a spacing metric makes maps play WORSE**
(`--spacing-scale` lifted spacing *metrics* but relocated objects so flow/
readability degraded — HANDOFF §0, §7). The reward must therefore **saturate at
"indistinguishable from ranked" and give zero gradient for going more extreme.**

So the per-metric sub-score is a **flat-topped tent**: it equals `1.0` *anywhere
inside* the real `[p10, p90]` band and falls off linearly only **outside** it.
There is no slope that rewards pushing a metric past the real distribution — the
optimum is "land in the real band", not "be the most jumpy map possible". This is
strictly less gameable than the naive `−mean|z|` (which a degenerate map can farm
by maximising one z while wrecking play). The prototype encodes exactly this
(`_band_score` is flat at 1.0 in-band; `test_reward_hacking_overshoot_not_better_than_ranked`
asserts an extreme map cannot out-score a ranked-looking one).

### A.3 The formula (matches `src/eval/reward.py`)

```
# per-metric sub-score: flat-topped tent in [0,1]
band_score(v, p10, p90):
    if p10 <= v <= p90:            return 1.0           # flat top — no farming
    half = (p90 - p10) / 2
    dist = (p10 - v) if v < p10 else (v - p90)
    return max(0, 1 - dist / (BAND_FALLOFF * half))     # BAND_FALLOFF = 1.0

# pattern quality: mapper-weighted mean over metrics present in BOTH map & bucket
quality = sum_k w_k * band_score(m[k], ref[bucket][k])  /  sum_k w_k

# SR closeness: smooth, bounded, 1.0 at target, ~0.5 one tol away, 0 if unparseable
sr_close = 0.0 if achieved_sr is None else 1 / (1 + (|achieved_sr - target_sr| / tol)^2)

# blend (convex, NOT a product, so a momentary SR miss doesn't zero a good map)
R = (1 - sr_weight) * quality + sr_weight * sr_close      # sr_weight ~ 0.35
```

**Per-metric weights** (`METRIC_WEIGHTS`, mapper-prioritised — rhythm/spacing/
flow/grid-snap matter most):

| tier | metrics | weight | why |
|------|---------|--------|-----|
| rhythm/timing | `on_quarter_grid_ratio` | 2.0 | a map off the grid is unplayable; the strongest ranked/non-ranked discriminator |
|  | `density_per_s`, `stream_ratio` | 1.5 | wrong density = wrong map for the song |
| spacing/aim | `mean_spacing_px`, `std_spacing_px`, `jump_ratio` | 1.5 | the v8/v9 crux — weighted but **band-capped, never maximised** |
| flow | `mean_turn_angle_deg` | 1.0 | already ~real; kept so the reward *notices a regression* |
|  | `reversal_ratio` | 0.75 | |
| shape | `curved_slider_ratio`, `slider_ratio`, `new_combo_ratio`, `sv_changes_per_min` | 0.5–1.0 | structure, lower stakes |
| cosmetic | `kiai_ratio`, `hitsound_ratio` | 0.25 | handled by a separate v9 head; minor here |

`n_objects`, `bpm`, `duration_s` are deliberately **excluded** — they are scene
facts, not quality signals (and rewarding `n_objects` invites spam).

Weights are renormalised over whatever metrics exist in both the map and the
(possibly refreshed/partial) reference bucket, so the reward is robust to schema
changes — it never crashes on a missing key, it just scores fewer metrics.

### A.4 Reward hacking — the threat model and the guards
| hack | guard in this design |
|------|----------------------|
| maximise one metric to extremes | flat-topped band: zero reward gradient past the real p90 |
| spam objects to inflate density | `density` is band-capped; `n_objects` excluded |
| game spacing while wrecking flow | flow metrics are *in* the weighted set, so flow regression is penalised; **and** held-out play feedback is the final gate (below) |
| satisfy metrics with an unplayable mess | SR-closeness via rosu (a broken map mis-rates or fails to parse → `sr_close=0`); play feedback is decisive |
| over-fit to stale reference numbers | reward reads `reference_stats.json` at call time; refresh it when corpus changes (see Risks) |

**The hard guard the metrics cannot provide: in-game play feedback.** The project
already learned to **trust the mapper's play feedback over the metrics**. Every
phase below has a *play-feedback acceptance gate*, not just a metric gate. The
reward is a *filter and a training signal*, never the final arbiter.

### A.5 Optional richer reward: a learned discriminator
A small classifier `D(metric_vector | bucket) → P(ranked)` trained on **real
ranked maps (positive)** vs **generated + perturbed maps (negative)** gives a
richer, harder-to-game reward than fixed bands (it learns metric *interactions* —
e.g. "high jump_ratio is fine *only if* grid-snap stays high"). Trade-offs:
- **Pro:** captures correlations the per-metric bands miss; **differentiable**
  in the metric vector (relevant for DRaFT/AlignProp — see B.6).
- **Con:** a new moving part that goes **stale** as the generator improves (the
  classic adversarial-drift failure: the policy learns the discriminator's blind
  spots). Needs periodic retraining on fresh generations, and a held-out set.
- **Verdict:** **not phase 1.** Start with the transparent, debuggable band
  reward. Add the discriminator only if/when we reach DRaFT (which *needs* a
  differentiable reward — rosu SR and the parser are not differentiable). Even
  then, gate it on play feedback and refresh it each round.

---

## B. RL / post-train alignment survey + recommendation

Our constraints, stated plainly:
- **One RTX 4070 Ti, 12 GB.** v8 base-160 train peaks ~5–7 GB; a batch-2 CFG
  forward already ~doubles activation memory; long songs OOM without bf16. Any
  method that backprops through the **denoising chain** multiplies that.
- **Reward is non-differentiable** (rosu SR + the `.osu` parser + decode). Only
  a *learned surrogate* (the A.5 discriminator over metrics) is differentiable —
  and only w.r.t. the metric vector, not the raw signal.
- **The user runs the long GPU trains; I do eval / codegen / short drafts.** So
  the plan must be cheapest-first and front-load the no-train / short-train wins.
- **v-pred + zero-SNR + CFG + adaLN + QK-norm** denoiser (`diffusion.py`,
  `unet.py`). DDIM sampler. Sampling is the rollout for any policy-gradient method.

### B.1 Method survey

**Best-of-N / reward-ranked sampling** (no training). Generate N maps per song
(vary seed; optionally vary SR / guidance / density), score with `R`, keep the
top map(s). Immediate, free of training, directly attacks under-dispersion by
*selecting* the rare high-spacing tail the model already produces with low
probability (HANDOFF §7: extremes are *under-produced*, not *absent*). Also the
**data generator** for every training method below. Cost: N× sampling per song.

**Reward-weighted regression / RWR** (Peters & Schaal 2007; "reward-weighted
fine-tuning"). Self-generate a pool, then fine-tune `best.pt` on it with each
sample weighted by `exp(R/β)` (or simply supervised fine-tuning on the top-k —
"best-of-N distillation"). This is the **standard denoising loss we already
train**, just on reward-filtered self-generations → cheap, stable, no new code in
the diffusion core, no backprop-through-sampling. Memory ≈ normal training. The
single best cheap lever to *shift the output distribution toward the high-reward
tail* (i.e. directly attack mean-regression). Caveat: only amplifies what the
model can already sample (best-of-N's ceiling), and can narrow diversity if β too
low / k too small → keep a diversity guard (B.7).

**DDPO** (Black et al. 2023, *Training Diffusion Models with Reinforcement
Learning*). Treats the T-step denoising as an MDP and does policy gradient
(PPO-style, importance-weighted) on the per-step log-probs with the **terminal
reward** `R(x0)`. Most direct "optimise the sampler for the reward" method;
handles non-differentiable rewards (reward only needs to be *computable*, which
ours is). **Cost on 12 GB is the problem:** PPO needs log-probs over the sampling
trajectory and gradients through *multiple* denoiser forwards per update; on
base-160 at our crop length this is tight and slow. Feasible only with few DDIM
steps in the rollout (e.g. 10–20), small batch, grad-checkpointing, LoRA on the
adaLN/attention blocks rather than full fine-tune. Higher variance / less stable
than RWR; reward-hacking-prone (Black et al. report it).

**DPOK** (Fan et al. 2023). DDPO + KL-regularisation to the pretrained model
(online RL with a KL anchor). The KL term curbs the drift/collapse that plain
DDPO suffers. Same memory profile as DDPO (it *is* online policy gradient through
the chain). Better stability, same 12 GB pressure.

**Diffusion-DPO** (Wallace et al. 2023). Direct Preference Optimisation adapted
to diffusion: train on **preference pairs** (winner, loser) with a closed-form
loss over the diffusion ELBO — **no reward model, no sampling in the loop, no
policy-gradient variance.** We can manufacture pairs trivially: *(ranked real
map, generated map)* or *(higher-R generation, lower-R generation)* on the same
song/SR. Memory ≈ a normal fine-tune (two forwards: policy + frozen reference, on
a noised pair) — **the most 12 GB-friendly *training* method here**, and
empirically more stable than online RL. Strong candidate if RWR plateaus.

**Differentiable-reward backprop — DRaFT / AlignProp** (Clark et al. 2023;
Prabhudesai et al. 2023). Backprop the reward gradient *through the sampling
chain* into the weights. Highest sample-efficiency **but requires a
differentiable reward** — rosu SR and the parser are **not** differentiable. It
would need the A.5 learned-discriminator surrogate, *and* the decode (signal→.osu)
step is non-differentiable too, so you'd score the *signal/metrics* directly, not
the decoded map. Plus backprop-through-chain is the heaviest on memory (DRaFT-K
truncates to the last K steps to cope). **Verdict: not worth it for us** unless
everything cheaper is exhausted — too many moving parts (surrogate + truncation +
chain backprop) for a 12 GB single-GPU project.

**Reward-guided sampling** (classifier/reward guidance at inference, no retrain).
Steer each DDIM step toward higher reward via a guidance term. Needs a reward
gradient w.r.t. the *partially-denoised signal* — again only via a differentiable
surrogate, and noisy at high t. Complements training methods (apply on top of any
checkpoint) but shares DRaFT's "needs a differentiable surrogate" blocker. Treat
as a **later, optional** complement, not a primary lever. Best-of-N is the
no-train win we take first instead.

### B.2 Comparison table

| method | needs train? | needs differentiable reward? | 12 GB feasible? | stability | expected payoff |
|--------|:---:|:---:|:---:|:---:|---|
| **Best-of-N / reward ranking** | no | no | yes (sampling only) | n/a | medium, **immediate**; selects existing tail; data source |
| **RWR / reward-weighted FT** | yes (short) | no | yes (≈ normal train) | high | medium-high; shifts dist toward tail; cheap |
| **Diffusion-DPO** | yes | no | **yes** (2 fwd, no rollout) | high | medium-high; stable; pairs from ranked-vs-gen |
| **DDPO** | yes (RL) | no | tight (LoRA + few steps) | medium | high ceiling, hack-prone, costly |
| **DPOK** (DDPO + KL) | yes (RL) | no | tight (same as DDPO) | medium-high | high ceiling, more stable than DDPO |
| **DRaFT / AlignProp** | yes | **YES** (surrogate needed) | heavy (chain backprop) | medium | high but **blocked** by non-diff reward + decode |
| **Reward-guided sampling** | no | **YES** (surrogate) | yes | medium | complement only; needs surrogate |

### B.3 Recommended phased v9 plan (cheapest-first)

**Phase 0 — wire the reward + a best-of-N harness (I do this; no GPU train).**
- Promote `src/eval/reward.py` to the canonical reward; add a thin
  `generate`-loop wrapper that samples N per (song, SR), scores with `R`, writes
  the top map + a JSON of the full `RewardBreakdown` (already structured for
  audit). Reuse `load_model`/`prepare_audio` (one model load, N samples).
- **Cost:** my time + the user's sampling GPU-minutes (N× normal inference). No
  training. **Acceptance:** on a held-out jump song (e.g. Happppy) best-of-N
  measurably lifts spacing/jump_ratio *toward* the real band **and the user
  confirms the picked map plays better** than the single-sample baseline. If
  best-of-N alone closes the gap to the user's satisfaction, **we may not need
  RL at all** — that is a real and welcome outcome.

**Phase 1 — RWR / best-of-N distillation fine-tune (USER runs a SHORT train).**
- Generate a reward-ranked self-corpus across many songs/SRs; fine-tune `best.pt`
  on the top-reward maps (weight `exp(R/β)`, or supervised on top-k). Same
  training loop, same memory, ~a few epochs (short vs the 5–6 h full train).
- **Cost:** my codegen + corpus build; the user runs a short fine-tune.
  **Acceptance:** the *base* (single-sample, no best-of-N) checkpoint now
  produces per-song spacing/jumps closer to the real band on held-out songs, with
  **no flow/readability regression in play feedback** and SR still on target.
  Diversity guard: per-SR spacing spread must not collapse (B.7).

**Phase 2 — Diffusion-DPO (USER runs a moderate train) — only if RWR plateaus.**
- Build preference pairs: *(real ranked map, generated)* and *(high-R gen,
  low-R gen)* per song/SR; run the DPO loss (policy + frozen reference). Most
  12 GB-friendly *training* upgrade and more stable than online RL.
- **Acceptance:** beats the RWR checkpoint on held-out play feedback at matched
  SR; no collapse.

**Phase 3 — DDPO/DPOK (LoRA, few-step rollout) — only if DPO is insufficient and
the reward has proven trustworthy in play.** This is the expensive, hack-prone
frontier; enter it only with a hardened reward (and likely the A.5 discriminator)
because online RL maximises whatever you give it. DRaFT/reward-guidance stay
**out of scope** unless a differentiable surrogate is independently justified.

### B.4 RL vs the planned v9 per-song conditioning — the honest call
These attack the **same** root problem (mean-regression / under-produced per-song
extremes) from **opposite ends**, and they are **synergistic, not competing**:
- **v9 conditioning** adds *new information at the input* (an audio-inferred
  aim-intensity / target-spacing scalar, fed like `--density`). RESEARCH §10.11
  is explicit that the spacing channel failed *because it shared the cursor's
  conditioning and had no extra per-song info*; conditioning supplies exactly
  that missing information. It tells the model **which song wants jumps**.
- **RL/RWR** reshapes the *output distribution* away from the conditional mean
  toward the high-reward tail. It tells the model **to actually produce the
  extreme** instead of regressing to `E[·|cond]`.

**Highest-leverage bet: per-song conditioning first.** Reasoning:
1. It addresses the *diagnosed* cause. §10.11's own conclusion is that the lever
   must be "NEW information at the input" — conditioning *is* that lever; RL is
   the *alternative* ("an objective that samples extremes"). The diagnosis points
   at conditioning first.
2. RL can only amplify the tail the model *can already sample*; if the model has
   no per-song signal, RWR/DPO will push spacing up **globally** (the same failure
   as uniform `--spacing-scale`, which **hurt** in play) rather than *per song*.
   Conditioning is what makes "more jumps **on the jumpy song**" expressible at
   all — it gives RL a per-song axis to optimise along.
3. Cost/risk: conditioning is a reprocess + one train the project already knows
   how to do; RL adds reward-hacking and stability risk on top.

**But sequence them, don't pick one.** Best plan: **(a) ship v9 per-song
conditioning**, then **(b) run best-of-N + RWR on the conditioned model** to
sharpen the tail the conditioning unlocks. Conditioning makes the per-song extreme
*expressible*; RL makes the model *commit* to it. Phase 0 (best-of-N) can and
should run *before/alongside* conditioning anyway — it's free and tells us how
much tail already exists to select from.

---

## C. Risks and guards
| risk | symptom | guard |
|------|---------|-------|
| **reward hacking** | metrics climb, maps play worse | flat-topped band reward (no overshoot gradient); multi-metric incl. flow; **play-feedback gate every phase**; prefer DPO/RWR over online RL |
| **distribution collapse** | all maps look the same; spacing variance drops | KL/reference anchor (DPOK, DPO's frozen ref); RWR temperature β not too low; **diversity metric** (per-SR std of `std_spacing_px`) must not fall vs baseline |
| **reward-model staleness** | (if A.5 discriminator) reward stops discriminating as gen improves | retrain D each round on fresh gens; hold-out set; prefer the fixed band reward until DRaFT forces a surrogate |
| **stale reference stats** | reward calibrated to an old corpus | reward reads `reference_stats.json` at call time; refresh after any corpus change; band reward degrades gracefully on schema changes (renormalises over present metrics) |
| **global vs per-song over-spacing** | RL lifts spacing everywhere (the `--spacing-scale`-hurt failure) | do per-song **conditioning first** so "more jumps" is conditioned per song; reward's SR-closeness + flow terms penalise blanket over-spacing |
| **12 GB OOM in online RL** | DDPO/DPOK won't fit | LoRA + few-step (10–20) DDIM rollouts + grad-checkpoint; default to DPO (no rollout) instead |

---

## D. Key papers (for follow-up)
- **DDPO** — Black, Janner, Du, Kostrikov, Levine, *Training Diffusion Models with
  Reinforcement Learning* (2023). Denoising-as-MDP policy gradient.
- **DPOK** — Fan et al., *DPOK: Reinforcement Learning for Fine-tuning
  Text-to-Image Diffusion Models* (2023). DDPO + KL regularisation.
- **Diffusion-DPO** — Wallace et al., *Diffusion Model Alignment Using Direct
  Preference Optimization* (2023). Preference pairs, no reward model / no rollout.
- **DRaFT** — Clark et al., *Directly Fine-tuning Diffusion Models on Differentiable
  Rewards* (2023). Backprop through (truncated) sampling; needs a diff. reward.
- **AlignProp** — Prabhudesai et al., *Aligning Text-to-Image Diffusion Models with
  Reward Backpropagation* (2023). Same family as DRaFT.
- **RWR** — Peters & Schaal, *Reinforcement Learning by Reward-Weighted Regression*
  (2007); modern diffusion form ≈ reward-weighted / best-of-N self-distillation.
- Supporting (already in this repo's stack): Salimans & Ho 2022 (v-prediction);
  Lin et al. 2023 (zero-terminal-SNR + guidance rescale); Hang et al. 2023
  (Min-SNR-γ loss weighting).

---

## E. Summary
- **Reward (one line):** `R = 0.65 · weighted-mean band-membership of the map's
  metrics inside the real per-SR p10–p90 bands + 0.35 · rosu SR-closeness` —
  flat-topped so it **cannot be farmed by going more extreme**, and always gated
  by in-game play feedback.
- **Phase-1 action:** wire `src/eval/reward.py` into a **best-of-N reward-ranked
  sampling** harness (no training, immediate) → then a short **RWR fine-tune** on
  reward-ranked self-generations.
- **Highest-leverage bet:** **v9 per-song aim-intensity *conditioning* first**
  (it supplies the *missing per-song information* §10.11 identified as the actual
  cause), **then RL/RWR on the conditioned model** to commit to the tail
  conditioning unlocks. They are synergistic; conditioning is the higher-leverage
  *first* move, RL is the multiplier.
