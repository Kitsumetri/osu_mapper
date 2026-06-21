# v9 — policy-gradient (DDPO / DPOK) feasibility deep-dive

**Status:** research (user request #3). Goes DEEPER than the existing survey
(`docs/v9/task3_rl_alignment.md`, RESEARCH §10.12.3) on **policy-gradient
feasibility specifically**, then either green-lights a concrete plan or falls back
to the existing RWR/DPO recommendation with evidence.

**Prototype:** `src/rl/sample_logprob.py` (+ `tests/test_rl_logprob.py`, 8 tests,
green + ruff-clean). It implements the *one missing primitive* — a stochastic-step
log-prob with gradients into the denoiser output — on toy tensors. It is **NOT
wired into training or `diffusion.py`** (another concern owns the model files); it
is the proof that the per-step log-prob math is correct and the score-function
gradient flows. `diffusion.py` changes are specified in §6 for the orchestrator.

---

## 1. One-line verdict

> **Policy gradient (DDPO/DPOK) is *technically* feasible on the 12 GB 4070 Ti —
> but only with LoRA + a very short (8–12-step) stochastic rollout + grad-checkpoint
> + batch-1×accum — and it is NOT the right first move.** The denoising-as-MDP
> requires a stochastic sampler we do not have (production DDIM is `eta=0`,
> deterministic, `@torch.no_grad()`), the reward is high-variance with a flat-topped
> ceiling that *starves* the policy gradient of signal exactly where it should
> saturate, and the same "big search space" that motivates RL is better and far more
> cheaply exploited by **best-of-N → RWR (reward-weighted / best-of-N distillation)
> → Diffusion-DPO**, which need no rollout, no log-prob, no new sampler, and ~normal
> training memory. **Recommended phase-1 RL action: RWR / best-of-N distillation on
> the (per-song-conditioned) model.** DDPO/DPOK stay a *phase-3 frontier*, entered
> only if DPO plateaus and the reward has proven trustworthy in play.

The rest of this doc earns that verdict: §2 frames the MDP, §3 works the log-prob
math (the crux), §4 the 12 GB memory budget, §5 the reward-hacking / stability
risks, §6 the exact `diffusion.py` additions a real DDPO would need, §7 the
verdict + phased plan, §8 papers.

---

## 2. The denoising-as-MDP framing (DDPO)

DDPO (Black et al. 2023) casts the `T`-step reverse process as a finite-horizon MDP
and runs policy gradient with the **terminal reward** `R(x_0)`:

| MDP element | here |
|---|---|
| state `s_t` | `(x_t, t, cond=mel, ctx=difficulty)` — the noisy signal + conditioning |
| action `a_t` | `x_{t-1}` — the next, less-noisy signal |
| policy `π_θ(a_t | s_t)` | `N(x_{t-1}; μ_θ(x_t,t,cond,ctx), σ_t² I)` — one reverse step |
| transition | deterministic given the action (`s_{t-1}` just unpacks `x_{t-1}`) |
| reward | `0` for `t>0`, `R(x_0)` at the last step (our `reward.py` scalar) |

Policy gradient (REINFORCE / score-function estimator):

```
∇_θ J(θ) = E_{τ~π_θ} [ Σ_{t} ∇_θ log π_θ(x_{t-1} | x_t) · (R(x_0) − b) ]
```

where `b` is a baseline (running mean reward) for variance reduction. DDPO uses the
PPO-clipped, importance-weighted form so it can take several gradient steps per
rollout batch:

```
L(θ) = E [ Σ_t min( ρ_t · Â,  clip(ρ_t, 1±ε) · Â ) ],
  ρ_t = π_θ(x_{t-1}|x_t) / π_{θ_old}(x_{t-1}|x_t),   Â = (R − mean)/std
```

**The non-obvious requirement this exposes:** `log π_θ` is only well-defined if the
step is *stochastic*. Our production sampler (`diffusion.ddim_sample`) defaults to
`eta=0`, i.e. `σ_t = 0`, i.e. a **Dirac** transition — `log π` is `−∞`/undefined and
`∇_θ log π` is identically zero. **DDPO/DPOK cannot run on the deterministic
sampler.** We must sample with **DDIM `eta>0`** (or DDPM ancestral, `p_sample`).
This is the single biggest "what would we have to add" item, and it is exactly what
the prototype demonstrates.

DPOK (Fan et al. 2023) = DDPO + a per-step KL anchor `β·KL(π_θ ‖ π_ref)` to the
frozen pretrained policy, which curbs the reward-hacking/collapse DDPO is prone to.
Same memory profile (it *is* online policy gradient through the chain), strictly
more stable. Both are in scope here; DPOK is the one you'd actually run if you ran
either.

---

## 3. The per-step log-prob — full math (the crux)

This is worked against `diffusion.py:ddim_sample` term-for-term so a real
implementation is a copy of the existing update with `eta>0` + a log-prob line. The
prototype `src/rl/sample_logprob.py` is the executable version; its tests assert the
mean is bit-identical to the production deterministic update.

### 3.1 v-prediction → (x0, eps)

The model predicts velocity `v`. At a step with `a_t = √ᾱ_t`, `s_t = √(1−ᾱ_t)`,
`diffusion._to_x0_eps` (for `objective='v'`) gives:

```
x0  = a_t · x_t − s_t · v
eps = s_t · x_t + a_t · v
```

Both are **affine in `v`** (the denoiser output), so any gradient that reaches `x0`
or `eps` reaches `v` and hence θ. This is why the policy gradient is computable at
all for v-pred — there is no non-differentiable step between the network output and
the transition mean.

### 3.2 The DDIM Gaussian transition (eta>0)

Reproducing the `k>0` branch of `ddim_sample`, but keeping the `eta` noise:

```
σ_t   = eta · √[ (1−ᾱ_prev)/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_prev) ]        # the code's `sigma`
μ_θ   = √ᾱ_prev · x0  +  √(1 − ᾱ_prev − σ_t²) · eps              # √(...)*eps is `dir_xt`
x_{prev} = μ_θ + σ_t · z,   z ~ N(0, I)
```

So `π_θ(x_{prev}|x_t) = N(x_{prev}; μ_θ, σ_t² I)` — **isotropic** (one scalar `σ_t`
shared over all channels and frames). The mean `μ_θ` is *bit-identical* to the
deterministic update's deterministic part; `eta` only adds `σ_t·z`. (Prototype test
`test_transition_mean_matches_deterministic_ddim` asserts exactly this.)

### 3.3 The log-prob, and which distribution it is

For a realised action `x_{prev}`, the diagonal-Gaussian log-density, summed over
channels `C` and frames `T` (so it is **per batch element**, `(B,)`, because the
reward is per map):

```
log π_θ(x_{prev}|x_t) = −½ Σ_{c,t} [ (x_{prev}−μ_θ)²/σ_t²  +  log(2π σ_t²) ]
```

(Prototype `gaussian_logprob`; test asserts it equals `torch.distributions.Normal`.)

**CFG — which distribution?** The production sampler combines the conditional and
unconditional *outputs* **before** the v→(x0,eps) conversion:
`out = out_u + g·(out_c − out_u)`, then converts `out`. The policy we roll out — and
therefore the one whose log-prob we score — is the **guided** sampling distribution
built from this combined `out`. That is the correct choice for DDPO/DPOK: *you
optimise the distribution you actually sample from.* Concretely, in the rollout
`out` (and thus `μ_θ`) is a function of **two** forward passes (cond + uncond), so a
single policy-gradient step backprops through **both** — a 2× forward cost per
denoising step that the memory budget (§4) must carry. (Black et al. and most DDPO
re-implementations fine-tune at a fixed guidance and treat the guided sampler as the
policy; this matches our setup.)

### 3.4 Interaction with zero-terminal-SNR (the edge steps)

Zero-SNR rescales the schedule so `ᾱ_T = 0` (`_rescale_zero_terminal_snr`). Two
consequences for the log-prob:

1. **The very first sampled step** (largest `t`, `ᾱ_t → 0`): `a_t = √ᾱ_t ≈ 0`,
   `s_t ≈ 1`, so `x0 = a_t·x_t − s_t·v ≈ −v` and `eps ≈ x_t`. `σ_t` is finite and
   the log-prob is well-defined; no division blow-up (the `s_t` in the denominator
   of the eps-path never appears in v-pred — v→x0/eps multiplies, never divides). So
   zero-SNR is *benign* for the log-prob, unlike eps-prediction (which is undefined
   at SNR=0 — the very reason the project uses v-pred).
2. **The last step (`k==0`)** returns `x0` **deterministically** (the `if k==0:
   x = x0; break` branch) — there is no Gaussian transition, so it carries **no
   log-prob**. The DDPO sum runs over the `K−1` stochastic transitions; the terminal
   `x0` is the action whose reward we score, not a log-prob-bearing step.

The prototype takes `a_t, s_t, ᾱ_t, ᾱ_prev` as explicit inputs precisely so the
caller controls these edge steps; the helper is correct for every interior step.

### 3.5 What the prototype proves

`tests/test_rl_logprob.py` (8 tests, hermetic, CPU):

- the v→(x0,eps) conversion is bit-identical to `diffusion._to_x0_eps`;
- the transition **mean** equals the production deterministic (eta=0) update — so
  turning on `eta` *only* opens the log-prob, it doesn't move sampling;
- `gaussian_logprob` matches `torch.distributions.Normal`;
- **the score-function gradient flows into the denoiser output** (`out.grad` is
  non-zero after `(advantage · log_prob).sum().backward()`) — i.e. the policy
  gradient is computable, which is the whole feasibility question;
- `eta=0` degenerates to the deterministic update (σ=0, action==mean);
- DPOK's closed-form per-step KL is correct (0 iff the policies coincide).

**Conclusion of §3: the math is sound and the gradient flows.** Feasibility now
turns entirely on **memory/throughput (§4)** and **reward signal quality (§5)**.

---

## 4. Memory & throughput budget on 12 GB (the real gate)

Grounding numbers (HANDOFF §7, RESEARCH §10.11): v8 = **base-160, 101.7M params,
21-ch, crop 4096**; full-train peak **~5–7 GB**; a **batch-2 CFG forward already
~2× activation memory**; an 8-min song at base-160 OOMs in fp32 (~14 GB) and needs
bf16 to fit (~11.5 GB). Weights are ~5% of peak, activations ~80% → **activation
memory is the binding constraint**, and DDPO multiplies it by the rollout length.

### 4.1 Why naive DDPO does NOT fit

A policy-gradient update must backprop through the denoiser forwards that produced
`μ_θ` along the rollout. Per stochastic step that is **2 forwards** (CFG cond +
uncond, §3.3). The activation memory of a *single training forward* at base-160 /
crop-4096 is already ~5–6 GB. Holding the graph for even a handful of steps is
multiples of that:

```
peak ≈ (per-step forward activations) × (rollout steps kept in graph) × (CFG factor 2)
```

A 20-step rollout × 2 (CFG) × ~5 GB ≫ 12 GB by an order of magnitude. **Naive
full-parameter DDPO with the graph over the whole rollout is impossible here.** This
matches Black et al.'s own practice (they LoRA-tune SD and use gradient
checkpointing / short effective rollouts).

### 4.2 What makes it fit — the levers, quantified

1. **Detach the rollout; recompute log-probs (the standard DDPO loop).** Do not
   hold the sampling graph. Roll out under `no_grad` (cheap, = normal inference),
   store `(x_t, x_{t-1}, t)` per step, then in the *update* phase recompute
   `log π_θ(x_{t-1}|x_t)` **one step at a time** with grad and call `.backward()`
   per step (gradient accumulation over steps). Peak graph = **one** step's forward,
   not the whole rollout. This is the key structural move; the prototype's
   "score a fixed action" path (`x_prev=` argument) is exactly this recompute.

2. **LoRA on adaLN + attention** (the conditioning + long-range blocks). Full
   fine-tune holds optimizer state (AdamW = 2× params in fp32) + grads for 101.7M
   params ≈ ~1.2 GB just for opt/grad. LoRA (rank 8–16 on `ResBlock1d.ada`,
   `AttnBlock1d.qkv/proj`, the `ctx_mlp`) cuts trainable params ~100× → opt/grad
   becomes negligible, and **only LoRA activations need grad** (the frozen backbone
   runs in inference mode / bf16). This is the difference between "tight" and
   "fits".

3. **Grad-checkpoint** (`UNet1d` already has `grad_ckpt` + `_run` checkpointing) —
   trade compute for the ~80%-of-peak activation term during the per-step recompute.

4. **Short rollout: 8–12 DDIM steps** (vs the production 100). Policy-gradient
   variance and cost both scale with the step count; few-step DDIM is standard for
   DDPO. Fewer steps = fewer log-prob terms = less recompute. (Quality cost: 8–12
   steps is coarser than 100, but RL fine-tunes a pretrained sampler, and best-of-N
   already showed the tail exists at modest budgets.)

5. **batch-1 + gradient accumulation** for the policy-gradient batch; accumulate
   advantages over many rolled-out maps before stepping the optimizer.

6. **bf16 autocast** around the forwards (the project's proven regime; enables flash
   SDPA → O(T) attention memory, the long-song fix).

### 4.3 A concrete config I believe fits 12 GB

```
sampler:        DDIM eta=1.0 (full ancestral-equivalent), 10 steps, guidance fixed at 2.0
trainable:      LoRA rank 16 on {ResBlock.ada, AttnBlock.qkv, AttnBlock.proj, ctx_mlp}
                (backbone frozen, bf16)
rollout:        no_grad, batch 4 maps/song, crop 4096 (or shorter 2048 crops for RL)
update:         per-step log-prob recompute with grad_checkpoint, batch 1,
                accumulate over 4 maps × 10 steps = 40 micro-backwards / optimizer step
precision:      bf16 autocast on forwards; fp32 schedule math + log-prob (as production)
KL anchor:      DPOK β·KL(π_θ‖π_ref) per step (closed-form, prototype gaussian_kl)
```

**Estimated peak:** rollout phase ≈ inference (~5–7 GB at base-160 crop-4096, bf16).
Update phase ≈ one checkpointed forward (~2–3 GB activations) + LoRA grads
(negligible) + frozen weights (~0.4 GB) ≈ **well under 12 GB**. The risk is not
OOM with this config; it is **throughput** — 40 micro-backwards per optimizer step,
each a base-160 forward, makes each RL update **~40× a normal training step**, and
online RL needs thousands of updates. On a single 4070 Ti this is **days, not
hours**, of GPU time per experiment — and the USER runs the trains. *That* is the
practical disqualifier, not memory.

**Verdict of §4:** it *fits* (with LoRA + detached-rollout + recompute + 10 steps +
checkpointing), but it is **slow and operationally heavy** for a single-GPU,
USER-runs-the-train project — and §5 shows the reward gives it little to work with.

---

## 5. Reward-hacking & stability risks specific to online RL here

The reward (`src/eval/reward.py`) was *designed* to be best-of-N/RWR-safe; online
policy gradient stresses it differently:

1. **The flat-topped band *starves* the policy gradient.** The reward is `1.0`
   anywhere inside the real `[p10,p90]` band with **zero gradient** inside it
   (the anti-hacking core). Best-of-N/RWR love this (saturating = safe). But a
   policy gradient `∇J = E[∇logπ · (R−b)]` needs **reward variance** to move; once
   most samples land in-band, advantages collapse toward zero and learning stalls —
   except on the *out-of-band tail*, where the only remaining gradient points
   *inward*. So online RL would mostly learn to pull outliers back into the band,
   not to robustly hit the per-song extreme. The "big search space" (best-of-N
   reward range 0.63–0.83 at SR5) is real headroom, but a flat-topped reward
   converts it into a **plateau**, which suits *selection* (best-of-N) and
   *reweighting* (RWR) far better than *gradient ascent* (DDPO).

2. **Non-differentiable, parse-dependent reward → high variance + brittle.** Each
   reward needs a full sample → decode → `.osu` parse → rosu SR → metrics. A parse
   failure or a NaN/clamp edge zeros the reward (`sr_close=0`), injecting large
   spurious negative advantages. DDPO is already high-variance; a noisy terminal
   reward makes the variance worse, and there's no cheap variance reduction beyond
   the running-mean baseline.

3. **Global over-spacing — the `--spacing-scale` failure, again.** The project
   already learned that *maximising* spacing **hurts play** even when metrics rise.
   Online RL without per-song conditioning would push spacing **globally** toward
   whatever raises reward — the exact uniform-respace failure mode — because the
   policy has no per-song axis to push *only the jumpy song*. This is why
   **per-song conditioning must precede RL** (RESEARCH §10.12.3, restated in §7).

4. **Distribution collapse.** Online RL can collapse diversity (all maps → the
   reward-maximising mode). Guards: DPOK's KL anchor (closed-form, prototype
   `gaussian_kl`), and the project's diversity metric (per-SR std of
   `std_spacing_px` must not fall vs baseline).

5. **Reward staleness / surrogate drift.** Not unique to RL, but online RL exploits
   blind spots fastest. The band reward reads `reference_stats.json` at call time;
   if a learned discriminator is ever added (A.5 in task3), it must be refreshed
   each round.

**The hard guard the metrics cannot provide remains in-game play feedback** —
decisive for every phase. Online RL's appetite for whatever you give it makes the
play-feedback gate *more* important, not less.

---

## 6. Exact `diffusion.py` additions a real DDPO would need (spec — I do NOT edit it)

For the orchestrator/USER to implement later (the model files are another concern's;
the prototype is the reference implementation):

1. **A stochastic, grad-capable sampler** `sample_with_logprob(...)` that:
   - is **not** `@torch.no_grad()` (or has a `no_grad` rollout mode + a grad
     recompute mode);
   - takes `eta>0`; for each interior step computes `σ_t`, `μ_θ`, samples
     `x_{prev}=μ_θ+σ_t z`, and returns **per-step** `(x_t, x_{prev}, t, log_prob,
     μ_θ, σ_t)` (the prototype's `StepLogProb` is the shape);
   - combines CFG **outputs** before the v→(x0,eps) conversion (as `ddim_sample`
     already does) so the scored policy is the guided one (§3.3);
   - keeps the `k==0` terminal `x0` deterministic and **log-prob-free** (§3.4);
   - returns the trajectory so the DDPO loop can detach it and recompute log-probs.

2. **A log-prob recompute entry** `step_logprob(out, x_t, x_prev, schedule_terms,
   eta)` for the PPO update phase (score a *fixed* action) — the prototype's
   `ddim_step_with_logprob(..., x_prev=fixed_action)` path, verbatim.

3. **(DPOK) the per-step Gaussian KL** to a frozen reference policy — the
   prototype's `gaussian_kl` (closed form, equal-σ reduces to `‖μ−μ_ref‖²/2σ²`).

Everything else (LoRA wrapping, the PPO outer loop, advantage normalisation,
optimizer) lives in a new `src/rl/` trainer, **not** in `diffusion.py`. The
prototype already isolates the parts that are pure math; only the
sampler-without-`no_grad` belongs in the model file.

---

## 7. Verdict + phased plan

### 7.1 The honest call

**Policy gradient is feasible-but-not-worth-it as the first RL lever**, for three
compounding reasons:

- **Operational:** even the fit-on-12 GB config (§4.3) is ~40× a training step per
  update × thousands of updates = days of single-GPU time per experiment, run by the
  USER, with PPO's tuning surface (clip ε, KL β, advantage norm, rollout length) on
  top. RWR/DPO are ~normal-training cost.
- **Signal:** the flat-topped reward (§5.1) gives gradient ascent a plateau, which
  *selection* (best-of-N) and *reweighting* (RWR) exploit better than a policy
  gradient does. The "big search space" is an argument for **harvesting the tail**
  (best-of-N/RWR), not for gradient-climbing a saturating reward.
- **Diagnosis:** the root cause is *mean-regression for lack of per-song
  information* (RESEARCH §10.11). RL alone pushes spacing **globally** (the
  `--spacing-scale`-hurt failure). **Per-song conditioning must come first**; only
  then does "more jumps **on the jumpy song**" become expressible for *any* RL
  method to optimise.

DDPO/DPOK remain a **legitimate phase-3 frontier** — the prototype shows the math
is sound and the gradient flows, and the §4.3 config fits — to be entered **only if**
Diffusion-DPO plateaus *and* the reward has proven trustworthy in play *and* a
hardened (likely learned) reward is in place. Not before.

### 7.2 Phased plan (cheapest-first; supersedes nothing in task3, sharpens its B.3)

| phase | method | who runs | train? | rollout/log-prob? | 12 GB | gate |
|---|---|---|---|---|---|---|
| **0** | best-of-N reward ranking (DONE, `src/best_of_n.py`) | USER samples | no | no | yes | USER play-feedback on a jump song |
| **0.5** | **v9 per-song aim-intensity conditioning** (the diagnosed primary fix) | USER | reprocess+train | no | yes | per-song spacing tracks the song, no flow regression |
| **1** | **RWR / best-of-N distillation** on the *conditioned* model (← phase-1 RL action) | USER (short FT) | yes (≈normal) | no | yes | base sample lands in-band per song; diversity not collapsed |
| **2** | Diffusion-DPO (pairs: real-vs-gen, high-R-vs-low-R) | USER (moderate) | yes (2 fwd) | no | yes | beats RWR in play at matched SR |
| **3** | **DDPO/DPOK** (LoRA r16, 10-step eta=1 rollout, recompute log-probs, DPOK KL) — §4.3 | USER (long, days) | yes (RL) | **YES** | tight-but-fits | only if DPO plateaus + reward proven; KL + diversity guards |

**Phase-1 RL action (the deliverable's headline):** **RWR / best-of-N distillation
on the per-song-conditioned model.** Reasoning: it (a) directly exploits the same
high-variance "big search space" the user cited — by reweighting toward the
high-reward tail best-of-N already surfaces — (b) needs **no** new sampler, **no**
log-prob, **no** rollout, just the existing denoising loss on reward-filtered
self-generations, so it is ~normal training memory and the USER can run it short,
and (c) is stable (no PPO variance, no online reward-hacking dynamics). It is the
natural first RL step regardless of whether DDPO is ever reached.

The prototype is kept as the **de-risked seed for phase 3**: if the project ever
gets there, the hard part (correct per-step log-prob for v-pred + zero-SNR + CFG,
with gradients flowing) is already written and tested.

---

## 8. Papers

- **DDPO** — Black, Janner, Du, Kostrikov, Levine, *Training Diffusion Models with
  Reinforcement Learning* (2023). Denoising-as-MDP policy gradient (PPO,
  importance-weighted); LoRA + grad-checkpoint in practice.
- **DPOK** — Fan et al., *DPOK: RL for Fine-tuning Text-to-Image Diffusion Models*
  (2023). DDPO + per-step KL anchor to the pretrained policy.
- **DDIM** — Song, Meng, Ermon, *Denoising Diffusion Implicit Models* (2021). The
  `η`-parameterised family; `η=0` deterministic, `η=1` ≈ DDPM ancestral. The σ_t
  formula `√[(1−ᾱ_prev)/(1−ᾱ_t)·(1−ᾱ_t/ᾱ_prev)]` is theirs (and the code's).
- **Diffusion-DPO** — Wallace et al. (2023). Preference pairs, no reward model / no
  rollout — the recommended phase-2 upgrade.
- **RWR** — Peters & Schaal (2007); modern diffusion form ≈ reward-weighted /
  best-of-N self-distillation — the recommended phase-1 lever.
- Project stack: Salimans & Ho 2022 (v-prediction); Lin et al. 2023 (zero-terminal-
  SNR + guidance rescale); Hang et al. 2023 (Min-SNR-γ).

---

## 9. Summary

- **Verdict (one line):** DDPO/DPOK *fit* 12 GB (LoRA + detached rollout + per-step
  log-prob recompute + 10 DDIM steps + grad-checkpoint — §4.3) and the math is
  correct (prototype, §3), **but** the deterministic-sampler blocker, the
  flat-topped reward that starves the policy gradient, the days-of-single-GPU cost,
  and the diagnosis (per-song conditioning first) make policy gradient the **wrong
  first move**.
- **Phase-1 RL action:** **RWR / best-of-N distillation on the per-song-conditioned
  model** — no sampler change, no log-prob, ~normal memory, stable, and it exploits
  the very "big search space" that motivated the request.
- **Prototype:** `src/rl/sample_logprob.py` (+ 8 tests) proves the per-step log-prob
  for v-pred + zero-SNR + CFG is correct and differentiable — the de-risked seed if
  the project ever reaches phase-3 DDPO/DPOK. Not wired into training; `diffusion.py`
  additions spec'd in §6 for the owning concern.
