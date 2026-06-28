"""Per-step DDIM log-prob for policy-gradient RL (DDPO/DPOK) — PROTOTYPE ONLY.

This is the single missing primitive for running DDPO (Black et al. 2023) or DPOK
(Fan et al. 2023) on this model: the *log-probability of one stochastic sampler
step*, with gradients flowing into the denoiser output. The full feasibility
analysis (math, memory budget, verdict) is ``docs/v9/task_rl_policy_gradient.md``.

It is **deliberately separate from** ``src/model/diffusion.py`` and **not wired
into anything** — the production ``ddim_sample`` is ``@torch.no_grad()`` with
``eta=0`` (deterministic: zero per-step variance → no log-prob). DDPO/DPOK need a
*stochastic* sampling MDP, i.e. DDIM with ``eta>0`` (or DDPM ancestral). This
module re-derives the same DDIM transition the production sampler uses, but keeps
``eta>0`` and computes the Gaussian transition's log-prob.

The denoising-as-MDP view (DDPO):

* **state**  ``s_t = (x_t, t, cond, ctx)``   (the noisy signal + conditioning),
* **action** ``a_t = x_{t-1}``                (the next, less-noisy signal),
* **policy** ``π_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)``,
* **reward** ``0`` for ``t>0`` and ``R(x_0)`` at the final step (terminal reward).

The policy-gradient estimator is ``∇_θ J = E[ Σ_t ∇_θ logπ_θ(a_t|s_t) · (R - b) ]``
(REINFORCE; DDPO uses the PPO-clipped importance-weighted form). The per-step
``∇_θ logπ_θ`` is what this file makes computable: ``logπ`` depends on θ only
through ``μ_θ`` (the mean), which is an affine function of the **denoiser output**
``out = v_θ(x_t,t,cond,ctx)`` — so a backward pass through ``ddim_step_with_logprob``
flows the gradient straight into the model.

Mathematical contract (matches ``diffusion.py:ddim_sample`` term-for-term):

Let ``a_t = sqrt(ᾱ_t)``, ``s_t = sqrt(1-ᾱ_t)`` at the current timestep and
``ᾱ_prev`` at the next (less-noisy) timestep in the step subsequence.

* **v→(x0, eps)** (v-prediction; identical to ``diffusion._to_x0_eps`` for ``v``)::

      x0  = a_t * x_t - s_t * v
      eps = s_t * x_t + a_t * v

* **CFG** combines conditional/unconditional *outputs* before the conversion
  (exactly as the production sampler does), so the guided ``out`` is::

      out = out_u + guidance * (out_c - out_u)

  The policy whose log-prob we score is therefore the **guided** sampling
  distribution — the one actually rolled out — which is the correct choice for
  DDPO/DPOK (you optimise the distribution you sample from). See the doc, §3.3.

* **DDIM stochastic transition** (eta>0)::

      σ_t   = eta * sqrt((1-ᾱ_prev)/(1-ᾱ_t) * (1-ᾱ_t/ᾱ_prev))
      μ_θ   = sqrt(ᾱ_prev) * x0 + sqrt(1-ᾱ_prev-σ_t²) * eps
      x_{prev} ~ N(μ_θ, σ_t² I)

  Mean ``μ_θ`` is bit-identical to the deterministic (eta=0) update's deterministic
  part; eta only adds the ``σ_t·z`` noise and opens up the log-prob.

* **log-prob** of the realised ``x_{prev}`` under that diagonal Gaussian::

      logπ = -0.5 * Σ [ (x_prev-μ)²/σ² + log(2π σ²) ]   (summed over C,T)

Terminal-step note (zero-terminal-SNR): the *last* step (k==0) of the production
sampler returns ``x0`` deterministically (no transition) and at the **first**
sampled step ``ᾱ_t→0`` under zero-SNR, so the first transition's ``μ`` divides by
``s_t=sqrt(1-ᾱ_t)≈1`` safely, but ``a_t≈0`` — the helper takes ``a_t, s_t,
ᾱ_prev`` as explicit inputs so the caller controls those edge steps (the doc, §3.4
spells out which steps carry a log-prob).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class StepLogProb:
    """One stochastic DDIM step: the sampled next state and its log-prob.

    ``x_prev`` is the sampled action ``x_{t-1}`` (detached is the caller's choice;
    here it is returned *with* graph history so a toy test can check gradients, but
    in a real DDPO loop you detach the rollout and recompute log-probs in the
    update — see the doc, §4.2). ``log_prob`` is summed over channels+time, per
    batch element: shape ``(B,)``. ``mean``/``std`` are exposed for testing and for
    the KL term DPOK needs (KL between two diagonal Gaussians is closed-form).
    """

    x_prev: torch.Tensor      # (B, C, T) sampled next (less-noisy) signal
    log_prob: torch.Tensor    # (B,) log π(x_prev | x_t), summed over C,T
    mean: torch.Tensor        # (B, C, T) transition mean μ_θ
    std: torch.Tensor         # scalar tensor σ_t (isotropic)


def v_to_x0_eps(out: torch.Tensor, x_t: torch.Tensor, a_t: float, s_t: float):
    """v-prediction → (x0, eps), matching ``diffusion._to_x0_eps`` for ``objective='v'``.

    ``a_t = sqrt(ᾱ_t)``, ``s_t = sqrt(1-ᾱ_t)``. Differentiable in ``out``.
    """
    x0 = a_t * x_t - s_t * out
    eps = s_t * x_t + a_t * out
    return x0, eps


def gaussian_logprob(x: torch.Tensor, mean: torch.Tensor,
                     std: torch.Tensor | float) -> torch.Tensor:
    """log N(x; mean, std² I), summed over all non-batch dims → (B,).

    ``std`` is isotropic (scalar or scalar-tensor), matching the DDIM transition's
    single ``σ_t``. Computed in fp32 for stability (the production schedule math is
    fp32 too). Gradients flow through ``mean`` (and ``std`` if it carries grad).
    """
    x = x.float()
    mean = mean.float()
    if not torch.is_tensor(std):
        std = torch.tensor(float(std), dtype=x.dtype, device=x.device)
    std = std.float().clamp(min=1e-12)
    var = std * std
    # per-element Gaussian log-density, then sum over channels + time
    log_norm = math.log(2.0 * math.pi)
    per_elem = -0.5 * (((x - mean) ** 2) / var + torch.log(var) + log_norm)
    return per_elem.flatten(start_dim=1).sum(dim=1)


def ddim_sigma(acp_t: float, acp_prev: float, eta: float) -> float:
    """The DDIM stochastic-transition std σ_t (Song et al. 2021), matching the
    ``sigma = eta * sqrt(...)`` line in ``diffusion.ddim_sample``. eta=0 → 0
    (deterministic, no log-prob). Clamped to be safe at the schedule edges."""
    if eta <= 0.0:
        return 0.0
    ratio = (1.0 - acp_prev) / max(1.0 - acp_t, 1e-12)
    inner = ratio * (1.0 - acp_t / max(acp_prev, 1e-12))
    return float(eta) * math.sqrt(max(inner, 0.0))


def ddim_step_with_logprob(
    out: torch.Tensor,
    x_t: torch.Tensor,
    *,
    a_t: float,
    s_t: float,
    acp_t: float,
    acp_prev: float,
    eta: float,
    x_prev: torch.Tensor | None = None,
    objective: str = "v",
    generator: torch.Generator | None = None,
) -> StepLogProb:
    """One stochastic DDIM step + its log-prob, given the (already-CFG-combined)
    denoiser output ``out``.

    This mirrors the production update exactly:

        x0, eps = v_to_x0_eps(out, x_t, a_t, s_t)          # v-pred conversion
        σ_t     = ddim_sigma(acp_t, acp_prev, eta)
        μ       = sqrt(acp_prev) * x0 + sqrt(1-acp_prev-σ²) * eps
        x_prev  ~ N(μ, σ² I)
        logπ    = gaussian_logprob(x_prev, μ, σ)

    Args:
        out:       guided denoiser output ``out_u + g*(out_c-out_u)`` (B,C,T). Carry
                   grad here to get the policy gradient.
        x_t:       current noisy signal (B,C,T). Detached in a real rollout.
        a_t, s_t:  ``sqrt(ᾱ_t)``, ``sqrt(1-ᾱ_t)`` (the current step's scales).
        acp_t, acp_prev: ``ᾱ_t`` and ``ᾱ_prev`` (cumprod alphas) for σ_t.
        eta:       DDIM stochasticity (>0 required for a non-degenerate log-prob).
        x_prev:    if given, score *this* action's log-prob (the off-policy/PPO
                   recompute path); if None, sample a fresh action from N(μ,σ²).
        objective: only ``'v'`` is implemented (this project's ckpts); ``'eps'``
                   would set ``x0=(x_t-s_t*out)/a_t, eps=out``.
        generator: optional RNG for reproducible sampling of the action.

    Returns a ``StepLogProb``.
    """
    if objective != "v":
        raise NotImplementedError(
            "prototype implements v-prediction only (this project's ckpts use v); "
            "see docstring for the eps conversion")
    x0, eps = v_to_x0_eps(out, x_t, a_t, s_t)
    # match the production clamp on the predicted x0 (diffusion.ddim_sample)
    x0 = x0.clamp(-1.5, 1.5)

    sigma = ddim_sigma(acp_t, acp_prev, eta)
    sqrt_acp_prev = math.sqrt(max(acp_prev, 0.0))
    dir_coef = math.sqrt(max(1.0 - acp_prev - sigma * sigma, 0.0))
    mean = sqrt_acp_prev * x0 + dir_coef * eps

    std = torch.tensor(sigma, dtype=torch.float32, device=out.device)
    if x_prev is None:
        if sigma > 0.0:
            noise = torch.randn(mean.shape, generator=generator,
                                device=mean.device, dtype=mean.dtype)
            x_prev = (mean + sigma * noise)
        else:
            x_prev = mean
    log_prob = gaussian_logprob(x_prev, mean, std)
    return StepLogProb(x_prev=x_prev, log_prob=log_prob, mean=mean, std=std)


def gaussian_kl(mean_p: torch.Tensor, std_p: torch.Tensor | float,
                mean_q: torch.Tensor, std_q: torch.Tensor | float) -> torch.Tensor:
    """KL( N(mean_p, std_p²I) || N(mean_q, std_q²I) ), summed over C,T → (B,).

    The closed-form per-step KL DPOK uses to anchor the fine-tuned policy to the
    frozen pretrained sampler (Fan et al. 2023). For equal isotropic σ (the usual
    case — both share the schedule's σ_t), it reduces to ``||μ_p-μ_q||² / (2σ²)``.
    """
    mp, mq = mean_p.float(), mean_q.float()
    sp = (std_p if torch.is_tensor(std_p) else torch.tensor(float(std_p))).float().clamp(min=1e-12)
    sq = (std_q if torch.is_tensor(std_q) else torch.tensor(float(std_q))).float().clamp(min=1e-12)
    vp, vq = sp * sp, sq * sq
    n = float(mp[0].numel())
    per = 0.5 * (((mp - mq) ** 2) / vq).flatten(start_dim=1).sum(dim=1)
    per = per + 0.5 * n * (vp / vq + torch.log(vq / vp) - 1.0)
    return per
