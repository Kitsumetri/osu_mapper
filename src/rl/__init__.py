"""RL / policy-gradient feasibility prototypes for the diffusion sampler.

This package contains **prototype, non-production** helpers that demonstrate the
per-step log-probability computation a policy-gradient method (DDPO / DPOK,
Black et al. 2023 / Fan et al. 2023) needs over a *stochastic* DDIM sampler.

It is **not wired into training or generation** — the production sampler
(``src/model/diffusion.py:ddim_sample``) is ``@torch.no_grad()`` with ``eta=0``
(deterministic), which has **no per-step stochasticity and therefore no policy
log-prob to differentiate**. The design rationale, math derivation, memory
budget, and the fit-or-not verdict live in ``docs/v9/task_rl_policy_gradient.md``.

The helper here mirrors the production DDIM update exactly (v-prediction +
zero-terminal-SNR + classifier-free guidance) but with ``eta>0`` so each step is
a Gaussian transition, and it returns the per-step log-prob with gradients
flowing into the model output — the score-function estimator's inner term. It is
exercised only on toy tensors in ``tests/test_rl_logprob.py``.
"""

from .sample_logprob import (
    StepLogProb,
    ddim_sigma,
    ddim_step_with_logprob,
    gaussian_kl,
    gaussian_logprob,
    v_to_x0_eps,
)

__all__ = [
    "StepLogProb",
    "ddim_sigma",
    "ddim_step_with_logprob",
    "gaussian_kl",
    "gaussian_logprob",
    "v_to_x0_eps",
]
