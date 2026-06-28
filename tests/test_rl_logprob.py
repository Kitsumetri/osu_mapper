"""Hermetic tests for the DDPO/DPOK per-step log-prob prototype (src/rl).

Tiny, CPU-only, no model/GPU/dataset. These verify the prototype's correctness
contract (the doc, docs/v9/task_rl_policy_gradient.md):

1. its transition MEAN is bit-identical to the production sampler's deterministic
   (eta=0) update — so adding eta>0 only opens the log-prob, it does not change
   where sampling goes;
2. its Gaussian log-prob matches torch.distributions.Normal;
3. the score-function gradient flows into the denoiser OUTPUT (the policy gradient
   is computable — the whole point);
4. eta=0 degenerates to the deterministic update (no stochasticity);
5. the closed-form per-step KL (DPOK's anchor) is correct.
"""
import math

import torch

from src.model.diffusion import GaussianDiffusion
from src.rl.sample_logprob import (
    ddim_sigma,
    ddim_step_with_logprob,
    gaussian_kl,
    gaussian_logprob,
    v_to_x0_eps,
)

B, C, T = 2, 4, 16


def _schedule():
    # a real (small) zero-SNR v-pred schedule, same construction as production
    return GaussianDiffusion(timesteps=20, device="cpu", objective="v", zero_snr=True)


def test_v_to_x0_eps_matches_production():
    """v→(x0,eps) is bit-identical to diffusion._to_x0_eps for objective='v'."""
    diff = _schedule()
    i = 12                            # current timestep
    a_t = float(diff.sqrt_acp[i])
    s_t = float(diff.sqrt_one_minus_acp[i])
    x_t = torch.randn(B, C, T)
    out = torch.randn(B, C, T)
    x0, eps = v_to_x0_eps(out, x_t, a_t, s_t)
    a = diff.sqrt_acp[i]
    s = diff.sqrt_one_minus_acp[i]
    x0_ref, eps_ref = diff._to_x0_eps(out, x_t, a, s)
    assert torch.allclose(x0, x0_ref, atol=1e-5)
    assert torch.allclose(eps, eps_ref, atol=1e-5)


def test_transition_mean_matches_deterministic_ddim():
    """The helper's μ equals the production eta=0 DDIM update's deterministic part.

    Reproduces diffusion.ddim_sample's k>0 branch (eta=0 → sigma=0 → dir=sqrt(1-acp_prev)*eps)
    and checks our mean lands on the same x_{prev}.
    """
    diff = _schedule()
    i, j = 12, 8
    a_t = float(diff.sqrt_acp[i])
    s_t = float(diff.sqrt_one_minus_acp[i])
    acp_t = float(diff.alphas_cumprod[i])
    acp_prev = float(diff.alphas_cumprod[j])
    x_t = torch.randn(B, C, T)
    out = torch.randn(B, C, T)

    # production deterministic update (eta=0), inlined from ddim_sample
    x0_ref, eps_ref = diff._to_x0_eps(out, x_t, diff.sqrt_acp[i], diff.sqrt_one_minus_acp[i])
    x0_ref = x0_ref.clamp(-1.5, 1.5)
    dir_ref = torch.sqrt(torch.tensor(1 - acp_prev)) * eps_ref
    x_prev_ref = torch.sqrt(torch.tensor(acp_prev)) * x0_ref + dir_ref

    step = ddim_step_with_logprob(out, x_t, a_t=a_t, s_t=s_t, acp_t=acp_t,
                                  acp_prev=acp_prev, eta=0.0)
    assert torch.allclose(step.mean, x_prev_ref, atol=1e-5)
    # eta=0 → deterministic: sampled action == mean, std == 0
    assert float(step.std) == 0.0
    assert torch.allclose(step.x_prev, step.mean, atol=1e-6)


def test_logprob_matches_torch_normal():
    """gaussian_logprob equals torch.distributions.Normal summed over C,T."""
    mean = torch.randn(B, C, T)
    std = 0.37
    x = mean + std * torch.randn(B, C, T)
    lp = gaussian_logprob(x, mean, std)
    ref = torch.distributions.Normal(mean, std).log_prob(x).flatten(1).sum(1)
    assert lp.shape == (B,)
    assert torch.allclose(lp, ref, atol=1e-4)


def test_logprob_peaks_at_the_mean():
    """log-prob is maximal when the action equals the transition mean."""
    mean = torch.randn(B, C, T)
    std = 0.5
    at_mean = gaussian_logprob(mean, mean, std)
    off_mean = gaussian_logprob(mean + 1.0, mean, std)
    assert (at_mean > off_mean).all()


def test_score_function_gradient_flows_into_output():
    """The crux of feasibility: ∂ logπ / ∂ out exists and is non-zero.

    This is the inner term of the DDPO/REINFORCE estimator ∇θ E[R] = E[R·∇θ logπ].
    With a stochastic step (eta>0), logπ depends on the denoiser output through the
    transition mean, so a backward pass reaches `out` (→ in a real loop, the model
    weights). If this gradient were zero/None, policy gradient would be impossible.
    """
    diff = _schedule()
    i, j = 12, 8
    a_t = float(diff.sqrt_acp[i])
    s_t = float(diff.sqrt_one_minus_acp[i])
    acp_t = float(diff.alphas_cumprod[i])
    acp_prev = float(diff.alphas_cumprod[j])
    x_t = torch.randn(B, C, T)
    out = torch.randn(B, C, T, requires_grad=True)

    g = torch.Generator().manual_seed(0)
    step = ddim_step_with_logprob(out, x_t, a_t=a_t, s_t=s_t, acp_t=acp_t,
                                  acp_prev=acp_prev, eta=1.0, generator=g)
    # detach the realised action (the PPO recompute path scores a FIXED action):
    fixed_action = step.x_prev.detach()
    scored = ddim_step_with_logprob(out, x_t, a_t=a_t, s_t=s_t, acp_t=acp_t,
                                    acp_prev=acp_prev, eta=1.0, x_prev=fixed_action)
    # surrogate "R * logπ" with a fake advantage; backprop must reach `out`
    advantage = torch.tensor([1.0, -1.0])
    (advantage * scored.log_prob).sum().backward()
    assert out.grad is not None
    assert out.grad.abs().sum() > 0


def test_eta_scales_sigma_monotonically():
    """σ_t grows with eta and is 0 at eta=0 (matches diffusion.ddim_sample)."""
    diff = _schedule()
    i, j = 12, 8
    acp_t = float(diff.alphas_cumprod[i])
    acp_prev = float(diff.alphas_cumprod[j])
    assert ddim_sigma(acp_t, acp_prev, 0.0) == 0.0
    s_half = ddim_sigma(acp_t, acp_prev, 0.5)
    s_full = ddim_sigma(acp_t, acp_prev, 1.0)
    assert 0.0 < s_half < s_full
    # matches the production sigma formula exactly
    ref = float(math.sqrt((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)))
    assert abs(s_full - ref) < 1e-6


def test_kl_zero_for_identical_and_positive_otherwise():
    """DPOK's per-step KL anchor: 0 iff the two policies coincide."""
    mean_p = torch.randn(B, C, T)
    mean_q = mean_p.clone()
    std = 0.4
    kl_same = gaussian_kl(mean_p, std, mean_q, std)
    assert torch.allclose(kl_same, torch.zeros(B), atol=1e-5)
    kl_diff = gaussian_kl(mean_p, std, mean_p + 0.5, std)
    assert (kl_diff > 0).all()
    # equal-σ closed form: ||μp-μq||² / (2σ²)
    ref = ((mean_p - (mean_p + 0.5)) ** 2).flatten(1).sum(1) / (2 * std * std)
    assert torch.allclose(kl_diff, ref, atol=1e-4)


def test_logprob_is_per_batch_element():
    """Shapes: log-prob reduces over C,T but keeps the batch axis (per-sample R)."""
    diff = _schedule()
    i, j = 10, 6
    step = ddim_step_with_logprob(
        torch.randn(B, C, T), torch.randn(B, C, T),
        a_t=float(diff.sqrt_acp[i]), s_t=float(diff.sqrt_one_minus_acp[i]),
        acp_t=float(diff.alphas_cumprod[i]), acp_prev=float(diff.alphas_cumprod[j]),
        eta=1.0)
    assert step.log_prob.shape == (B,)
    assert step.mean.shape == (B, C, T)
