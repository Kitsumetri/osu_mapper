"""Gaussian DDPM helper: schedule, q_sample (forward noising), p_sample loop.

Supports two training objectives and an optional zero-terminal-SNR schedule:

- ``objective="eps"`` (default): predict the noise (the original v1-v6 behaviour).
- ``objective="v"``: predict the velocity ``v = a_t*eps - s_t*x0`` (Salimans & Ho
  2022). Sharper outputs at low SNR -> less mean-regression / under-dispersion,
  which is the v7 target (spacing + slider curvature collapse, RESEARCH §10.7).
- ``zero_snr=True``: rescale the schedule so ``alpha_bar_T = 0`` (Lin et al. 2023),
  removing the train/test gap where sampling starts from pure noise but training
  never sees SNR=0. Requires ``objective="v"`` (eps is undefined at SNR=0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
        objective: str = "eps",
        zero_snr: bool = False,
    ):
        assert objective in ("eps", "v"), objective
        if zero_snr and objective != "v":
            raise ValueError("zero_snr requires objective='v' (eps is undefined at SNR=0)")
        self.timesteps = timesteps
        self.device = device
        self.objective = objective
        self.zero_snr = zero_snr

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        acp = torch.cumprod(1.0 - betas, dim=0)
        if zero_snr:
            acp = _rescale_zero_terminal_snr(acp)
            # re-derive alphas/betas from the rescaled cumprod (terminal beta -> 1)
            acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)
            betas = 1.0 - acp / acp_prev

        self.betas = betas
        self.alphas_cumprod = acp
        self.sqrt_acp = torch.sqrt(acp.clamp(min=0.0))
        self.sqrt_one_minus_acp = torch.sqrt((1.0 - acp).clamp(min=0.0))
        self.sqrt_recip_alphas = torch.sqrt(1.0 / (1.0 - betas).clamp(min=1e-8))
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp).clamp(min=1e-8)

    def q_sample(self, x0, t, noise):
        a = self.sqrt_acp[t][:, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None]
        return a * x0 + b * noise

    def target(self, x0, t, noise):
        """The regression target for the chosen objective (eps -> noise, v -> velocity)."""
        if self.objective == "v":
            a = self.sqrt_acp[t][:, None, None]
            s = self.sqrt_one_minus_acp[t][:, None, None]
            return a * noise - s * x0
        return noise

    def loss_weight(self, t, gamma: float):
        """Per-sample Min-SNR-gamma loss weight (Hang et al. 2023): caps the loss on
        easy low-noise steps so training balances across noise levels. Returns (B,)."""
        acp = self.alphas_cumprod[t]
        snr = acp / (1.0 - acp).clamp(min=1e-8)
        capped = torch.clamp(snr, max=gamma)
        if self.objective == "v":
            return capped / (snr + 1.0)        # v-pred normalisation
        return capped / snr.clamp(min=1e-8)    # eps

    def _to_x0_eps(self, out, x_t, a, s):
        """Convert a model output (eps or v) at scale (a=sqrt_acp, s=sqrt_1macp) to (x0, eps)."""
        if self.objective == "v":
            x0 = a * x_t - s * out
            eps = s * x_t + a * out
        else:
            eps = out
            x0 = (x_t - s * eps) / a.clamp(min=1e-8)
        return x0, eps

    @torch.no_grad()
    def p_sample(self, model, cond, shape, steps: int | None = None):
        """Full ancestral DDPM sampling (reference only; ``generate`` uses ``ddim_sample``).

        Unconditional (no ctx/CFG) and assumes ``objective='eps'`` — wire those
        through before using it in production.
        """
        b = shape[0]
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            eps = model(x, cond, t)
            beta = self.betas[i]
            sqrt_one_minus = self.sqrt_one_minus_acp[i]
            mean = self.sqrt_recip_alphas[i] * (x - beta / sqrt_one_minus * eps)
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(self.posterior_var[i]) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, steps: int = 100, eta: float = 0.0,
                    ctx=None, guidance: float = 1.0, guidance_rescale: float = 0.0):
        """DDIM sampling: correct accelerated sampling over a step subsequence.

        eta=0 is deterministic. ``ctx`` is the difficulty context; ``guidance`` > 1
        applies classifier-free guidance. ``guidance_rescale`` in (0,1] rescales the
        guided x0 toward the conditional x0's std (Lin et al.) to curb the
        over-saturation that high guidance + zero-SNR can cause. Returns x0 (B,C,T).
        """
        b = shape[0]
        x = torch.randn(shape, device=self.device)
        seq = torch.linspace(0, self.timesteps - 1, steps, device=self.device).long()
        seq = torch.unique(seq).tolist()
        use_cfg = ctx is not None and guidance != 1.0
        for k in reversed(range(len(seq))):
            i = seq[k]
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            a_t = self.sqrt_acp[i]
            s_t = self.sqrt_one_minus_acp[i]
            if use_cfg:
                out_c = model(x, cond, t, ctx=ctx)
                out_u = model(x, cond, t, ctx=None)
                out = out_u + guidance * (out_c - out_u)
            else:
                out = out_c = model(x, cond, t, ctx=ctx)
            x0, eps = self._to_x0_eps(out, x, a_t, s_t)
            if guidance_rescale > 0 and use_cfg:
                # per-sample std (over channel+time) so rescaling is correct under
                # batched sampling, not just the B=1 inference path.
                x0_c, _ = self._to_x0_eps(out_c, x, a_t, s_t)
                std_c = x0_c.std(dim=(1, 2), keepdim=True)
                std_g = x0.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                x0 = guidance_rescale * (x0 * std_c / std_g) + (1 - guidance_rescale) * x0
            x0 = x0.clamp(-1.5, 1.5)
            if k == 0:
                x = x0
                break
            acp_prev = self.alphas_cumprod[seq[k - 1]]
            acp_t = self.alphas_cumprod[i]
            sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev))
            dir_xt = torch.sqrt((1 - acp_prev - sigma**2).clamp(min=0.0)) * eps
            x = torch.sqrt(acp_prev) * x0 + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x


def _rescale_zero_terminal_snr(acp: torch.Tensor) -> torch.Tensor:
    """Rescale alpha_bar so sqrt(alpha_bar) hits 0 at the last step (Lin et al. 2023).

    Keeps sqrt(alpha_bar_0), maps sqrt(alpha_bar_T) -> 0, linearly in between.
    """
    sqrt_acp = torch.sqrt(acp.clamp(min=0.0))
    a0, aT = sqrt_acp[0].clone(), sqrt_acp[-1].clone()
    sqrt_acp = (sqrt_acp - aT) * (a0 / (a0 - aT))
    return (sqrt_acp**2).clamp(min=0.0)
