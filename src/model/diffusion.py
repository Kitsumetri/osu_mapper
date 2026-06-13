"""Gaussian DDPM helper: schedule, q_sample (forward noising), p_sample loop."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cuda"):
        self.timesteps = timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = acp
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, x0, t, noise):
        a = self.sqrt_acp[t][:, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None]
        return a * x0 + b * noise

    @torch.no_grad()
    def p_sample(self, model, cond, shape, steps: int | None = None):
        """Full ancestral DDPM sampling (uses every timestep). For fewer steps
        use :meth:`ddim_sample`, which is mathematically correct when steps are
        skipped (plain strided ancestral sampling is not)."""
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
    def ddim_sample(self, model, cond, shape, steps: int = 100, eta: float = 0.0):
        """DDIM sampling: correct accelerated sampling over a step subsequence.

        eta=0 is deterministic. Returns the final x0 estimate (B,C,T).
        """
        b = shape[0]
        x = torch.randn(shape, device=self.device)
        # evenly spaced subsequence of timesteps, descending
        seq = torch.linspace(0, self.timesteps - 1, steps, device=self.device).long()
        seq = torch.unique(seq).tolist()
        for k in reversed(range(len(seq))):
            i = seq[k]
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            eps = model(x, cond, t)
            acp_t = self.alphas_cumprod[i]
            x0 = (x - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)
            x0 = x0.clamp(-1.5, 1.5)
            if k == 0:
                x = x0
                break
            acp_prev = self.alphas_cumprod[seq[k - 1]]
            sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev))
            dir_xt = torch.sqrt(1 - acp_prev - sigma ** 2) * eps
            x = torch.sqrt(acp_prev) * x0 + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x
