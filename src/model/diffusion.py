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
                    ctx=None, guidance: float = 1.0, guidance_rescale: float = 0.0,
                    progress: bool = False, batch_cfg: bool = True, amp: bool = False,
                    monitor=None):
        """DDIM sampling: correct accelerated sampling over a step subsequence.

        eta=0 is deterministic. ``ctx`` is the difficulty context; ``guidance`` > 1
        applies classifier-free guidance. ``guidance_rescale`` in (0,1] rescales the
        guided x0 toward the conditional x0's std (Lin et al.) to curb the
        over-saturation that high guidance + zero-SNR can cause. ``progress`` shows a
        tqdm bar over the steps. ``batch_cfg`` runs the CFG conditional+unconditional as
        one batch-2 forward (~2× faster) — but that ~doubles peak activation memory, so
        turn it off for very long songs near the VRAM limit. Returns x0 (B,C,T).

        ``monitor`` (optional): a callback ``monitor(k_index, frac_done, x0) -> bool``
        invoked AFTER the predicted clean signal ``x0`` is computed (and clamped) at
        each step. ``k_index`` is the remaining-step index (counts down to 0 = final
        step), ``frac_done`` in [0, 1] is how far through the reverse process we are
        (0 at the first/noisiest step, ~1 at the last), and ``x0`` is the current
        predicted clean signal (B,C,T). If it returns truthy the loop BREAKS early and
        returns the current ``x0`` (the abort path — used by best-of-N to drop doomed
        candidates cheaply). The monitor cannot change ``x0`` and is purely an
        early-exit gate, so with ``monitor=None`` the sampling is byte-for-byte
        unchanged. The caller records the abort itself (e.g. via the closure); the
        return type is always the tensor so existing callers are unaffected.
        """
        b = shape[0]
        x = torch.randn(shape, device=self.device)
        seq = torch.linspace(0, self.timesteps - 1, steps, device=self.device).long()
        seq = torch.unique(seq).tolist()
        use_cfg = ctx is not None and guidance != 1.0
        # batched classifier-free guidance: run the conditional + unconditional passes
        # as ONE batch-2 forward instead of two — ctx_drop nulls the second half, which
        # the UNet maps to exactly the null embedding (== ctx=None), so the result is
        # bit-identical while ~halving the forward count (RESEARCH §11 5.4). It ~doubles
        # peak memory though (batch 2), so ``batch_cfg=False`` keeps the low-memory
        # two-forward path for marathon-length songs that would otherwise OOM.
        batched = use_cfg and batch_cfg
        if batched:
            cond2 = torch.cat([cond, cond], 0)
            ctx2 = torch.cat([ctx, ctx], 0)
            drop2 = torch.cat([torch.zeros(b, dtype=torch.bool, device=self.device),
                               torch.ones(b, dtype=torch.bool, device=self.device)])
        # bf16 autocast around only the model forward (x + the schedule math stay fp32):
        # enables the flash-attention SDPA kernel (O(T) instead of O(T²) memory) so long
        # songs don't OOM materialising the attention matrix, and ~halves activation memory
        # / ~2× the speed. Inference matches the bf16-autocast training regime.
        def _fwd(*a, **kw):
            if amp and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(*a, **kw)
            return model(*a, **kw)
        steps_iter = list(reversed(range(len(seq))))
        if progress:
            from tqdm import tqdm
            steps_iter = tqdm(steps_iter, desc="ddim", unit="step")
        for k in steps_iter:
            i = seq[k]
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            a_t = self.sqrt_acp[i]
            s_t = self.sqrt_one_minus_acp[i]
            if batched:
                out2 = _fwd(torch.cat([x, x], 0), cond2, torch.cat([t, t], 0),
                            ctx=ctx2, ctx_drop=drop2)
                out_c, out_u = out2[:b], out2[b:]
                out = out_u + guidance * (out_c - out_u)
            elif use_cfg:  # low-memory two-forward path (marathon songs)
                out_c = _fwd(x, cond, t, ctx=ctx)
                out_u = _fwd(x, cond, t, ctx=None)
                out = out_u + guidance * (out_c - out_u)
            else:
                out = out_c = _fwd(x, cond, t, ctx=ctx)
            out = out.float()
            x0, eps = self._to_x0_eps(out, x, a_t, s_t)
            if guidance_rescale > 0 and use_cfg:
                # per-sample std (over channel+time) so rescaling is correct under
                # batched sampling, not just the B=1 inference path.
                x0_c, _ = self._to_x0_eps(out_c, x, a_t, s_t)
                std_c = x0_c.std(dim=(1, 2), keepdim=True)
                std_g = x0.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                x0 = guidance_rescale * (x0 * std_c / std_g) + (1 - guidance_rescale) * x0
            x0 = x0.clamp(-1.5, 1.5)
            if monitor is not None:
                # frac_done: 0 at the first (noisiest) step, 1 at the last. Single-step
                # seq -> treat as fully done. The monitor inspects the (clamped)
                # predicted clean x0 and may request an early abort; it cannot mutate x0.
                denom = len(seq) - 1
                frac_done = 1.0 if denom <= 0 else 1.0 - k / denom
                if monitor(k, frac_done, x0):
                    x = x0
                    break
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
