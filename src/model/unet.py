"""1D conditional U-Net denoiser for beatmap-signal diffusion.

Input  : noisy signal x_t (B, C_sig, T) + mel condition (B, n_mels, T) + t
Output : predicted noise eps (B, C_sig, T)

The mel condition is concatenated channel-wise with the noisy signal. Timestep
information is injected into every residual block via a FiLM-style shift from a
sinusoidal embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        g = math.gcd(groups, in_ch)
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(t_dim, out_ch)
        g2 = math.gcd(groups, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock1d(nn.Module):
    """Multi-head self-attention over the time axis (long-range structure).

    Uses query/key RMS-normalisation ("QK-norm") which keeps attention logits
    bounded and prevents the divergence seen with bf16 + plain dot-product
    attention. The output projection is zero-initialised so the block starts as
    an identity and eases in during training.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(math.gcd(8, ch), ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)
        # learnable attention temperature (CLIP-style), since QK-norm makes raw
        # logits cosine-similarities in [-1, 1].
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, t = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        # (b, heads, t, c/heads)
        q, k, v = (z.reshape(b, self.heads, c // self.heads, t).transpose(-1, -2)
                   for z in (q, k, v))
        # QK-norm: unit-norm queries/keys keep logits in a stable range; the
        # learnable temperature is folded into q (sdpa needs scale=1.0 float).
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = out.transpose(-1, -2).reshape(b, c, t)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet1d(nn.Module):
    def __init__(
        self,
        sig_channels: int,
        cond_channels: int,
        base: int = 64,
        mults=(1, 2, 4, 8),
        t_dim: int = 256,
        attn: bool = True,
        ctx_dim: int = 0,
        attn_levels: int = 2,
    ):
        super().__init__()
        self.sig_channels = sig_channels
        self.t_dim = t_dim
        self.ctx_dim = ctx_dim
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        if ctx_dim > 0:
            # difficulty context -> added to the timestep embedding (FiLM path).
            # A learned "null" embedding enables classifier-free guidance.
            self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, t_dim), nn.SiLU(),
                                         nn.Linear(t_dim, t_dim))
            self.null_ctx = nn.Parameter(torch.zeros(t_dim))
        in_ch = sig_channels + cond_channels
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)

        chs = [base * m for m in mults]
        self.downs = nn.ModuleList()
        self.down_samps = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        prev = base
        skip_chs = [base]
        for i, ch in enumerate(chs):
            self.downs.append(ResBlock1d(prev, ch, t_dim))
            # attention at the ``attn_levels`` deepest (coarsest) levels to bound
            # cost; raise it to give the model longer-range pattern context.
            use_attn = attn and i >= len(chs) - attn_levels
            self.down_attn.append(AttnBlock1d(ch) if use_attn else nn.Identity())
            skip_chs.append(ch)
            self.down_samps.append(Down(ch))
            prev = ch

        self.mid1 = ResBlock1d(prev, prev, t_dim)
        self.mid_attn = AttnBlock1d(prev) if attn else nn.Identity()
        self.mid2 = ResBlock1d(prev, prev, t_dim)

        self.up_samps = nn.ModuleList()
        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.up_samps.append(Up(prev))
            self.ups.append(ResBlock1d(prev + ch, ch, t_dim))
            prev = ch

        self.out_norm = nn.GroupNorm(math.gcd(8, prev), prev)
        self.out_conv = nn.Conv1d(prev, sig_channels, 3, padding=1)

    def forward(self, x_t, cond, t, ctx=None, ctx_drop=None):
        """x_t (B,C,T), cond=mel (B,Cc,T), t (B,), ctx (B,ctx_dim) difficulty.

        ctx=None uses the null embedding (unconditioned). ctx_drop (B,) bool mask
        replaces those rows with the null embedding (classifier-free guidance).
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))
        if self.ctx_dim > 0:
            b = x_t.shape[0]
            if ctx is None:
                c_emb = self.null_ctx.expand(b, -1)
            else:
                c_emb = self.ctx_mlp(ctx)
                if ctx_drop is not None:
                    c_emb = torch.where(ctx_drop[:, None], self.null_ctx.expand(b, -1), c_emb)
            t_emb = t_emb + c_emb
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        skips = [h]
        for block, attn, ds in zip(self.downs, self.down_attn, self.down_samps):
            h = block(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = ds(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        for us, block in zip(self.up_samps, self.ups):
            h = us(h)
            skip = skips.pop()
            # guard against off-by-one length mismatch from striding
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))
