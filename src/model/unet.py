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
    def __init__(self, sig_channels: int, cond_channels: int,
                 base: int = 64, mults=(1, 2, 4, 8), t_dim: int = 256):
        super().__init__()
        self.sig_channels = sig_channels
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        in_ch = sig_channels + cond_channels
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)

        chs = [base * m for m in mults]
        self.downs = nn.ModuleList()
        self.down_samps = nn.ModuleList()
        prev = base
        skip_chs = [base]
        for ch in chs:
            self.downs.append(ResBlock1d(prev, ch, t_dim))
            skip_chs.append(ch)
            self.down_samps.append(Down(ch))
            prev = ch

        self.mid1 = ResBlock1d(prev, prev, t_dim)
        self.mid2 = ResBlock1d(prev, prev, t_dim)

        self.up_samps = nn.ModuleList()
        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.up_samps.append(Up(prev))
            self.ups.append(ResBlock1d(prev + ch, ch, t_dim))
            prev = ch

        self.out_norm = nn.GroupNorm(math.gcd(8, prev), prev)
        self.out_conv = nn.Conv1d(prev, sig_channels, 3, padding=1)

    def forward(self, x_t, cond, t):
        """x_t (B,C,T), cond (B,Cc,T), t (B,)"""
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        skips = [h]
        for block, ds in zip(self.downs, self.down_samps):
            h = block(h, t_emb)
            skips.append(h)
            h = ds(h)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        for us, block in zip(self.up_samps, self.ups):
            h = us(h)
            skip = skips.pop()
            # guard against off-by-one length mismatch from striding
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))
