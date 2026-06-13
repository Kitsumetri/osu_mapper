"""Model/diffusion tests. Tiny + CPU-only so they don't touch the GPU."""

import torch

from src.model.diffusion import GaussianDiffusion
from src.model.unet import UNet1d, timestep_embedding

DEV = "cpu"
C_SIG, C_COND, T, B = 6, 64, 64, 2


def _model(attn=False):
    return UNet1d(sig_channels=C_SIG, cond_channels=C_COND, base=16, mults=(1, 2),
                  t_dim=32, attn=attn)


def test_unet_forward_with_attention():
    m = _model(attn=True)
    out = m(torch.randn(B, C_SIG, T), torch.randn(B, C_COND, T), torch.randint(0, 1000, (B,)))
    assert out.shape == (B, C_SIG, T)


def test_unet_difficulty_conditioning():
    ctx_dim = 6
    m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=False, ctx_dim=ctx_dim)
    x, cond, t = torch.randn(B, C_SIG, T), torch.randn(B, C_COND, T), torch.randint(0, 1000, (B,))
    ctx = torch.rand(B, ctx_dim)
    # conditioned, unconditioned (None), and CFG-dropped all run and match shape
    assert m(x, cond, t, ctx=ctx).shape == (B, C_SIG, T)
    assert m(x, cond, t, ctx=None).shape == (B, C_SIG, T)
    drop = torch.tensor([True, False])
    assert m(x, cond, t, ctx=ctx, ctx_drop=drop).shape == (B, C_SIG, T)
    # ctx changes the output (conditioning actually wired in)
    a = m(x, cond, t, ctx=ctx)
    b = m(x, cond, t, ctx=None)
    assert not torch.allclose(a, b)


def test_timestep_embedding_shape():
    t = torch.arange(5)
    emb = timestep_embedding(t, 32)
    assert emb.shape == (5, 32)


def test_unet_forward_shape():
    m = _model()
    x = torch.randn(B, C_SIG, T)
    cond = torch.randn(B, C_COND, T)
    t = torch.randint(0, 1000, (B,))
    out = m(x, cond, t)
    assert out.shape == (B, C_SIG, T)


def test_unet_backward():
    m = _model()
    x = torch.randn(B, C_SIG, T)
    cond = torch.randn(B, C_COND, T)
    t = torch.randint(0, 1000, (B,))
    loss = m(x, cond, t).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_q_sample_shape_and_endpoints():
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    x0 = torch.randn(B, C_SIG, T)
    noise = torch.randn_like(x0)
    # at t=0 result should be ~x0; at t=last ~pure noise
    t0 = torch.zeros(B, dtype=torch.long)
    out0 = diff.q_sample(x0, t0, noise)
    assert torch.allclose(out0, x0, atol=0.2)
    assert diff.q_sample(x0, torch.full((B,), 99), noise).shape == x0.shape


def test_ddim_sample_shape():
    m = _model()
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    cond = torch.randn(1, C_COND, T)
    out = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=10)
    assert out.shape == (1, C_SIG, T)
    assert torch.isfinite(out).all()


def test_ddim_deterministic_when_eta_zero():
    torch.manual_seed(0)
    m = _model()
    m.eval()
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    cond = torch.randn(1, C_COND, T)
    torch.manual_seed(123)
    a = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=10, eta=0.0)
    torch.manual_seed(123)
    b = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=10, eta=0.0)
    assert torch.allclose(a, b)
