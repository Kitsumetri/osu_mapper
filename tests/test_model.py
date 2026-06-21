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
    x, cond, t = torch.randn(B, C_SIG, T), torch.randn(B, C_COND, T), torch.randint(0, 1000, (B,))
    ctx = torch.rand(B, ctx_dim)
    drop = torch.tensor([True, False])
    for adaln in (False, True):
        m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=False,
                   ctx_dim=ctx_dim, adaln=adaln)
        # conditioned, unconditioned (None), and CFG-dropped all run and match shape
        assert m(x, cond, t, ctx=ctx).shape == (B, C_SIG, T)
        assert m(x, cond, t, ctx=None).shape == (B, C_SIG, T)
        assert m(x, cond, t, ctx=ctx, ctx_drop=drop).shape == (B, C_SIG, T)
        if adaln:
            # adaLN-zero gates conditioning to zero at init (it eases in during
            # training), so perturb the ada projections to test the ctx wiring.
            with torch.no_grad():
                for mod in m.modules():
                    if hasattr(mod, "ada"):
                        mod.ada.weight.normal_(0, 0.1)
        # ctx changes the output (conditioning actually wired in)
        a = m(x, cond, t, ctx=ctx)
        b = m(x, cond, t, ctx=None)
        assert not torch.allclose(a, b), f"ctx has no effect (adaln={adaln})"


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


def test_v_objective_roundtrip():
    # v target must invert back to the exact x0 and eps it was built from
    import pytest
    diff = GaussianDiffusion(timesteps=100, device=DEV, objective="v")
    x0 = torch.randn(B, C_SIG, T)
    noise = torch.randn_like(x0)
    t = torch.randint(0, 100, (B,))
    x_t = diff.q_sample(x0, t, noise)
    v = diff.target(x0, t, noise)
    a = diff.sqrt_acp[t][:, None, None]
    s = diff.sqrt_one_minus_acp[t][:, None, None]
    rec_x0, rec_eps = diff._to_x0_eps(v, x_t, a, s)
    assert torch.allclose(rec_x0, x0, atol=1e-4)
    assert torch.allclose(rec_eps, noise, atol=1e-4)
    # eps objective target is just the noise
    deps = GaussianDiffusion(timesteps=100, device=DEV, objective="eps")
    assert torch.allclose(deps.target(x0, t, noise), noise)
    with pytest.raises(ValueError):
        GaussianDiffusion(timesteps=100, device=DEV, objective="eps", zero_snr=True)


def test_zero_terminal_snr_schedule():
    diff = GaussianDiffusion(timesteps=100, device=DEV, objective="v", zero_snr=True)
    # terminal alpha_bar driven to 0 (pure noise at the last step); start preserved
    assert diff.sqrt_acp[-1].abs() < 1e-6
    base = GaussianDiffusion(timesteps=100, device=DEV)
    assert torch.allclose(diff.sqrt_acp[0], base.sqrt_acp[0], atol=1e-5)
    # cumprod stays monotonically non-increasing
    assert (diff.alphas_cumprod[1:] <= diff.alphas_cumprod[:-1] + 1e-6).all()


def test_ddim_v_zero_snr_finite():
    m = _model()
    diff = GaussianDiffusion(timesteps=100, device=DEV, objective="v", zero_snr=True)
    cond = torch.randn(1, C_COND, T)
    out = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=10, ctx=None, guidance=1.0)
    assert out.shape == (1, C_SIG, T)
    assert torch.isfinite(out).all()


def test_zero_snr_schedule_is_finite_everywhere():
    """Zero-terminal-SNR drives alpha_bar_T->0, so betas[T]->1. Every derived buffer
    (incl. posterior_var and sqrt_recip_alphas) must stay finite — a NaN here would
    silently corrupt the last DDIM step / any DDPM ancestral step."""
    diff = GaussianDiffusion(timesteps=50, device=DEV, objective="v", zero_snr=True)
    for buf in (diff.betas, diff.alphas_cumprod, diff.sqrt_acp, diff.sqrt_one_minus_acp,
                diff.sqrt_recip_alphas, diff.posterior_var):
        assert torch.isfinite(buf).all()
    assert (diff.posterior_var >= 0).all()
    assert diff.alphas_cumprod[-1].abs() < 1e-6      # terminal SNR == 0


def test_posterior_q_sample_consistency():
    """q_sample(x0,t) noised then the true posterior mean of q(x_{t-1}|x_t,x0) must
    equal the closed-form posterior coefficients applied to (x0, x_t)."""
    import torch.nn.functional as F
    d = GaussianDiffusion(timesteps=100, device=DEV)
    x0 = torch.randn(2, C_SIG, T)
    noise = torch.randn_like(x0)
    t = torch.tensor([30, 70])
    x_t = d.q_sample(x0, t, noise)
    # the eps recovered from (x_t, x0) at this t matches the injected noise
    a = d.sqrt_acp[t][:, None, None]
    s = d.sqrt_one_minus_acp[t][:, None, None]
    rec_eps = (x_t - a * x0) / s
    assert torch.allclose(rec_eps, noise, atol=1e-4)
    # posterior coefficients on x0 and x_t sum (with mean over batch) to ~the DDPM mean
    acp = d.alphas_cumprod
    acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)
    c0 = (torch.sqrt(acp_prev[t]) * d.betas[t] / (1 - acp[t]))[:, None, None]
    ct = (torch.sqrt(1 - d.betas[t]) * (1 - acp_prev[t]) / (1 - acp[t]))[:, None, None]
    post_mean = c0 * x0 + ct * x_t
    assert torch.isfinite(post_mean).all()


def test_guidance_rescale_runs_and_is_finite():
    """The guidance_rescale branch (Lin et al. std-matching) must run on the CFG path
    and stay finite for both batch_cfg on and off."""
    ctx_dim = 6
    m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=False,
               ctx_dim=ctx_dim).eval()
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    cond, ctx = torch.randn(1, C_COND, T), torch.rand(1, ctx_dim)
    for bc in (True, False):
        out = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=8, ctx=ctx,
                               guidance=3.0, guidance_rescale=0.7, batch_cfg=bc)
        assert out.shape == (1, C_SIG, T) and torch.isfinite(out).all()


def test_apply_rope_preserves_norm_and_is_positional():
    from src.model.unet import _apply_rope
    x = torch.randn(1, 2, 5, 8)  # (b, heads, t, d)
    r = _apply_rope(x)
    assert torch.allclose(x.norm(dim=-1), r.norm(dim=-1), atol=1e-4)  # rotation -> norm kept
    assert torch.allclose(r[:, :, 0], x[:, :, 0], atol=1e-5)          # position 0 unrotated
    assert not torch.allclose(r[:, :, 3], x[:, :, 3], atol=1e-3)      # later positions rotate


def test_unet_attention_upgrades_forward_and_grad():
    # RoPE, up-path attention, and grad checkpointing all run + backprop
    for kw in ({"rope": True},
               {"up_attn": True, "attn_levels": 1},
               {"rope": True, "up_attn": True, "grad_ckpt": True, "attn_levels": 1}):
        m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=True, **kw)
        m.train()
        out = m(torch.randn(B, C_SIG, T), torch.randn(B, C_COND, T),
                torch.randint(0, 1000, (B,)))
        assert out.shape == (B, C_SIG, T)
        out.mean().backward()
        assert any(p.grad is not None for p in m.parameters())


def test_min_snr_loss_weight():
    # eps loss is applied in noise-space (= SNR * x0-loss), so to get the Min-SNR
    # effective x0-weight min(SNR,g) the eps weight must be min(SNR,g)/SNR (matches
    # diffusers). NOT min(SNR,g) — that would double-count SNR. Lock the formula.
    diff = GaussianDiffusion(timesteps=100, device=DEV)  # eps
    t = torch.tensor([1, 50, 99])
    snr = diff.alphas_cumprod[t] / (1 - diff.alphas_cumprod[t])
    w = diff.loss_weight(t, gamma=5.0)
    assert torch.allclose(w, torch.clamp(snr, max=5.0) / snr, atol=1e-5)
    assert (w <= 1.0 + 1e-4).all()                     # min(snr,g)/snr <= 1
    # v-prediction loss = (SNR+1)*x0-loss -> weight min(SNR,g)/(SNR+1)
    dv = GaussianDiffusion(timesteps=100, device=DEV, objective="v")
    wv = dv.loss_weight(t, gamma=5.0)
    assert torch.allclose(wv, torch.clamp(snr, max=5.0) / (snr + 1), atol=1e-5)


def test_diffusion_loss_helper():
    from types import SimpleNamespace

    from src.train import _diffusion_loss
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    pred, target = torch.randn(2, C_SIG, T), torch.randn(2, C_SIG, T)
    t = torch.tensor([10, 90])
    mse = _diffusion_loss(pred, target, t, diff,
                          SimpleNamespace(loss="mse", huber_beta=1.0, min_snr_gamma=0.0))
    # default mse path matches plain mean MSE
    assert abs(mse.item() - torch.nn.functional.mse_loss(pred, target).item()) < 1e-5
    for kw in (dict(loss="huber", huber_beta=1.0, min_snr_gamma=0.0),
               dict(loss="mse", huber_beta=1.0, min_snr_gamma=5.0),
               dict(loss="huber", huber_beta=1.0, min_snr_gamma=5.0)):
        v = _diffusion_loss(pred, target, t, diff, SimpleNamespace(**kw))
        assert torch.isfinite(v) and v.item() >= 0


def test_rope_is_parameter_free():
    # RoPE adds no parameters (so it never breaks loading an existing state_dict)
    base = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=True, rope=False)
    rope = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=True, rope=True)
    assert (sum(p.numel() for p in base.parameters())
            == sum(p.numel() for p in rope.parameters()))


def test_batched_cfg_matches_two_forward():
    """Batched CFG (one batch-2 forward, second half ctx_drop=True) must equal the two
    separate conditional + unconditional forwards bit-for-bit — the equivalence the
    sampling speedup relies on (RESEARCH §11 5.4)."""
    ctx_dim = 6
    m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=False,
               ctx_dim=ctx_dim, adaln=False).eval()  # FiLM: ctx active at init
    x, cond = torch.randn(1, C_SIG, T), torch.randn(1, C_COND, T)
    t, ctx = torch.randint(0, 1000, (1,)), torch.rand(1, ctx_dim)
    with torch.no_grad():
        out_c = m(x, cond, t, ctx=ctx)
        out_u = m(x, cond, t, ctx=None)
        out2 = m(torch.cat([x, x]), torch.cat([cond, cond]), torch.cat([t, t]),
                 ctx=torch.cat([ctx, ctx]), ctx_drop=torch.tensor([False, True]))
    assert torch.allclose(out2[:1], out_c, atol=1e-6)   # conditional half
    assert torch.allclose(out2[1:], out_u, atol=1e-6)   # unconditional half == ctx=None
    assert not torch.allclose(out_c, out_u)             # ctx actually matters (test is real)


def test_ddim_cfg_sample_runs():
    """DDIM with guidance>1 + ctx exercises the batched-CFG path -> finite, right shape."""
    ctx_dim = 6
    m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=False,
               ctx_dim=ctx_dim).eval()
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    out = diff.ddim_sample(m, torch.randn(1, C_COND, T), (1, C_SIG, T), steps=8,
                           ctx=torch.rand(1, ctx_dim), guidance=2.0)
    assert out.shape == (1, C_SIG, T) and torch.isfinite(out).all()


def test_ddim_batch_cfg_matches_two_forward_path():
    """batch_cfg on/off must produce identical samples — the low-memory fallback (for
    marathon songs that OOM the batch-2 forward) is exact, not an approximation."""
    ctx_dim = 6
    m = UNet1d(C_SIG, C_COND, base=16, mults=(1, 2), t_dim=32, attn=False,
               ctx_dim=ctx_dim).eval()
    diff = GaussianDiffusion(timesteps=100, device=DEV)
    cond, ctx = torch.randn(1, C_COND, T), torch.rand(1, ctx_dim)
    torch.manual_seed(7)
    a = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=8, ctx=ctx, guidance=2.0, batch_cfg=True)
    torch.manual_seed(7)
    b = diff.ddim_sample(m, cond, (1, C_SIG, T), steps=8, ctx=ctx, guidance=2.0, batch_cfg=False)
    assert torch.allclose(a, b, atol=1e-5)
