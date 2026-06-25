from src.conditioning import (
    CONTEXT_DIM,
    context_from_manifest,
    context_vector,
    target_context,
)


def test_context_vector_dim_and_range():
    c = context_vector(sr=5.0, ar=9.0, od=8.0, hp=5.0, cs=4.0, density=3.5)
    assert len(c) == CONTEXT_DIM == 7   # v9: + aim slot (was 6)
    assert all(0.0 <= v <= 1.5 for v in c)


def test_context_vector_clamps_extremes():
    c = context_vector(sr=99, ar=99, od=99, hp=99, cs=99, density=999, aim=99)
    assert all(v == 1.5 for v in c)               # upper clamp
    c0 = context_vector(sr=0, ar=0, od=0, hp=0, cs=0, density=0, aim=0)
    assert all(v == 0.0 for v in c0)


def test_target_context_scales_with_sr():
    low = target_context(2.0)
    high = target_context(7.0)
    # SR component (index 0) increases; derived AR/density also increase
    assert high[0] > low[0]
    assert high[1] >= low[1]          # AR
    assert high[5] >= low[5]          # density


def test_target_context_overrides():
    c = target_context(5.0, ar=10.0, density=4.0)
    assert c[1] == context_vector(5, 10, 0, 0, 0, 0)[1]   # AR honoured


def test_target_settings_in_valid_osu_ranges():
    """AR/OD/CS/HP the model is conditioned on (and that get written into the .osu)
    must stay within osu!'s legal [0, 10] (CS practically <=7), across the SR range."""
    from src.conditioning import target_settings
    for sr in (1.0, 3.0, 5.0, 7.0, 9.0, 11.0):
        s = target_settings(sr)
        assert 0.0 <= s["ar"] <= 10.0
        assert 0.0 <= s["od"] <= 10.0
        assert 0.0 <= s["cs"] <= 7.0
        assert 0.0 <= s["hp"] <= 10.0
        assert s["density"] > 0.0


def test_context_dim_matches_unet_ctx_wiring():
    """CONTEXT_DIM is the single source of truth; a UNet built with ctx_dim=CONTEXT_DIM
    must accept a context vector of exactly that length (guards a silent dim drift)."""
    import torch

    from src.conditioning import CONTEXT_DIM, target_context
    from src.model.unet import UNet1d
    m = UNet1d(6, 16, base=16, mults=(1, 2), t_dim=32, attn=False, ctx_dim=CONTEXT_DIM).eval()
    ctx = torch.tensor([target_context(5.0)], dtype=torch.float32)
    assert ctx.shape[1] == CONTEXT_DIM
    x, cond, t = torch.randn(1, 6, 32), torch.randn(1, 16, 32), torch.randint(0, 1000, (1,))
    with torch.no_grad():
        assert m(x, cond, t, ctx=ctx).shape == (1, 6, 32)


def test_context_from_manifest_defaults():
    # missing fields fall back to sane defaults without crashing
    c = context_from_manifest({"n_objects": 300, "duration_s": 100.0})
    assert len(c) == CONTEXT_DIM
    # density = 300/100 = 3.0 -> 3.0/12 = 0.25
    assert abs(c[5] - 0.25) < 1e-6
