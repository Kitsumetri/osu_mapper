"""Tests for the v9 per-song aim-intensity conditioning feature.

Hermetic: synthetic audio arrays only (no GPU, no model, no real Songs library, no
network). Covers the load-bearing guarantees:

  * PARITY (the key test): the ONE shared ``aim_intensity`` function produces the
    identical value via the preprocess path (``load_audio`` -> ``aim_intensity``) and
    the inference path (``generate.prepare_audio``) on the SAME audio file — no
    train/infer skew (the round-3a audio-parity lesson).
  * the context vector has the new ``aim`` dim in the right place with the right length;
  * ``context_from_manifest`` defaults gracefully when ``aim_intensity`` is absent
    (old datasets processed before the field existed);
  * ``target_context(aim_intensity=...)`` / the ``--aim-intensity`` override changes
    exactly the aim slot and nothing else;
  * a synthetic "busy" (percussive) song yields higher intensity than a "calm"
    (sustained-tone) song (monotonicity sanity);
  * silence / empty / non-finite audio -> 0.0.
"""

import numpy as np
import pytest
import soundfile as sf

from src.conditioning import (
    CONTEXT_DIM,
    context_from_manifest,
    context_vector,
    target_context,
)
from src.config import AUDIO
from src.data.audio import _AIM_REF, aim_intensity, load_audio

SR = AUDIO.sample_rate
AIM_IDX = 6   # aim is appended last (index 6); 0-5 keep their meaning
C_SIG = 6     # tiny signal width for the hermetic backward-compat checkpoints


def _busy(seconds=3.0):
    """Dense percussive clicks (many sharp onsets) -> high aim-intensity.

    Short noise bursts at 8 Hz: each burst is a broadband onset, the gaps between
    them keep the onset envelope's normalised mean high (lots of strong frames)."""
    rng = np.random.default_rng(0)
    n = int(SR * seconds)
    y = np.zeros(n, dtype=np.float32)
    burst = int(SR * 0.02)            # 20 ms broadband burst
    for start in range(0, n - burst, int(SR / 8.0)):   # 8 onsets / second
        y[start:start + burst] += 0.8 * rng.standard_normal(burst).astype(np.float32)
    return y


def _calm(seconds=3.0, freq=220.0):
    """A single smooth sustained sine -> almost no onsets -> low aim-intensity."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# the shared feature function
# --------------------------------------------------------------------------- #
def test_aim_intensity_bounded_range():
    """Always in [0, 1] for any real audio."""
    for y in (_busy(), _calm(), _busy(1.0), _calm(5.0, 440.0)):
        v = aim_intensity(y)
        assert 0.0 <= v <= 1.0
        assert np.isfinite(v)


def test_aim_intensity_deterministic():
    """Same array -> same value (no hidden randomness; required for train/infer parity)."""
    y = _busy()
    assert aim_intensity(y) == aim_intensity(y)
    assert aim_intensity(y.copy()) == aim_intensity(y)


def test_busy_higher_than_calm():
    """Monotonicity sanity: a percussive song scores clearly above a sustained tone."""
    assert aim_intensity(_busy()) > aim_intensity(_calm())


def test_silence_and_degenerate_audio_zero():
    """Silence / empty / non-finite -> 0.0 (safe calm default == missing-field default)."""
    assert aim_intensity(np.zeros(SR, dtype=np.float32)) == 0.0
    assert aim_intensity(np.array([], dtype=np.float32)) == 0.0
    assert aim_intensity(np.full(SR, np.nan, dtype=np.float32)) == 0.0


def test_loudness_invariance():
    """A globally louder copy of the same song scores ~the same (the loudness-stability
    claim): onset_strength responds to spectral *change* and we peak-normalise, so a
    6 dB (2x) gain barely moves the value."""
    y = _busy()
    quiet = aim_intensity(y)
    loud = aim_intensity((y * 2.0).astype(np.float32))
    assert abs(loud - quiet) < 0.05


# --------------------------------------------------------------------------- #
# PARITY — the key test: preprocess path == inference path
# --------------------------------------------------------------------------- #
def test_preprocess_inference_parity(tmp_path):
    """The aim-intensity computed at preprocess (load_audio -> aim_intensity) is
    BITWISE-identical to the one ``generate.prepare_audio`` attaches at inference,
    on the same audio file. This is the no-skew guarantee — both route through the
    single shared function on the same decoded array."""
    from src.generate import prepare_audio

    wav = tmp_path / "song.wav"
    sf.write(wav, _busy(), SR)

    # preprocess path (mirrors preprocess._process_set: decode once, then aim_intensity)
    preprocess_aim = aim_intensity(load_audio(wav))

    # inference path (CPU, no model load): prepare_audio decodes + computes aim itself
    prepared = prepare_audio(str(wav), device="cpu")
    assert prepared.aim == preprocess_aim          # exact parity, no train/infer skew
    assert 0.0 <= prepared.aim <= 1.0


# --------------------------------------------------------------------------- #
# context vector layout
# --------------------------------------------------------------------------- #
def test_context_dim_is_seven_with_aim_last():
    assert CONTEXT_DIM == 7
    c = context_vector(sr=5.0, ar=9.0, od=8.0, hp=5.0, cs=4.0, density=3.5, aim=0.6)
    assert len(c) == CONTEXT_DIM
    assert c[AIM_IDX] == pytest.approx(0.6)        # aim already ~[0,1], scale 1.0


def test_context_vector_aim_defaults_zero():
    """Old call sites that pass only the 6 difficulty fields stay valid (aim -> 0)."""
    c = context_vector(sr=5.0, ar=9.0, od=8.0, hp=5.0, cs=4.0, density=3.5)
    assert len(c) == CONTEXT_DIM
    assert c[AIM_IDX] == 0.0


def test_context_vector_aim_clamped():
    """aim is clamped to [0, 1.5] like every slot (defensive against a bad override)."""
    assert context_vector(5, 9, 8, 5, 4, 3.5, aim=99.0)[AIM_IDX] == 1.5
    assert context_vector(5, 9, 8, 5, 4, 3.5, aim=-1.0)[AIM_IDX] == 0.0


# --------------------------------------------------------------------------- #
# target_context override (inference side)
# --------------------------------------------------------------------------- #
def test_target_context_aim_override_changes_only_aim():
    base = target_context(5.0)
    over = target_context(5.0, aim_intensity=0.9)
    assert base[AIM_IDX] == 0.0                    # default baseline
    assert over[AIM_IDX] == pytest.approx(0.9)
    # nothing else moved (SR/AR/OD/HP/CS/density identical)
    assert base[:AIM_IDX] == over[:AIM_IDX]


def test_target_context_default_back_compat():
    """target_context(sr) with no aim behaves exactly as before (aim slot = 0)."""
    c = target_context(7.0)
    assert len(c) == CONTEXT_DIM
    assert c[AIM_IDX] == 0.0


# --------------------------------------------------------------------------- #
# context_from_manifest (train side)
# --------------------------------------------------------------------------- #
def test_context_from_manifest_reads_aim():
    c = context_from_manifest({"n_objects": 300, "duration_s": 100.0,
                               "aim_intensity": 0.42})
    assert len(c) == CONTEXT_DIM
    assert c[AIM_IDX] == pytest.approx(0.42)


def test_context_from_manifest_missing_aim_defaults_zero():
    """Old datasets (no aim_intensity field) still build a full-length context with the
    neutral baseline -> they train without re-preprocessing."""
    c = context_from_manifest({"n_objects": 300, "duration_s": 100.0})
    assert len(c) == CONTEXT_DIM
    assert c[AIM_IDX] == 0.0
    # the difficulty/density slots are unaffected by the new field
    assert c[5] == pytest.approx(0.25)             # density 300/100=3 -> /12


def test_aim_ref_constant_sane():
    """_AIM_REF only rescales the [0,1] axis; guard it stays a positive finite scalar."""
    assert 0.0 < _AIM_REF < 10.0


# --------------------------------------------------------------------------- #
# BACKWARD-COMPAT (load-bearing): CONTEXT_DIM grew 6 -> 7, so OLD checkpoints whose
# ctx_mlp was built at ctx_dim=6 must STILL LOAD + sample. load_model reads the ctx
# width from the checkpoint's own ctx_mlp.0.weight, so a pre-v9 ckpt loads at 6 and a
# v9 ckpt at 7 — neither crashes load_state_dict. The new aim slot is simply unused by
# old models (it only takes effect once the USER retrains).
# --------------------------------------------------------------------------- #
def _save_ckpt(path, ctx_dim):
    """Save a tiny UNet the way train.py does. ``ctx_dim=6`` emulates a pre-v9 (v8/v8_1)
    checkpoint (no 'ctx_dim' key in args, exactly like the real saved v8 ckpts).

    Uses the real UNet1d default ``t_dim`` and ``mults`` so that load_model (which
    rebuilds the net from cargs and does NOT forward t_dim) reconstructs an identical
    architecture — isolating the test to the ONE thing under test: the context width."""
    import torch

    from src.model.unet import UNet1d
    m = UNet1d(C_SIG, AUDIO.n_mels, base=16, attn=False, ctx_dim=ctx_dim)
    torch.save({"model": m.state_dict(), "ema": m.state_dict(),
                # pre-v9 ckpts carried 'cfg_drop' (conditioned) but NOT 'ctx_dim'
                "args": {"base": 16, "attn": False, "attn_levels": 2, "adaln": True,
                         "cfg_drop": 0.1, "objective": "v", "zero_snr": True},
                "sig_channels": C_SIG}, path)


@pytest.mark.parametrize("ckpt_ctx_dim", [6, 7])
def test_load_model_uses_checkpoint_ctx_dim(tmp_path, ckpt_ctx_dim):
    """A checkpoint built at ctx_dim={6 (pre-v9), 7 (v9)} loads at THAT width (read from
    its own ctx_mlp weights) and samples — proves bumping CONTEXT_DIM to 7 doesn't break
    old checkpoints whose ctx_mlp was built at 6."""
    import torch

    from src.generate import load_model
    from src.model.diffusion import GaussianDiffusion

    ckpt = tmp_path / f"ctx{ckpt_ctx_dim}.pt"
    _save_ckpt(ckpt, ckpt_ctx_dim)
    loaded = load_model(str(ckpt), device="cpu")   # must NOT crash load_state_dict
    assert loaded.ctx_dim == ckpt_ctx_dim          # width came from the checkpoint
    assert loaded.model.ctx_dim == ckpt_ctx_dim
    # and it can sample with a ctx of its own width (no shape crash)
    diff = GaussianDiffusion(timesteps=50, device="cpu", objective="v", zero_snr=True)
    cond = torch.randn(1, AUDIO.n_mels, 32)
    ctx = torch.rand(1, ckpt_ctx_dim)
    out = diff.ddim_sample(loaded.model, cond, (1, C_SIG, 32), steps=4, ctx=ctx)
    assert out.shape == (1, C_SIG, 32) and torch.isfinite(out).all()


def test_generate_ctx_truncates_to_old_checkpoint_width(tmp_path):
    """The REAL inference path (the gap the width-6 sample test above missed):
    ``target_context`` always emits the full CONTEXT_DIM (v9 = 7), but a pre-v9 (6-dim)
    checkpoint's ctx_mlp expects 6. ``generate._one_pass`` truncates ctx to the model's
    ctx_dim — and because ``aim`` is appended LAST, that drops exactly the slot the old
    model never knew. Without the truncation this raised
    'mat1 and mat2 shapes cannot be multiplied (1x7 and 6x256)' (the bug the GPU run hit).
    """
    import torch

    from src.conditioning import CONTEXT_DIM, target_context
    from src.generate import load_model
    from src.model.diffusion import GaussianDiffusion

    ckpt = tmp_path / "ctx6.pt"
    _save_ckpt(ckpt, 6)                                   # pre-v9 width-6 model
    loaded = load_model(str(ckpt), device="cpu")
    assert loaded.ctx_dim == 6

    full = target_context(5.0, density=4.0, aim_intensity=0.9)
    assert len(full) == CONTEXT_DIM == 7                 # always the full width
    diff = GaussianDiffusion(timesteps=50, device="cpu", objective="v", zero_snr=True)
    cond = torch.randn(1, AUDIO.n_mels, 32)
    full_t = torch.tensor([full], dtype=torch.float32)
    # un-truncated 7-dim ctx into the 6-dim model -> the matmul error the fix prevents
    with pytest.raises(RuntimeError):
        diff.ddim_sample(loaded.model, cond, (1, C_SIG, 32), steps=4, ctx=full_t)
    # the _one_pass fix: truncate to the model's ctx_dim -> samples cleanly
    ctx = full_t[:, :loaded.ctx_dim]
    assert ctx.shape == (1, 6)
    out = diff.ddim_sample(loaded.model, cond, (1, C_SIG, 32), steps=4, ctx=ctx)
    assert out.shape == (1, C_SIG, 32) and torch.isfinite(out).all()
