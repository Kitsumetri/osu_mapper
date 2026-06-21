import argparse

import torch

from src.config import CH_CORNER, CH_CURX, CH_SPACING, N_SIGNAL_CHANNELS
from src.model.diffusion import GaussianDiffusion
from src.train import _diffusion_loss, _spatial_channel_weights


def test_spatial_channel_weights_mean_one_and_upweights():
    w1 = _spatial_channel_weights(1.0)
    assert torch.allclose(w1, torch.ones(N_SIGNAL_CHANNELS))      # 1.0 -> exact no-op
    w3 = _spatial_channel_weights(3.0)
    assert abs(float(w3.mean()) - 1.0) < 1e-6                     # overall scale preserved
    assert float(w3[CH_CURX]) > float(w3[0])                      # cursor > onset (non-spatial)
    assert float(w3[CH_SPACING]) > 1.0                            # spacing up-weighted
    assert float(w3[CH_CORNER]) > 1.0  # corner up-weighted in the loss, not re-encoded


def test_diffusion_loss_channel_weight_raises_spatial_error():
    args = argparse.Namespace(loss="mse", huber_beta=1.0, min_snr_gamma=0.0)
    diff = GaussianDiffusion(timesteps=10, device="cpu")
    b, c, t_len = 2, N_SIGNAL_CHANNELS, 8
    pred = torch.zeros(b, c, t_len)
    target = torch.zeros(b, c, t_len)
    target[:, CH_CURX] = 1.0                                      # error only in a spatial ch
    t = torch.zeros(b, dtype=torch.long)
    base = _diffusion_loss(pred, target, t, diff, args)
    w = _spatial_channel_weights(4.0).view(1, -1, 1)
    up = _diffusion_loss(pred, target, t, diff, args, channel_w=w)
    assert float(up) > float(base) * 1.5                         # spatial error weighted up
