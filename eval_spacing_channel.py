"""Probe the v8 spacing channel (RESEARCH 10.11). The v8 bet: the spacing *channel*
mean-regresses to the correct (larger) magnitude while the cursor *positions* collapse
toward centre, and `respace_by_magnitude` exploits that gap. So per generated map, compare
mean(decode_spacing) [the channel's predicted magnitude] vs mean(cursor head-to-head
spacing) [the model's actual positions]. ratio > 1 => the channel wants bigger spacing than
the positions show => respace will expand jumps (mechanism active). ratio ~ 1 => not yet
(undertrained, or the channel hasn't diverged from the positions).

  uv run python eval_spacing_channel.py --ckpt runs/<id>/ckpt/best.pt
"""
from __future__ import annotations

import argparse
from math import hypot

import numpy as np
import torch

from src.conditioning import target_context
from src.data.signal import decode_signal, decode_spacing
from src.generate import load_model, prepare_audio

AUDIO_2MIN = "C:/osu!/Songs/986934 JIN feat LiSA - Headphone Actor/audio.mp3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--srs", default="3,4,5,6")
    ap.add_argument("--guidance", type=float, default=2.0)
    args = ap.parse_args()

    loaded = load_model(args.ckpt)
    model, diff, _ctx_dim, device = loaded
    cond, t_len, t_full, _tp = prepare_audio(AUDIO_2MIN, device)
    torch.manual_seed(0)

    print(f"\n{'SR':>4} {'cursor_sp':>10} {'channel_sp':>11} {'ratio':>6} "
          f"{'cur_p90':>8} {'chan_p90':>9} {'n_obj':>6}")
    for sr in [float(s) for s in args.srs.split(",")]:
        ctx = torch.tensor([target_context(sr)], dtype=torch.float32, device=device)
        sig = diff.ddim_sample(model, cond, (1, model.sig_channels, t_full),
                               steps=100, ctx=ctx, guidance=args.guidance)
        sig = sig[0, :, :t_len].float().cpu().numpy()
        objs = sorted(decode_signal(sig), key=lambda o: o.time)
        cur = [hypot(b.x - a.x, b.y - a.y) for a, b in zip(objs, objs[1:])]
        mags = decode_spacing(sig, objs)
        chan = [m for m in mags[1:] if m > 0] if mags else []
        cm = float(np.mean(cur)) if cur else 0.0
        hm = float(np.mean(chan)) if chan else 0.0
        cp = float(np.percentile(cur, 90)) if cur else 0.0
        hp = float(np.percentile(chan, 90)) if chan else 0.0
        print(f"{sr:>4.1f} {cm:>10.1f} {hm:>11.1f} {hm / max(cm, 1):>6.2f} "
              f"{cp:>8.1f} {hp:>9.1f} {len(objs):>6}")


if __name__ == "__main__":
    main()
