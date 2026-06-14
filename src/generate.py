"""Generate a playable .osu file from an audio file using a trained model.

python -m src.generate --audio song.mp3 --ckpt checkpoints/model_last.pt --out out.osu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .conditioning import CONTEXT_DIM, target_context
from .config import AUDIO, N_SIGNAL_CHANNELS
from .data.audio import load_audio, log_mel
from .data.signal import decode_kiai, decode_signal
from .data.timing import estimate_timing_point
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d
from .parsing.beatmap import Beatmap, TimingPoint, write_osu
from .postprocess import (
    clamp_slider_endpoints,
    compute_breaks,
    snap_slider_ends,
    snap_to_grid,
    trim_isolated_ends,
)


def generate(audio_path, ckpt_path, out_path, steps=100, base=64, use_ema=True,
             snap=True, sr=None, guidance=2.0, match_sr=False, max_iter=3, tol=0.4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cargs = ckpt.get("args", {})
    base = cargs.get("base", base)
    attn = cargs.get("attn", False)        # old checkpoints had no attention
    attn_levels = cargs.get("attn_levels", 2)  # old ckpts: 2 deepest levels
    ctx_dim = CONTEXT_DIM if ("cfg_drop" in cargs or "ctx_dim" in cargs) else 0
    model = UNet1d(N_SIGNAL_CHANNELS, AUDIO.n_mels, base=base, attn=attn,
                   ctx_dim=ctx_dim, attn_levels=attn_levels).to(device)
    weights = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()
    diff = GaussianDiffusion(timesteps=cargs.get("timesteps", 1000), device=device)

    y = load_audio(audio_path)          # decode once, reuse for mel + timing
    mel = log_mel(y)                    # (n_mels, T)
    T = mel.shape[1]
    pad = (-T) % 16                    # multiple of 16 for clean U-Net striding
    mel_p = np.pad(mel, ((0, 0), (0, pad)))
    cond = torch.from_numpy(mel_p[None].astype(np.float32)).to(device)
    tp = estimate_timing_point(y)
    print(f"estimated timing: {60000.0 / tp.beat_length:.1f} BPM, offset {tp.time} ms")

    def _one_pass(sr_used):
        ctx = None
        if ctx_dim and sr_used is not None:
            ctx = torch.tensor([target_context(sr_used)], dtype=torch.float32, device=device)
        sig = diff.ddim_sample(model, cond, (1, N_SIGNAL_CHANNELS, mel_p.shape[1]),
                               steps=steps, ctx=ctx, guidance=guidance)
        sig = sig[0, :, :T].float().cpu().numpy()
        objects = decode_signal(sig)
        trim_isolated_ends(objects)
        bm = Beatmap(path=Path(out_path))
        bm.audio_filename = Path(audio_path).name
        bm.title = Path(audio_path).stem
        bm.version = f"AI {sr:.1f} star" if sr else "AI Generated"
        bm.approach_rate, bm.overall_difficulty, bm.hp, bm.circle_size = 8.0, 7.0, 5.0, 4.0
        if snap:
            snap_to_grid(objects, tp)                       # snap onsets to grid
            snap_slider_ends(objects, tp, bm.slider_multiplier)  # snap slider ends
        # keep slider tails inside the playfield (must run after length-changing
        # snap_slider_ends, before breaks/write which read end_time gaps)
        clamp_slider_endpoints(objects)
        breaks = compute_breaks(objects)
        timing = [tp]
        for ks, ke in decode_kiai(sig):
            timing.append(TimingPoint(ks, -100.0, tp.meter, False, effects=1))
            timing.append(TimingPoint(ke, -100.0, tp.meter, False, effects=0))
        timing.sort(key=lambda t: t.time)
        write_osu(bm, objects, out_path, timing_points=timing, breaks=breaks)
        return objects

    if match_sr and sr is not None and ctx_dim:
        # feedback loop: nudge the context SR until the *achieved* SR (rosu) hits
        # the requested target, correcting the model's systematic SR offset.
        from .difficulty import star_rating
        cur = sr
        for it in range(max_iter):
            objs = _one_pass(cur)
            achieved = star_rating(out_path)
            print(f"  [match-sr] iter {it}: ctx={cur:.2f} -> achieved {achieved}")
            if achieved is None or abs(achieved - sr) <= tol:
                break
            cur = float(min(11.0, max(1.0, cur + 0.8 * (sr - achieved))))
    else:
        if ctx_dim and sr is not None:
            print(f"conditioning on target star rating {sr:.2f}* (guidance {guidance})")
        objs = _one_pass(sr)

    print(f"wrote {out_path} ({len(objs)} hit objects)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default="checkpoints/model_last.pt")
    ap.add_argument("--out", default="generated.osu")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sr", type=float, default=None, help="target star rating")
    ap.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance scale")
    ap.add_argument("--match-sr", action="store_true",
                    help="iterate to hit the requested star rating (corrects SR offset)")
    ap.add_argument("--no-snap", action="store_true", help="disable beat-snapping")
    args = ap.parse_args()
    generate(args.audio, args.ckpt, args.out, steps=args.steps, snap=not args.no_snap,
             sr=args.sr, guidance=args.guidance, match_sr=args.match_sr)


if __name__ == "__main__":
    main()
