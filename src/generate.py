"""Generate a playable .osu file from an audio file using a trained model.

python -m src.generate --audio song.mp3 --ckpt checkpoints/model_last.pt --out out.osu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import AUDIO, N_SIGNAL_CHANNELS
from .data.audio import load_audio, log_mel
from .data.signal import decode_signal
from .data.timing import estimate_timing_point
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d
from .parsing.beatmap import Beatmap, write_osu
from .postprocess import snap_to_grid, trim_isolated_ends


def generate(audio_path, ckpt_path, out_path, steps=200, window=2048, base=64,
             use_ema=True, snap=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cargs = ckpt.get("args", {})
    base = cargs.get("base", base)
    attn = cargs.get("attn", False)  # old checkpoints had no attention
    model = UNet1d(N_SIGNAL_CHANNELS, AUDIO.n_mels, base=base, attn=attn).to(device)
    weights = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()
    diff = GaussianDiffusion(timesteps=cargs.get("timesteps", 1000), device=device)

    y = load_audio(audio_path)          # decode once, reuse for mel + timing
    mel = log_mel(y)                    # (n_mels, T)
    T = mel.shape[1]
    # round up to multiple of 16 for clean U-Net striding
    pad = (-T) % 16
    mel_p = np.pad(mel, ((0, 0), (0, pad)))
    cond = torch.from_numpy(mel_p[None].astype(np.float32)).to(device)

    sig = diff.ddim_sample(model, cond, (1, N_SIGNAL_CHANNELS, mel_p.shape[1]), steps=steps)
    sig = sig[0, :, :T].float().cpu().numpy()

    objects = decode_signal(sig)
    trimmed = trim_isolated_ends(objects)
    print(f"generated {len(objects)} hit objects ({trimmed} dangling end notes trimmed)")

    bm = Beatmap(path=Path(out_path))
    bm.audio_filename = Path(audio_path).name
    bm.title = Path(audio_path).stem
    bm.version = "AI Generated"
    # difficulty defaults from dataset means (std maps): AR~8, OD~7, HP~4.7, CS~3.8.
    # The model has no difficulty conditioning yet, so these are fixed for now.
    bm.approach_rate = 8.0
    bm.overall_difficulty = 7.0
    bm.hp = 5.0
    bm.circle_size = 4.0

    # estimate BPM + offset from the audio (best-effort; see data/timing.py)
    tp = estimate_timing_point(y)
    bpm = 60000.0 / tp.beat_length
    print(f"estimated timing: {bpm:.1f} BPM, offset {tp.time} ms")
    if snap:
        moved = snap_to_grid(objects, tp)
        print(f"beat-snapped {moved}/{len(objects)} objects to the grid")
    write_osu(bm, objects, out_path, timing_points=[tp])
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default="checkpoints/model_last.pt")
    ap.add_argument("--out", default="generated.osu")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--no-snap", action="store_true", help="disable beat-snapping")
    args = ap.parse_args()
    generate(args.audio, args.ckpt, args.out, steps=args.steps, snap=not args.no_snap)


if __name__ == "__main__":
    main()
