"""Generate a playable .osu file from an audio file using a trained model.

  python -m src.generate --audio song.mp3 --ckpt checkpoints/model_last.pt --out out.osu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import AUDIO, N_SIGNAL_CHANNELS
from .data.audio import audio_to_mel
from .data.signal import decode_signal
from .model.unet import UNet1d
from .model.diffusion import GaussianDiffusion
from .parsing.beatmap import Beatmap, write_osu, TimingPoint


def generate(audio_path, ckpt_path, out_path, steps=200, window=2048, base=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    base = ckpt.get("args", {}).get("base", base)
    model = UNet1d(N_SIGNAL_CHANNELS, AUDIO.n_mels, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diff = GaussianDiffusion(timesteps=ckpt.get("args", {}).get("timesteps", 1000),
                             device=device)

    mel = audio_to_mel(audio_path)               # (n_mels, T)
    T = mel.shape[1]
    # round up to multiple of 16 for clean U-Net striding
    pad = (-T) % 16
    mel_p = np.pad(mel, ((0, 0), (0, pad)))
    cond = torch.from_numpy(mel_p[None].astype(np.float32)).to(device)

    sig = diff.ddim_sample(model, cond, (1, N_SIGNAL_CHANNELS, mel_p.shape[1]), steps=steps)
    sig = sig[0, :, :T].float().cpu().numpy()

    objects = decode_signal(sig)
    print(f"generated {len(objects)} hit objects")

    bm = Beatmap(path=Path(out_path))
    bm.audio_filename = Path(audio_path).name
    bm.title = Path(audio_path).stem
    bm.version = "AI Generated"
    tps = [TimingPoint(0, 500.0, 4, True)]
    write_osu(bm, objects, out_path, timing_points=tps)
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default="checkpoints/model_last.pt")
    ap.add_argument("--out", default="generated.osu")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    generate(args.audio, args.ckpt, args.out, steps=args.steps)


if __name__ == "__main__":
    main()
