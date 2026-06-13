"""Train the conditional diffusion model on preprocessed osu! data.

python -m src.train --data data/processed --epochs 50 --batch 16 --crop 1024
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import AUDIO, N_SIGNAL_CHANNELS
from .data.dataset import OsuSignalDataset
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    ds = OsuSignalDataset(args.data, crop_frames=args.crop)
    print(f"dataset: {len(ds)} difficulties")
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )

    model = UNet1d(sig_channels=N_SIGNAL_CHANNELS, cond_channels=AUDIO.n_mels, base=args.base).to(
        device
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")
    diff = GaussianDiffusion(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for sig, mel in dl:
            sig = sig.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            b = sig.shape[0]
            t = torch.randint(0, diff.timesteps, (b,), device=device)
            noise = torch.randn_like(sig)
            x_t = diff.q_sample(sig, t, noise)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(x_t, mel, t)
                loss = torch.nn.functional.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}")
        avg = running / max(1, len(dl))
        dt = time.time() - t0
        print(f"epoch {epoch} done avg_loss {avg:.4f} ({dt:.1f}s)")
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = out / f"model_e{epoch + 1}.pt"
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "model_last.pt")
            print(f"  saved {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
