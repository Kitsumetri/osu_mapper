"""Train the conditional diffusion model on a preprocessed osu! dataset.

  python -m src.train --data data/processed/std-v1 --epochs 200 --batch 12 \
      --crop 3072 --base 160 --tag std-v1-base160

Features: bf16 autocast, EMA weights, cosine LR with warmup, gradient
accumulation, and self-contained run logging under runs/<run_id>/.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .conditioning import CONTEXT_DIM
from .config import AUDIO, N_SIGNAL_CHANNELS
from .data.dataset import OsuSignalDataset
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d


class EMA:
    """Exponential moving average of model parameters for cleaner samples."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.tag}"
    run_dir = Path(args.runs) / run_id
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run={run_dir}")

    ds = OsuSignalDataset(args.data, crop_frames=args.crop, min_objects=args.min_objects)
    print(f"dataset: {len(ds)} difficulties")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)

    model = UNet1d(N_SIGNAL_CHANNELS, AUDIO.n_mels, base=args.base, attn=args.attn,
                   ctx_dim=CONTEXT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params (base={args.base}, attn={args.attn})")
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None
    diff = GaussianDiffusion(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    config = {**vars(args), "run_id": run_id, "n_params": n_params,
              "git_commit": _git_commit(), "amp_dtype": str(amp_dtype),
              "dataset_size": len(ds)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = open(run_dir / "metrics.csv", "w", newline="")  # noqa: SIM115
    writer = csv.writer(metrics)
    writer.writerow(["epoch", "avg_loss", "lr", "sec"])

    steps_per_epoch = max(1, len(dl) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup, total_steps // 10)
    gstep = 0
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for i, (sig, mel, ctx) in enumerate(dl):
            sig = sig.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True)
            b = sig.shape[0]
            # classifier-free guidance: randomly drop the difficulty context
            ctx_drop = torch.rand(b, device=device) < args.cfg_drop
            t = torch.randint(0, diff.timesteps, (b,), device=device)
            noise = torch.randn_like(sig)
            x_t = diff.q_sample(sig, t, noise)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                pred = model(x_t, mel, t, ctx=ctx, ctx_drop=ctx_drop)
                loss = torch.nn.functional.mse_loss(pred, noise) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                lr = _lr_at(gstep, total_steps, warmup, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
                gstep += 1
                if gstep % args.log_every == 0:
                    cur = loss.item() * args.accum
                    print(f"  e{epoch} step {gstep}/{total_steps} loss {cur:.4f} lr {lr:.2e}")
        avg = running / max(1, len(dl))
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch} avg_loss {avg:.4f} lr {lr_now:.2e} ({dt:.1f}s)")
        writer.writerow([epoch, f"{avg:.5f}", f"{lr_now:.3e}", f"{dt:.1f}"])
        metrics.flush()

        def _ckpt(path, epoch=epoch):
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict() if ema else None,
                        "args": vars(args), "epoch": epoch,
                        "git_commit": config["git_commit"]}, path)

        _ckpt(ckpt_dir / "last.pt")
        if avg < best:
            best = avg
            _ckpt(ckpt_dir / "best.pt")
        if (epoch + 1) % args.save_every == 0:
            _ckpt(ckpt_dir / f"epoch_{epoch + 1}.pt")
    metrics.close()
    print(f"done. best avg_loss {best:.4f}  ->  {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/std-v1")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--crop", type=int, default=3072)
    ap.add_argument("--base", type=int, default=160)
    ap.add_argument("--attn", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--cfg-drop", type=float, default=0.15,
                    help="prob. of dropping difficulty context (classifier-free guidance)")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-objects", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
