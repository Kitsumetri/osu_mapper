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
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .conditioning import CONTEXT_DIM
from .config import (
    AUDIO,
    CH_CURX,
    CH_CURY,
    CH_SLIDER_ANCHORS,
    CH_SLIDES,
    CH_SPACING,
    N_SIGNAL_CHANNELS,
)
from .data.dataset import OsuSignalDataset
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d


class _Tee:
    """Duplicate a stream (stdout/stderr) into a log file so the run folder keeps
    a full transcript of prints, warnings, and tracebacks."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()


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


def _diffusion_loss(pred, target, t, diff, args, channel_w=None):
    """Diffusion loss with optional Huber distance + Min-SNR-gamma weighting (computed
    in fp32 for stability). reduction matches the old mean MSE when all are default.
    ``channel_w`` (1,C,1) up-weights chosen channels before the spatial reduction."""
    if args.loss == "huber":
        per = torch.nn.functional.smooth_l1_loss(
            pred.float(), target.float(), reduction="none", beta=args.huber_beta)
    else:
        per = (pred.float() - target.float()) ** 2
    if channel_w is not None:
        per = per * channel_w                         # (B,C,T) * (1,C,1): up-weight spatial
    per = per.mean(dim=tuple(range(1, per.ndim)))     # per-sample (B,)
    if args.min_snr_gamma > 0:
        per = per * diff.loss_weight(t, args.min_snr_gamma)
    return per.mean()


def _spatial_channel_weights(weight: float, n_channels: int = N_SIGNAL_CHANNELS):
    """Per-channel loss-weight vector (mean 1) that up-weights the **spatial** channels —
    cursor x/y, the slider anchor offsets, and the v8 spacing magnitude — by ``weight``.
    The easy piecewise channels (SV/curve/corner/hitsounds/holds) are 'solved' early and
    dominate the averaged MSE, so the hard position channels stay underfit and the model
    hedges them to the mean -> under-dispersion (RESEARCH 10.10/10.11). Renormalised so the
    mean weight is 1 (overall loss scale unchanged). ``weight=1.0`` -> all ones (no-op)."""
    w = torch.ones(n_channels)
    spatial = [CH_CURX, CH_CURY, *range(CH_SLIDER_ANCHORS, CH_SLIDES), CH_SPACING]
    w[spatial] = weight
    return w * n_channels / w.sum()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = min((step - warmup) / max(1, total - warmup), 1.0)  # clamp: never let LR rise back
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cheap, safe perf settings (no effect on architecture / ckpt compatibility):
    # TF32 matmul + autotuned conv kernels for our fixed crop size.
    torch.set_float32_matmul_precision("high")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    # --resume continues an interrupted run *in place* (same run dir, appended
    # metrics, restored optimizer/EMA/step so the LR schedule stays continuous).
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location=device, weights_only=False)
        run_dir = Path(args.resume).resolve().parent.parent
        run_id = run_dir.name
        print(f"resuming {args.resume} (epoch {resume_ck.get('epoch')}) -> {run_dir}")
    else:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.tag}"
        run_dir = Path(args.runs) / run_id
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # tee stdout/stderr (prints, warnings, tracebacks) into runs/<id>/train.log
    log_fh = open(run_dir / "train.log", "a", encoding="utf-8")  # noqa: SIM115
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    print(f"device={device}  run={run_dir}  ({datetime.now():%Y-%m-%d %H:%M:%S})")

    # train/val split: hold out a deterministic slice for validation. The val set
    # uses a non-augmented dataset view so the metric is stable across epochs.
    ds = OsuSignalDataset(args.data, crop_frames=args.crop, min_objects=args.min_objects,
                          augment=args.augment)
    val_ds_full = OsuSignalDataset(args.data, crop_frames=args.crop,
                                   min_objects=args.min_objects, augment=False)
    n_total = len(ds)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(1234)).tolist()
    n_val = int(n_total * args.val_frac) if args.val_frac > 0 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = Subset(ds, train_idx) if n_val else ds
    val_ds = Subset(val_ds_full, val_idx) if n_val else None
    print(f"dataset: {n_total} difficulties (train {len(train_ds)}, val {n_val})")
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
    val_dl = (DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, drop_last=False, pin_memory=True)
              if val_ds is not None else None)

    model = UNet1d(N_SIGNAL_CHANNELS, AUDIO.n_mels, base=args.base, attn=args.attn,
                   ctx_dim=CONTEXT_DIM, attn_levels=args.attn_levels,
                   adaln=args.adaln, rope=args.rope, up_attn=args.up_attn,
                   grad_ckpt=args.grad_checkpoint).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params (base={args.base}, attn={args.attn})")
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None
    diff = GaussianDiffusion(timesteps=args.timesteps, device=device,
                             objective=args.objective, zero_snr=args.zero_snr)
    print(f"diffusion: objective={args.objective} zero_snr={args.zero_snr}")
    # v8: optionally up-weight the spatial channels (cursor/anchors/spacing) so patterns
    # aren't hedged to the mean. None (weight 1.0) keeps the exact old unweighted loss.
    channel_w = (None if args.spatial_loss_weight == 1.0 else
                 _spatial_channel_weights(args.spatial_loss_weight).to(device).view(1, -1, 1))
    if channel_w is not None:
        print(f"spatial loss weight: {args.spatial_loss_weight}x on cursor/anchors/spacing")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            fused=(device == "cuda"))
    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    steps_per_epoch = max(1, len(dl) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup, total_steps // 10)
    gstep = 0
    best = float("inf")
    start_epoch = 0
    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
        if ema and resume_ck.get("ema"):
            ema.shadow.load_state_dict(resume_ck["ema"])
        if resume_ck.get("opt"):
            opt.load_state_dict(resume_ck["opt"])
        if resume_ck.get("scaler") and use_scaler:
            scaler.load_state_dict(resume_ck["scaler"])  # keep fp16 scale across resume
        start_epoch = int(resume_ck.get("epoch", -1)) + 1
        gstep = int(resume_ck.get("gstep", start_epoch * steps_per_epoch))
        best = float(resume_ck.get("best", best))
        print(f"  restored: start_epoch={start_epoch} gstep={gstep} best={best:.5f}")

    config = {**vars(args), "run_id": run_id, "n_params": n_params,
              "git_commit": _git_commit(), "amp_dtype": str(amp_dtype),
              "dataset_size": len(ds)}
    if resume_ck is None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # append on resume so the existing metrics history is preserved
    new_metrics = resume_ck is None or not (run_dir / "metrics.csv").exists()
    metrics = open(run_dir / "metrics.csv", "a" if not new_metrics else "w", newline="")  # noqa: SIM115
    writer = csv.writer(metrics)
    if new_metrics:
        writer.writerow(["epoch", "avg_loss", "val_loss", "lr", "sec"])

    @torch.no_grad()
    def _validate():
        """Average diffusion loss over the held-out set. Uses a fixed RNG for the
        timestep/noise draws so the number is comparable across epochs."""
        if val_dl is None:
            return None
        model.eval()
        g = torch.Generator(device=device).manual_seed(1234)
        tot, nb = 0.0, 0
        for sig, mel, ctx in val_dl:
            sig = sig.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True)
            b = sig.shape[0]
            t = torch.randint(0, diff.timesteps, (b,), device=device, generator=g)
            noise = torch.randn(sig.shape, device=device, generator=g)
            x_t = diff.q_sample(sig, t, noise)
            target = diff.target(sig, t, noise)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                pred = model(x_t, mel, t, ctx=ctx)      # full conditioning (no CFG drop)
            loss = _diffusion_loss(pred, target, t, diff, args, channel_w)
            tot += loss.item() * b      # sample-weight so a smaller last batch isn't over-counted
            nb += b
        model.train()
        return tot / max(1, nb)

    # torch.compile wraps the model for the training forward only; EMA, optimizer,
    # validation and checkpoints all use the *raw* `model`, so saved state_dicts
    # never carry the compiled `_orig_mod.` prefix (resume/generate stay compatible).
    # NOTE: on Windows this needs `triton-windows` + MSVC Build Tools (cl.exe);
    # without them torch.compile raises at the first step. Default off.
    fwd_model = torch.compile(model) if (args.compile and device == "cuda") else model
    if args.compile:
        print("torch.compile enabled (first step compiles; needs triton + a C compiler)")

    for epoch in range(start_epoch, args.epochs):
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
            target = diff.target(sig, t, noise)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                pred = fwd_model(x_t, mel, t, ctx=ctx, ctx_drop=ctx_drop)
            loss = _diffusion_loss(pred, target, t, diff, args, channel_w) / args.accum
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
        val = _validate()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        val_str = f"{val:.4f}" if val is not None else "n/a"
        print(f"epoch {epoch} avg_loss {avg:.4f} val_loss {val_str} lr {lr_now:.2e} ({dt:.1f}s)")
        writer.writerow([epoch, f"{avg:.5f}", f"{val:.5f}" if val is not None else "",
                         f"{lr_now:.3e}", f"{dt:.1f}"])
        metrics.flush()

        # select the best checkpoint by val loss when available, else train loss
        score = val if val is not None else avg

        def _ckpt(path, epoch=epoch, best=best, gstep=gstep):
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict() if ema else None,
                        "opt": opt.state_dict(), "gstep": gstep, "best": best,
                        "scaler": scaler.state_dict() if use_scaler else None,
                        "args": vars(args), "epoch": epoch,
                        "sig_channels": N_SIGNAL_CHANNELS,
                        "git_commit": config["git_commit"]}, path)

        _ckpt(ckpt_dir / "last.pt")
        if score < best:
            best = score
            _ckpt(ckpt_dir / "best.pt", best=best)
        if (epoch + 1) % args.save_every == 0:
            _ckpt(ckpt_dir / f"epoch_{epoch + 1}.pt")
    metrics.close()
    print(f"done. best score {best:.4f}  ->  {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/std-v1")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--crop", type=int, default=3072)
    # base 128 is the proven-stable size; base 160 + bf16 diverged twice (v2 @e21,
    # v3 @e12) even with QK-norm. Keep LR/clip conservative.
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--attn", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--attn-levels", type=int, default=2,
                    help="apply self-attention at the N deepest U-Net levels "
                         "(2=default; 3 gives finer-resolution pattern context)")
    ap.add_argument("--adaln", type=lambda s: s.lower() != "false", default=True,
                    help="adaLN-zero conditioning (v6 default; 'false' = additive FiLM)")
    ap.add_argument("--rope", action="store_true",
                    help="rotary position embeddings in attention (relative-time; free params)")
    ap.add_argument("--up-attn", action="store_true",
                    help="add symmetric attention on the up path (more attention; audit S-5)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient-checkpoint blocks to fit finer attention / bigger nets in VRAM")
    ap.add_argument("--augment", type=lambda s: s.lower() != "false", default=True,
                    help="playfield h/v flip augmentation (default on; 'false' to disable)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the training forward (needs triton + a C "
                         "compiler; on Windows: triton-windows + MSVC Build Tools)")
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--cfg-drop", type=float, default=0.15,
                    help="prob. of dropping difficulty context (classifier-free guidance)")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--objective", choices=["eps", "v"], default="eps",
                    help="prediction target: eps (v1-v6) or v (velocity, v7 sharpness fix)")
    ap.add_argument("--zero-snr", action="store_true",
                    help="rescale schedule to zero terminal SNR (requires --objective v)")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse",
                    help="pointwise distance on the diffusion target (huber = robust/sharper)")
    ap.add_argument("--huber-beta", type=float, default=1.0, help="Huber transition point")
    ap.add_argument("--min-snr-gamma", type=float, default=0.0,
                    help="Min-SNR-gamma loss weighting (0 disables; ~5 typical)")
    ap.add_argument("--spatial-loss-weight", type=float, default=1.0,
                    help="up-weight the spatial channels (cursor/anchors/spacing) in the "
                         "loss so patterns aren't hedged to the mean (1.0=off; ~3 typical)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-objects", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="fraction of difficulties held out for validation (0 disables)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", default=None,
                    help="resume an interrupted run from a checkpoint (e.g. "
                         "runs/<id>/ckpt/last.pt); continues in the same run dir")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
