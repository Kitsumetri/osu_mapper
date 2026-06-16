"""Generate a playable .osu file from an audio file using a trained model.

python -m src.generate --audio song.mp3 --ckpt checkpoints/model_last.pt --out out.osu
"""

from __future__ import annotations

import argparse
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from .conditioning import CONTEXT_DIM, target_context, target_settings
from .config import AUDIO, N_SIGNAL_CHANNELS
from .data.audio import load_audio, log_mel
from .data.signal import decode_kiai, decode_signal
from .data.timing import estimate_timing_point
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d
from .parsing.beatmap import Beatmap, TimingPoint, parse_beatmap, write_osu
from .postprocess import (
    clamp_slider_endpoints,
    compute_breaks,
    snap_slider_ends,
    snap_to_grid,
    trim_isolated_ends,
)

# a loaded denoiser + its sampler, ready to reuse across many generate() calls
LoadedModel = namedtuple("LoadedModel", "model diff ctx_dim device")
# prepared audio conditioning: mel batch, true len T, padded len, timing point
PreparedAudio = namedtuple("PreparedAudio", "cond t_len t_full tp")


def load_model(ckpt_path, base=64, use_ema=True, device=None) -> LoadedModel:
    """Load a checkpoint into an eval-ready U-Net + diffusion sampler.

    Reuse the result across generate() calls (e.g. an SR sweep) to avoid
    reloading the ~1 GB checkpoint each time.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cargs = ckpt.get("args", {})
    base = cargs.get("base", base)
    attn = cargs.get("attn", False)        # old checkpoints had no attention
    attn_levels = cargs.get("attn_levels", 2)  # old ckpts: 2 deepest levels
    adaln = cargs.get("adaln", False)          # pre-v6 ckpts used additive FiLM
    ctx_dim = CONTEXT_DIM if ("cfg_drop" in cargs or "ctx_dim" in cargs) else 0
    model = UNet1d(N_SIGNAL_CHANNELS, AUDIO.n_mels, base=base, attn=attn,
                   ctx_dim=ctx_dim, attn_levels=attn_levels, adaln=adaln).to(device)
    weights = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()
    diff = GaussianDiffusion(timesteps=cargs.get("timesteps", 1000), device=device)
    return LoadedModel(model, diff, ctx_dim, device)


def prepare_audio(audio_path, device, timing_ref=None) -> PreparedAudio:
    """Decode audio once into the mel conditioning + a timing point.

    Reuse across generate() calls for the same song (e.g. an SR sweep) to avoid
    re-decoding + re-running beat estimation each time.
    """
    y = load_audio(audio_path)          # decode once, reuse for mel + timing
    mel = log_mel(y)                    # (n_mels, T)
    t_len = mel.shape[1]
    pad = (-t_len) % 16                # multiple of 16 for clean U-Net striding
    mel_p = np.pad(mel, ((0, 0), (0, pad)))
    cond = torch.from_numpy(mel_p[None].astype(np.float32)).to(device)
    # timing: use a reference map's exact BPM+offset if given (the librosa estimate
    # is only ~28% exact and the offset may be off a beat), else estimate.
    tp = None
    if timing_ref:
        rb = parse_beatmap(timing_ref)
        ref = next((t for t in rb.timing_points if t.uninherited and t.beat_length > 0), None)
        if ref:
            tp = TimingPoint(ref.time, ref.beat_length, ref.meter, True)
            print(f"reference timing: {60000.0 / tp.beat_length:.1f} BPM, offset {tp.time} ms")
    if tp is None:
        tp = estimate_timing_point(y)
        print(f"estimated timing: {60000.0 / tp.beat_length:.1f} BPM, offset {tp.time} ms")
    return PreparedAudio(cond, t_len, mel_p.shape[1], tp)


def generate(audio_path, ckpt_path=None, out_path="generated.osu", steps=100, base=64,
             use_ema=True, snap=True, sr=None, guidance=2.0, match_sr=False, max_iter=3,
             tol=0.4, snap_divisors=(4, 8, 6), timing_ref=None, loaded=None, prepared=None):
    if loaded is None:
        loaded = load_model(ckpt_path, base=base, use_ema=use_ema)
    model, diff, ctx_dim, device = loaded
    if prepared is None:
        prepared = prepare_audio(audio_path, device, timing_ref)
    cond, T, t_full, tp = prepared

    def _one_pass(sr_used):
        ctx = None
        if ctx_dim and sr_used is not None:
            ctx = torch.tensor([target_context(sr_used)], dtype=torch.float32, device=device)
        sig = diff.ddim_sample(model, cond, (1, N_SIGNAL_CHANNELS, t_full),
                               steps=steps, ctx=ctx, guidance=guidance)
        sig = sig[0, :, :T].float().cpu().numpy()
        objects = decode_signal(sig)
        trim_isolated_ends(objects)
        bm = Beatmap(path=Path(out_path))
        bm.audio_filename = Path(audio_path).name
        bm.title = Path(audio_path).stem
        bm.version = f"AI {sr:.1f} star" if sr else "AI Generated"
        # write AR/OD/HP/CS consistent with what the model was conditioned on
        # (target_settings), so the file's difficulty matches the generated map and
        # the rosu SR read-back is scored on the right settings. Fall back to a
        # mid-difficulty default when generating unconditioned.
        if ctx_dim and sr_used is not None:
            s = target_settings(sr_used)
            bm.approach_rate, bm.overall_difficulty = s["ar"], s["od"]
            bm.hp, bm.circle_size = s["hp"], s["cs"]
        else:
            bm.approach_rate, bm.overall_difficulty, bm.hp, bm.circle_size = 8.0, 7.0, 5.0, 4.0
        if snap:
            # snap onsets to the nearest of 1/4, 1/8, 1/6 — the model places notes
            # on the 1/8 and 1/6 grids too, so a 1/4-only snap drags those onto the
            # wrong 1/4 line and wrecks the rhythm (play feedback).
            snap_to_grid(objects, tp, divisors=snap_divisors)
            snap_slider_ends(objects, tp, bm.slider_multiplier)  # snap slider ends (SV=1)
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
        # SR is stochastic per sample, so keep the *closest* pass, not the last.
        import shutil

        from .difficulty import star_rating
        best_path = str(out_path) + ".best"
        best_objs, best_err = None, float("inf")
        cur = sr
        for it in range(max_iter):
            objs = _one_pass(cur)
            achieved = star_rating(out_path)
            print(f"  [match-sr] iter {it}: ctx={cur:.2f} -> achieved {achieved}")
            if achieved is None:
                break
            err = abs(achieved - sr)
            if err < best_err:
                best_err, best_objs = err, objs
                shutil.copyfile(out_path, best_path)
            if err <= tol:
                break
            cur = float(min(11.0, max(1.0, cur + 0.8 * (sr - achieved))))
        if best_objs is not None:
            shutil.move(best_path, out_path)
            objs = best_objs
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
    ap.add_argument("--timing-from", default=None,
                    help="read exact BPM+offset from this reference .osu instead of "
                         "estimating (use when the song already has a known map)")
    ap.add_argument("--match-sr", action="store_true",
                    help="iterate to hit the requested star rating (corrects SR offset)")
    ap.add_argument("--match-iter", type=int, default=3,
                    help="max --match-sr iterations (raise for high/noisy SR targets)")
    ap.add_argument("--no-snap", action="store_true", help="disable beat-snapping")
    args = ap.parse_args()
    generate(args.audio, args.ckpt, args.out, steps=args.steps, snap=not args.no_snap,
             sr=args.sr, guidance=args.guidance, match_sr=args.match_sr,
             max_iter=args.match_iter, timing_ref=args.timing_from)


if __name__ == "__main__":
    main()
