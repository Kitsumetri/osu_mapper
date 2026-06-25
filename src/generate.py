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
from .data.audio import aim_intensity, load_audio, log_mel
from .data.signal import decode_kiai, decode_signal, decode_spacing, decode_sv
from .data.timing import estimate_timing_point
from .model.diffusion import GaussianDiffusion
from .model.unet import UNet1d
from .parsing.beatmap import Beatmap, TimingPoint, parse_beatmap, write_osu
from .postprocess import (
    clamp_objects_to_playfield,
    clamp_slider_endpoints,
    compute_breaks,
    respace_by_magnitude,
    snap_slider_ends,
    snap_to_grid,
    trim_isolated_ends,
)

# a loaded denoiser + its sampler, ready to reuse across many generate() calls
LoadedModel = namedtuple("LoadedModel", "model diff ctx_dim device")
# prepared audio conditioning: mel batch, true len T, padded len, timing point, and
# the v9 per-song aim-intensity (computed from the same decoded audio as the mel).
# ``aim`` defaults to None so manual constructions (e.g. train.py's val-reward probe,
# which only has the cached mel, not raw audio) stay valid -> the model sees the 0.0
# baseline for that slot, exactly like old data missing the manifest field.
PreparedAudio = namedtuple("PreparedAudio", "cond t_len t_full tp aim")
PreparedAudio.__new__.__defaults__ = (None,)  # aim optional (back-compat)


def _active_sv(t_ms: float, sv_secs: list[tuple[int, float]]) -> float:
    """SV multiplier active at time t (last section starting at/before t, else 1.0)."""
    sv = 1.0
    for start, s in sv_secs:
        if start <= t_ms:
            sv = s
        else:
            break
    return sv


def _merge_green_lines(base_tp, kiai_spans, sv_secs):
    """Unify SV sections + kiai spans into inherited (green) timing points.

    osu! carries both SV and kiai on the same green line, so at each change in either
    we emit one line with the active SV (beat_length = -100/SV) and kiai flag (effects
    bit 0), de-duplicating unchanged states. Green lines are clamped to just after the
    red line. ``sv_secs=[]`` (pre-v7) reduces to the old kiai-only timing exactly.
    """
    events = sorted({a for a, _ in sv_secs} | {t for span in kiai_spans for t in span})
    timing = [base_tp]
    last = (-100.0, 1 if base_tp.kiai else 0)
    for t in events:
        gt = max(int(t), int(base_tp.time) + 1)
        sv = _active_sv(gt, sv_secs)
        kiai = any(a <= gt < b for a, b in kiai_spans)
        bl = round(-100.0 / max(sv, 1e-3), 4)
        eff = 1 if kiai else 0
        if (bl, eff) == last:
            continue
        timing.append(TimingPoint(gt, bl, base_tp.meter, False, effects=eff))
        last = (bl, eff)
    timing.sort(key=lambda x: x.time)
    return timing


def load_model(ckpt_path, base=64, use_ema=True, device=None,
               compile_model=False) -> LoadedModel:
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
    # Build the UNet with the CHECKPOINT's OWN ctx_dim so older models still load when
    # CONTEXT_DIM grows (v9 added the aim slot: 6 -> 7). A pre-v9 ckpt's ctx_mlp.0.weight
    # is [t_dim, 6]; building at 7 would crash load_state_dict. Read it from the saved
    # weights (authoritative), else cargs["ctx_dim"], else (conditioned ckpt) CONTEXT_DIM,
    # else 0 (unconditioned). The extra aim slot is simply unused by old ckpts.
    weights = ckpt["ema"] if (use_ema and ckpt.get("ema")) else ckpt["model"]
    if "ctx_mlp.0.weight" in weights:
        ctx_dim = weights["ctx_mlp.0.weight"].shape[1]
    elif "ctx_dim" in cargs:
        ctx_dim = cargs["ctx_dim"]
    elif "cfg_drop" in cargs:
        ctx_dim = CONTEXT_DIM
    else:
        ctx_dim = 0
    # build the model with the checkpoint's own channel count so older 17-ch (v5/v6/v7)
    # checkpoints still load under the v7 18-ch global (decode handles the missing SV).
    n_sig = ckpt.get("sig_channels", N_SIGNAL_CHANNELS)
    model = UNet1d(n_sig, AUDIO.n_mels, base=base, attn=attn,
                   ctx_dim=ctx_dim, attn_levels=attn_levels, adaln=adaln,
                   rope=cargs.get("rope", False), up_attn=cargs.get("up_attn", False)).to(device)
    model.load_state_dict(weights)
    model.eval()
    # torch.compile fuses kernels for the repeated DDIM forwards (compiles once on the
    # first call, then reused across all steps / an SR sweep). Opt-in: the ~30-60s compile
    # cost only pays off over many forwards (long songs, sweeps); it specialises on the
    # song length, so a different-length song recompiles. (`model.sig_channels` still
    # resolves through the OptimizedModule wrapper.)
    if compile_model and device == "cuda":
        model = torch.compile(model)
    diff = GaussianDiffusion(timesteps=cargs.get("timesteps", 1000), device=device,
                             objective=cargs.get("objective", "eps"),
                             zero_snr=cargs.get("zero_snr", False))
    return LoadedModel(model, diff, ctx_dim, device)


def prepare_audio(audio_path, device, timing_ref=None) -> PreparedAudio:
    """Decode audio once into the mel conditioning + a timing point.

    Reuse across generate() calls for the same song (e.g. an SR sweep) to avoid
    re-decoding + re-running beat estimation each time.
    """
    y = load_audio(audio_path)          # decode once, reuse for mel + timing + aim
    mel = log_mel(y)                    # (n_mels, T)
    # v9 per-song aim-intensity from the SAME decoded array preprocess uses (parity).
    aim = aim_intensity(y)
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
    return PreparedAudio(cond, t_len, mel_p.shape[1], tp, aim)


def generate(audio_path, ckpt_path=None, out_path="generated.osu", steps=100, base=64,
             use_ema=True, snap=True, sr=None, guidance=2.0, match_sr=False, max_iter=3,
             tol=0.4, snap_divisors=(4, 8, 6), timing_ref=None, loaded=None, prepared=None,
             guidance_rescale=0.0, density=None, onset_threshold=0.3, spacing_scale=0.0,
             compile_model=False, batch_cfg=True, amp=False, aim_override=None):
    if loaded is None:
        loaded = load_model(ckpt_path, base=base, use_ema=use_ema, compile_model=compile_model)
    model, diff, ctx_dim, device = loaded
    if prepared is None:
        prepared = prepare_audio(audio_path, device, timing_ref)
    cond, T, t_full, tp, aim_audio = prepared
    # v9 per-song aim-intensity fed into the context like --density: default is the
    # value computed from THIS song's audio (prepare_audio); --aim-intensity overrides
    # it to push (higher = jumpier) or dampen. None on both -> 0.0 baseline (old ckpts
    # ignore the slot anyway; see load_model's ctx_dim from the checkpoint).
    aim_eff = aim_override if aim_override is not None else aim_audio

    def _one_pass(sr_used):
        ctx = None
        if ctx_dim and sr_used is not None:
            # density override conditions the model denser/sparser than the SR default
            # (e.g. to push streams on a stream-heavy song); aim_eff supplies the
            # per-song aim-intensity (audio-derived, overridable).
            ctx = torch.tensor([target_context(sr_used, density=density,
                                                aim_intensity=aim_eff)],
                               dtype=torch.float32, device=device)
            # back-compat: target_context emits the full CONTEXT_DIM (v9 = 7, incl. the
            # appended `aim` slot), but an OLDER checkpoint's ctx_mlp expects its own
            # narrower width (ctx_dim, read from the ckpt by load_model — 6 pre-v9). The
            # new fields are appended LAST, so truncating to ctx_dim drops exactly the
            # slot(s) that model never knew (aim) and rebuilds its training-time vector.
            # (load_model builds the UNet at ctx_dim, so without this the (1,7) ctx hits a
            # Linear(6,*) -> "2x7 and 6x256" matmul error on v8/v8_1.)
            ctx = ctx[:, :ctx_dim]
        sig = diff.ddim_sample(model, cond, (1, model.sig_channels, t_full),
                               steps=steps, ctx=ctx, guidance=guidance,
                               guidance_rescale=guidance_rescale, progress=True,
                               batch_cfg=batch_cfg, amp=amp)
        sig = sig[0, :, :T].float().cpu().numpy()
        objects = decode_signal(sig, onset_threshold=onset_threshold)
        trim_isolated_ends(objects)
        # v8: rebuild positions to the spacing channel's per-gap magnitudes (keep the
        # model's direction, set the step length from the predicted magnitude) — breaks
        # the jump under-dispersion ceiling (RESEARCH 10.11). Off (0.0) for pre-v8 ckpts;
        # decode_spacing returns [] for an absent/untrained channel, so respacing is
        # skipped safely. Runs before snap/clamp so the slider-body clamp catches it.
        if spacing_scale > 0:
            mags = decode_spacing(sig, objects)
            if mags:
                respace_by_magnitude(objects, mags, alpha=spacing_scale)
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
        # SV + kiai are both carried by green (inherited) lines, so merge them: each
        # line sets the active SV (beat_length=-100/SV) and kiai flag. decode_sv is []
        # for pre-v7 ckpts -> SV=1 everywhere -> identical to the old kiai-only timing.
        # Built BEFORE snapping so slider-end snapping is SV-aware (else SV shifts ends
        # off the grid — every slider's duration = length/SV).
        timing = _merge_green_lines(tp, decode_kiai(sig), decode_sv(sig))
        if snap:
            # snap onsets to the nearest of 1/4, 1/8, 1/6 — the model places notes
            # on the 1/8 and 1/6 grids too, so a 1/4-only snap drags those onto the
            # wrong 1/4 line and wrecks the rhythm (play feedback).
            snap_to_grid(objects, tp, divisors=snap_divisors)
            snap_slider_ends(objects, timing, bm.slider_multiplier)  # SV-aware
        clamp_slider_endpoints(objects)
        clamp_objects_to_playfield(objects)  # final guard: all heads inside playfield
        breaks = compute_breaks(objects)
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
    ap.add_argument("--guidance-rescale", type=float, default=0.0,
                    help="rescale guided x0 toward conditional std (0-1; for v/zero-SNR ckpts)")
    ap.add_argument("--density", type=float, default=None,
                    help="override conditioned objects/sec (default SR-based; raise for streams)")
    ap.add_argument("--aim-intensity", type=float, default=None,
                    help="v9: override the per-song aim-intensity in [0,1] (default: "
                         "computed from the audio; raise to push jumps on a jumpy song, "
                         "lower to dampen). Only affects v9+ ckpts trained with the channel")
    ap.add_argument("--onset-threshold", type=float, default=0.3,
                    help="decode onset peak threshold (lower = more notes / denser streams)")
    ap.add_argument("--timing-from", default=None,
                    help="read exact BPM+offset from this reference .osu instead of "
                         "estimating (use when the song already has a known map)")
    ap.add_argument("--match-sr", action="store_true",
                    help="iterate to hit the requested star rating (corrects SR offset)")
    ap.add_argument("--match-iter", type=int, default=3,
                    help="max --match-sr iterations (raise for high/noisy SR targets)")
    ap.add_argument("--no-snap", action="store_true", help="disable beat-snapping")
    ap.add_argument("--spacing-scale", type=float, default=0.0,
                    help="v8: rebuild positions to the spacing channel (0=off; ~1.0 for "
                         "v8 ckpts; lower = gentler, raise >1 for more extreme jumps)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model for faster sampling (compiles once; "
                         "worth it for long songs / SR sweeps, needs triton + a C compiler)")
    ap.add_argument("--no-batch-cfg", action="store_true",
                    help="disable batched CFG (use the low-memory two-forward path; "
                         "needed for marathon-length songs that OOM the batch-2 forward)")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast sampling: enables flash-attention (O(T) memory, "
                         "fixes long-song OOM) + ~2x faster (matches the training regime)")
    args = ap.parse_args()
    generate(args.audio, args.ckpt, args.out, steps=args.steps, snap=not args.no_snap,
             sr=args.sr, guidance=args.guidance, match_sr=args.match_sr,
             max_iter=args.match_iter, timing_ref=args.timing_from,
             guidance_rescale=args.guidance_rescale, density=args.density,
             onset_threshold=args.onset_threshold, spacing_scale=args.spacing_scale,
             compile_model=args.compile, batch_cfg=not args.no_batch_cfg, amp=args.amp,
             aim_override=args.aim_intensity)


if __name__ == "__main__":
    main()
