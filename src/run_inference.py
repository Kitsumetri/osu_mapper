#!/usr/bin/env python
"""Generate an osu!standard beatmap from an audio file — then print its stats and drop
it straight into your osu! Songs folder so it's playable in-game. One friendly command.

EXAMPLES
  # 5-star map for a song you already have mapped (uses the existing map for timing + art)
  uv run python main.py infer \
      --audio "C:/osu!/Songs/123 Artist - Song/audio.mp3" \
      --reference "C:/osu!/Songs/123 Artist - Song/Artist - Song (Mapper) [Insane].osu" \
      --sr 5

  # several difficulties at once
  uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 4 5 6

  # RECOMMENDED: reward-ranked best-of-8 (sample N, keep the highest-reward winner per SR)
  uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 5 6 7 --best-of-n 8

  # jumpier output (low aim = bigger spacing); raise --aim-intensity for denser/streamier
  uv run python main.py infer --audio song.mp3 --reference ref.osu --sr 6 --aim-intensity 0.1

  # pick a checkpoint, just write the .osu (don't copy into Songs)
  uv run python main.py infer --audio song.mp3 --sr 5 --ckpt runs/<id>/ckpt/best.pt --no-package

TIPS
  * --reference is an existing .osu for the same song. It gives exact BPM/offset (much better
    timing than auto-estimate) AND lets the map be packaged with the right audio/art. Optional,
    but recommended — without it the map is written to --out-dir and you place it yourself.
  * Star rating is conditioned, not exact; pass --match-sr to iterate toward the target (slower).
  * Long songs (>4 min) automatically use bf16 + the low-memory path so they don't run out of VRAM.
  * --best-of-n N samples N candidates per SR, scores each with the reward function, and keeps
    the winner. Requires artifacts/reference_stats.json (build with corpus_stats). The winner
    flows through the same packaging path as a normal single generation.
  * --aim-intensity 0..1 is the v9 per-song density/spacing dial (default: inferred from the
    audio). LOW (~0.1) = sparser + bigger spacing (jumpier); HIGH (~0.9) = denser + streamier.
    It trades note-density against spacing rather than forcing jumps outright.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _find_latest_ckpt() -> str | None:
    cks = sorted(Path("runs").glob("*/ckpt/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(cks[-1]) if cks else None


def _bar(title: str = "") -> None:
    print("=" * 64 if not title else f"  {title}")


def _print_stats(osu_path: Path, target_sr: float, gen_secs: float) -> float | None:
    from src.difficulty import star_rating
    from src.metrics import compute_metrics_for_osu

    m = compute_metrics_for_osu(osu_path)
    sr = star_rating(osu_path)
    sr_str = f"{sr:.2f}*" if sr is not None else "n/a"

    def pct(k):
        return f"{m.get(k, 0) * 100:.0f}%"

    _bar()
    print(f"  STATS   {osu_path.name}")
    _bar()
    print(f"  star rating   target {target_sr:.1f}*   ->   got {sr_str}")
    print(f"  objects       {m['n_objects']} over {_fmt_dur(m['duration_s'])}  "
          f"({m['density_per_s']:.1f}/sec)")
    print(f"  rhythm        BPM {m['bpm']:.0f}   on-1/4-grid {pct('on_quarter_grid_ratio')}")
    print(f"  spacing       mean {m['mean_spacing_px']:.0f}px (+/-{m['std_spacing_px']:.0f})   "
          f"jumps {pct('jump_ratio')}   streams {pct('stream_ratio')}")
    print(f"  sliders       {pct('slider_ratio')} sliders, {pct('curved_slider_ratio')} curved, "
          f"{m['sv_changes_per_min']:.1f} SV/min")
    print(f"  accents       hitsounds {pct('hitsound_ratio')}   kiai {pct('kiai_ratio')}")
    print(f"  generated in  {gen_secs:.0f}s")
    _bar()
    return sr


def _print_bon_summary(sr: float, win_reward: float, all_rewards: list[float],
                       elapsed: float) -> None:
    """Print the per-SR best/mean/lift line after a best-of-N run."""
    mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    lift = win_reward - mean
    print(f"  [bon] SR {sr:g}: best R={win_reward:.4f} / mean R={mean:.4f} "
          f"(lift {lift:+.4f}) in {elapsed:.0f}s  n={len(all_rewards)}")


def _load_ref_stats_for_infer(path: str) -> dict:
    """Load reference_stats.json, raising a user-friendly error if missing."""
    import json
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"ERROR: --best-of-n requires reward reference stats, "
            f"but '{path}' is missing.\n"
            f"Build them first:\n"
            f'  uv run python -m src.corpus_stats --songs "C:/osu!/Songs" --out {path}')
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate, analyse and package an osu! map from audio.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="path to the song audio (mp3/ogg/wav)")
    ap.add_argument("--sr", type=float, nargs="+", required=True, metavar="STARS",
                    help="target star rating(s), e.g. --sr 5  or  --sr 4 5 6")
    ap.add_argument("--ckpt", default=None,
                    help="model checkpoint (default: newest runs/*/ckpt/best.pt)")
    ap.add_argument("--reference", default=None,
                    help="an existing .osu for this song: exact timing + enables packaging")
    ap.add_argument("--out-dir", default="artifacts/generated", help="where to write the .osu")
    ap.add_argument("--prefix", default="[AI]", help="label for the packaged difficulty/folder")
    ap.add_argument("--songs", default="C:/osu!/Songs", help="your osu! Songs folder")
    # generation knobs (sensible defaults from play-testing)
    ap.add_argument("--spacing-scale", type=float, default=0.0,
                    help="amplify jump spacing (0=off, the default; >1 hurt play-feel in testing)")
    ap.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance strength")
    ap.add_argument("--steps", type=int, default=100, help="DDIM sampling steps (fewer = faster)")
    ap.add_argument("--density", type=float, default=None,
                    help="override objects/sec (raise to push streams on a busy song)")
    ap.add_argument("--aim-intensity", type=float, default=None, metavar="0..1",
                    help="v9 per-song density/spacing dial (default: inferred from the "
                         "audio). LOW (~0.1) = sparser, bigger spacing / jumpier; HIGH "
                         "(~0.9) = denser, more streams. (Trades density vs spacing; it is "
                         "NOT a pure 'more jumps' knob.)")
    ap.add_argument("--match-sr", action="store_true",
                    help="iterate to actually hit the target star rating (slower)")
    ap.add_argument("--no-package", action="store_true",
                    help="just write the .osu; don't copy into your Songs folder")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (faster over many SRs; needs a C compiler)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="bf16 sampling (faster, lower memory). Default: auto-on for long songs")
    ap.add_argument("--batch-cfg", action=argparse.BooleanOptionalAction, default=None,
                    help="batched guidance (~2x faster, +memory); default auto-off for long songs")
    # best-of-N flags
    ap.add_argument("--best-of-n", type=int, default=1, metavar="N",
                    help="sample N candidates per SR, keep the highest-reward winner "
                         "(default 1 = single-sample, unchanged behaviour). "
                         "Requires --ref-stats / artifacts/reference_stats.json.")
    ap.add_argument("--ref-stats", default="artifacts/reference_stats.json",
                    metavar="PATH",
                    help="reference_stats.json for the reward function "
                         "(only needed with --best-of-n N>1; "
                         "build with: uv run python -m src.corpus_stats)")
    ap.add_argument("--bon-seed", type=int, default=0,
                    help="RNG seed for best-of-N candidate sampling (default 0)")
    ap.add_argument("--keep-candidates", action="store_true",
                    help="keep all N candidate .osu files on disk (default: only the winner)")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"ERROR: audio not found: {audio}", file=sys.stderr)
        return 1
    ckpt = args.ckpt or _find_latest_ckpt()
    if not ckpt or not Path(ckpt).exists():
        print("ERROR: no checkpoint. Pass --ckpt runs/<id>/ckpt/best.pt "
              "(none found under runs/).", file=sys.stderr)
        return 1
    reference = Path(args.reference) if args.reference else None
    if reference and not reference.exists():
        print(f"WARNING: --reference not found, falling back to estimated timing: {reference}")
        reference = None
    if args.no_package:
        do_package = False
    elif reference is None:
        print("NOTE: no --reference, so the map can't be auto-packaged; writing the .osu only.")
        do_package = False
    else:
        do_package = True

    use_bon = args.best_of_n > 1

    # load ref_stats early (fail fast before any expensive model load)
    ref_stats: dict | None = None
    if use_bon:
        ref_stats = _load_ref_stats_for_infer(args.ref_stats)
        print(f"reward calibrated on {ref_stats.get('n_maps')} ranked maps ({args.ref_stats})")

    # heavy imports after arg-validation so --help / bad paths fail fast
    from src.config import AUDIO
    from src.generate import generate, load_model, prepare_audio

    print(f"\nloading model: {ckpt}")
    loaded = load_model(ckpt, compile_model=args.compile)
    ref_str = str(reference) if reference else None
    prepared = prepare_audio(str(audio), loaded.device, timing_ref=ref_str)
    duration_s = prepared.t_len * AUDIO.ms_per_frame / 1000.0

    # long songs materialise a big attention matrix -> auto-use bf16 (fixes OOM) + the
    # low-memory two-forward guidance, unless the user said otherwise.
    long_song = duration_s > 240
    amp = long_song if args.amp is None else args.amp
    batch_cfg = (not long_song) if args.batch_cfg is None else args.batch_cfg
    print(f"song: {_fmt_dur(duration_s)}  |  amp(bf16)={amp}  batched-guidance={batch_cfg}"
          + ("   [long song -> memory-safe mode]" if long_song and args.amp is None else ""))

    if use_bon:
        print(f"best-of-{args.best_of_n} mode: sampling {args.best_of_n} candidates per SR "
              f"(seed={args.bon_seed}), keeping highest-reward winner")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated, diff_names = [], []

    # shared generation kwargs (passed through to generate / best_of_n)
    gen_kwargs = dict(
        guidance=args.guidance,
        steps=args.steps,
        match_sr=args.match_sr,
        density=args.density,
        aim_override=args.aim_intensity,   # v9 per-song aim dial (None -> audio-derived)
        spacing_scale=args.spacing_scale,
        amp=amp,
        batch_cfg=batch_cfg,
    )

    bon_summary: list[tuple[float, float, float]] = []  # (sr, best_r, mean_r) per SR

    for sr in args.sr:
        _bar()
        print(f"  GENERATING  {sr:.1f}*  ({audio.stem})"
              + (f"  [best-of-{args.best_of_n}]" if use_bon else ""))
        _bar()
        out_path = out_dir / f"{audio.stem}_sr{sr:g}.osu"
        t0 = time.time()

        if use_bon:
            from src.best_of_n import best_of_n
            _, win_bd, all_bds = best_of_n(
                str(audio), sr=sr, ref_stats=ref_stats,
                out_path=str(out_path),
                n=args.best_of_n, seed=args.bon_seed,
                loaded=loaded, prepared=prepared,
                keep_candidates=args.keep_candidates,
                **gen_kwargs,
            )
            elapsed = time.time() - t0
            rewards = [b.reward for b in all_bds]
            _print_bon_summary(sr, win_bd.reward, rewards, elapsed)
            bon_summary.append((sr, win_bd.reward, sum(rewards) / len(rewards)))
        else:
            generate(str(audio), out_path=str(out_path), sr=sr,
                     loaded=loaded, prepared=prepared, **gen_kwargs)
            elapsed = time.time() - t0

        _print_stats(out_path, sr, elapsed)
        generated.append(out_path)
        diff_names.append(f"AI {sr:g}star")

    if bon_summary:
        _bar()
        print(f"  BEST-OF-{args.best_of_n} SUMMARY")
        _bar()
        for sr, best_r, mean_r in bon_summary:
            lift = best_r - mean_r
            print(f"  SR {sr:g}*   best R={best_r:.4f} / mean R={mean_r:.4f} "
                  f"(lift {lift:+.4f})")

    _bar()
    if do_package:
        # all difficulties go into ONE beatmapset folder (shared audio), the osu! way
        from src.package_map import package_set
        folder = package_set(generated, reference, Path(args.songs),
                             set_prefix=args.prefix, diff_names=diff_names)
        print(f"  DONE - {len(generated)} difficulty(ies) in ONE folder:")
        print(f"    {folder}")
        print("  Open osu! and press F5 (or restart) to see them under one song.")
    else:
        print(f"  DONE - wrote {len(generated)} .osu file(s) to {out_dir}/")
        print("  (pass --reference <existing.osu> to auto-package them into osu!.)")
    _bar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
