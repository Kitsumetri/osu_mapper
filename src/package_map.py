"""Package a generated map into a playable osu! Songs folder.

Copies the original song's audio (and background) into a new, clearly-labelled
folder and writes the generated hit objects with metadata borrowed from the
original beatmap so it loads and is testable in-game.

  python -m src.package_map \
      --generated data/final_generated.osu \
      --original "C:/osu!/Songs/1000309 .../[Expert].osu" \
      --songs "C:/osu!/Songs" --prefix "[AI-GEN]"
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from .parsing.beatmap import Beatmap, parse_beatmap, write_osu


def _safe(name: str) -> str:
    """Make a string safe for a Windows folder/file name."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return name.strip().rstrip(".")[:120] or "track"


def _find_background(osu_path: Path) -> str | None:
    """Pull the background image filename from the original [Events] section."""
    text = osu_path.read_text(encoding="utf-8", errors="ignore")
    in_events = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_events = s == "[Events]"
            continue
        if in_events:
            # e.g.  0,0,"bg.jpg",0,0
            m = re.match(r'0,0,"([^"]+)"', s)
            if m:
                return m.group(1)
    return None


def _copy_assets(original: Path, orig: Beatmap, out_dir: Path) -> str | None:
    """Copy the original song's audio (and background image) into ``out_dir`` once.
    Returns the background filename if one was found + copied, else None."""
    src_audio = original.parent / orig.audio_filename
    if src_audio.exists():
        shutil.copy2(src_audio, out_dir / orig.audio_filename)
    else:
        print(f"WARNING: audio not found at {src_audio}")
    bg_name = _find_background(original)
    if bg_name:
        src_bg = original.parent / bg_name
        if src_bg.exists():
            shutil.copy2(src_bg, out_dir / bg_name)
        else:
            bg_name = None
    return bg_name


def _write_difficulty(out_dir: Path, artist: str, title: str, gen: Beatmap,
                      version: str, audio_name: str, bg_name: str | None) -> Path:
    """Write one generated map as a single difficulty (``[version]``) in ``out_dir``."""
    bm = Beatmap(path=out_dir / "generated.osu")
    bm.audio_filename = audio_name
    bm.title, bm.artist, bm.creator, bm.version = title, artist, "osu_mapper", version
    # KEEP the generated map's own difficulty settings — generate writes AR/OD/HP/CS to
    # match the conditioned target SR. Borrowing the original's would make an easy map
    # play at the original's (often harder) AR/OD and change its computed star rating.
    # SliderMultiplier is kept too (slider lengths are beat-snap-calibrated to it).
    bm.circle_size = gen.circle_size
    bm.approach_rate = gen.approach_rate
    bm.overall_difficulty = gen.overall_difficulty
    bm.hp = gen.hp
    bm.slider_multiplier = gen.slider_multiplier
    out_osu = out_dir / f"{_safe(artist + ' - ' + title)} (osu_mapper) [{_safe(version)}].osu"
    write_osu(bm, gen.hit_objects, out_osu, timing_points=gen.timing_points)
    if bg_name:  # inject the background event so the map shows art in-game
        text = out_osu.read_text(encoding="utf-8")
        out_osu.write_text(text.replace("[Events]\n", f'[Events]\n0,0,"{bg_name}",0,0\n'),
                           encoding="utf-8")
    return out_osu


def package_set(generated: list, original: Path, songs_dir: Path,
                set_prefix: str = "[AI]", diff_names: list | None = None) -> Path:
    """Package several generated maps as difficulties of ONE beatmapset — a single folder
    with one shared audio/background and one ``[Version]`` .osu per map (the osu!
    convention for a mapset, so all difficulties appear under the same song in-game)."""
    orig = parse_beatmap(original)
    artist = orig.artist or "Unknown"
    title = orig.title or Path(original).stem
    out_dir = songs_dir / _safe(f"{set_prefix} {artist} - {title}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_name = _copy_assets(Path(original), orig, out_dir)
    for i, gpath in enumerate(generated):
        gen = parse_beatmap(gpath)
        version = (diff_names[i] if diff_names and i < len(diff_names)
                   else f"{set_prefix} {i + 1}")
        _write_difficulty(out_dir, artist, title, gen, version, orig.audio_filename, bg_name)
        print(f"  + [{version}]  {len(gen.hit_objects)} objects  "
              f"CS{gen.circle_size} AR{gen.approach_rate}")
    print(f"packaged {len(generated)} difficulty(ies) -> {out_dir}"
          f"{'  (bg: ' + bg_name + ')' if bg_name else ''}")
    return out_dir


def package(generated: Path, original: Path, songs_dir: Path,
            prefix: str = "[AI-GEN]") -> Path:
    """Single-difficulty convenience wrapper around :func:`package_set`."""
    return package_set([generated], original, songs_dir, set_prefix=prefix,
                       diff_names=[f"{prefix} AI"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--songs", default=r"C:/osu!/Songs")
    ap.add_argument("--prefix", default="[AI-GEN]")
    args = ap.parse_args()
    package(Path(args.generated), Path(args.original), Path(args.songs), args.prefix)


if __name__ == "__main__":
    main()
