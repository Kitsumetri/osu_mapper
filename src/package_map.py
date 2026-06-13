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


def package(generated: Path, original: Path, songs_dir: Path,
            prefix: str = "[AI-GEN]") -> Path:
    gen = parse_beatmap(generated)
    orig = parse_beatmap(original)

    artist = orig.artist or "Unknown"
    title = orig.title or original.stem
    folder_name = _safe(f"{prefix} {artist} - {title}")
    out_dir = songs_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy audio
    src_audio = original.parent / orig.audio_filename
    audio_name = orig.audio_filename
    if src_audio.exists():
        shutil.copy2(src_audio, out_dir / audio_name)
    else:
        print(f"WARNING: audio not found at {src_audio}")

    # copy background if present
    bg_name = _find_background(original)
    if bg_name:
        src_bg = original.parent / bg_name
        if src_bg.exists():
            shutil.copy2(src_bg, out_dir / bg_name)
        else:
            bg_name = None

    # build the packaged beatmap with metadata from the original
    bm = Beatmap(path=out_dir / "generated.osu")
    bm.audio_filename = audio_name
    bm.title = title
    bm.artist = artist
    bm.creator = "osu_mapper"
    bm.version = f"{prefix} AI Generated"
    # borrow difficulty settings from the original for sane gameplay values
    bm.circle_size = orig.circle_size
    bm.approach_rate = orig.approach_rate
    bm.overall_difficulty = orig.overall_difficulty
    bm.hp = orig.hp
    bm.slider_multiplier = orig.slider_multiplier

    out_osu = out_dir / f"{_safe(artist + ' - ' + title)} ({bm.creator}) [{prefix} AI].osu"
    write_osu(bm, gen.hit_objects, out_osu, timing_points=gen.timing_points)

    # inject the background event so the map shows art in-game
    if bg_name:
        text = out_osu.read_text(encoding="utf-8")
        text = text.replace("[Events]\n", f'[Events]\n0,0,"{bg_name}",0,0\n')
        out_osu.write_text(text, encoding="utf-8")

    print(f"packaged -> {out_dir}")
    print(f"  audio: {audio_name}{'  bg: ' + bg_name if bg_name else ''}")
    print(f"  objects: {len(gen.hit_objects)}  |  diff CS{bm.circle_size} AR{bm.approach_rate}")
    return out_dir


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
