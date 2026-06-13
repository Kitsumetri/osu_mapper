"""Robust osu! beatmap parser/writer focused on the osu!standard mode.

Compared to the original prototype this module:
  * Correctly decodes the hit-object ``type`` bitfield (circle/slider/spinner,
    new-combo flag).
  * Parses slider curve type / control points / slides / pixel length and
    computes slider *duration* from the timing points + SliderMultiplier.
  * Fixes the ``uninherited`` boolean bug (the prototype used ``bool(str)``
    which is always True).
  * Can serialise a list of hit objects back into a valid ``.osu`` file.

Only the fields needed for the ML pipeline are kept; everything else is parsed
loosely so malformed community maps don't crash the crawler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import bisect

# --- hit object type bitflags -------------------------------------------------
TYPE_CIRCLE = 1 << 0      # 1
TYPE_SLIDER = 1 << 1      # 2
TYPE_NEW_COMBO = 1 << 2   # 4
TYPE_SPINNER = 1 << 3     # 8
TYPE_MANIA_HOLD = 1 << 7  # 128

PLAYFIELD_W = 512
PLAYFIELD_H = 384


@dataclass
class TimingPoint:
    time: float
    beat_length: float       # ms/beat (uninherited) or -100/SV (inherited)
    meter: int
    uninherited: bool

    @property
    def sv(self) -> float:
        """Slider-velocity multiplier for an *inherited* point."""
        if self.uninherited:
            return 1.0
        # inherited: beat_length is a negative inverse SV percentage
        return -100.0 / self.beat_length if self.beat_length < 0 else 1.0


@dataclass
class HitObject:
    x: int
    y: int
    time: int                       # start time, ms
    type: int                       # raw bitfield
    hit_sound: int = 0
    # slider-specific
    curve_type: Optional[str] = None
    curve_points: List[Tuple[int, int]] = field(default_factory=list)
    slides: int = 1
    length: float = 0.0             # pixel length
    end_time: int = 0               # computed for sliders/spinners

    @property
    def is_circle(self) -> bool:
        return bool(self.type & TYPE_CIRCLE)

    @property
    def is_slider(self) -> bool:
        return bool(self.type & TYPE_SLIDER)

    @property
    def is_spinner(self) -> bool:
        return bool(self.type & TYPE_SPINNER)

    @property
    def is_new_combo(self) -> bool:
        return bool(self.type & TYPE_NEW_COMBO)


@dataclass
class Beatmap:
    path: Path
    audio_filename: str = "audio.mp3"
    mode: int = 0
    title: str = ""
    artist: str = ""
    creator: str = ""
    version: str = ""               # difficulty name
    slider_multiplier: float = 1.4
    circle_size: float = 5.0
    overall_difficulty: float = 5.0
    approach_rate: float = 5.0
    hp: float = 5.0
    timing_points: List[TimingPoint] = field(default_factory=list)
    hit_objects: List[HitObject] = field(default_factory=list)

    @property
    def audio_path(self) -> Path:
        return self.path.parent / self.audio_filename

    # --- slider timing helpers ------------------------------------------------
    def _uninherited_at(self, time: float) -> TimingPoint:
        """Most recent uninherited (BPM) timing point at/before ``time``."""
        chosen = None
        for tp in self.timing_points:
            if tp.uninherited and tp.time <= time:
                chosen = tp
            elif tp.time > time:
                break
        if chosen is None:
            # fall back to first uninherited or a sane default
            for tp in self.timing_points:
                if tp.uninherited:
                    return tp
            return TimingPoint(0, 500.0, 4, True)
        return chosen

    def _sv_at(self, time: float) -> float:
        sv = 1.0
        for tp in self.timing_points:
            if tp.time > time:
                break
            if not tp.uninherited:
                sv = tp.sv
            else:
                sv = 1.0  # uninherited resets SV
        return sv

    def slider_duration(self, obj: HitObject) -> float:
        """Duration in ms of one slider (all slides)."""
        beat_len = self._uninherited_at(obj.time).beat_length
        sv = self._sv_at(obj.time)
        velocity = self.slider_multiplier * 100.0 * sv  # px per beat
        if velocity <= 0:
            return 0.0
        beats = obj.length / velocity
        return beats * beat_len * obj.slides


def _section_iter(text: str):
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        yield section, line


def parse_beatmap(path: str | Path) -> Beatmap:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    bm = Beatmap(path=path)

    for section, line in _section_iter(text):
        if section in ("General", "Metadata", "Difficulty"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            try:
                if key == "AudioFilename":
                    bm.audio_filename = value
                elif key == "Mode":
                    bm.mode = int(value)
                elif key == "Title":
                    bm.title = value
                elif key == "Artist":
                    bm.artist = value
                elif key == "Creator":
                    bm.creator = value
                elif key == "Version":
                    bm.version = value
                elif key == "SliderMultiplier":
                    bm.slider_multiplier = float(value)
                elif key == "CircleSize":
                    bm.circle_size = float(value)
                elif key == "OverallDifficulty":
                    bm.overall_difficulty = float(value)
                elif key == "ApproachRate":
                    bm.approach_rate = float(value)
                elif key == "HPDrainRate":
                    bm.hp = float(value)
            except ValueError:
                continue

        elif section == "TimingPoints":
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                time = float(parts[0])
                beat_length = float(parts[1])
                meter = int(parts[2]) if len(parts) > 2 else 4
                # field 6 is "uninherited" (1/0); older maps may omit it
                uninherited = (parts[6].strip() == "1") if len(parts) > 6 else (beat_length > 0)
                bm.timing_points.append(TimingPoint(time, beat_length, meter, uninherited))
            except (ValueError, IndexError):
                continue

        elif section == "HitObjects":
            obj = _parse_hit_object(line)
            if obj is not None:
                bm.hit_objects.append(obj)

    bm.timing_points.sort(key=lambda t: t.time)
    bm.hit_objects.sort(key=lambda o: o.time)

    # compute end times now that timing is available
    for obj in bm.hit_objects:
        if obj.is_spinner:
            pass  # end_time already parsed
        elif obj.is_slider:
            obj.end_time = int(round(obj.time + bm.slider_duration(obj)))
        else:
            obj.end_time = obj.time
    return bm


def _parse_hit_object(line: str) -> Optional[HitObject]:
    parts = line.split(",")
    if len(parts) < 4:
        return None
    try:
        x, y, time, type_ = int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(parts[3])
    except ValueError:
        return None
    hit_sound = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
    obj = HitObject(x=x, y=y, time=time, type=type_, hit_sound=hit_sound)

    if obj.is_slider and len(parts) > 7:
        curve = parts[5]
        if "|" in curve:
            pieces = curve.split("|")
            obj.curve_type = pieces[0]
            for p in pieces[1:]:
                if ":" in p:
                    cx, cy = p.split(":")[:2]
                    try:
                        obj.curve_points.append((int(float(cx)), int(float(cy))))
                    except ValueError:
                        pass
        try:
            obj.slides = max(1, int(parts[6]))
            obj.length = float(parts[7])
        except ValueError:
            pass
    elif obj.is_spinner and len(parts) > 5:
        try:
            obj.end_time = int(float(parts[5]))
        except ValueError:
            obj.end_time = time
    return obj


def write_osu(bm: Beatmap, hit_objects: List[HitObject], out_path: str | Path,
              timing_points: Optional[List[TimingPoint]] = None) -> Path:
    """Write a minimal but valid .osu file from generated hit objects."""
    out_path = Path(out_path)
    tps = timing_points if timing_points is not None else bm.timing_points
    if not tps:
        tps = [TimingPoint(0, 500.0, 4, True)]

    lines = ["osu file format v14", "", "[General]",
             f"AudioFilename: {bm.audio_filename}",
             "AudioLeadIn: 0", "PreviewTime: -1", "Countdown: 0",
             "SampleSet: Normal", "StackLeniency: 0.7", "Mode: 0",
             "LetterboxInBreaks: 0", "WidescreenStoryboard: 0", "",
             "[Metadata]",
             f"Title:{bm.title or 'Generated'}",
             f"TitleUnicode:{bm.title or 'Generated'}",
             f"Artist:{bm.artist or 'Unknown'}",
             f"ArtistUnicode:{bm.artist or 'Unknown'}",
             f"Creator:osu_mapper",
             f"Version:{bm.version or 'AI'}",
             "Source:", "Tags:ai generated", "BeatmapID:0", "BeatmapSetID:-1", "",
             "[Difficulty]",
             f"HPDrainRate:{bm.hp}", f"CircleSize:{bm.circle_size}",
             f"OverallDifficulty:{bm.overall_difficulty}",
             f"ApproachRate:{bm.approach_rate}",
             f"SliderMultiplier:{bm.slider_multiplier}", "SliderTickRate:1", "",
             "[Events]", "", "[TimingPoints]"]
    for tp in tps:
        uninh = 1 if tp.uninherited else 0
        lines.append(f"{int(tp.time)},{tp.beat_length},{tp.meter},1,0,50,{uninh},0")
    lines += ["", "[HitObjects]"]
    for o in hit_objects:
        if o.is_spinner:
            lines.append(f"{o.x},{o.y},{o.time},{o.type},{o.hit_sound},{o.end_time},0:0:0:0:")
        elif o.is_slider and o.curve_points:
            pts = "|".join(f"{cx}:{cy}" for cx, cy in o.curve_points)
            ctype = o.curve_type or "L"
            lines.append(f"{o.x},{o.y},{o.time},{o.type},{o.hit_sound},"
                         f"{ctype}|{pts},{o.slides},{o.length},0:0|0:0,0:0:0:0:")
        else:
            lines.append(f"{o.x},{o.y},{o.time},{o.type},{o.hit_sound},0:0:0:0:")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
