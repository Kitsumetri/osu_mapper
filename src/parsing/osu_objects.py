from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class BasicData:
    def __repr__(self):
        field_strings = []
        for f in fields(self):
            field_name = f.name
            field_value = getattr(self, field_name)
            field_strings.append(f"* {field_name}: {field_value}")
        result_str = "\n".join(field_strings)
        sep = '-' * 70 + '\n'
        return f'{sep}- {str(self.__class__).split(".")[-1][:-2] + ":":^35}\n{sep}{result_str}'


@dataclass
class GeneralData(BasicData):
    AudioFilename: str = field(default='audio.mp3')
    AudioLeadIn: int = field(default=0)
    AudioHash: str = field(default='example_hash')  # Deprecated
    PreviewTime: int = field(default=-1)
    Countdown: int = field(default=1)
    SampleSet: str = field(default='Normal')  # 'Normal', 'Soft', 'Drum'
    StackLeniency: float = field(default=0.7)

    # 0 = osu!, 1 = osu!taiko, 2 = osu!catch, 3 = osu!mania
    Mode: int = field(default=0)

    LetterboxInBreaks: bool = field(default=0)
    StoryFireInFront: bool = field(default=1)  # Deprecated
    UseSkinSprites: bool = field(default=0)
    AlwaysShowPlayfield: bool = field(default=0)  # Deprecated

    # NoChange = use skin setting,
    # Below = draw overlays under numbers,
    # Above = draw overlays on top of numbers
    OverlayPosition: str = field(default='NoChange')

    SkinPreference: str = field(default='default')
    EpilepsyWarning: bool = field(default=0)
    CountdownOffset: int = field(default=0)
    SpecialStyle: bool = field(default=0)
    WidescreenStoryboard: bool = field(default=0)
    SamplesMatchPlaybackRate: bool = field(default=0)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class EditorData(BasicData):
    Bookmarks: List[int] = field(default_factory=lambda: [])
    DistanceSpacing: float = field(default=0)
    BeatDivisor: int = field(default=1)
    GridSize: int = field(default=1)
    TimelineZoom: float = field(default=1)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class MetaData(BasicData):
    Title: str = field(default='title_placeholder')
    TitleUnicode: str = field(default='title_unicode_placeholder')
    Artist: str = field(default='artist_placeholder')
    ArtistUnicode: str = field(default='artist_unicode_placeholder')
    Creator: str = field(default='creator_placeholder')
    Version: str = field(default='version_placeholder')
    Source: str = field(default='source_placeholder')
    Tags: List[str] = field(default_factory=lambda: [])
    BeatmapID: int = field(default=-1)
    BeatmapSetID: int = field(default=-1)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class DifficultyData(BasicData):
    HPDrainRate: float = field(default=5)
    CircleSize: float = field(default=5)
    OverallDifficulty: float = field(default=5)
    ApproachRate: float = field(default=5)
    SliderMultiplier: float = field(default=1)
    SliderTickRate: float = field(default=1)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class TimingPointNode:
    time: int
    beat_length: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    effects: int


@dataclass
class TimingPointsData(BasicData):
    timing_sections: List[TimingPointNode] = field(default_factory=lambda: [])

    def __repr__(self) -> str:
        return super().__repr__()

    def __getitem__(self, item: int) -> TimingPointNode:
        return self.timing_sections[item]


@dataclass
class ColoursData(BasicData):
    combo_colors: Dict

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class HitObjectNode:
    x: int
    y: int
    time: int
    type: int
    hit_sound: int
    object_params: Optional[str] = field(default=None)
    hit_sample: str = field(default="0:0:0:0:")


@dataclass
class HitObjectData(BasicData):
    object_sections: List[HitObjectNode] = field(default_factory=lambda: [])

    def __repr__(self) -> str:
        return super().__repr__()

    def __getitem__(self, item: int) -> HitObjectNode:
        return self.object_sections[item]


class OsuBeatmap:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.general = None
        self.editor = None
        self.metadata = None
        self.difficulty = None
        self.timing_points = None
        self.colours = None
        self.hit_objects = None

        self.__parse_file(self.file_path)

    def __repr__(self) -> str:
        result_str = f'{"Osu Beatmap":^70}\n'
        for obj in (self.general, self.editor, self.metadata, self.difficulty,
                    self.timing_points, self.colours, self.hit_objects):
            result_str += f"{repr(obj)}\n\n"
        return result_str

    @staticmethod
    def parse_key_value(line: str):
        if ":" in line:
            key, value = line.split(":", 1)
            return key.strip(), value.strip()
        return None, None

    def parse_general(self, lines: List[str]) -> GeneralData:
        general_data = {}
        for line in lines:
            key, value = self.parse_key_value(line)
            if key == "AudioFilename":
                general_data['AudioFilename'] = value
            elif key == "AudioLeadIn":
                general_data['AudioLeadIn'] = int(value)
            elif key == "PreviewTime":
                general_data['PreviewTime'] = int(value)
            elif key == "Countdown":
                general_data['Countdown'] = int(value)
            elif key == "SampleSet":
                general_data['SampleSet'] = value
            elif key == "StackLeniency":
                general_data['StackLeniency'] = float(value)
            elif key == "Mode":
                general_data['Mode'] = int(value)
            elif key == "LetterboxInBreaks":
                general_data['LetterboxInBreaks'] = bool(int(value))
            elif key == "WidescreenStoryboard":
                general_data['WidescreenStoryboard'] = bool(int(value))
        return GeneralData(**general_data)

    def parse_editor(self, lines: List[str]) -> EditorData:
        editor_data = {}
        for line in lines:
            key, value = self.parse_key_value(line)
            if key == "DistanceSpacing":
                editor_data['DistanceSpacing'] = float(value)
            elif key == "BeatDivisor":
                editor_data['BeatDivisor'] = int(value)
            elif key == "GridSize":
                editor_data['GridSize'] = int(value)
            elif key == "TimelineZoom":
                editor_data['TimelineZoom'] = float(value)
        return EditorData(**editor_data)

    def parse_metadata(self, lines: List[str]) -> MetaData:
        metadata = {}
        for line in lines:
            key, value = self.parse_key_value(line)
            if key == "Title":
                metadata['Title'] = value
            elif key == "TitleUnicode":
                metadata['TitleUnicode'] = value
            elif key == "Artist":
                metadata['Artist'] = value
            elif key == "ArtistUnicode":
                metadata['ArtistUnicode'] = value
            elif key == "Creator":
                metadata['Creator'] = value
            elif key == "Version":
                metadata['Version'] = value
            elif key == "BeatmapID":
                metadata["BeatmapID"] = value
            elif key == "BeatmapSetID":
                metadata["BeatmapSetID"] = value
            elif key == "Tags":
                metadata["Tags"] = value.split(' ')
            elif key == "Source":
                metadata["Source"] = value
        return MetaData(**metadata)

    def parse_difficulty(self, lines: List[str]) -> DifficultyData:
        difficulty_data = {}
        for line in lines:
            key, value = self.parse_key_value(line)
            if key == "HPDrainRate":
                difficulty_data['HPDrainRate'] = float(value)
            elif key == "CircleSize":
                difficulty_data['CircleSize'] = float(value)
            elif key == "OverallDifficulty":
                difficulty_data['OverallDifficulty'] = float(value)
            elif key == "ApproachRate":
                difficulty_data['ApproachRate'] = float(value)
            elif key == "SliderMultiplier":
                difficulty_data['SliderMultiplier'] = float(value)
            elif key == "SliderTickRate":
                difficulty_data['SliderTickRate'] = float(value)
        return DifficultyData(**difficulty_data)

    @staticmethod
    def parse_timing_points(lines: List[str]) -> TimingPointsData:
        timing_sections = []
        for line in lines:
            values = line.split(',')
            if len(values) >= 8:
                timing_point = TimingPointNode(
                    time=int(float(values[0])),
                    beat_length=float(values[1]),
                    meter=int(values[2]),
                    sample_set=int(values[3]),
                    sample_index=int(values[4]),
                    volume=int(values[5]),
                    uninherited=bool(values[6]),
                    effects=int(values[7])
                )
                timing_sections.append(timing_point)
        return TimingPointsData(timing_sections=timing_sections)

    def parse_colours(self, lines: List[str]) -> ColoursData:
        combo_colors = {}
        for line in lines:
            key, value = self.parse_key_value(line)
            if key.startswith("Combo"):
                color_values = tuple(map(int, value.split(',')))
                combo_colors[key] = color_values
        return ColoursData(combo_colors=combo_colors)

    @staticmethod
    def parse_hit_objects(lines: List[str]) -> HitObjectData:
        object_sections = []
        for line in lines:
            values = line.split(',')
            if len(values) >= 5:
                hit_object = HitObjectNode(
                    x=int(values[0]),
                    y=int(values[1]),
                    time=int(values[2]),
                    type=int(values[3]),
                    hit_sound=int(values[4]),
                    object_params=values[5] if len(values) > 5 else None,
                    hit_sample=values[6] if len(values) > 6 else "0:0:0:0:"
                )
                object_sections.append(hit_object)
        return HitObjectData(object_sections=object_sections)

    def __parse_file(self, file_path: str | Path):
        with open(file_path, 'r', encoding='utf-8') as f:
            current_section = None
            section_lines = []

            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue

                if line.startswith('[') and line.endswith(']'):
                    # Обрабатываем конец предыдущей секции
                    if current_section == 'General':
                        self.general = self.parse_general(section_lines)
                    elif current_section == 'Editor':
                        self.editor = self.parse_editor(section_lines)
                    elif current_section == 'Metadata':
                        self.metadata = self.parse_metadata(section_lines)
                    elif current_section == 'Difficulty':
                        self.difficulty = self.parse_difficulty(section_lines)
                    elif current_section == 'TimingPoints':
                        self.timing_points = self.parse_timing_points(section_lines)
                    elif current_section == 'Colours':
                        self.colours = self.parse_colours(section_lines)
                    elif current_section == 'HitObjects':
                        self.hit_objects = self.parse_hit_objects(section_lines)

                    # Начинаем новую секцию
                    current_section = line[1:-1]
                    section_lines = []
                else:
                    section_lines.append(line)

            # Обрабатываем последнюю секцию
            if current_section == 'General':
                self.general = self.parse_general(section_lines)
            elif current_section == 'Editor':
                self.editor = self.parse_editor(section_lines)
            elif current_section == 'Metadata':
                self.metadata = self.parse_metadata(section_lines)
            elif current_section == 'Difficulty':
                self.difficulty = self.parse_difficulty(section_lines)
            elif current_section == 'TimingPoints':
                self.timing_points = self.parse_timing_points(section_lines)
            elif current_section == 'Colours':
                self.colours = self.parse_colours(section_lines)
            elif current_section == 'HitObjects':
                self.hit_objects = self.parse_hit_objects(section_lines)
