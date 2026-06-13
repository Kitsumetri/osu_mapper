"""Shared fixtures: a small synthetic .osu file written to a temp dir.

Tests are hermetic — they never touch the real osu! Songs library or a GPU.
"""
import textwrap
import pytest

# a tiny but valid osu!standard map: one circle, one slider, one spinner
SAMPLE_OSU = textwrap.dedent("""\
    osu file format v14

    [General]
    AudioFilename: audio.mp3
    Mode: 0

    [Metadata]
    Title:Test Song
    Artist:Tester
    Creator:unit
    Version:Normal

    [Difficulty]
    HPDrainRate:5
    CircleSize:4
    OverallDifficulty:6
    ApproachRate:7
    SliderMultiplier:1.4
    SliderTickRate:1

    [TimingPoints]
    0,500,4,2,0,50,1,0
    1000,-50,4,2,0,50,0,0

    [HitObjects]
    256,192,0,1,0,0:0:0:0:
    100,100,1000,2,0,L|300:100,1,200,0:0|0:0,0:0:0:0:
    256,192,3000,12,0,5000,0:0:0:0:
    """)


@pytest.fixture
def sample_osu(tmp_path):
    p = tmp_path / "test (unit) [Normal].osu"
    p.write_text(SAMPLE_OSU, encoding="utf-8")
    return p
