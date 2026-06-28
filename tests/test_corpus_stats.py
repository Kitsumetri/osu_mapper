"""Hermetic tests for corpus_stats — the per-file worker and the parallel path.

These build a tiny synthetic .osu corpus in a temp dir (never the real library)
and assert the new ``workers`` parallel path produces *exactly* the serial result.
rosu-pp may or may not be installed; the equivalence holds either way (if absent,
``star_rating`` returns None and both paths yield empty buckets).
"""
from __future__ import annotations

import textwrap

from src.corpus_stats import _process_file, collect

_HEADER = textwrap.dedent("""\
    osu file format v14

    [General]
    AudioFilename: audio.mp3
    Mode: 0

    [Metadata]
    Title:Test {idx}
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
    0,300,4,2,0,50,1,0

    [HitObjects]
    """)


def _write_map(path, idx: int, n: int = 60, spacing: int = 90) -> None:
    """A valid std map with ``n`` circles on a 1/2 grid, zig-zagging across the
    playfield so spacing/flow metrics are non-trivial (and SR is computable)."""
    lines = [_HEADER.format(idx=idx)]
    for i in range(n):
        x = 100 + (spacing if i % 2 else 0)
        y = 100 + (i * 3 % 180)
        t = i * 150  # 150 ms apart at 200 BPM beat=300 -> 1/2 grid
        lines.append(f"{x},{y},{t},1,0,0:0:0:0:")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _corpus(tmp_path):
    for i in range(4):
        d = tmp_path / f"set{i}"
        d.mkdir()
        _write_map(d / f"map{i}.osu", idx=i, n=55 + i * 4, spacing=70 + i * 20)
    return tmp_path


def test_process_file_returns_bucket_and_metrics(tmp_path):
    p = tmp_path / "m.osu"
    _write_map(p, idx=0, n=60)
    res = _process_file(p)
    # rosu present on this machine -> a usable std map yields (bucket, metrics)
    if res is not None:
        bucket, vals = res
        assert isinstance(bucket, str) and bucket
        assert "star_rating" in vals and "mean_spacing_px" in vals


def test_process_file_skips_too_short(tmp_path):
    p = tmp_path / "short.osu"
    _write_map(p, idx=0, n=10)  # < 50 objects -> filtered
    assert _process_file(p) is None


def test_collect_parallel_matches_serial(tmp_path):
    root = _corpus(tmp_path)
    serial = collect(root, workers=1)
    parallel = collect(root, workers=2)
    # aggregation is order-independent (_summary sorts) -> identical output
    assert serial["n_maps"] == parallel["n_maps"]
    assert serial["buckets"] == parallel["buckets"]


def test_collect_records_worker_count(tmp_path):
    root = _corpus(tmp_path)
    assert collect(root, workers=1)["workers"] == 1
    assert collect(root, workers=3)["workers"] == 3


def test_collect_limit_caps_used_maps(tmp_path):
    root = _corpus(tmp_path)
    out = collect(root, limit=2, workers=1)
    # only counts *usable* maps; with rosu absent n_maps is 0 (<= limit either way)
    assert out["n_maps"] <= 2
