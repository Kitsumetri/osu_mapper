"""Hermetic tests for the gold-map reward measurement tool (src/eval/measure_reward.py).

Exercise the pure aggregation helpers (no real osu! library / osu!.db needed).
The full ``measure`` path needs rosu + a real Songs dir and is covered by the CLI.
"""
from src.eval.measure_reward import _percentile, _summary, sample_gold_paths


def test_percentile_and_summary():
    vs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert _percentile(vs, 0.0) == 0.1
    assert _percentile(vs, 1.0) == 1.0
    s = _summary(vs)
    assert s["n"] == 10
    assert abs(s["mean"] - 0.55) < 1e-9
    assert s["min"] == 0.1 and s["max"] == 1.0
    assert s["p10"] <= s["median"] <= s["p90"]


def test_summary_empty():
    assert _summary([]) == {"n": 0}


def test_sample_gold_paths_random_fallback(tmp_path):
    # no osu!.db -> random sample of every .osu under the dir, capped at limit
    songs = tmp_path / "Songs"
    songs.mkdir()
    for i in range(10):
        (songs / f"map_{i}.osu").write_text("osu file format v14\n", encoding="utf-8")
    got = sample_gold_paths(songs, db_path=None, limit=4, seed=0)
    assert len(got) == 4
    assert all(p.suffix == ".osu" for p in got)
    # deterministic for a fixed seed
    assert sample_gold_paths(songs, None, 4, seed=0) == got


def test_sample_gold_paths_missing_db_path(tmp_path):
    # a db_path that doesn't exist still falls back cleanly (no crash)
    songs = tmp_path / "Songs"
    songs.mkdir()
    (songs / "a.osu").write_text("osu file format v14\n", encoding="utf-8")
    got = sample_gold_paths(songs, db_path=tmp_path / "nope.db", limit=10, seed=1)
    assert len(got) == 1
