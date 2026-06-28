"""Hermetic tests for the best-of-N ranking logic (no model/GPU/audio).

``best_of_n`` itself needs a trained model + audio, so it is exercised live, not
here. The selection/report logic and the infer-side helpers are pure and tested here."""
from __future__ import annotations

import json

import pytest

from src.best_of_n import _candidate_row, select_winner
from src.eval.reward import RewardBreakdown


def _bd(reward: float, **kw) -> RewardBreakdown:
    return RewardBreakdown(
        reward=reward, quality=kw.get("quality", reward),
        sr_closeness=kw.get("sr_closeness", 1.0), achieved_sr=kw.get("achieved_sr", 5.0),
        target_sr=kw.get("target_sr", 5.0), bucket=kw.get("bucket", "Insane"),
        per_metric=kw.get("per_metric", {}), n_objects=kw.get("n_objects", 100))


def test_select_winner_picks_max_reward():
    bds = [_bd(0.40), _bd(0.72), _bd(0.55)]
    assert select_winner(bds) == 1


def test_select_winner_ties_break_to_lowest_index():
    bds = [_bd(0.5), _bd(0.6), _bd(0.6)]
    assert select_winner(bds) == 1  # first of the tied 0.6s


def test_select_winner_empty():
    assert select_winner([]) == -1


def test_select_winner_single():
    assert select_winner([_bd(0.33)]) == 0


def test_candidate_row_shape():
    row = _candidate_row(3, _bd(0.61, per_metric={"jump_ratio": 0.8}, n_objects=222))
    assert row["idx"] == 3
    assert row["reward"] == 0.61
    assert row["n_objects"] == 222
    assert row["per_metric"] == {"jump_ratio": 0.8}
    # every audit field present so reward hacking is inspectable
    assert set(row) >= {"idx", "reward", "quality", "sr_closeness",
                        "achieved_sr", "bucket", "n_objects", "per_metric"}


# ---------------------------------------------------------------------------
# Tests for helpers added in run_inference.py (imported here to stay hermetic)
# ---------------------------------------------------------------------------

from src.run_inference import _load_ref_stats_for_infer, _print_bon_summary


def test_load_ref_stats_missing_raises_system_exit(tmp_path):
    """_load_ref_stats_for_infer should raise SystemExit with a helpful message."""
    missing = str(tmp_path / "does_not_exist.json")
    with pytest.raises(SystemExit) as exc_info:
        _load_ref_stats_for_infer(missing)
    msg = str(exc_info.value)
    assert "corpus_stats" in msg  # points the user at the right fix


def test_load_ref_stats_returns_dict(tmp_path):
    """_load_ref_stats_for_infer should parse and return the JSON as a dict."""
    stats_path = tmp_path / "reference_stats.json"
    payload = {"n_maps": 42, "some_key": [1, 2, 3]}
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    result = _load_ref_stats_for_infer(str(stats_path))
    assert result == payload


def test_print_bon_summary_lift_calculation(capsys):
    """_print_bon_summary should compute lift = best - mean and surface it."""
    rewards = [0.4, 0.6, 0.5]   # mean = 0.5
    win_reward = 0.6
    _print_bon_summary(sr=5.0, win_reward=win_reward, all_rewards=rewards, elapsed=30.0)
    out = capsys.readouterr().out
    # check that the winner reward, mean and lift all appear in output
    assert "0.6000" in out
    assert "0.5000" in out
    assert "+0.1000" in out


def test_print_bon_summary_n_shown(capsys):
    """_print_bon_summary should mention the number of candidates."""
    _print_bon_summary(sr=6.0, win_reward=0.7, all_rewards=[0.5, 0.7, 0.6], elapsed=10.0)
    out = capsys.readouterr().out
    assert "n=3" in out


def test_print_bon_summary_single_candidate(capsys):
    """Edge case: N=1 -> lift is 0."""
    _print_bon_summary(sr=4.0, win_reward=0.55, all_rewards=[0.55], elapsed=5.0)
    out = capsys.readouterr().out
    assert "+0.0000" in out
