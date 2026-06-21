"""Hermetic tests for the best-of-N ranking logic (no model/GPU/audio).

``best_of_n`` itself needs a trained model + audio, so it is exercised live, not
here. The selection/report logic is pure and is what we test."""
from __future__ import annotations

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
