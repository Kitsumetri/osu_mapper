"""Hermetic tests for early-abort sampling (no GPU / rosu / network / audio).

Covers the four contracts from the task spec:

1. ``GaussianDiffusion.ddim_sample(monitor=None)`` is byte-for-byte unchanged, and a
   monitor that never aborts cannot move the result (the monitor is an exit gate, not
   a steerer).
2. The cheap quality proxy + ``EarlyAbortMonitor`` FIRE on a synthetic always-low
   candidate and do NOT fire on a good one.
3. The step-relative rule + absolute floor + late-window gate behave exactly as
   designed (each branch tested in isolation).
4. With a deterministic fake sampler, ``best_of_n``'s selected winner is IDENTICAL
   with and without early-abort (we only ever abort would-be losers) — the critical
   correctness property — while ``n_aborted`` / ``steps_saved`` are surfaced.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.best_of_n import (
    EA_ABS_FLOOR,
    EA_MARGIN,
    EA_MONITOR_FRAC,
    EarlyAbortMonitor,
    best_of_n,
    cheap_quality,
)
from src.model.diffusion import GaussianDiffusion
from src.parsing.beatmap import HitObject


# --- tiny fakes -------------------------------------------------------------
class _ConstModel(torch.nn.Module):
    """A denoiser stub returning a fixed-shape constant; deterministic & cheap."""

    def __init__(self, value: float = 0.1):
        super().__init__()
        self.value = value
        self.sig_channels = 4

    def forward(self, x, cond, t, ctx=None, ctx_drop=None):
        return torch.full_like(x, self.value)


def _circles(n: int, *, x: int = 256, y: int = 192, step_ms: int = 300) -> list[HitObject]:
    """n stacked circles `step_ms` apart (a valid-but-boring object stream)."""
    return [HitObject(x=x, y=y, time=i * step_ms, type=1, end_time=i * step_ms)
            for i in range(n)]


def _fake_quality(value: float):
    """A quality_score-shaped fn ignoring its inputs and returning a constant."""
    def q(metrics, ref_stats, bucket):
        return value, {}, {}
    return q


# ============================================================================
# 1. ddim_sample(monitor=None) unchanged + monitor cannot steer
# ============================================================================
def _run_sample(monitor=..., seed: int = 0):
    torch.manual_seed(seed)
    diff = GaussianDiffusion(timesteps=50, device="cpu", objective="v", zero_snr=True)
    model = _ConstModel(0.1)
    kw = {} if monitor is ... else {"monitor": monitor}
    return diff.ddim_sample(model, cond=None, shape=(1, 4, 8), steps=10, **kw)


def test_ddim_sample_default_unchanged_when_monitor_absent_or_none():
    a = _run_sample()              # monitor arg not passed at all
    b = _run_sample(monitor=None)  # monitor=None
    assert torch.equal(a, b)


def test_monitor_never_aborting_is_byte_identical():
    calls = []

    def watch(k, frac, x0):
        calls.append((k, round(float(frac), 4)))
        return False  # never abort

    base = _run_sample(monitor=None)
    with_mon = _run_sample(monitor=watch)
    assert torch.equal(base, with_mon)         # monitor cannot move x0
    assert calls, "monitor should be invoked every step"
    # frac_done runs 0..1 monotonically; last step is ~1.0
    fracs = [f for _, f in calls]
    assert fracs[0] == pytest.approx(0.0)
    assert fracs[-1] == pytest.approx(1.0)
    assert fracs == sorted(fracs)


def test_monitor_abort_breaks_early_and_returns_current_x0():
    seen = {}

    def watch(k, frac, x0):
        seen["x0"] = x0.clone()
        return frac >= 0.5  # abort at the first late step

    out = _run_sample(monitor=watch)
    # returned tensor is exactly the x0 the monitor last saw (the partial clean signal)
    assert torch.equal(out, seen["x0"])


# ============================================================================
# 2 + 4 (proxy). cheap_quality fires low / passes high; no rosu / no I/O
# ============================================================================
def test_cheap_quality_too_few_objects_is_zero():
    sig = np.zeros((4, 8), dtype=np.float32)
    q = cheap_quality(sig, ref_stats={}, bucket="Insane", bpm=180.0,
                      decode=lambda s, onset_threshold=0.3: _circles(1),
                      metrics_fn=lambda bm: {"n_objects": 1},
                      quality_fn=_fake_quality(0.9))
    assert q == 0.0  # < 2 objects -> "going nowhere"


def test_cheap_quality_uses_injected_quality_no_rosu():
    sig = np.zeros((4, 8), dtype=np.float32)
    q = cheap_quality(sig, ref_stats={}, bucket="Insane", bpm=180.0,
                      decode=lambda s, onset_threshold=0.3: _circles(10),
                      metrics_fn=lambda bm: {"n_objects": 10},
                      quality_fn=_fake_quality(0.73))
    assert q == pytest.approx(0.73)


def test_cheap_quality_real_decode_metrics_path_runs_in_memory():
    """End-to-end proxy through the REAL decode_signal + compute_metrics on a
    synthetic signal — proves the in-memory Beatmap wrapping works (no file I/O,
    no rosu) and returns a finite quality in [0, 1]."""
    rng = np.random.default_rng(0)
    sig = rng.uniform(-1, 1, size=(21, 256)).astype(np.float32)
    sig[0, ::8] = 1.0  # plant periodic onsets so decode finds objects
    q = cheap_quality(sig, ref_stats={"buckets": {}}, bucket="Insane", bpm=180.0)
    assert 0.0 <= q <= 1.0  # empty ref bucket -> quality 0.0, but must not raise


# ============================================================================
# 3. EarlyAbortMonitor: step-relative rule + floor + late-window + arming
# ============================================================================
def _monitor(best_by_step, n_completed, **kw):
    return EarlyAbortMonitor(
        ref_stats={}, bucket="Insane", bpm=180.0,
        best_by_step=best_by_step, n_completed=n_completed,
        quality_fn=_fake_quality(kw.pop("q", 0.1)),
        decode=lambda s, onset_threshold=0.3: _circles(10),
        metrics_fn=lambda bm: {"n_objects": 10}, **kw)


def _x0():  # any tensor; the fake decode/quality ignore its contents
    return torch.zeros(1, 4, 8)


def test_no_abort_before_late_window():
    # quality is rock-bottom and the cache says everyone else is great, but we are
    # still early in sampling -> never abort (early x0 is blurry/meaningless).
    m = _monitor({3: 0.95}, n_completed=5, q=0.0, monitor_frac=0.7)
    assert m(k=3, frac_done=0.5, x0=_x0()) is False
    assert m.aborted is False


def test_no_abort_until_min_candidates_completed():
    # late + low + below floor, but not enough completed candidates to trust the
    # cache yet -> disarmed (the first candidates are never aborted blindly).
    m = _monitor({3: 0.95}, n_completed=1, q=0.0, min_candidates=2)
    assert m(k=3, frac_done=0.9, x0=_x0()) is False
    assert m.aborted is False


def test_no_abort_when_above_floor_even_if_trailing():
    # trails best (0.95) by > margin, but its own quality is still respectable
    # (>= abs_floor) -> NOT aborted (err toward keeping; it could still win).
    m = _monitor({3: 0.95}, n_completed=3, q=0.70,
                 margin=0.15, abs_floor=0.55)
    assert 0.70 < 0.95 - 0.15  # genuinely trailing by the margin...
    assert m(k=3, frac_done=0.9, x0=_x0()) is False  # ...but above floor -> keep
    assert m.aborted is False


def test_no_abort_when_within_margin_even_if_below_floor():
    # below the absolute floor, but only just behind the best -> NOT aborted (the
    # step-relative drop hasn't been exceeded; the cache best is also low here).
    m = _monitor({3: 0.50}, n_completed=3, q=0.45,
                 margin=0.15, abs_floor=0.55)
    assert 0.45 >= 0.50 - 0.15  # within margin of the (low) best
    assert m(k=3, frac_done=0.9, x0=_x0()) is False
    assert m.aborted is False


def test_abort_when_below_floor_and_trailing_by_margin():
    # late + below floor + trails the best by > margin -> the one genuine
    # bottom-feeder case: ABORT.
    m = _monitor({3: 0.95}, n_completed=3, q=0.05,
                 margin=0.15, abs_floor=0.55)
    assert m(k=3, frac_done=0.9, x0=_x0()) is True
    assert m.aborted is True
    assert m.abort_k == 3
    assert m.last_quality == pytest.approx(0.05)


def test_good_candidate_never_aborts_across_late_steps():
    # a strong candidate (q above floor & near best) is never aborted at any late step.
    m = _monitor({3: 0.95, 2: 0.95, 1: 0.95, 0: 0.95}, n_completed=4, q=0.92,
                 margin=0.15, abs_floor=0.55)
    for k, frac in [(3, 0.7), (2, 0.8), (1, 0.9), (0, 1.0)]:
        assert m(k=k, frac_done=frac, x0=_x0()) is False
    assert m.aborted is False


def test_monitor_defaults_are_conservative():
    # sanity on the shipped defaults: floor below 1, margin modest, late window late.
    assert 0.0 < EA_MARGIN < 0.5
    assert 0.0 < EA_ABS_FLOOR < 1.0
    assert 0.5 <= EA_MONITOR_FRAC < 1.0


# ============================================================================
# 5. Winner IDENTICAL with and without early-abort (deterministic fake sampler)
# ============================================================================
class _FakeDiff:
    """A deterministic ddim_sample stand-in.

    Per candidate it replays a designed quality trajectory through the monitor (so
    the abort logic runs for real), then returns a constant x0. ``self.seq_len`` makes
    ``len(seq)`` look like a real run so steps_saved is non-trivial. The trajectory is
    keyed off a per-instance counter so candidate i gets quality ``self.qualities[i]``.
    """

    def __init__(self, qualities, steps=10):
        self.qualities = qualities
        self.steps = steps
        self.i = -1

    def ddim_sample(self, model, cond, shape, *, monitor=None, **kw):
        self.i += 1
        # the cheap proxy reads x0 via the injected decode/metrics/quality on the
        # monitor, so the *contents* don't matter; only the monitor's quality_fn does.
        x0 = torch.zeros(*shape)
        if monitor is not None:
            # walk the late steps k = steps-1 .. 0 with frac 0..1 like the real loop.
            denom = self.steps - 1
            for k in reversed(range(self.steps)):
                frac = 1.0 - k / denom
                if monitor(k, frac, x0):
                    break
        return x0


def _make_loaded(fake_diff):
    from collections import namedtuple
    Loaded = namedtuple("Loaded", "model diff ctx_dim device")
    return Loaded(model=_ConstModel(), diff=fake_diff, ctx_dim=6, device="cpu")


def _install_bon_fakes(monkeypatch, qualities, rewards):
    """Patch best_of_n's generate + reward so it runs with no model/rosu/audio.

    - ``generate`` writes a candidate file whose bytes encode the candidate index
      (so identical candidates across the two runs are byte-identical) and triggers
      the patched ddim_sample (firing the monitor).
    - ``reward_from_osu`` returns a deterministic reward per candidate index.
    The monitor's quality proxy is driven by per-candidate ``qualities`` via a
    quality_fn we attach to every EarlyAbortMonitor through a patched constructor.
    """
    import src.best_of_n as bon

    call = {"i": -1}

    def fake_generate(audio_path, out_path, sr, loaded, prepared, **kw):
        call["i"] += 1
        idx = call["i"]
        # fire the (possibly monitor-patched) sampler so abort logic runs
        loaded.diff.ddim_sample(loaded.model, None, (1, 4, 8))
        # deterministic per-candidate content -> identical bytes across runs
        from pathlib import Path
        Path(out_path).write_text(f"CAND idx={idx}\n", encoding="utf-8")
        return out_path

    def fake_reward(osu_path, ref_stats, target_sr, **kw):
        from pathlib import Path

        from src.eval.reward import RewardBreakdown
        idx = int(Path(osu_path).read_text(encoding="utf-8").split("idx=")[1])
        r = rewards[idx]
        return RewardBreakdown(reward=r, quality=r, sr_closeness=1.0,
                               achieved_sr=target_sr, target_sr=target_sr,
                               bucket="Insane", per_metric={}, n_objects=100)

    # make every monitor use the per-candidate scripted quality
    orig_ctor = bon.EarlyAbortMonitor

    def patched_ctor(*a, **kw):
        idx = call["i"] + 1   # the candidate ABOUT to be generated
        kw["quality_fn"] = _fake_quality(qualities[idx])
        kw["decode"] = lambda s, onset_threshold=0.3: _circles(10)
        kw["metrics_fn"] = lambda bm: {"n_objects": 10}
        return orig_ctor(*a, **kw)

    monkeypatch.setattr(bon, "generate", fake_generate)
    monkeypatch.setattr(bon, "reward_from_osu", fake_reward)
    monkeypatch.setattr(bon, "EarlyAbortMonitor", patched_ctor)
    return call


def _prepared_stub():
    from collections import namedtuple

    from src.parsing.beatmap import TimingPoint
    Prepared = namedtuple("Prepared", "cond t_len t_full tp aim")
    return Prepared(cond=None, t_len=8, t_full=8,
                    tp=TimingPoint(0.0, 333.33, 4, True), aim=None)


def test_winner_identical_with_and_without_early_abort(tmp_path, monkeypatch):
    # 6 candidates. Two are clear bottom-feeders (low quality AND low reward); the
    # winner is candidate 4 with the top reward. Abort must drop the losers and leave
    # the winner unchanged.
    qualities = [0.90, 0.92, 0.05, 0.91, 0.95, 0.03]   # proxy quality per candidate
    rewards   = [0.70, 0.74, 0.20, 0.72, 0.85, 0.18]   # final reward per candidate
    ref_stats = {"n_maps": 1, "buckets": {}}

    # --- run WITHOUT early-abort ---
    call = _install_bon_fakes(monkeypatch, qualities, rewards)
    call["i"] = -1
    out_a = tmp_path / "a.osu"
    loaded = _make_loaded(_FakeDiff(qualities))
    _, win_a, bds_a = best_of_n(
        "song.mp3", sr=5.0, ref_stats=ref_stats, out_path=str(out_a),
        loaded=loaded, prepared=_prepared_stub(), n=6, seed=0, early_abort=False)
    bytes_a = out_a.read_bytes()
    report_a = json.loads((tmp_path / "a.osu.bon.json").read_text(encoding="utf-8"))

    # --- run WITH early-abort ---
    call["i"] = -1
    out_b = tmp_path / "b.osu"
    loaded2 = _make_loaded(_FakeDiff(qualities))
    _, win_b, bds_b = best_of_n(
        "song.mp3", sr=5.0, ref_stats=ref_stats, out_path=str(out_b),
        loaded=loaded2, prepared=_prepared_stub(), n=6, seed=0, early_abort=True,
        ea_min_candidates=2, ea_margin=0.15, ea_abs_floor=0.55, ea_monitor_frac=0.7)
    bytes_b = out_b.read_bytes()
    report_b = json.loads((tmp_path / "b.osu.bon.json").read_text(encoding="utf-8"))

    # winner reward + winner bytes identical (THE correctness property)
    assert win_a.reward == win_b.reward == 0.85
    assert report_a["winner"] == report_b["winner"]
    assert bytes_a == bytes_b
    # early-abort actually saved compute on the genuine losers
    assert report_b["n_aborted"] >= 1
    assert report_b["steps_saved"] > 0
    assert report_a["n_aborted"] == 0          # OFF path reports zero aborts
    # the aborted candidates were the low-reward ones (sanity: never the winner)
    aborted_idxs = {int(k) for k in report_b["aborted"]}
    assert report_b["winner"] not in aborted_idxs
    assert aborted_idxs.issubset({2, 5})       # only the two bottom-feeders


def test_first_candidate_never_aborts(tmp_path, monkeypatch):
    # Even with the worst-possible first candidate (quality 0.0, monitoring armed from
    # the very first step), it must COMPLETE — there is no completed baseline to trail,
    # so the cache is empty and the step-relative rule cannot fire. This is why
    # best_of_n always has >=1 completed candidate (the SystemExit guard is purely
    # defensive). Here candidate 0 is the worst and still survives + wins by default.
    qualities = [0.0, 0.0]
    rewards = [0.40, 0.10]   # cand 0 is best on reward despite worst proxy quality
    call = _install_bon_fakes(monkeypatch, qualities, rewards)
    call["i"] = -1
    loaded = _make_loaded(_FakeDiff(qualities))
    out = tmp_path / "f.osu"
    _, win, _ = best_of_n(
        "song.mp3", sr=5.0, ref_stats={"n_maps": 1, "buckets": {}},
        out_path=str(out), loaded=loaded, prepared=_prepared_stub(), n=2, seed=0,
        early_abort=True, ea_min_candidates=0, ea_margin=0.05, ea_abs_floor=0.9,
        ea_monitor_frac=0.0)
    report = json.loads((tmp_path / "f.osu.bon.json").read_text(encoding="utf-8"))
    assert report["winner"] == 0           # the un-abortable first candidate wins
    assert win.reward == 0.40
    assert 0 not in {int(k) for k in report["aborted"]}
