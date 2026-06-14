from src.conditioning import (
    CONTEXT_DIM,
    context_from_manifest,
    context_vector,
    target_context,
)


def test_context_vector_dim_and_range():
    c = context_vector(sr=5.0, ar=9.0, od=8.0, hp=5.0, cs=4.0, density=3.5)
    assert len(c) == CONTEXT_DIM == 6
    assert all(0.0 <= v <= 1.5 for v in c)


def test_context_vector_clamps_extremes():
    c = context_vector(sr=99, ar=99, od=99, hp=99, cs=99, density=999)
    assert all(v == 1.5 for v in c)               # upper clamp
    c0 = context_vector(sr=0, ar=0, od=0, hp=0, cs=0, density=0)
    assert all(v == 0.0 for v in c0)


def test_target_context_scales_with_sr():
    low = target_context(2.0)
    high = target_context(7.0)
    # SR component (index 0) increases; derived AR/density also increase
    assert high[0] > low[0]
    assert high[1] >= low[1]          # AR
    assert high[5] >= low[5]          # density


def test_target_context_overrides():
    c = target_context(5.0, ar=10.0, density=4.0)
    assert c[1] == context_vector(5, 10, 0, 0, 0, 0)[1]   # AR honoured


def test_context_from_manifest_defaults():
    # missing fields fall back to sane defaults without crashing
    c = context_from_manifest({"n_objects": 300, "duration_s": 100.0})
    assert len(c) == CONTEXT_DIM
    # density = 300/100 = 3.0 -> 3.0/12 = 0.25
    assert abs(c[5] - 0.25) < 1e-6
