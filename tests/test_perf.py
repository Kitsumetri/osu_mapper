"""Hermetic test for the torch.compile perf/cache config (no GPU needed)."""
import os


def test_configure_compile_sets_cache_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    import src._perf as perf
    monkeypatch.setattr(perf, "_DONE", False)   # reset the module-level guard for the test

    cache = tmp_path / "tc"
    perf.configure_compile(cache_dir=cache)

    # the persistent Inductor cache dir is created and exported for the next process
    assert cache.exists()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache.resolve())
    # dynamo can hold a graph per distinct input shape (>= our raised limit)
    import torch
    assert torch._dynamo.config.cache_size_limit >= 64

    # idempotent: once configured, a second call (even with a different dir) is a no-op
    perf.configure_compile(cache_dir=tmp_path / "other")
    assert not (tmp_path / "other").exists()
