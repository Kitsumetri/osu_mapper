"""One-stop torch.compile perf + caching config — call once before the first compile.

Makes torch.compile both FAST (TF32) and cheap to RE-USE across runs (a persistent
on-disk Inductor cache). The persistent cache is the big win for our per-song inference
CLI: without it every fresh ``main.py infer`` process recompiles the model (~30-60 s);
with it the compile is paid ONCE EVER per shape (+ torch/triton/GPU version) and reloaded
from disk by later runs. (Training, which has static shapes, benefits too: resumes/re-runs
skip the recompile.)
"""
from __future__ import annotations

import os
from pathlib import Path

_DONE = False


def configure_compile(cache_dir: str | Path = "artifacts/torch_compile_cache") -> None:
    """Idempotent. Set TF32 + a raised dynamo cache limit + a PERSISTENT Inductor cache.

    - **TF32 matmuls** (Ada/Ampere): ~free speedup for fp32 ops (bf16/autocast paths are
      unaffected). No-op off CUDA.
    - **dynamo ``cache_size_limit`` 8 -> 64**: at inference each distinct song length gets
      its own cached graph; the default 8 falls back to slow eager after 8 lengths in one
      process (e.g. a multi-song batch). 64 covers a big batch.
    - **persistent Inductor cache**: ``TORCHINDUCTOR_CACHE_DIR`` -> a stable project folder
      (under the git-ignored ``artifacts/``) + the FX-graph cache, so compiled graphs
      survive process exit and are reused by the next run instead of recompiling. Keyed on
      graph + shape + torch/triton/GPU version, so it self-invalidates on an upgrade or a
      model-code change (no stale-cache risk).

    Safe to call from both train and inference; the first call wins (cache_dir is created).
    """
    global _DONE
    if _DONE:
        return
    import torch

    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # must be visible before the first compile so Inductor picks it up; setdefault so an
    # explicit user TORCHINDUCTOR_CACHE_DIR (e.g. a ramdisk) still wins.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))
    try:                                  # cross-process compiled-graph cache
        import torch._inductor.config as ind_cfg
        ind_cfg.fx_graph_cache = True
    except Exception:                     # pragma: no cover - older torch without the knob
        pass
    try:                                  # one cached graph per distinct input shape
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(
            64, torch._dynamo.config.cache_size_limit)
    except Exception:                     # pragma: no cover
        pass
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    _DONE = True
