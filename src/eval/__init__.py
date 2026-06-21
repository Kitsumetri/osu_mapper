"""Analysis / eval probes (dev tools, not part of the inference path).

  python -m src.eval.analyze_phase1 --ckpt runs/<id>/ckpt/best.pt    # real-vs-generated patterns
  python -m src.eval.eval_spacing_channel --ckpt runs/<id>/ckpt/best.pt  # v8 spacing-channel probe
"""
