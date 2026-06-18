"""Timing model — BPM + offset estimation for novel songs (RESEARCH §10.8).

A separate package from the diffusion model (`src/data`, `src/model`): a beat/
downbeat/tempo tracker whose output becomes the uninherited `.osu` timing point
when no reference map exists (`generate --timing-from` stays the exact path for
known songs). CPU foundation first (labels + eval); model/training need a GPU.
"""
