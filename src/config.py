"""Shared configuration for the osu_mapper ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256  # -> 22050/256 = 86.13 frames/sec (~11.6 ms/frame)
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 11025.0

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 / self.frame_rate

    def time_to_frame(self, t_ms: float) -> float:
        return t_ms / self.ms_per_frame

    def frame_to_time(self, frame: float) -> float:
        return frame * self.ms_per_frame


# Map-signal channel layout (the diffusion target). v3: 10 channels. v5: +7
# slider channels (indices appended so 0-9 keep their meaning / older models).
N_SLIDER_ANCHORS = 3  # K control-point slots carried per slider (head-relative)
SIGNAL_CHANNELS = [
    "onset",  # 0: bump at circle/slider-head start
    "slider_hold",  # 1: +1 during slider body
    "spinner_hold",  # 2: +1 during spinner
    "new_combo",  # 3: bump at new-combo objects
    "cursor_x",  # 4: normalised playfield x in [-1, 1]
    "cursor_y",  # 5: normalised playfield y in [-1, 1]
    "kiai_hold",  # 6: +1 during kiai (chorus/hype) sections
    "whistle",  # 7: bump at objects with the whistle hitsound (bit 2)
    "finish",  # 8: bump at objects with the finish hitsound (bit 4)
    "clap",  # 9: bump at objects with the clap hitsound (bit 8)
    # v5 slider-shape channels: control-point offsets from the slider head,
    # normalised by playfield size, held constant over the slider span (baseline 0).
    "slider_dx1",  # 10
    "slider_dy1",  # 11
    "slider_dx2",  # 12
    "slider_dy2",  # 13
    "slider_dx3",  # 14
    "slider_dy3",  # 15
    "slides",  # 16: repeat count held over the slider span (reverse sliders)
    # v7 slider-velocity: the SV multiplier timeline (log2(SV)/2, baseline 0 = SV 1.0),
    # piecewise-constant like real green-line sections. Decode quantises it to a few
    # stable green lines (RESEARCH 10.7 P4-A). Appended -> 0-16 keep their meaning.
    "sv",  # 17
]
N_SIGNAL_CHANNELS = len(SIGNAL_CHANNELS)

# Channel indices grouped by decode behaviour.
CH_ONSET, CH_SLIDER, CH_SPINNER, CH_NEWCOMBO, CH_CURX, CH_CURY = 0, 1, 2, 3, 4, 5
CH_KIAI, CH_WHISTLE, CH_FINISH, CH_CLAP = 6, 7, 8, 9
CH_SLIDER_ANCHORS = 10                       # first of 2*N_SLIDER_ANCHORS dx/dy channels
CH_SLIDES = CH_SLIDER_ANCHORS + 2 * N_SLIDER_ANCHORS  # 16
CH_SV = CH_SLIDES + 1                         # 17: slider-velocity timeline

AUDIO = AudioConfig()
