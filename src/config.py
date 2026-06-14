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


# Map-signal channel layout (the diffusion target). v3: 10 channels.
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
]
N_SIGNAL_CHANNELS = len(SIGNAL_CHANNELS)

# Channel indices grouped by decode behaviour.
CH_ONSET, CH_SLIDER, CH_SPINNER, CH_NEWCOMBO, CH_CURX, CH_CURY = 0, 1, 2, 3, 4, 5
CH_KIAI, CH_WHISTLE, CH_FINISH, CH_CLAP = 6, 7, 8, 9

AUDIO = AudioConfig()
