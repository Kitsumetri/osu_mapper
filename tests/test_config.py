from src.config import AUDIO, SIGNAL_CHANNELS, N_SIGNAL_CHANNELS


def test_frame_time_roundtrip():
    for t in (0.0, 123.4, 60000.0):
        f = AUDIO.time_to_frame(t)
        assert abs(AUDIO.frame_to_time(f) - t) < 1e-6


def test_frame_rate_consistency():
    assert abs(AUDIO.frame_rate - AUDIO.sample_rate / AUDIO.hop_length) < 1e-9
    assert abs(AUDIO.ms_per_frame - 1000.0 / AUDIO.frame_rate) < 1e-9


def test_channel_count_matches():
    assert N_SIGNAL_CHANNELS == len(SIGNAL_CHANNELS)
    assert SIGNAL_CHANNELS[0] == "onset"
