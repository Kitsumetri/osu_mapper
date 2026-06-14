from src.difficulty import SR_BANDS, sr_bucket, star_rating


def test_sr_bucket_bands():
    assert sr_bucket(1.5) == "Easy"
    assert sr_bucket(2.6) == "Normal"
    assert sr_bucket(3.7) == "Hard"
    assert sr_bucket(5.0) == "Insane"
    assert sr_bucket(6.0) == "Expert"
    assert sr_bucket(8.0) == "Expert+"


def test_sr_bands_contiguous():
    # bands must tile [0, inf) with no gaps/overlaps
    for (_, hi, _), (lo2, *_ ) in zip(SR_BANDS, SR_BANDS[1:]):
        assert hi == lo2


def test_star_rating_on_sample(sample_osu):
    sr = star_rating(sample_osu)
    # rosu-pp may or may not be installed; if it is, a tiny map has a low SR
    if sr is not None:
        assert 0.0 <= sr < 10.0
