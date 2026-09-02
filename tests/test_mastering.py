"""Mastering headroom for the delivery encode.

The bug: audio was limited to exactly the delivery ceiling, but QC measures the finished
MP4, and mux() re-encodes to AAC. A lossy codec reconstructs inter-sample peaks above the
PCM ceiling it was handed - a mixdown limited to -1.0 dBFS measured -0.3 dBFS once encoded
and QC correctly failed the episode. The limiter was never wrong; it was aiming at the
delivery target instead of a master target.
"""
import pytest

from asa.media.audio.mixer import AAC_ENCODE_HEADROOM_DB, master_ceiling_db


def test_master_sits_below_the_delivery_ceiling():
    assert master_ceiling_db(-1.0) == pytest.approx(-2.0)


@pytest.mark.parametrize("delivery", [-0.5, -1.0, -2.0, -3.0])
def test_headroom_is_applied_at_any_ceiling(delivery):
    assert master_ceiling_db(delivery) == pytest.approx(delivery - AAC_ENCODE_HEADROOM_DB)
    assert master_ceiling_db(delivery) < delivery


def test_headroom_covers_the_observed_overshoot():
    # Measured on this pipeline: -1.0 dBFS mixdown came back at -0.3 (0.7 dB overshoot),
    # another run at -0.7 (0.3 dB). The allowance has to cover the worst of those.
    worst_observed = 0.7
    assert AAC_ENCODE_HEADROOM_DB >= worst_observed


def test_headroom_stays_within_qc_tolerance():
    # QC fails above delivery + 0.3. Mastering to delivery - headroom must leave the
    # encoded file inside that band even with zero overshoot, or quiet masters would
    # start failing a *lower* bound that does not exist - but it must not be so deep
    # that the master is needlessly quiet.
    assert 0.5 <= AAC_ENCODE_HEADROOM_DB <= 2.0
