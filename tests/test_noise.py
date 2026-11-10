import numpy as np
import pytest

from shrutlekh.noise import add_noise, make_noise, snr_db


def tone(seconds=1.0, sr=16000, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.mark.parametrize("snr", [20.0, 10.0, 5.0, 0.0])
@pytest.mark.parametrize("kind", ["white", "pink"])
def test_added_noise_has_the_requested_snr(snr, kind):
    clean = tone(amp=0.05)  # quiet enough that the mix never needs rescaling
    mixed = add_noise(clean, snr, kind, seed=1)
    assert snr_db(clean, mixed - clean) == pytest.approx(snr, abs=0.05)


def test_loud_mix_is_rescaled_without_clipping_and_keeps_its_snr():
    clean = tone(amp=0.95)
    mixed = add_noise(clean, 0.0, "white", seed=2)
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6
    # speech and noise are scaled down together, so the ratio between them is unchanged
    est_clean = clean * (np.dot(mixed, clean) / np.dot(clean, clean))
    assert snr_db(est_clean, mixed - est_clean) == pytest.approx(0.0, abs=0.3)


def test_same_seed_same_noise():
    assert np.array_equal(make_noise(100, "pink", 7), make_noise(100, "pink", 7))


def test_silence_is_left_alone():
    silent = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(add_noise(silent, 10), silent)


def test_unknown_kind():
    with pytest.raises(ValueError):
        make_noise(10, "brown")
