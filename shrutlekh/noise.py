"""Add background noise at an exact signal-to-noise ratio (SNR).

SNR in decibels = 10 * log10(signal power / noise power).
20 dB is a quiet room, 10 dB a busy office, 0 dB means the noise is as loud
as the speech.
"""

from __future__ import annotations

import numpy as np

NOISE_KINDS = ("white", "pink")


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise made by shaping white noise in the frequency domain.
    It has more low-frequency energy, closer to fans, traffic and crowd hum."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(len(spectrum), dtype=np.float64)
    freqs[0] = 1.0
    spectrum /= np.sqrt(freqs)
    return np.fft.irfft(spectrum, n)


def make_noise(n: int, kind: str = "white", seed: int = 0) -> np.ndarray:
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}; choose from {NOISE_KINDS}")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) if kind == "white" else _pink(n, rng)
    return noise.astype(np.float32)


def power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64)))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    return 10.0 * np.log10(power(signal) / power(noise))


def add_noise(signal: np.ndarray, snr: float, kind: str = "white", seed: int = 0) -> np.ndarray:
    """Return signal + noise, with the noise scaled so the mix has exactly `snr` dB SNR."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0 or power(signal) == 0.0:
        return signal.copy()
    noise = make_noise(signal.size, kind, seed)
    target_noise_power = power(signal) / (10.0 ** (snr / 10.0))
    noise *= np.float32(np.sqrt(target_noise_power / power(noise)))
    mixed = signal + noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:  # scale both together so clipping never changes the SNR
        mixed /= peak
    return mixed.astype(np.float32)
