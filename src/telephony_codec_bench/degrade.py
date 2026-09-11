"""Telephony-style degradation presets (local fallback; Colab uses noisekit)."""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy import signal


class Preset(str, Enum):
    CLEAN = "clean_reference"
    TELECOM = "telecom"
    NOISE_TELECOM = "noise_telecom"


def _to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav.astype(np.float32, copy=False)
    return wav.mean(axis=0).astype(np.float32, copy=False)


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    n = int(round(len(wav) * target_sr / orig_sr))
    return signal.resample(wav, n).astype(np.float32)


def _mu_law_encode(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def _mu_law_decode(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu


def _telecom_narrowband(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """8 kHz band-limited + mu-law companding, upsampled back (PSTN-ish)."""
    wb = _resample(wav, sample_rate, 8000)
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
    nb = signal.sosfilt(sos, wb)
    nb = _mu_law_decode(_mu_law_encode(nb))
    return _resample(nb, 8000, sample_rate)


def _add_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(wav.shape[0]).astype(np.float32)
    sig_pow = np.mean(wav**2) + 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise *= np.sqrt(noise_pow / (np.mean(noise**2) + 1e-12))
    out = wav + noise
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.95).astype(np.float32)


def apply_preset(
    wav: np.ndarray,
    sample_rate: int,
    preset: Preset | str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return degraded mono float32 waveform at ``sample_rate``."""
    p = Preset(preset) if not isinstance(preset, Preset) else preset
    x = _to_mono(wav)
    if p == Preset.CLEAN:
        return x
    if p == Preset.TELECOM:
        return _telecom_narrowband(x, sample_rate)
    if p == Preset.NOISE_TELECOM:
        noisy = _add_noise(x, snr_db=10.0, seed=seed)
        return _telecom_narrowband(noisy, sample_rate)
    raise ValueError(f"unknown preset: {preset}")
