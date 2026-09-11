"""Reconstruction metrics: SNR, STOI (optional PESQ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    snr_db: float
    stoi: float | None
    pesq: float | None


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    noise = ref - est
    sig_pow = np.mean(ref**2) + 1e-12
    noise_pow = np.mean(noise**2) + 1e-12
    return float(10.0 * np.log10(sig_pow / noise_pow))


def compute_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    *,
    use_pesq: bool = False,
) -> MetricResult:
    ref = reference.astype(np.float32)
    est = estimate.astype(np.float32)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]

    stoi_val: float | None = None
    try:
        from pystoi import stoi

        stoi_val = float(stoi(ref, est, sample_rate, extended=False))
    except ImportError:
        pass

    pesq_val: float | None = None
    if use_pesq:
        try:
            from pesq import pesq

            mode = "nb" if sample_rate <= 8000 else "wb"
            pesq_val = float(pesq(sample_rate, ref, est, mode))
        except Exception:
            pesq_val = None

    return MetricResult(snr_db=snr_db(ref, est), stoi=stoi_val, pesq=pesq_val)
