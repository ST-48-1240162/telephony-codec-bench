"""Reconstruction metrics: SNR, STOI (optional PESQ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import resample_mono


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


def _pesq_mode(sample_rate: int, *, pesq_mode: str) -> str | None:
    if pesq_mode == "off":
        return None
    if pesq_mode == "nb":
        return "nb"
    if pesq_mode == "wb":
        return "wb"
    # auto: telephony narrowband at 8 kHz, wideband otherwise
    return "nb" if sample_rate <= 8000 else "wb"


def _pesq_sample_rate(mode: str) -> int:
    return 8000 if mode == "nb" else 16000


def compute_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    *,
    use_pesq: bool = False,
    pesq_mode: str = "auto",
    eval_sr: int | None = None,
) -> MetricResult:
    """Compute metrics, optionally in a telephony evaluation band (``eval_sr=8000``)."""
    metric_sr = eval_sr if eval_sr is not None else sample_rate
    ref = resample_mono(reference, sample_rate, metric_sr)
    est = resample_mono(estimate, sample_rate, metric_sr)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]

    stoi_val: float | None = None
    try:
        from pystoi import stoi

        stoi_val = float(stoi(ref, est, metric_sr, extended=False))
    except ImportError:
        pass

    pesq_val: float | None = None
    if use_pesq:
        mode = _pesq_mode(metric_sr, pesq_mode=pesq_mode)
        if mode is not None:
            try:
                from pesq import pesq

                pesq_sr = _pesq_sample_rate(mode)
                ref_p = resample_mono(ref, metric_sr, pesq_sr)
                est_p = resample_mono(est, metric_sr, pesq_sr)
                n_p = min(len(ref_p), len(est_p))
                pesq_val = float(pesq(pesq_sr, ref_p[:n_p], est_p[:n_p], mode))
            except Exception:
                pesq_val = None

    return MetricResult(snr_db=snr_db(ref, est), stoi=stoi_val, pesq=pesq_val)
