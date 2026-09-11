"""SNAC coarse-scale-only ablation (token-rate vs quality trade-off)."""

from __future__ import annotations

import numpy as np

from .base import RoundtripStats
from .snac import SnacCodec


class SnacCoarseCodec:
    name = "snac_24khz_coarse"
    sample_rate = 24000

    def __init__(self, device: str = "cpu") -> None:
        self._inner = SnacCodec(device=device)

    def roundtrip(self, wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, RoundtripStats]:
        recon, stats = self._inner.roundtrip(wav, sample_rate, coarse_only=True)
        stats.codec = self.name
        return recon, stats
