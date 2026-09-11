"""Identity baseline: no neural codec (upper bound for metrics)."""

from __future__ import annotations

import numpy as np

from .base import RoundtripStats


class PassthroughCodec:
    name = "passthrough"
    sample_rate = 0  # follow input rate

    def roundtrip(self, wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, RoundtripStats]:
        mono = wav.mean(axis=0) if wav.ndim > 1 else wav
        out = mono.astype(np.float32, copy=True)
        stats = RoundtripStats(codec=self.name, token_count=0, encode_ms=0.0, decode_ms=0.0)
        return out, stats
