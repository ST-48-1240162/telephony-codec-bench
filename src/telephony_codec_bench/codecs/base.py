"""Codec round-trip protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class RoundtripStats:
    codec: str
    token_count: int | None = None
    code_levels: list[int] = field(default_factory=list)
    encode_ms: float | None = None
    decode_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "codec": self.codec,
            "token_count": self.token_count,
            "code_levels": self.code_levels,
            "encode_ms": self.encode_ms,
            "decode_ms": self.decode_ms,
        }


class CodecRoundtrip(Protocol):
    name: str
    sample_rate: int

    def roundtrip(self, wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, RoundtripStats]:
        """Encode then decode; return reconstructed mono float32 at ``self.sample_rate``."""
