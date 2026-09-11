"""Codec registry."""

from __future__ import annotations

from .base import CodecRoundtrip


def available_codecs() -> list[str]:
    names: list[str] = []
    try:
        import snac  # noqa: F401

        names.append("snac_24khz")
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401

        names.append("encodec_24khz")
    except ImportError:
        pass
    return names


def get_codec(name: str, *, device: str = "cpu") -> CodecRoundtrip:
    if name == "snac_24khz":
        from .snac import SnacCodec

        return SnacCodec(device=device)
    if name == "encodec_24khz":
        from .encodec import EnCodecCodec

        return EnCodecCodec(device=device)
    raise ValueError(f"unknown codec {name!r}; available: {available_codecs()}")
