"""Codec registry."""

from __future__ import annotations

import re

from .base import CodecRoundtrip

_CODEC_ALIASES: dict[str, tuple[str, dict]] = {
    "passthrough": ("passthrough", {}),
    "snac_24khz": ("snac_24khz", {}),
    "snac_24khz_coarse": ("snac_24khz_coarse", {}),
    "encodec_24khz": ("encodec_24khz", {}),
    "encodec_24khz_bw6": ("encodec_24khz", {"bandwidth": 6.0}),
    "encodec_24khz_bw12": ("encodec_24khz", {"bandwidth": 12.0}),
}


def available_codecs() -> list[str]:
    names = ["passthrough"]
    try:
        import snac  # noqa: F401

        names.extend(["snac_24khz", "snac_24khz_coarse"])
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401

        names.extend(["encodec_24khz", "encodec_24khz_bw6", "encodec_24khz_bw12"])
    except ImportError:
        pass
    return names


def _parse_codec_name(name: str) -> tuple[str, dict]:
    if name in _CODEC_ALIASES:
        return _CODEC_ALIASES[name]
    m = re.fullmatch(r"encodec_24khz_bw(\d+(?:\.\d+)?)", name)
    if m:
        return "encodec_24khz", {"bandwidth": float(m.group(1))}
    return name, {}


def get_codec(name: str, *, device: str = "cpu", **overrides) -> CodecRoundtrip:
    base, kwargs = _parse_codec_name(name)
    kwargs.update(overrides)

    if base == "passthrough":
        from .passthrough import PassthroughCodec

        return PassthroughCodec()
    if base == "snac_24khz":
        from .snac import SnacCodec

        return SnacCodec(device=device)
    if base == "snac_24khz_coarse":
        from .snac_coarse import SnacCoarseCodec

        return SnacCoarseCodec(device=device)
    if base == "encodec_24khz":
        from .encodec import EnCodecCodec

        return EnCodecCodec(device=device, **kwargs)
    raise ValueError(f"unknown codec {name!r}; available: {available_codecs()}")
