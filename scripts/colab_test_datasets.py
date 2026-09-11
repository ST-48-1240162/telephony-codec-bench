#!/usr/bin/env python3
"""Smoke-test HF datasets audio decode (catches numpy/pyarrow ufunc bugs)."""

from __future__ import annotations

import sys


def main() -> int:
    import numpy as np

    print("numpy", np.__version__)

    import pyarrow

    print("pyarrow", pyarrow.__version__)

    from datasets import load_dataset, Audio

    print("loading google/fleurs (1 row)...")
    ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=True))
    row = next(iter(ds))
    audio = row["audio"]
    assert "array" in audio, f"unexpected audio payload: {type(audio)}"
    print("fleurs ok", audio["sampling_rate"], len(audio["array"]))

    print("loading musan noise (1 row)...")
    musan = load_dataset("Aynursusuz/musan-audio-dataset", split="train", streaming=True)
    musan = musan.cast_column("audio", Audio(decode=True))
    nrow = next(iter(musan))
    na = nrow["audio"]
    assert "array" in na, f"unexpected musan audio: {type(na)}"
    print("musan ok", na["sampling_rate"], len(na["array"]))
    print("datasets decode OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
