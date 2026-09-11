#!/usr/bin/env python3
"""Prefetch MUSAN noise WAVs for noisekit (--noise-dir workaround)."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/root/.cache/noisekit/noise/musan_ambient"),
    )
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob("musan_noise_*.wav"))
    if len(existing) >= args.count:
        print(f"MUSAN cache ready: {len(existing)} files in {args.out}")
        return 0

    ds = load_dataset("Aynursusuz/musan-audio-dataset", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=True))

    saved = 0
    for row in ds:
        audio = row["audio"]
        if not isinstance(audio, dict) or "array" not in audio:
            continue
        out_path = args.out / f"musan_noise_{saved:04d}.wav"
        sf.write(out_path, audio["array"], audio["sampling_rate"], subtype="PCM_16")
        saved += 1
        print(f"saved {out_path.name}")
        if saved >= args.count:
            break

    if saved < args.count:
        raise SystemExit(f"only saved {saved}/{args.count} MUSAN clips")
    print(f"MUSAN cache ready: {saved} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
