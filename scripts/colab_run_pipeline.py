#!/usr/bin/env python3
"""Generate noisekit data + run SNAC vs EnCodec benchmark on Colab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "telephony_speech"
MUSAN_DIR = Path("/root/.cache/noisekit/noise/musan_ambient")


def run(cmd: list[str] | str, *, cwd: Path = ROOT) -> None:
    if isinstance(cmd, str):
        print(f"\n$ {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    else:
        print(f"\n$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=cwd)


def prefetch_musan() -> None:
    run([sys.executable, "scripts/colab_prefetch_musan.py"])


def generate(samples: int) -> None:
    prefetch_musan()
    cmd = [
            "noisekit",
            "generate",
            "--dataset",
            "google/fleurs",
            "--config",
            "en_us",
            "--split",
            "test",
            "--samples",
            str(samples),
            "--preset",
            "clean_reference",
            "--preset",
            "telecom",
            "--preset",
            "noise_telecom",
            "--output",
            str(DATA_DIR),
            "--seed",
            "42",
            "--noise-dir",
            str(MUSAN_DIR),
        ]
    run(cmd)


def benchmark(samples: int, out: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--data-dir",
            str(DATA_DIR),
            "--device",
            "cuda",
            "--max-samples",
            str(samples),
            "--eval-sr",
            "8000",
            "--pesq",
            "--pesq-mode",
            "nb",
            "--out",
            str(out),
        ]
    )


def print_summary(path: Path) -> None:
    import json

    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n=== {path.name} ===")
    print(f"{'codec|preset':40} {'STOI':>8} {'PESQ':>8} {'enc_ms':>10} {'dec_ms':>10}")
    print("-" * 80)
    for key, row in sorted(data["summary"].items()):
        stoi = row.get("stoi_mean")
        pesq = row.get("pesq_mean")
        stoi_s = f"{stoi:8.4f}" if stoi is not None else f"{'n/a':>8}"
        pesq_s = f"{pesq:8.3f}" if pesq is not None else f"{'n/a':>8}"
        enc = row.get("encode_ms_mean") or 0
        dec = row.get("decode_ms_mean") or 0
        print(f"{key:40} {stoi_s} {pesq_s} {enc:10.2f} {dec:10.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="benchmark JSON path (default: reports/benchmark_{samples}.json)",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    out = args.out or (ROOT / "reports" / f"benchmark_{args.samples}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline: samples={args.samples} out={out}", flush=True)
    if not args.skip_generate:
        generate(args.samples)
    if not args.skip_benchmark:
        benchmark(args.samples, out)
    print_summary(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
