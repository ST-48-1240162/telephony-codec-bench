"""CLI: folder benchmark over telephony-degraded WAVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import run_folder_benchmark
from .codecs.registry import available_codecs
from .cpu_limits import apply_cpu_limits
from .degrade import Preset


def main(argv: list[str] | None = None) -> int:
    apply_cpu_limits(threads=1)
    parser = argparse.ArgumentParser(
        description="Telephony neural codec benchmark (SNAC vs EnCodec)."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--data-dir", type=str, required=True, help="WAV root directory")
    parser.add_argument(
        "--codec",
        action="append",
        dest="codecs",
        help="Codec name (repeatable). Default: snac_24khz + encodec_24khz when installed.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        choices=[p.value for p in Preset],
        help="Degradation preset (repeatable)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--pesq", action="store_true", help="Compute PESQ (needs pesq package)")
    args = parser.parse_args(argv)

    presets = args.presets or [Preset.CLEAN.value, Preset.TELECOM.value]
    codecs = args.codecs or [c for c in ("snac_24khz", "encodec_24khz") if c in available_codecs()]
    if not codecs:
        print("no neural codecs installed; pip install -e '.[bench]'", file=sys.stderr)
        return 2

    report = run_folder_benchmark(
        Path(args.data_dir),
        codec_names=codecs,
        presets=presets,
        max_samples=args.max_samples,
        device=args.device,
        use_pesq=args.pesq,
    )

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, row in payload["summary"].items():
            stoi = row.get("stoi_mean")
            stoi_s = f"{stoi:.3f}" if stoi is not None else "n/a"
            print(f"{row['codec']:16} {row['preset']:18} n={row['n']} STOI={stoi_s} SNR={row['snr_db_mean']:.1f}dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
