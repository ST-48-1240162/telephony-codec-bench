#!/usr/bin/env python3
"""Full benchmark driver for Colab / GPU VM. Writes JSON + CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telephony_codec_bench.benchmark import run_folder_benchmark
from telephony_codec_bench.cpu_limits import apply_cpu_limits
from telephony_codec_bench.degrade import Preset


def main() -> int:
    apply_cpu_limits(threads=4)
    parser = argparse.ArgumentParser(description="Telephony codec benchmark (folder mode).")
    parser.add_argument("--data-dir", type=Path, required=True, help="WAV folder (e.g. noisekit output)")
    parser.add_argument(
        "--codec",
        action="append",
        default=["snac_24khz", "encodec_24khz"],
        help="Codec names",
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        default=[p.value for p in Preset],
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pesq", action="store_true")
    parser.add_argument(
        "--pesq-mode",
        choices=["auto", "nb", "wb", "off"],
        default="auto",
        help="PESQ mode (use nb with --eval-sr 8000 for telephony)",
    )
    parser.add_argument(
        "--eval-sr",
        type=int,
        default=None,
        help="Metric evaluation sample rate (e.g. 8000 for telephony band)",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available; falling back to cpu", file=sys.stderr)
            device = "cpu"

    use_pesq = args.pesq and args.pesq_mode != "off"
    report = run_folder_benchmark(
        args.data_dir,
        codec_names=args.codec,
        presets=args.presets,
        max_samples=args.max_samples,
        device=device,
        use_pesq=use_pesq,
        pesq_mode=args.pesq_mode,
        eval_sr=args.eval_sr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "codec",
                "preset",
                "n",
                "stoi_mean",
                "pesq_mean",
                "snr_db_mean",
                "encode_ms_mean",
                "decode_ms_mean",
            ],
        )
        writer.writeheader()
        for row in payload["summary"].values():
            writer.writerow(row)

    print(f"Wrote {args.out}")
    print(f"Wrote {csv_path}")
    for row in payload["summary"].values():
        stoi = row.get("stoi_mean")
        pesq = row.get("pesq_mean")
        stoi_s = f"{stoi:.3f}" if stoi is not None else "n/a"
        pesq_s = f"{pesq:.3f}" if pesq is not None else "n/a"
        print(
            f"  {row['codec']:16} {row['preset']:18} STOI={stoi_s} PESQ={pesq_s} "
            f"encode={row.get('encode_ms_mean')}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
