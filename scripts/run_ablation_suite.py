#!/usr/bin/env python3
"""Run the recommended eval + inference + ablation benchmark matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_BENCH = ROOT / "scripts" / "run_benchmark.py"

# name -> extra CLI args for run_benchmark.py
SUITE: dict[str, list[str]] = {
    "baseline_codec24k": [],
    "telephony_eval8k": ["--eval-sr", "8000", "--pesq", "--pesq-mode", "nb"],
    "ablation_snac_coarse": ["--codec", "snac_24khz", "--codec", "snac_24khz_coarse"],
    "ablation_encodec_bw": [
        "--codec",
        "encodec_24khz",
        "--codec",
        "encodec_24khz_bw6",
        "--codec",
        "encodec_24khz_bw12",
    ],
    "with_passthrough": ["--codec", "passthrough", "--codec", "snac_24khz", "--codec", "encodec_24khz"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-utterances", type=int, default=50)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=list(SUITE),
        choices=list(SUITE),
        help="Which ablation configs to run",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/ablations"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    for name in args.runs:
        out = args.out_dir / f"{name}.json"
        cmd = [
            sys.executable,
            str(RUN_BENCH),
            "--data-dir",
            str(args.data_dir),
            "--device",
            args.device,
            "--max-utterances",
            str(args.max_utterances),
            "--out",
            str(out),
            *SUITE[name],
        ]
        print(f"\n=== {name} ===", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        index[name] = str(out)

    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
