#!/usr/bin/env python3
"""SNAC hierarchical ablation: full codes vs coarse-only (L2.5 interview story)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telephony_codec_bench.cpu_limits import apply_cpu_limits
from telephony_codec_bench.metrics import compute_metrics
from telephony_codec_bench.codecs.registry import get_codec
import soundfile as sf
import numpy as np
import torch
import torchaudio


def main() -> int:
    apply_cpu_limits(threads=4)
    parser = argparse.ArgumentParser(description="SNAC full vs coarse-only ablation.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("reports/snac_ablation.json"))
    args = parser.parse_args()

    codec = get_codec("snac_24khz", device=args.device)
    wavs = sorted(args.data_dir.rglob("*.wav"))[: args.max_samples]
    rows = []
    for path in wavs:
        ref, sr = sf.read(path, always_2d=False)
        ref = np.asarray(ref)
        degraded = ref  # use noisekit preset subdirs as-is when present
        for coarse_only, tag in ((False, "full"), (True, "coarse")):
            recon, stats = codec.roundtrip(degraded, sr, coarse_only=coarse_only)
            ref_eval = degraded.mean(axis=0) if degraded.ndim > 1 else degraded
            if sr != codec.sample_rate:
                ref_t = torch.from_numpy(ref_eval.astype(np.float32)).unsqueeze(0)
                ref_eval = (
                    torchaudio.functional.resample(ref_t, sr, codec.sample_rate)
                    .squeeze()
                    .numpy()
                )
            m = compute_metrics(ref_eval, recon, codec.sample_rate)
            rows.append(
                {
                    "source": str(path),
                    "variant": tag,
                    "stoi": m.stoi,
                    "snr_db": m.snr_db,
                    "stats": stats.to_dict(),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"samples": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
