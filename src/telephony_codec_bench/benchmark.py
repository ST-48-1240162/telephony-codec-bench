"""Run codec × degradation benchmark over waveforms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .degrade import Preset, apply_preset
from .metrics import MetricResult, compute_metrics
from .codecs.base import CodecRoundtrip
from .codecs.registry import get_codec


@dataclass
class SampleResult:
    preset: str
    codec: str
    metrics: MetricResult
    stats: dict
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "codec": self.codec,
            "source": self.source,
            "snr_db": self.metrics.snr_db,
            "stoi": self.metrics.stoi,
            "pesq": self.metrics.pesq,
            "stats": self.stats,
        }


@dataclass
class BenchmarkReport:
    samples: list[SampleResult] = field(default_factory=list)

    def summary(self) -> dict:
        """Mean metrics grouped by (codec, preset)."""
        buckets: dict[tuple[str, str], list[SampleResult]] = {}
        for s in self.samples:
            buckets.setdefault((s.codec, s.preset), []).append(s)
        out: dict[str, dict] = {}
        for (codec, preset), rows in sorted(buckets.items()):
            key = f"{codec}|{preset}"
            stoi_vals = [r.metrics.stoi for r in rows if r.metrics.stoi is not None]
            pesq_vals = [r.metrics.pesq for r in rows if r.metrics.pesq is not None]
            out[key] = {
                "codec": codec,
                "preset": preset,
                "n": len(rows),
                "snr_db_mean": float(np.mean([r.metrics.snr_db for r in rows])),
                "stoi_mean": float(np.mean(stoi_vals)) if stoi_vals else None,
                "pesq_mean": float(np.mean(pesq_vals)) if pesq_vals else None,
                "encode_ms_mean": _mean_or_none([r.stats.get("encode_ms") for r in rows]),
                "decode_ms_mean": _mean_or_none([r.stats.get("decode_ms") for r in rows]),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "summary": self.summary(),
        }


def _mean_or_none(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def _preset_label(preset: Preset | str) -> str:
    return preset.value if isinstance(preset, Preset) else Preset(preset).value


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    return np.asarray(wav), int(sr)


def _load_noisekit_manifest(data_dir: Path) -> list[tuple[Path, str]] | None:
    """Return (wav path, preset) rows when ``data_dir`` is a noisekit output folder."""
    manifest = data_dir / "metadata.jsonl"
    if not manifest.is_file():
        return None
    rows: list[tuple[Path, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        rel = entry.get("file_name")
        preset = entry.get("preset")
        if not rel or not preset:
            continue
        path = data_dir / rel
        if path.is_file():
            rows.append((path, str(preset)))
    return rows or None


def _benchmark_wav(
    report: BenchmarkReport,
    *,
    path: Path,
    preset: str,
    wav: np.ndarray,
    sr: int,
    codec_names: list[str],
    device: str,
    use_pesq: bool,
) -> None:
    degraded = np.asarray(wav)
    for codec_name in codec_names:
        codec = get_codec(codec_name, device=device)
        recon, stats = codec.roundtrip(degraded, sr)
        eval_sr = getattr(codec, "sample_rate", sr)
        ref_eval = degraded
        if eval_sr != sr:
            import torch
            import torchaudio

            ref_t = torch.from_numpy(degraded.astype(np.float32)).unsqueeze(0)
            ref_eval = torchaudio.functional.resample(ref_t, sr, eval_sr).squeeze().numpy()
        m = compute_metrics(ref_eval, recon, eval_sr, use_pesq=use_pesq)
        report.samples.append(
            SampleResult(
                preset=preset,
                codec=codec.name,
                metrics=m,
                stats=stats.to_dict(),
                source=str(path),
            )
        )


def run_folder_benchmark(
    data_dir: Path,
    *,
    codec_names: list[str],
    presets: list[Preset | str] | None = None,
    max_samples: int | None = None,
    device: str = "cpu",
    use_pesq: bool = False,
) -> BenchmarkReport:
    """Benchmark WAV files under ``data_dir`` (flat or noisekit ``metadata.jsonl`` layout)."""
    report = BenchmarkReport()
    noisekit_rows = _load_noisekit_manifest(data_dir)
    if noisekit_rows is not None:
        if max_samples is not None:
            noisekit_rows = noisekit_rows[:max_samples]
        for path, preset in noisekit_rows:
            ref, sr = _load_wav(path)
            _benchmark_wav(
                report,
                path=path,
                preset=preset,
                wav=ref,
                sr=sr,
                codec_names=codec_names,
                device=device,
                use_pesq=use_pesq,
            )
        return report

    presets = presets or list(Preset)
    wavs = sorted(data_dir.rglob("*.wav"))
    if max_samples is not None:
        wavs = wavs[:max_samples]
    if not wavs:
        raise FileNotFoundError(f"no WAV files under {data_dir}")

    for path in wavs:
        ref, sr = _load_wav(path)
        for preset in presets:
            degraded = apply_preset(ref, sr, preset, seed=hash(path.name) % 2**31)
            _benchmark_wav(
                report,
                path=path,
                preset=_preset_label(preset),
                wav=degraded,
                sr=sr,
                codec_names=codec_names,
                device=device,
                use_pesq=use_pesq,
            )
    return report
