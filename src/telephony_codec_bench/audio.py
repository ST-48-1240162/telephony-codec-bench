"""Audio helpers: resampling and noisekit manifest utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

NOISEKIT_PRESETS = ("clean_reference", "telecom", "noise_telecom")


def utterance_id_from_path(path: Path) -> str:
    """Extract utterance id from noisekit ``{id}_{preset}.wav`` filenames."""
    stem = path.stem
    for preset in sorted(NOISEKIT_PRESETS, key=len, reverse=True):
        suffix = f"_{preset}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resample_mono(wav: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    if sample_rate == target_sr:
        mono = wav.mean(axis=0) if wav.ndim > 1 else wav
        return mono.astype(np.float32, copy=False)
    import torch
    import torchaudio

    mono = wav.mean(axis=0) if wav.ndim > 1 else wav
    tensor = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0)
    out = torchaudio.functional.resample(tensor, sample_rate, target_sr).squeeze().numpy()
    return out.astype(np.float32, copy=False)


def limit_noisekit_rows(
    rows: list[tuple[Path, str]],
    *,
    max_rows: int | None = None,
    max_utterances: int | None = None,
) -> list[tuple[Path, str]]:
    """Limit manifest rows, optionally keeping balanced preset coverage per utterance."""
    if max_utterances is not None:
        grouped: dict[str, list[tuple[Path, str]]] = {}
        order: list[str] = []
        for path, preset in rows:
            uid = utterance_id_from_path(path)
            if uid not in grouped:
                grouped[uid] = []
                order.append(uid)
            grouped[uid].append((path, preset))
        out: list[tuple[Path, str]] = []
        for uid in order[:max_utterances]:
            out.extend(grouped[uid])
        return out
    if max_rows is not None:
        return rows[:max_rows]
    return rows
