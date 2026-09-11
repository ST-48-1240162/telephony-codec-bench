"""Audio helpers: resampling."""

from __future__ import annotations

import numpy as np


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
