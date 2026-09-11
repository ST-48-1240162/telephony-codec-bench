"""SNAC multi-scale neural codec (Bland-adjacent audio tokenizer family)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torchaudio


from .base import RoundtripStats


class SnacCodec:
    name = "snac_24khz"
    sample_rate = 24000

    def __init__(self, device: str = "cpu") -> None:
        from snac import SNAC

        self.device = torch.device(device)
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

    def roundtrip(
        self,
        wav: np.ndarray,
        sample_rate: int,
        *,
        coarse_only: bool = False,
    ) -> tuple[np.ndarray, RoundtripStats]:
        mono = wav.mean(axis=0) if wav.ndim > 1 else wav
        tensor = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if sample_rate != self.sample_rate:
            tensor = torchaudio.functional.resample(tensor, sample_rate, self.sample_rate)
        tensor = tensor.to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            codes = self.model.encode(tensor)
        encode_ms = (time.perf_counter() - t0) * 1000.0

        decode_codes = codes
        if coarse_only and len(codes) > 1:
            # Ablation: keep coarsest scale only; zero fine-scale codebooks.
            decode_codes = [codes[0]] + [torch.zeros_like(c) for c in codes[1:]]

        t1 = time.perf_counter()
        with torch.inference_mode():
            recon = self.model.decode(decode_codes)
        decode_ms = (time.perf_counter() - t1) * 1000.0

        out = recon.squeeze().detach().cpu().numpy().astype(np.float32)
        levels = [int(c.shape[1]) for c in codes]
        stats = RoundtripStats(
            codec=self.name + ("_coarse" if coarse_only else ""),
            token_count=int(sum(levels)),
            code_levels=levels,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
        )
        return out, stats
