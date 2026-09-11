"""Meta EnCodec RVQ baseline."""

from __future__ import annotations

import time

import numpy as np
import torch
import torchaudio

from .base import RoundtripStats


class EnCodecCodec:
    sample_rate = 24000

    def __init__(self, device: str = "cpu", *, bandwidth: float | None = None) -> None:
        from transformers import EncodecModel, AutoProcessor

        self.device = torch.device(device)
        self.bandwidth = bandwidth
        if bandwidth is None:
            self.name = "encodec_24khz"
        else:
            bw = int(bandwidth) if float(bandwidth).is_integer() else bandwidth
            self.name = f"encodec_24khz_bw{bw}"
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval().to(self.device)

    def roundtrip(self, wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, RoundtripStats]:
        mono = wav.mean(axis=0) if wav.ndim > 1 else wav
        tensor = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0)
        if sample_rate != self.sample_rate:
            tensor = torchaudio.functional.resample(tensor, sample_rate, self.sample_rate)
        # HF processor expects 1-D mono float32 (samples,), not (1, samples).
        inputs = self.processor(
            raw_audio=tensor.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        encode_kwargs: dict = {}
        if self.bandwidth is not None:
            encode_kwargs["bandwidth"] = self.bandwidth

        t0 = time.perf_counter()
        with torch.inference_mode():
            encoded = self.model.encode(
                inputs["input_values"],
                padding_mask=inputs["padding_mask"],
                **encode_kwargs,
            )
        encode_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with torch.inference_mode():
            decoded = self.model.decode(
                encoded.audio_codes,
                encoded.audio_scales,
                padding_mask=inputs["padding_mask"],
                last_frame_pad_length=getattr(encoded, "last_frame_pad_length", 0) or 0,
            )
        decode_ms = (time.perf_counter() - t1) * 1000.0

        audio = decoded.audio_values if hasattr(decoded, "audio_values") else decoded[0]
        out = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        codes = encoded.audio_codes
        token_count = int(codes.numel()) if codes is not None else None
        stats = RoundtripStats(
            codec=self.name,
            token_count=token_count,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
        )
        return out, stats
