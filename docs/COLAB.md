# Telephony Codec Benchmark: Colab Guide

Run the full pipeline on a **Google Colab T4** if you do not have a local GPU. You end up with telephony-degraded WAVs and SNAC vs EnCodec numbers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ST-48-1240162/telephony-codec-bench/blob/main/docs/Telephony_Codec_Bench.ipynb)

**Notebook:** [Telephony_Codec_Bench.ipynb](./Telephony_Codec_Bench.ipynb)

---

## What you get

| Metric | Where | Example line |
|--------|-------|--------------|
| STOI mean | `reports/benchmark.csv` | `STOI 0.82 → 0.71 under telecom (SNAC 24k)` |
| PESQ mean | same file (`--pesq`) | `nb-PESQ 3.1 clean / 2.4 telecom @ 8 kHz` |
| Token count | JSON `stats.token_count` | `SNAC hierarchical codes ~N tokens/utt` |
| Latency | JSON `encode_ms` / `decode_ms` | `SNAC encode+decode ~12 ms/1s clip (T4)` |

---

## Google Colab (recommended)

### 1. Runtime

**Runtime → Change runtime type → T4 GPU**

### 2. Cell 1: Clone

```python
REPO = "https://github.com/ST-48-1240162/telephony-codec-bench.git"
!git clone {REPO}
%cd telephony-codec-bench
```

### 3. Cell 2: Dependencies

Pins are in [`docs/colab-requirements.txt`](./colab-requirements.txt) and [`.python-env-compat.toml`](../.python-env-compat.toml). The install cell runs `pip check` and `scripts/verify_colab_env.py`.

```python
import sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU. Colab: Runtime → Change runtime type → T4 GPU, "
        "then Runtime → Restart session, rerun §0 and this cell."
    )

!apt-get -qq install -y libsndfile1 ffmpeg
!{sys.executable} -m pip install -q -r docs/colab-requirements.txt
!{sys.executable} -m pip install -q --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
!{sys.executable} -m pip install -q -e . --no-deps
!{sys.executable} -m pip install -q "fsspec==2025.3.0" --force-reinstall --no-deps
!{sys.executable} scripts/verify_colab_env.py --no-pip-check
```

**Colab compatibility**

| Problem | Fix |
|---------|-----|
| `datasets` 4.x needs torchcodec | pin `datasets>=2.20,<4.0` |
| `datasets` vs `fsspec==2025.12.0` | pin `fsspec==2025.3.0` (datasets 3.x cap) |
| `numpy.ufunc` decode errors | pin `numpy==2.2.2`, reinstall `pyarrow>=19` and `numba>=0.61` |
| `pip install [bench]` upgrades torch | use `-e . --no-deps` plus the requirements file |
| torchvision / torch mismatch | install vision/audio from `download.pytorch.org` only |
| `transformers` 5.x / hub 2.x | pin `transformers<5`, `huggingface-hub<1` |
| `audiomentations` / librosa | pin `librosa>=0.10.1,<0.12.0` |

On a **local GPU box** (not Colab), install CUDA-matched torch/torchaudio first, then `pip install -e '.[bench]'`.

### 4. Cell 3: Generate telephony data (noisekit)

200 utterances × 3 presets takes about 45-90 min (includes NISQA). For a quick test, use `--samples 50`.

```python
!noisekit generate \
  --dataset google/fleurs \
  --config en_us \
  --split test \
  --samples 200 \
  --preset clean_reference \
  --preset telecom \
  --preset noise_telecom \
  --output ./data/telephony_speech \
  --seed 42
```

> LibriTTS works too but needs a HuggingFace license: `--dataset mythicinfinity/libritts --config clean --split test.clean`

### 5. Cell 4: Run benchmark

```python
!python scripts/run_benchmark.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-samples 200 \
  --eval-sr 8000 \
  --pesq \
  --pesq-mode nb \
  --out reports/benchmark.json
```

### 6. Cell 5: View summary

```python
import json
from pathlib import Path
p = Path("reports/benchmark.json")
data = json.loads(p.read_text())
for k, row in data["summary"].items():
    print(row)
```

Download `reports/benchmark.json` and `reports/benchmark.csv` when you are done.

### 7. Cell 6 (optional): SNAC coarse-only ablation

```python
!python scripts/run_snac_ablation.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-samples 50 \
  --out reports/snac_ablation.json
```

This strips fine-scale SNAC tokens and shows how much STOI you lose. Handy if you want to talk about hierarchical tokenizers in an interview.

---

## Research question

> After simulated PSTN degradation (8 kHz, μ-law, ambient noise), how much intelligibility do **SNAC multi-scale tokens** keep vs **EnCodec RVQ** on a round-trip, and what does encode/decode latency look like?

## Reference repos

| Repo | Why |
|------|-----|
| [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac) | SNAC tokenizer |
| [facebookresearch/encodec](https://github.com/facebookresearch/encodec) | RVQ baseline |
| [karamouche/noisekit](https://github.com/karamouche/noisekit) | Telephony dataset generation |
| [voidful/Codec-SUPERB](https://github.com/voidful/Codec-SUPERB) | Metric methodology |
