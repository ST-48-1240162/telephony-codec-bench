# Telephony Codec Benchmark: Colab Guide

> For **GPU runs without a local machine**: generate telephony-degraded data on **Google Colab T4** and produce SNAC vs EnCodec numbers you can cite on a CV.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ST-48-1240162/telephony-codec-bench/blob/main/docs/Telephony_Codec_Bench.ipynb)

**One-click notebook:** [Telephony_Codec_Bench.ipynb](./Telephony_Codec_Bench.ipynb)

---

## Deliverables (L2 interview-ready)

| Metric | Source | CV example |
|------|------|---------|
| STOI mean | `reports/benchmark.csv` | `STOI 0.82 → 0.71 under telecom (SNAC 24k)` |
| PESQ mean | same (`--pesq`) | `nb-PESQ 3.1 clean / 2.4 telecom @ 8 kHz` |
| Token count | JSON `stats.token_count` | `SNAC hierarchical codes ~N tokens/utt` |
| Latency | JSON `encode_ms` / `decode_ms` | `SNAC encode+decode ~12 ms/1s clip (T4)` |

---

## Option A: Google Colab (recommended)

### 1. Runtime

**Runtime → Change runtime type → T4 GPU**

### 2. Cell 1: Clone

```python
REPO = "https://github.com/ST-48-1240162/telephony-codec-bench.git"
!git clone {REPO}
%cd telephony-codec-bench
```

### 3. Cell 2: Dependencies

Version pins: [`docs/colab-requirements.txt`](./colab-requirements.txt) and [`.python-env-compat.toml`](../.python-env-compat.toml). The install cell runs `pip check` + `scripts/verify_colab_env.py` (reads the manifest).

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

| Conflict | Fix |
|------|------|
| `datasets` 4.x requires torchcodec | pin `datasets>=2.20,<4.0` |
| `datasets` vs `fsspec==2025.12.0` | pin `fsspec==2025.3.0` (datasets 3.x upper bound) |
| `numpy.ufunc` decode errors | pin `numpy==2.2.2` + reinstall `pyarrow>=19` + `numba>=0.61` |
| `pip install [bench]` upgrades torch | Colab: `-e . --no-deps` + requirements file |
| torchvision / torch version mismatch | install vision/audio only from `download.pytorch.org` |
| `transformers` 5.x / hub 2.x | pin `transformers<5`, `huggingface-hub<1` |
| `audiomentations` / librosa | pin `librosa>=0.10.1,<0.12.0` |

**Local GPU machine** (not Colab): install CUDA-matched torch/torchaudio first, then `pip install -e '.[bench]'`.

### 4. Cell 3: Generate telephony data (noisekit)

200 utterances × 3 presets ≈ 45–90 min (includes NISQA). Quick smoke: `--samples 50`.

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

> LibriTTS alternative (HuggingFace license required): `--dataset mythicinfinity/libritts --config clean --split test.clean`

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

Download `reports/benchmark.json` and `reports/benchmark.csv` for your records.

### 7. Cell 6 (optional L2.5): SNAC coarse-only ablation

```python
!python scripts/run_snac_ablation.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-samples 50 \
  --out reports/snac_ablation.json
```

Interview angle: quantify how much STOI drops when fine-scale tokens are removed — shows you understand hierarchical tokenizers.

---

## Research question for your CV

> Under simulated PSTN degradation (8 kHz, μ-law, ambient noise), how much intelligibility do **SNAC multi-scale tokens** retain vs **EnCodec RVQ** on round-trip reconstruction, and what is the encode/decode latency trade-off?

## Reference repos

| Repo | Role |
|------|------|
| [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac) | Bland-adjacent tokenizer |
| [facebookresearch/encodec](https://github.com/facebookresearch/encodec) | RVQ baseline |
| [karamouche/noisekit](https://github.com/karamouche/noisekit) | Telephony dataset |
| [voidful/Codec-SUPERB](https://github.com/voidful/Codec-SUPERB) | Metric methodology |
