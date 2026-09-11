# Telephony Codec Benchmark: Colab 教程

> 面向 **不在本机跑 GPU** 的场景：在 **Google Colab T4** 上生成 telephony 数据并产出 SNAC vs EnCodec 可写进 CV 的数字。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ST-48-1240162/telephony-codec-bench/blob/main/docs/Telephony_Codec_Bench.ipynb)

**一键 notebook：** [Telephony_Codec_Bench.ipynb](./Telephony_Codec_Bench.ipynb)

---

## 产出目标（L2 面试可用）

| 指标 | 来源 | CV 示例 |
|------|------|---------|
| STOI mean | `reports/benchmark.csv` | `STOI 0.82 → 0.71 under telecom (SNAC 24k)` |
| PESQ mean | 同上（`--pesq`） | `WB-PESQ 3.1 clean / 2.4 telecom` |
| Token count | JSON `stats.token_count` | `SNAC hierarchical codes ~N tokens/utt` |
| Latency | JSON `encode_ms` / `decode_ms` | `SNAC encode+decode ~12 ms/1s clip (T4)` |

---

## 方案 A: Google Colab（推荐）

### 1. Runtime

**Runtime → Change runtime type → T4 GPU**

### 2. Cell 1: 克隆

```python
REPO = "https://github.com/ST-48-1240162/telephony-codec-bench.git"
!git clone {REPO}
%cd telephony-codec-bench
```

### 3. Cell 2: 依赖

版本 pin 见 [`docs/colab-requirements.txt`](./colab-requirements.txt) 与 [`.python-env-compat.toml`](../.python-env-compat.toml)。Install cell 会跑 `pip check` + `scripts/verify_colab_env.py`（读取 manifest）。

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

**Colab 兼容策略**

| 冲突 | 处理 |
|------|------|
| `datasets` 4.x 要 torchcodec | pin `datasets>=2.20,<4.0` |
| `datasets` 与 `fsspec==2025.12.0` 冲突 | pin `fsspec==2025.3.0`（datasets 3.x 上限） |
| `numpy.ufunc` decode 报错 | pin `numpy==2.2.2` + `pyarrow>=19` + `numba>=0.61` 一起重装 |
| `pip install [bench]` 升级 torch | Colab 用 `-e . --no-deps` + requirements 文件 |
| torchvision 与 torch 版本不一致 | 只从 `download.pytorch.org` 装 vision/audio |
| `transformers` 5.x / hub 2.x | pin `transformers<5`, `huggingface-hub<1` |
| `audiomentations` / librosa | pin `librosa>=0.10.1,<0.12.0` |

**本地 GPU 机器**（非 Colab）：先装匹配 CUDA 的 torch/torchaudio，再 `pip install -e '.[bench]'`。

### 4. Cell 3: 生成 telephony 数据（noisekit）

200 条 × 3 preset ≈ 45-90 min（含 NISQA）。快速试跑可 `--samples 50`。

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

> LibriTTS 备选（需 HuggingFace license）：`--dataset mythicinfinity/libritts --config clean --split test.clean`

### 5. Cell 4: 跑 benchmark

```python
!python scripts/run_benchmark.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-samples 200 \
  --pesq \
  --out reports/benchmark.json
```

### 6. Cell 5: 查看 summary

```python
import json
from pathlib import Path
p = Path("reports/benchmark.json")
data = json.loads(p.read_text())
for k, row in data["summary"].items():
    print(row)
```

下载 `reports/benchmark.json` 和 `reports/benchmark.csv` 存档。

### 7. Cell 6（可选 L2.5）: SNAC coarse-only ablation

```python
!python scripts/run_snac_ablation.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-samples 50 \
  --out reports/snac_ablation.json
```

面试话术：fine-scale tokens 掉多少 STOI，证明你理解 hierarchical tokenizer。

---

## 写进 CV 的 research question

> Under simulated PSTN degradation (8 kHz, μ-law, ambient noise), how much intelligibility do **SNAC multi-scale tokens** retain vs **EnCodec RVQ** on round-trip reconstruction, and what is the encode/decode latency trade-off?

## 参考 repo

| Repo | 用途 |
|------|------|
| [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac) | Bland-adjacent tokenizer |
| [facebookresearch/encodec](https://github.com/facebookresearch/encodec) | RVQ baseline |
| [karamouche/noisekit](https://github.com/karamouche/noisekit) | Telephony dataset |
| [voidful/Codec-SUPERB](https://github.com/voidful/Codec-SUPERB) | Metric methodology |
