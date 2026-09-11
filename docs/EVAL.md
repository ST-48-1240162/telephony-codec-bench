# Eval + ablation walkthrough

Optional extension on branch [`feat/eval-ablations`](https://github.com/ST-48-1240162/telephony-codec-bench/tree/feat/eval-ablations). The default benchmark on `main` (see [COLAB.md](./COLAB.md)) already compares SNAC vs EnCodec with 8 kHz nb metrics. Use this guide when you want the **full eval matrix**: passthrough upper bound, telephony eval domain, SNAC coarse-only, and EnCodec bandwidth ablations.

**Driver:** `scripts/run_ablation_suite.py` (inference only; no training).

---

## What gets run

| Run name | What it isolates | Codecs / flags |
|----------|------------------|----------------|
| `baseline_codec24k` | Default round-trip @ 24 kHz scoring | SNAC + EnCodec (defaults) |
| `telephony_eval8k` | Phone-band metric domain | same codecs + `--eval-sr 8000 --pesq --pesq-mode nb` |
| `ablation_snac_coarse` | SNAC full vs coarse-only tokens | `snac_24khz`, `snac_24khz_coarse` |
| `ablation_encodec_bw` | EnCodec target bandwidth | `encodec_24khz`, `_bw6`, `_bw12` |
| `with_passthrough` | Codec loss vs no-op upper bound | `passthrough`, SNAC, EnCodec |

Each run writes `reports/ablations/<run_name>.json` plus a CSV sibling from `run_benchmark.py`. An `index.json` lists all outputs.

Use **`--max-utterances`** (balanced presets per utterance), not `--max-samples` (raw manifest rows).

---

## Prerequisites

1. Checkout the eval branch:

```sh
git clone https://github.com/ST-48-1240162/telephony-codec-bench.git
cd telephony-codec-bench
git checkout feat/eval-ablations
```

2. Install the bench extra (same as [COLAB.md](./COLAB.md)): matching `torch` / `torchaudio`, then `pip install -e '.[bench]'`.

3. Telephony WAVs under `./data/telephony_speech` with `metadata.jsonl` from noisekit. If you already ran the main Colab pipeline, reuse that folder. Otherwise generate data first (Colab §4 below, or [COLAB.md](./COLAB.md)).

4. **GPU recommended.** Codec inference on CPU works but is very slow (many hours for 50 utterances × 5 runs).

---

## Colab (T4)

### 1. Runtime

**Runtime → Change runtime type → T4 GPU**

### 2. Clone eval branch

```python
REPO = "https://github.com/ST-48-1240162/telephony-codec-bench.git"
!git clone -b feat/eval-ablations {REPO}
%cd telephony-codec-bench
```

### 3. Install

Copy the install cell from [COLAB.md §2](./COLAB.md) (system libs, `docs/colab-requirements.txt`, CUDA-matched `torch`/`torchaudio`, `pip install -e . --no-deps`, then `verify_colab_env.py --no-pip-check`). On `feat/eval-ablations`, `.python-env-compat.toml` sets `requires_cuda = true`, so **use a T4 runtime** before running verify — CPU-only sessions will fail here even though `run_benchmark.py` can fall back to CPU.

### 4. Generate telephony data (if needed)

Skip if `./data/telephony_speech/metadata.jsonl` already exists. Same as [COLAB.md §4](./COLAB.md):

```python
SAMPLES = 50  # quick test; use 200 for a larger matrix

import subprocess, sys
subprocess.run([
    "noisekit", "generate",
    "--dataset", "google/fleurs",
    "--config", "en_us",
    "--split", "test",
    "--samples", str(SAMPLES),
    "--preset", "clean_reference",
    "--preset", "telecom",
    "--preset", "noise_telecom",
    "--output", "./data/telephony_speech",
    "--seed", "42",
], check=True)
```

50 utterances × 3 presets takes about 20–45 min on T4 (includes NISQA).

### 5. Run the ablation suite

```python
!python scripts/run_ablation_suite.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-utterances 50 \
  --out-dir reports/ablations
```

Subset only (example: telephony eval + SNAC coarse):

```python
!python scripts/run_ablation_suite.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-utterances 50 \
  --runs telephony_eval8k ablation_snac_coarse \
  --out-dir reports/ablations
```

### 6. Download results

Grab `reports/ablations/` (`*.json`, `*.csv`, `index.json`).

---

## Local GPU

```sh
git checkout feat/eval-ablations
pip install -e '.[bench]'

# after noisekit data exists:
python scripts/run_ablation_suite.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-utterances 50 \
  --out-dir reports/ablations
```

Single-run equivalent (manual — must match `SUITE` flags in `run_ablation_suite.py`):

```sh
# ablation_snac_coarse (24 kHz metric domain, default codecs subset)
python scripts/run_benchmark.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-utterances 50 \
  --codec snac_24khz --codec snac_24khz_coarse \
  --out reports/ablations/ablation_snac_coarse.json

# telephony_eval8k (8 kHz nb + PESQ; default SNAC + EnCodec)
python scripts/run_benchmark.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --max-utterances 50 \
  --eval-sr 8000 --pesq --pesq-mode nb \
  --out reports/ablations/telephony_eval8k.json
```

---

## Read the outputs

```python
import json
from pathlib import Path

idx = json.loads(Path("reports/ablations/index.json").read_text())
for name, path in idx.items():
    data = json.loads(Path(path).read_text())
    print(f"\n=== {name} ===")
    for key, row in sorted(data["summary"].items()):
        stoi = row.get("stoi_mean")
        pesq = row.get("pesq_mean")
        print(f"{key:40} STOI={stoi} PESQ={pesq}")
```

Compare across runs to answer:

- How much does codec round-trip cost vs `passthrough`?
- Does 8 kHz nb eval change the SNAC vs EnCodec story vs 24 kHz scoring?
- How much STOI/PESQ do SNAC fine-scale tokens buy over coarse-only?
- Which EnCodec bandwidth hits the best telephony trade-off?

---

## Related

| Doc | Contents |
|-----|----------|
| [COLAB.md](./COLAB.md) | Main baseline benchmark (required path) |
| [README.md](../README.md) | Results table, optional ablations summary |
| [PROJECT.md](./PROJECT.md) | Architecture and codec registry |
