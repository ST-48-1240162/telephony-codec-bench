# telephony-codec-bench

Phone audio is narrowband, companded, and often noisy. After that degradation, a neural codec round-trip (encode then decode) removes additional STOI, PESQ, and bandwidth. This benchmark quantifies the loss.

Benchmark compares **SNAC** (multi-scale tokens, the kind used in LLM-TTS stacks like Bland) with **EnCodec** (Meta's RVQ baseline). Metrics include STOI, PESQ, SNR, and encode/decode latency.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ST-48-1240162/telephony-codec-bench/blob/main/docs/Telephony_Codec_Bench.ipynb)

**Notebook:** [docs/Telephony_Codec_Bench.ipynb](docs/Telephony_Codec_Bench.ipynb)  
**Walkthrough:** [docs/COLAB.md](docs/COLAB.md)

## What runs where

| Step | Where |
|------|-------|
| Build telephony WAVs (FLEURS + noisekit) | Colab T4, about 2-3 h for 200 utterances × 3 presets |
| SNAC vs EnCodec benchmark | Colab T4 or any CUDA machine |

## Install

**Colab:** run the notebook install cell. Pins live in [`docs/colab-requirements.txt`](docs/colab-requirements.txt). Do not `pip install torch` from PyPI on Colab.

**Local GPU:** install a matching `torch` / `torchaudio` pair first, then:

```sh
python3.11 -m venv ~/.venvs/telephony-codec-bench
~/.venvs/telephony-codec-bench/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
~/.venvs/telephony-codec-bench/bin/pip install -e '/path/to/telephony-codec-bench[bench]'
python scripts/verify_colab_env.py
```

## Full benchmark

```sh
pip install -e '.[bench]'
python scripts/run_benchmark.py \
  --data-dir ./data/telephony_speech \
  --device cuda \
  --out reports/benchmark.json
```

Generate `./data/telephony_speech` with noisekit first (see the Colab notebook).

## Pipeline

```mermaid
flowchart TB
    classDef input fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef preset fill:#f9fafb,stroke:#9ca3af,color:#374151
    classDef snac fill:#fffbeb,stroke:#d97706,stroke-width:2px,color:#92400e
    classDef encodec fill:#fdf2f8,stroke:#db2777,stroke-width:2px,color:#9d174d
    classDef metrics fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#065f46

    IN["FLEURS + noisekit<br/>telephony WAVs"]:::input

    subgraph DEG["Degradation presets"]
        direction LR
        P1[clean_reference]:::preset
        P2[telecom]:::preset
        P3[noise_telecom]:::preset
        P1 ~~~ P2 ~~~ P3
    end

    IN --> DEG
    DEG --> FORK{{round-trip}}

    subgraph SNAC["SNAC @ 24 kHz"]
        direction LR
        S1[encode]:::snac --> S2[multi-scale tokens]:::snac --> S3[decode]:::snac
    end

    subgraph ENCODEC["EnCodec @ 24 kHz"]
        direction LR
        E1[encode]:::encodec --> E2[RVQ codes]:::encodec --> E3[decode]:::encodec
    end

    FORK --> S1
    FORK --> E1
    S3 --> MET
    E3 --> MET

    MET["STOI, PESQ, SNR<br/>8 kHz nb eval + latency"]:::metrics
```

For a quick local test, `degrade.py` applies a simple 8 kHz bandpass, μ-law, and upsample. Full benchmark runs use [noisekit](https://github.com/karamouche/noisekit) `telecom` presets so numbers stay reproducible.

Codecs: [SNAC](https://github.com/hubertsiuzdak/snac) and EnCodec via HuggingFace `facebook/encodec_24khz`.

## Results

200 FLEURS utterances × 3 noisekit presets, codec round-trip at 24 kHz, metrics at **8 kHz nb** (`--eval-sr 8000 --pesq-mode nb`). Full numbers in [`reports/baseline_pesq/`](reports/baseline_pesq/).

| Preset | SNAC STOI | EnCodec STOI | SNAC PESQ | EnCodec PESQ |
|--------|-----------|--------------|-----------|--------------|
| `clean_reference` | **0.795** | 0.784 | **2.58** | 2.27 |
| `telecom` | **0.804** | 0.784 | 2.32 | **2.34** |
| `noise_telecom` | **0.770** | 0.767 | 1.99 | **2.17** |

On `telecom`, mean encode latency is about **10 ms** (SNAC) vs **60 ms** (EnCodec) on T4.

**Comparison.** SNAC STOI is higher on every preset and encode on telecom is roughly 6× faster. EnCodec PESQ leads once the input is already phone-band or noisy: a small edge on `telecom`, a larger one on `noise_telecom`. Clean-reference PESQ still favors SNAC.

**Conclusion.** For low-latency streaming tokenizers, SNAC leads on STOI and encode latency. On degraded phone channels, EnCodec RVQ tends to score higher in PESQ even when STOI is close. Metric choice should match the deployment, not a single leaderboard column.

## References

1. [SNAC (2024)](https://arxiv.org/abs/2410.14411)
2. [EnCodec (2022)](https://arxiv.org/abs/2210.13438)
3. [Bland on SNAC + LLM TTS](https://www.bland.ai/blog/new-tts-announcement)
4. [noisekit](https://github.com/karamouche/noisekit)
5. [Codec-SUPERB](https://github.com/voidful/Codec-SUPERB)

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
