#!/usr/bin/env bash
# Set GitHub About box: description, homepage, topics. Requires: gh auth login
set -euo pipefail

REPO="${1:-ST-48-1240162/telephony-codec-bench}"

command -v gh >/dev/null || { echo "Install: pacman -S github-cli"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run: gh auth login"; exit 1; }

DESC="Benchmark SNAC vs EnCodec on telephony-degraded speech (FLEURS + noisekit): round-trip STOI, nb-PESQ @ 8 kHz, SNR, GPU latency. Colab-ready; baseline results in reports/."

HOMEPAGE="https://colab.research.google.com/github/ST-48-1240162/telephony-codec-bench/blob/main/docs/Telephony_Codec_Bench.ipynb"

TOPICS=(
  neural-audio-codec
  speech-codec
  snac
  encodec
  telephony
  pstn
  pesq
  stoi
  speech-quality
  audio-benchmark
  voice-ai
  text-to-speech
  pytorch
  google-colab
  fleurs
  narrowband
  round-trip
  noise-robustness
  rvq
)

args=(--description "$DESC" --homepage "$HOMEPAGE")
for t in "${TOPICS[@]}"; do
  args+=(--add-topic "$t")
done

echo "Updating $REPO ..."
gh repo edit "$REPO" "${args[@]}"

echo "Done. Verify: https://github.com/${REPO}"
