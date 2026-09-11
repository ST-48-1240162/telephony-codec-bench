#!/usr/bin/env bash
# Set GitHub About box: description, homepage, topics. Requires: gh auth login
set -euo pipefail

REPO="${1:-ST-48-1240162/telephony-codec-bench}"

command -v gh >/dev/null || { echo "Install: pacman -S github-cli"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run: gh auth login"; exit 1; }

DESC="Benchmark SNAC vs EnCodec on telephony-degraded speech (FLEURS + noisekit): round-trip STOI, nb-PESQ @ 8 kHz, SNR, GPU latency. Colab-ready; baseline results in reports/."

HOMEPAGE="https://colab.research.google.com/github/ST-48-1240162/telephony-codec-bench/blob/main/docs/Telephony_Codec_Bench.ipynb"

TOPICS=(
  snac
  encodec
  telephony
  audio-benchmark
  pesq
  google-colab
  fleurs
)

echo "Updating $REPO ..."
gh repo edit "$REPO" --description "$DESC" --homepage "$HOMEPAGE"

# Replace topics wholesale (gh repo edit --add-topic does not remove stale ones).
topic_json="$(printf '%s\n' "${TOPICS[@]}" | jq -R . | jq -s '{names: .}')"
gh api -X PUT "repos/${REPO}/topics" \
  -H "Accept: application/vnd.github+json" \
  --input - <<<"$topic_json"

echo "Done. Verify: https://github.com/${REPO}"
