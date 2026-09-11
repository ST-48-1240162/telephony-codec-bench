#!/usr/bin/env bash
# One-shot Colab GPU setup + compat audit via google-colab-cli.
# Prerequisite: complete OAuth once (see docs/COLAB_CLI.md).
set -euo pipefail

SESSION="${1:-bench}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SETUP="$ROOT/scripts/colab_remote_setup.py"

command -v colab >/dev/null || { echo "Install: uv tool install google-colab-cli"; exit 1; }

echo "[1/4] Provision T4 session: $SESSION"
colab new -s "$SESSION" --gpu T4

echo "[2/4] Remote install + verify (may take several minutes)"
colab exec -s "$SESSION" -f "$SETUP" --timeout 900

echo "[3/4] Session status"
colab status -s "$SESSION"

echo "[4/4] Export log"
colab log -s "$SESSION" -o "$ROOT/reports/colab-debug.md" || true

echo "Done. Attach with: colab exec -s $SESSION  |  colab console -s $SESSION"
echo "Stop VM: colab stop -s $SESSION"
