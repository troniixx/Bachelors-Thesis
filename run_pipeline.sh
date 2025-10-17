#!/usr/bin/env bash
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.
set -euo pipefail

# --- Config ---

# Toggle transformer fine tuning
RUN_TRANSFORMER="${RUN_TRANSFORMER:-false}"

# --- Datasets ---
BASELINE_CSV="data/baseline_spam-ham.csv"
SA_CSV="data/spam_assassin_cleaned.csv"
ENRON_CSV="data/enron_emails.csv"
ZENODO_CSV="data/zenodo_cleaned.csv"

# --- Sample for lime rendering ---
SAMPLE_TEXT="Congratulations! You've won a free ticket to Bahamas. Click here to claim your prize."

# --- Transformer config (only if RUN_TRANSFORMER=true) ---
TRANSFORMER_MODEL="distilroberta-base"
TRANSFORMER_EPOCHS=3
TRANSFORMER_MAXLEN=256

# --- repo root ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT=""

for d in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
    if [ -d "$d/src" ]; then REPO_ROOT="$(cd "$d" && pwd)"; break; fi
done
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find repository root directory." ; exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

