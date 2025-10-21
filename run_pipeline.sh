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

# --- Timestamping and output dir ---
RUN_ID="$(date + %Y%m%d_%H%M%S )"
RUN_DIR="runs/run_$RUN_ID"
ART_DIR="$RUN_DIR/artifacts"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$ART_DIR" "$LOG_DIR" "models" "feedback"

echo "=== Starting pipeline run ==="
echo "REPO: $REPO_ROOT"
echo "RUN ID: $RUN_ID"
echo "RUN DIR: $RUN_DIR"
echo

# --- Resolving Datasets ---
DATASETS=()
for f in "$BASELINE_CSV" "$SA_CSV" "$ZENODO_CSV"; do
    [ -f "$f" ] && DATASETS+=("$f")
done
if [ "${#DATASETS[@]}" == "" ]; then
    echo "No training datasets found. Checked:"
    echo " - $BASELINE_CSV"
    echo " - $SA_CSV"
    echo " - $ZENODO_CSV"
    exit 1
fi
if [ ! -f "$ENRON_CSV" ]; then
    echo "Enron holdout not found at $ENRON_CSV. Skipping cross domain test."
    ENRON_CSV=""
fi

echo "[DATASETS]"
printf ' - %s\n' "${DATASETS[@]}"
[ -n "$ENRON_CSV" ] && echo " - (holdout) $ENRON_CSV"
echo


# Helper: output tag under models/
m_out() { echo "runs/$RUN_ID/$1"; }

# 1) Train classical baselines
echo "[1/5] Training classical baselines …"

# A) TF-IDF ONLY + Logistic Regression (fast baseline)
OUT_TAG_A="$(m_out tfidf_lr)"
python3 -m src.models.train \
    --datasets "${DATASETS[@]}" \
    --model logistic_regression \
    --use_tfidf --no_sbert --no_fc \
    --out "$OUT_TAG_A" \
    | tee "$LOG_DIR/train_tfidf_lr.log"

# B) SBERT + FactChecker + Logistic Regression (stronger baseline)
OUT_TAG_B="$(m_out sbert_fc_lr)"
python3 -m src.models.train \
    --datasets "${DATASETS[@]}" \
    --model logistic_regression \
    --out "$OUT_TAG_B" \
    | tee "$LOG_DIR/train_sbert_fc_lr.log"

echo "Classical models trained."
echo


# 2) (Optional) Fine-tune Transformer
TRANS_DIR=""
if [ "$RUN_TRANSFORMER" = "true" ]; then
    echo "[2/5] Fine-tuning transformer ($TRANSFORMER_MODEL) …"
    TRANS_OUT="runs/$RUN_ID/transformer_${TRANSFORMER_MODEL//\//_}"
    python3 -m src.models.transformer_train \
        --datasets "${DATASETS[@]}" \
        --model_name "$TRANSFORMER_MODEL" \
        --out_dir "models/$TRANS_OUT" \
        --epochs "$TRANSFORMER_EPOCHS" \
        --max_len "$TRANSFORMER_MAXLEN" \
        | tee "$LOG_DIR/transformer_train.log"
    TRANS_DIR="models/$TRANS_OUT"
    echo "✔ Transformer saved at: $TRANS_DIR"
    echo
else
    echo "[2/5] Skipping transformer fine-tuning (set RUN_TRANSFORMER=true to enable)."
    echo
fi

# Resolve model dirs (under models/)
MODEL_DIR_TFIDF="models/$(m_out tfidf_lr)"
MODEL_DIR_SBERT="models/$(m_out sbert_fc_lr)"
if [ ! -d "$MODEL_DIR_TFIDF" ] || [ ! -d "$MODEL_DIR_SBERT" ]; then
    echo "Expected model directories not found under models/. Aborting."
    exit 1
fi

# 3) Predict on Enron (holdout)
echo "[3/5] Enron predictions …"
if [ -n "$ENRON_CSV" ]; then
    mkdir -p "$ART_DIR/preds"
    python3 -m src.models.predict \
        --model_dir "$MODEL_DIR_TFIDF" \
        --input_csv "$ENRON_CSV" \
        --out_csv "$ART_DIR/preds/enron_preds_tfidf_lr.csv" \
        | tee "$LOG_DIR/predict_enron_tfidf_lr.log"

    python3 -m src.models.predict \
        --model_dir "$MODEL_DIR_SBERT" \
        --input_csv "$ENRON_CSV" \
        --out_csv "$ART_DIR/preds/enron_preds_sbert_fc_lr.csv" \
        | tee "$LOG_DIR/predict_enron_sbert_fc_lr.log"

    if [ -n "$TRANS_DIR" ]; then
    echo "ℹ Transformer prediction step not implemented in this script."
    fi
    else
    echo " Skipped (no Enron holdout CSV)."
    fi
    echo