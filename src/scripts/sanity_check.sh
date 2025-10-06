#!/usr/bin/env bash
set -euo pipefail

# --- find repo root (folder that contains 'src') ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for d in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.."; do
    if [ -d "$d/src" ]; then REPO_ROOT="$(cd "$d" && pwd)"; break; fi
done
[ -z "${REPO_ROOT:-}" ] && echo "Cannot find repo root" && exit 1
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

echo "=== SANITY CHECK RUNNING ==="
echo "Repo root: $REPO_ROOT"
echo

DATA_DIR="data"
WANTED_DIR="models/tmp_sanity"   # standard output dir name
SAMPLE_DATA="$DATA_DIR/spam_assassin_cleaned.csv"

if [ -z "${SAMPLE_DATA:-}" ] || [ ! -f "$SAMPLE_DATA" ]; then
    echo "❌ SAMPLE_DATA missing or not a file: '$SAMPLE_DATA'"
    echo "   Expected something like: data/spam_assassin_cleaned.csv"
    exit 1
fi


echo "DATA_DIR:    $DATA_DIR"
echo "WANTED_DIR:  $WANTED_DIR"
echo "SAMPLE_DATA: $SAMPLE_DATA"
echo

echo "[1/5] Cleaning previous artifacts..."
rm -rf "$WANTED_DIR"
mkdir -p "$(dirname "$WANTED_DIR")"
echo "Done."
echo

echo "[2/5] Training baseline model (SBERT+FC + LogisticRegression)..."
python3 -m src.models.train \
    --datasets "$SAMPLE_DATA" \
    --model logistic_regression \
    --out tmp_sanity

if [ ! -f "$WANTED_DIR/pipeline.joblib" ]; then
    echo "Expected model not found at $WANTED_DIR/pipeline.joblib"
    echo "Hint: run  'find models -name pipeline.joblib -print'"
    exit 1
fi
echo "Using model dir: $WANTED_DIR"
echo

echo "[3/5] Running predictions..."
python3 -m src.models.predict \
    --model_dir "$WANTED_DIR" \
    --input_csv "$SAMPLE_DATA" \
    --out_csv "$WANTED_DIR/predictions.csv"
head -n 5 "$WANTED_DIR/predictions.csv" || true
echo "Predictions OK."
echo

echo "[4/5] Generating LIME + SHAP explanations..."
SAMPLE_TEXT="Free money!!! Click here to claim your prize."
python3 -m src.explain.explain \
    --model_dir "$WANTED_DIR" \
    --text "$SAMPLE_TEXT" \
    --sample_csv "$SAMPLE_DATA" \
    --sample_size 100 \
    --save_global_shap \
    --save_local_shap
echo "Explanations OK."
echo

echo "[5/5] Checking artifacts..."
ls -lh "$WANTED_DIR" | grep -E "pipeline|lime|shap|predictions|metrics" || true
echo
echo "=== SANITY CHECK COMPLETED ==="
