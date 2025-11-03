RUN_ID=20251022-124353
ART_DIR="runs/run_${RUN_ID}/artifacts"
LOG_DIR="runs/run_${RUN_ID}/logs"
ENRON_CSV="data/enron_emails.csv"

mkdir -p "$ART_DIR/preds"

# Predict for each classical model
find "models/runs/$RUN_ID" -maxdepth 1 -mindepth 1 -type d \( -name 'tfidf_*' -o -name 'sbertfc_*' \) -print0 | while IFS= read -r -d '' dir; do
  NAME="$(basename "$dir")"
  OUT_CSV="$ART_DIR/preds/enron_${NAME}.csv"
  LOG="$LOG_DIR/predict_enron_${NAME}.log"
  echo " -> $NAME"
  python3 -m src.models.predict \
    --model_dir "$dir" \
    --input_csv "$ENRON_CSV" \
    --out_csv "$OUT_CSV" | tee "$LOG" >/dev/null
done

TRANS_DIR="models/runs/${RUN_ID}/transformer_distilroberta-base"
TNAME="$(basename "$TRANS_DIR")"
OUT_CSV="$ART_DIR/preds/enron_${TNAME}.csv"
LOG="$LOG_DIR/predict_enron_${TNAME}.log"

python3 -m src.models.transformer_predict \
  --model_dir "$TRANS_DIR" \
  --input_csv "$ENRON_CSV" \
  --out_csv "$OUT_CSV" \
  --max_len 256 | tee "$LOG" >/dev/null