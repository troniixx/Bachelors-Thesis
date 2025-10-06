# CLI to load a saved model and predict on new data
import argparse, joblib, pandas as pd, json
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Load a saved model and predict on new data.")
    ap.add_argument("--model_dir", required=True, help="Path to the saved model directory.")
    ap.add_argument("--input_csv", required=True, help="Path to input CSV file with 'text' and 'sender' columns.")
    ap.add_argument("--out_csv", required=True, help="Path to output CSV file for predictions.")
    
    return ap.parse_args()

def main():
    args = parse_args()
    p = Path(args.model_dir)
    path = p / "pipeline.joblib"
    if not path.exists():
        path = p / "model.joblib"

    pipe = joblib.load(path)
    df = pd.read_csv(args.input_csv)
    
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    if "sender" not in df.columns:
        df["sender"] = ""
    
    ypred = pipe.predict(df[["text", "sender"]])
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        yscore = pipe.predict_proba(df[["text", "sender"]])[:, 1]
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        yscore = pipe.decision_function(df[["text", "sender"]])
    else:
        yscore = ypred
        
    df_out = df.copy()
    df_out["pred"] = ypred
    df_out["score"] = yscore
    
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.out_csv, index=False)
        print(f"Predictions saved to {args.out_csv}")
    else:
        print(df_out.head(10).to_string(index=False))
        
if __name__ == "__main__":
    main()