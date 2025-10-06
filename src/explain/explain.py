"""
Thin CLI wrapper that uses lime_explainer.py and shap_explainer.py
to generate local (LIME) and global/local (SHAP) explanations
for a trained sklearn Pipeline.

Examples:
  # LIME for a single text
  python -m src.models.explain --model_dir models/sbert_fc \
      --text "Your account is suspended. Verify within 24 hours."

  # SHAP global on a CSV sample (and optional local SHAP on first row)
  python -m src.models.explain --model_dir models/sbert_fc \
      --sample_csv data/spam_assassin.csv --sample_size 200 --save_local_shap
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

from .lime_explain import explain_with_lime, save_lime_html
from .shap_explain import shap_explanation_global, shap_explain_local

def load_pipeline(model_dir_or_path: str):
    p = Path(model_dir_or_path)
    model_path = p if p.suffix == ".joblib" else (p / "pipeline.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return load(model_path)

def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out

def parse_args():
    ap = argparse.ArgumentParser(description="Generate LIME and SHAP explanations for a trained model.")
    ap.add_argument("--model_dir", required=True, help="Path to the saved model directory or pipeline.joblib file.")
    ap.add_argument("--out_dir", default=None, help="Directory to save explanations")
    
    # LIME
    ap.add_argument("--text", default=None, help="Single text input for LIME explanation.")
    ap.add_argument("--lime_features", type=int, default=10, help="Number of features for LIME explanation.")
    
    # SHAP
    ap.add_argument("--sample_csv", default=None, help="CSV file with sample data for SHAP global explanation.")
    ap.add_argument("--sample_size", type=int, default=200, help="Number of rows to sample from CSV for SHAP.")
    ap.add_argument("--save_global_shap", action="store_true", help="Whether to save the SHAP global summary plot.")
    
    return ap.parse_args()

def main():
    args = parse_args()
    pipe = load_pipeline(args.model_dir)
    out_dir = ensure_outdir(args.out_dir or args_model_dir)
    
    # LIME
    if args.text:
        exp = explain_with_lime(pipeline=pipe, text=args.text, num_features=args.lime_features)
        html_path = out_dir / "lime_explanation.html"
        save_lime_html(exp, str(html_path))
        print(f"[LIME] explanation saved to {html_path}")
        print("[LIME] Top features:", exp.as_list())
        
    # SHAP global + optional Local
    if args.sample_csv:
        df = pd.read_csv(args.sample_csv)
        if "text" not in df.columns:
            raise ValueError("--sample_csv must contain a 'text' column.")
        if "sender" not in df.columns:
            df["sender"] = ""
            
        sample = df.sample(min(len(df), args.sample_size), random_state=42)[["text", "sender"]]

        shap_sum_path = out_dir / "shap_summary.png"
        shap_explain_global(pipe, sample, save_path=str(shap_sum_path))
        print(f"[SHAP] global summary plot saved to {shap_sum_path}")
        
        # Optional Local shap on first row
        if args.save_local_shap and len(sample) > 0:
            first = sample.iloc[[0]]
            shap_loc_path = out_dir / "shap_local.png"
            shap_explain_local(pipe, first, save_path=str(shap_loc_path))
            print(f"[SHAP] local explanation for first sample saved to {shap_loc_path}")
            
    if not args.text and not args.sample_csv:
        print("No --text or --sample_csv provided; nothing to explain.")
        
if __name__ == "__main__":
    main()