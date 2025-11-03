#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

"""
Thin CLI wrapper that uses lime_explainer.py and shap_explainer.py
to generate local (LIME) and global/local (SHAP) explanations
for a trained sklearn Pipeline.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Set
import pandas as pd
from joblib import load

from .lime_explain import explain_with_lime, save_lime_html
from .shap_explain import shap_explain_global, shap_explain_local


# ---------- helpers ----------

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


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    """
    Robust CSV reader for messy text data:
    - tolerates commas in fields, odd quoting, and bad lines
    - uses the Python engine (more forgiving than C)
    """
    try:
        df = pd.read_csv(
            path,
            engine="python",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="skip",
        )
    except Exception:
        # Last resort: let pandas try defaults, still skip bad lines where supported
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return df


def coerce_text_sender(df: pd.DataFrame, require: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Ensure DataFrame has 'text' and 'sender' columns.
    - If 'text' missing, try subject+body fallbacks.
    - Fill NaNs and coerce to str.
    """
    require = require or {"text", "sender"}

    cols = set(df.columns)
    if "text" not in cols:
        # Common fallback: subject + body
        subj = df["subject"].astype(str) if "subject" in cols else pd.Series("", index=df.index)
        body = df["body"].astype(str) if "body" in cols else pd.Series("", index=df.index)
        if "subject" in cols or "body" in cols:
            df["text"] = (subj + " " + body).str.strip()
        else:
            raise ValueError("--sample_csv/--background_csv must contain 'text' "
                            "or ('subject' and/or 'body').")

    if "sender" not in cols:
        df["sender"] = ""

    # Clean up types / NaNs
    df["text"] = df["text"].astype(str).fillna("")
    df["sender"] = df["sender"].astype(str).fillna("")

    # Return only the columns explainer expects (order matters for ColumnTransformer)
    return df[["text", "sender"]]


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Generate LIME and SHAP explanations for a trained model.")
    ap.add_argument("--model_dir", required=True, help="Path to saved model directory or pipeline.joblib file.")
    ap.add_argument("--out_dir", default=None, help="Directory to save explanations")

    # LIME
    ap.add_argument("--text", default=None, help="Single text input for LIME explanation.")
    ap.add_argument("--lime_features", type=int, default=10, help="Number of features for LIME explanation.")

    # SHAP
    ap.add_argument("--sample_csv", default=None,
                    help="CSV for SHAP global/local (must contain 'text' or ('subject'/'body'); 'sender' optional).")
    ap.add_argument("--background_csv", default=None,
                    help="Optional lightweight CSV used ONLY as SHAP background. "
                        "If omitted, --sample_csv provides both background and evaluation sample.")
    ap.add_argument("--sample_size", type=int, default=200, help="Rows to sample from CSV for SHAP.")
    ap.add_argument("--save_global_shap", action="store_true", help="Save the SHAP global summary plot.")
    ap.add_argument("--save_local_shap", action="store_true", help="Also save local SHAP for first sampled row.")
    return ap.parse_args()


# ---------- main ----------

def main():
    args = parse_args()
    pipe = load_pipeline(args.model_dir)
    out_dir = ensure_outdir(args.out_dir or args.model_dir)

    # ---- LIME (single text) ----
    if args.text:
        exp = explain_with_lime(pipeline=pipe, text=args.text, num_features=args.lime_features)
        html_path = out_dir / "lime_explanation.html"
        save_lime_html(exp, str(html_path))
        print(f"[LIME] explanation saved to {html_path}")
        print("[LIME] Top features:", exp.as_list())

    # ---- SHAP (global + optional local) ----
    if args.sample_csv:
        # Load evaluation sample
        df_eval_raw = safe_read_csv(args.sample_csv)
        if len(df_eval_raw) == 0:
            raise ValueError(f"--sample_csv appears empty after parsing: {args.sample_csv}")
        df_eval = coerce_text_sender(df_eval_raw)

        # Background: explicit file or reuse eval sample
        if args.background_csv:
            df_bg_raw = safe_read_csv(args.background_csv)
            if len(df_bg_raw) == 0:
                raise ValueError(f"--background_csv appears empty after parsing: {args.background_csv}")
            df_bg = coerce_text_sender(df_bg_raw)
        else:
            df_bg = df_eval

        # Subsample (deterministic)
        eval_sample = df_eval.sample(min(len(df_eval), args.sample_size), random_state=42)
        bg_sample = df_bg.sample(min(len(df_bg), max(50, args.sample_size // 2)), random_state=17)

        # Safety: ensure we have something to plot
        if len(eval_sample) == 0 or len(bg_sample) == 0:
            raise ValueError("Not enough rows for SHAP after sampling. Provide a different CSV or increase sample_size.")

        # Global SHAP
        if args.save_global_shap:
            shap_sum_path = out_dir / "shap_summary.png"
            shap_explain_global(pipe, eval_sample, save_path=str(shap_sum_path))
            print(f"[SHAP] global summary plot saved to {shap_sum_path}")

        # Local SHAP on the first row
        if args.save_local_shap and len(eval_sample) > 0:
            first = eval_sample.iloc[[0]]
            shap_loc_path = out_dir / "shap_local.png"
            shap_explain_local(pipe, first, save_path=str(shap_loc_path))
            print(f"[SHAP] local explanation for first sample saved to {shap_loc_path}")

    if not args.text and not args.sample_csv:
        print("No --text or --sample_csv provided; nothing to explain.")


if __name__ == "__main__":
    main()
