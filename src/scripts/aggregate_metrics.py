#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import argparse, json
from pathlib import Path
import pandas as pd

NUM_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

def read_metrics_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            js = json.load(f)
        agg = js.get("aggregate", {})
        out = {k: agg.get(k, None) for k in NUM_KEYS}
        return out
    except Exception:
        return {k: None for k in NUM_KEYS}

def enron_fp_rate(preds_csv: Path):
    """Assumes Enron holdout is HAM-only → FP = predicted spam (label 1)."""
    try:
        df = pd.read_csv(preds_csv)
        if "pred" not in df.columns:
            return None, None, None
        n = len(df)
        if n == 0:
            return 0.0, 0, 0.0
        fp = (df["pred"] == 1).sum()
        fp_rate = fp / n
        mean_score = df["score"].mean() if "score" in df.columns else None
        return fp_rate, int(n), mean_score
    except Exception:
        return None, None, None

def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics.json and Enron FP rates into a single CSV.")
    ap.add_argument("--run_dir", required=True, help="Path like runs/run_YYYYmmdd-HHMMSS")
    ap.add_argument("--out_csv", default=None, help="Optional explicit output path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    models_root = Path("models") / run_dir
    preds_root = Path(run_dir) / "artifacts" / "preds"
    out_dir = Path(run_dir) / "artifacts" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Each model dir under models/runs/<RUN_ID>/*
    for model_dir in sorted(models_root.glob("*")):
        if not model_dir.is_dir():
            continue
        name = model_dir.name  # e.g., tfidf_logistic_regression, sbertfc_ridge_classifier
        family = "tfidf" if name.startswith("tfidf_") else ("sbertfc" if name.startswith("sbertfc_") else "other")

        metrics_path = model_dir / "metrics.json"
        m = read_metrics_json(metrics_path) if metrics_path.exists() else {k: None for k in NUM_KEYS}

        enron_csv = preds_root / f"enron_{name}.csv"
        fp_rate, enron_n, enron_mean_score = (None, None, None)
        if enron_csv.exists():
            fp_rate, enron_n, enron_mean_score = enron_fp_rate(enron_csv)

        rows.append({
            "model_name": name,
            "family": family,
            **m,
            "enron_n": enron_n,
            "enron_fp_rate": fp_rate,         # proportion predicted as spam on ham-only
            "enron_mean_score": enron_mean_score,
            "model_dir": str(model_dir),
            "metrics_json": str(metrics_path if metrics_path.exists() else ""),
            "enron_preds_csv": str(enron_csv if enron_csv.exists() else "")
        })

    df = pd.DataFrame(rows)
    out_csv = args.out_csv or (out_dir / "metrics_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[aggregate] Wrote summary to {out_csv}")
    if len(df):
        print(df[["model_name", "family", "f1", "roc_auc", "enron_fp_rate"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
