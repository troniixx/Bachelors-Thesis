#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# CLI to train, save artifacts, etc
import argparse
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from .config import DATA_DIR, MODELS_DIR, RANDOM_STATE, N_FOLDS, PROJECT_ROOT
from .datasets import load_many
from .vectorize import build_feature_space
from .baselines import get_model, AVAILABLE
from .evaluate import compute_metrics, save_json

def parse_args():
    ap = argparse.ArgumentParser(description="Train spam/phishing classifiers with fact-check features.")
    ap.add_argument("--datasets", nargs="+", required=True, help="Paths to CSV files under data/")
    ap.add_argument("--model", default="logistic_regression", choices=AVAILABLE)
    ap.add_argument("--out", default=None, help="Output dir under models/. Default auto timestamp.")
    # Feature toggles
    ap.add_argument("--use_tfidf", action="store_true", help="Include TF-IDF features")
    ap.add_argument("--no_sbert", action="store_true", help="Exclude SBERT embeddings")
    ap.add_argument("--no_fc", action="store_true", help="Exclude FactChecker features")
    return ap.parse_args()

def main():
    args = parse_args()
    
    # after parse_args(); replace your current path loop with this
    paths = []
    for p in args.datasets:
        pth = Path(p)
        if not pth.is_absolute():
            pth = (PROJECT_ROOT / pth).resolve()
        if not pth.exists():
            raise FileNotFoundError(f"[train] Dataset not found: {pth}")
        if not pth.is_file():
            raise ValueError(f"[train] Expected a FILE for --datasets, got directory: {pth}")
        paths.append(pth)
    print("[train] Using datasets:", [str(p) for p in paths])

    
    df = load_many(paths)
    print(f"[train] Loaded {len(df)} rows")
    
    X = df[["text", "sender"]]
    y = df["label"].values

    feature_space = build_feature_space(
        use_tfidf=args.use_tfidf,
        use_sbert=not args.no_sbert,
        use_factchecker=not args.no_fc
    )
    clf = get_model(args.model)
    pipe = Pipeline([("features", feature_space), ("clf", clf)])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        pipe.fit(Xtr, ytr)

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            scr = pipe.predict_proba(Xva)[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            scr = pipe.decision_function(Xva)
        else:
            scr = pipe.predict(Xva)
        ypred = pipe.predict(Xva)
        m = compute_metrics(yva, scr, ypred)
        m["fold"] = fold
        fold_metrics.append(m)

    # Aggregate only numeric metrics
    NUMERIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    agg = {}
    for k in NUMERIC_KEYS:
        vals = []
        for fm in fold_metrics:
            v = fm.get(k, None)
            if v is None:
                continue
            if isinstance(v, (int, float, np.floating)):
                vals.append(float(v))
        agg[k] = float(np.nanmean(vals)) if vals else None

    print("[train] Average metrics:", agg)

    # Save
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / (args.out if args.out else f"{args.model}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")

    metrics_path = out_dir / "metrics.json"   # (name can stay as-is)
    save_json({"folds": fold_metrics, "aggregate": agg}, metrics_path)
    print(f"Saved metrics to {metrics_path}")
    
if __name__ == "__main__":
    main()