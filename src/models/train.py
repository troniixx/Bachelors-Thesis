# CLI to train, save artifacts, etc
import argparse
import time
from pathlib import Path
import joblib
import numpy as np
from skelearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from .config import DATA_DIR, MODELS_DIR, RANDOM_STATE, N_FOLDS
from .datasets import load_many
from .vectorize import build_feature_space
from .baselines import get_model, AVAILABLE
from .evaluate import compute_metrics, save_json

def parse_args():
    ap = argparse.ArgumentParser(description="Train spam/phishing classifiers with fact-check features.")
    ap.add_argument("--datasets", nargs="+", required=True, help="Paths to CSV files under data/")
    ap.add_argument("--model", default="logreg", choices=AVAILABLE)
    ap.add_argument("--out", default=None, help="Output dir under models/. Default auto timestamp.")
    # Feature toggles
    ap.add_argument("--use_tfidf", action="store_true", help="Include TF-IDF features")
    ap.add_argument("--no_sbert", action="store_true", help="Exclude SBERT embeddings")
    ap.add_argument("--no_fc", action="store_true", help="Exclude FactChecker features")
    return ap.parse_args()

def main():
    args = parse_args()
    paths = [DATA_DIR / p if not Path(p).is_absolute() else Path(p) for p in args.datasets]
    df = load_many(paths)
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

    # Aggregate
    agg = {k: np.nanmean([fm[k] for fm in fold_metrics if fm.get(k) is not None])
            for k in fold_metrics[0].keys() if k != "fold"}
    print("Average metrics:", agg)

    # Save
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / (args.out if args.out else f"{args.model}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")

    metrics_path = out_dir / "metrics.json"
    save_json({"folds": fold_metrics, "aggregate": agg}, metrics_path)
    print(f"Saved metrics to {metrics_path}")