# combines transformer prediction scores with fact-checker features using a meta classifier (improved robustness)
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from .datasets import load_many
from .fc_features import FactCheckerFeaturizer
from .evaluate import compute_metrics

def transformer_scores(model_dir, texts, batch_size=32, max_len=256):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to("mps" if torch.backends.mps.is_available() else "cpu")
    
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        scores.extend(probs)
        
    return np.array(scores)

def run_fusion(data_paths, model_dir):
    df = load_many(data_paths)
    X = df[["text", "sender"]]
    y = df["label"].values
    fc = FactCheckerFeaturizer()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        
        s_tr = transformer_scores(model_dir, Xtr["text"].tolist())
        s_va = transformer_scores(model_dir, Xva["text"].tolist())
        
        f_tr = fc.fit_transform(Xtr)
        f_va = fc.transform(Xva)
        
        Z_tr = np.column_stack([s_tr, f_tr])
        Z_va = np.column_stack([s_va, f_va])
        
        meta = LogisticRegression(max_iter=2000, solver="liblinear")
        meta.fit(Z_tr, ytr)
        ypred = meta.predict(Z_va)
        m = compute_metrics(yva, meta.predict_proba(Z_va)[:, 1], ypred)
        metrics.append(m)
    
    avg_f1 = np.mean([m["f1"] for m in metrics])
    print("Fusion avg F1:", avg_f1)
    return metrics