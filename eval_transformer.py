import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Paths
MODEL_DIR = "models/runs/20251022-124353/transformer_distilroberta-base"
DATA_FILES = [
    "/Users/merterol/Desktop/Bachelors-Thesis/data/zenodo_cleaned.csv",
    "/Users/merterol/Desktop/Bachelors-Thesis/data/spam_assassin_cleaned.csv",
    "/Users/merterol/Desktop/Bachelors-Thesis/data/baseline_spam-ham.csv"
]

MAX_LEN = 256
BATCH_SIZE = 32

def load_data(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f, on_bad_lines='skip')  # Add this argument
        if "text" not in df.columns or "label" not in df.columns:
            continue
        dfs.append(df[["text", "label"]])
    return pd.concat(dfs, ignore_index=True)

def predict(model, tokenizer, texts):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
            if logits.shape[1] == 1:
                p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            else:
                p1 = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                p1 = p1[:, 1]
            batch_preds = (p1 >= 0.5).astype(int)
            preds.extend(batch_preds)
    return np.array(preds)

def main():
    df = load_data(DATA_FILES)
    print(f"Loaded {len(df)} rows from all datasets.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    texts = df["text"].astype(str).fillna("").tolist()
    true_labels = df["label"].astype(int).values
    pred_labels = predict(model, tokenizer, texts)
    #print("Accuracy:", accuracy_score(true_labels, pred_labels))
    #print("Precision:", precision_score(true_labels, pred_labels))
    #print("Recall:", recall_score(true_labels, pred_labels))
    #print("F1:", f1_score(true_labels, pred_labels))
    print("ROC AUC:", roc_auc_score(true_labels, pred_labels))
    print("PR AUC:", average_precision_score(true_labels, pred_labels))

if __name__ == "__main__":
    main()