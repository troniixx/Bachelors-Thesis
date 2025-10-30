#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# === CONFIGURATION ===
INPUT_DIR = "runs/run_20251022-124353/artifacts/preds"
OUTPUT_FILE = "runs/run_20251022-124353/artifacts/enron_summary.csv"
PLOT_FILE = "runs/run_20251022-124353/artifacts/enron_false_positive_rates.png"

# === LOAD CSV FILES ===
csv_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("enron_") and f.endswith(".csv")]

summary = []

for file in csv_files:
    model_name = file.replace("enron_", "").replace(".csv", "")
    path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(path)

    required_cols = {"label", "pred", "score"}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {file}: missing required columns.")
        continue

    n_samples = len(df)
    n_spam = (df["label"] == 1).sum()
    n_ham = (df["label"] == 0).sum()

    acc = accuracy_score(df["label"], df["pred"])
    tn, fp, fn, tp = confusion_matrix(df["label"], df["pred"], labels=[0, 1]).ravel()

    # === Robustness metrics ===
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_positive_pct = false_positive_rate * 100
    false_positives_per_1k = (fp / n_samples) * 1000

    pred_spam = (df["pred"] == 1).sum()
    pred_ham = (df["pred"] == 0).sum()
    pred_spam_ratio = pred_spam / n_samples * 100

    summary.append({
        "model_name": model_name,
        "n_samples": n_samples,
        "true_spam": n_spam,
        "true_ham": n_ham,
        "pred_spam": pred_spam,
        "pred_ham": pred_ham,
        "accuracy": acc,
        "false_positive_rate": false_positive_rate,
        "false_positive_%": false_positive_pct,
        "false_positives_per_1k": false_positives_per_1k,
        "pred_spam_ratio_%": pred_spam_ratio
    })

# === SAVE SUMMARY CSV ===
summary_df = pd.DataFrame(summary).sort_values(by="false_positive_rate", ascending=True)
summary_df.to_csv(OUTPUT_FILE, index=False)
print(f"Enron robustness summary saved to {OUTPUT_FILE}\n")

# === CREATE BAR PLOT ===
plt.figure(figsize=(10, 6))
bars = plt.barh(
    summary_df["model_name"],
    summary_df["false_positive_%"],
    color="steelblue",
    edgecolor="black"
)

plt.xlabel("False Positive Rate (%)", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.title("False Positive Rate on Enron (All Legitimate Emails)", fontsize=14, pad=15)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()

# Label bars with percentage values
for bar in bars:
    plt.text(
        bar.get_width() + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.2f}%",
        va="center",
        fontsize=9
    )

plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
print(f"Plot saved to {PLOT_FILE}")
