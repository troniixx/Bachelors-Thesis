cd "/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis"
/usr/local/bin/python3 - <<'PY'
from pathlib import Path
from src.models.datasets import load_csv, load_many
import pandas as pd

print("Baseline sample:")
print(load_csv(Path("data/baseline_spam-ham.csv")).head())

print("\nSpamAssassin sample:")
print(load_csv(Path("data/spam_assassin_cleaned.csv")).head())

print("\nZenodo sample:")
print(load_csv(Path("data/zenodo.csv")).head())

print("\nMerged:")
df = load_many([Path("data/baseline_spam-ham.csv"),
                Path("data/spam_assassin_cleaned.csv"),
                Path("data/zenodo.csv")])
print(df.sample(5))
print(df.label.value_counts())
PY