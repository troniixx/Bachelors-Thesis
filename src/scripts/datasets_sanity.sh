#!/usr/bin/env bash
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

cd "/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis"
/usr/local/bin/python3 - <<'PY'
from pathlib import Path
from src.models.datasets import load_csv, load_many
import pandas as pd

print("Baseline sample:")
df_base = load_csv(Path("data/baseline_spam-ham.csv"))
print(df_base.head())

print("\nSpamAssassin sample:")
df_sa = load_csv(Path("data/spam_assassin_cleaned.csv"))
print(df_sa.head())

print("\nZenodo sample:")
df_zenodo = load_csv(Path("data/zenodo_cleaned.csv"))
print(df_zenodo.head())

print("\nMerged:")
df = load_many([Path("data/baseline_spam-ham.csv"),
                Path("data/spam_assassin_cleaned.csv"),
                Path("data/zenodo_cleaned.csv")])
print(df.sample(5))
df_merged = pd.concat([df_base, df_sa, df_zenodo], ignore_index=True)
df_merged.to_csv("/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis/data/merged.csv", index=False)
print(df.label.value_counts())
PY