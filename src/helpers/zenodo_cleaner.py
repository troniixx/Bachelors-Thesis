#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import pandas as pd

# in zenodo.csv, clean sender so that names and '<' and '>' are removed

def clean_sender(sender: str) -> str:
    if pd.isna(sender):
        return ""
    if '<' in sender and '>' in sender:
        start = sender.index('<') + 1
        end = sender.index('>')
        return sender[start:end].strip()
    return sender.strip()

def clean_zenodo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sender'] = df['sender'].apply(clean_sender)
    return df

# save cleaned zenodo to new file
def save_cleaned_zenodo(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df_cleaned = clean_zenodo(df)
    df_cleaned.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    input_path = "/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis/data/zenodo.csv"
    output_path = "/Users/merterol/Desktop/UZH/CompLing:CompSci/CL/Sem 5/Bachelors Thesis/VSCode/Bachelors-Thesis/data/zenodo_cleaned.csv"
    save_cleaned_zenodo(input_path, output_path)