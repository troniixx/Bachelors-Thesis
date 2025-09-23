# loading & merging datasets (Enron, SpamAssassin, etc.)
# goal is to have one huge dataset with columns: text (string), sender (string, may be empty), label (0 = ham or 1 = spam/phish)

import ast
from pathlib import Path
import pandas as pd
from typing import List, Optional, Tuple

SPAM_STR = {"spam", "phish", "phishing", "malicious", "junk", "bad", "scam", "unsolicited"}
HAM_STR = {"ham", "legit", "legitimate", "good", "normal", "not spam"}

def normalize_label(val) -> int:
    """Normalize label to 0 (ham) or 1 (spam/phish)

    Args:
        val: Input label (various types)

    Returns:
        int: 0 for ham, 1 for spam/phish
    """
    if pd.isna(val):
        raise ValueError("Label is NaN")
    
    if isinstance(val, (int, float)) and str(val).isdigit():
        return 1 if int(val) == 1 else 0
    if isinstance(val, bool):
        return 1 if val else 0
    
    s = str(val).strip().lower()
    if s in SPAM_STR:
        return 1
    if s in HAM_STR:
        return 0
    
    # last resort: try parsing numeric strings
    try:
        n = int(float(s))
        return 1 if n != 0 else 0
    except Exception:
        raise ValueError(f"Cannot normalize label: {val}")
    
def safe_string(s):
    """Convert to string, handling NaN and None"""
    return "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)

def zenodo_urls_to_inline(urls_cell) -> str:
    """
    Convert a Zenodo 'urls' cell to inline text:
    - if it's a Python list string like "['http://a', 'http://b']" -> join
    - if it's a single URL string -> return as is
    - else -> empty
    """
    
    if urls_cell is None or (isinstance(urls_cell, float) and pd.isna(urls_cell)):
        return ""
    
    try:
        parsed = ast.literal_eval(urls_cell) if isinstance(urls_cell, str) else urls_cell
        
        if isinstance(parsed, (list, tuple)):
            return "\n".join(map(str, parsed))
        
    except Exception:
        pass
    
    return str(urls_cell)

# ---- Specific dataset loaders ----

def load_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Load the baseline dataset from CSV file"""
    
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("Baseline dataset must have 'label' and 'text' columns")
    
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "sender": "",
        "label": df["label"].apply(normalize_label).astype(int)
    })
    
    return out

def load_spam_assassin(df: pd.DataFrame) -> pd.DataFrame:
    """Load the SpamAssassin dataset from CSV file"""
    
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("SpamAssassin dataset must have 'text' and 'target' columns")
    
    out = pd.DataFrame({
        "text": df["text"].astype(str),
        "sender": "",
        "label": df["target"].apply(normalize_label).astype(int)
    })

    return out

def load_zenodo(df: pd.DataFrame) -> pd.DataFrame:
    """Load the Zenodo dataset from CSV file"""
    
    required = {"subject", "body", "label"}
    missing = required - set(df.columns)
    
    if missing:
        raise ValueError(f"Zenodo dataset is missing required columns: {missing}")
    
    subject = df["subject"].map(safe_string)
    body = df["body"].map(safe_string)
    
    urls_inline = ""
    if "urls" in df.columns:
        urls_inline = df["urls"].map(zenodo_urls_to_inline)
        text = subject.str.cat(body, sep = "\n\n").str.cat(urls_inline, sep = "\n\nURLS:\n")
    else:
        text = subject.str.cat(body, sep = "\n\n")
        
    sender = df["sender"].map(safe_string) if "sender" in df.columns else pd.Series([""] * len(df), index=df.index)
    
    out = pd.DataFrame({
        "text": text.astype(str),
        "sender": sender.astype(str),
        "label": df["label"].apply(normalize_label).astype(int)
    })
    
    return out

# ---- Merging multiple datasets ----

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame"""
    
    df = pd.read_csv(path)
    cols = set(df.columns)
    
    if {"label", "text"}.issubset(cols) and "target" not in cols:
        return load_baseline(df)
    if "target" in cols and "text" in cols:
        return load_spam_assassin(df)
    if {"subject", "body"}.issubset(cols) or "urls" in cols or "sender" in cols:
        return load_zenodo(df)
    
    raise ValueError(f"Unrecognized dataset format in {path}, columns: {cols}")

def load_many(paths: List[Path]) -> pd.DataFrame:
    """Load and merge multiple datasets from given CSV file paths"""
    
    dfs = [load_csv(Path(p)) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    
    # strip werid whitespace
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["sender"] = df["sender"].astype(str).fillna("")
    df["label"] = df["label"].astype(int)
    return df