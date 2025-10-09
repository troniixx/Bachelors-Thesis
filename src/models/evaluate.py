#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# metric, PR/ROC, confusion matrix, etc.

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

def compute_metrics(y_true, y_proba_or_decision, y_pred, labels=(0,1)):
    # y_proba_or_decision: probabilities or decision function scores in the range [0, 1]
    # y_pred: binary predictions (0 or 1)
    # y_true: true binary labels (0 or 1)
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # try for roc auc
    try:
        roc = roc_auc_score(y_true, y_proba_or_decision)
    except Exception:
        roc = None
        
    try:
        pr_auc = average_precision_score(y_true, y_proba_or_decision)
    except Exception:
        pr_auc = None
        
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, digits = 3)
    
    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "roc_auc": roc, "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report
    }

def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2))