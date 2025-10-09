#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# model registry (Naive Bayes, Log Reg, Random Forest, etc.)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

def get_model(name: str):
    name = name.lower()
    
    if name in {"naive_bayes", "nb", "multinomial_nb", "mnb"}:
        return MultinomialNB(alpha=0.1)
    if name in {"logistic_regression", "log_reg", "lr"}:
        return LogisticRegression(max_iter=2000, solver="liblinear", n_jobs=None)
    if name in {"random_forest", "rf"}:
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=None)
    if name in {"svm", "svc", "linear_svc", "linsvc"}:
        return LinearSVC()
    
    raise ValueError(f"Unknown model name: {name}")

AVAILABLE = ["naive_bayes", "logistic_regression", "random_forest", "svm"]