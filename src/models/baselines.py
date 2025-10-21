#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

"""
Model registry for classic baselines that plug into the existing sklearn Pipeline
without any other code changes. All returned models implement fit/predict and either
predict_proba or decision_function (your code already handles both).
"""

from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV


def get_model(name: str):
    name = name.lower().strip()

    # --- existing ones ---
    if name in {"naive_bayes", "nb", "multinomial_nb", "mnb"}:
        return MultinomialNB(alpha=0.1)

    if name in {"logistic_regression", "log_reg", "lr"}:
        return LogisticRegression(max_iter=2000, solver="liblinear")

    if name in {"random_forest", "rf"}:
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    if name in {"svm", "svc", "linear_svc", "linsvc"}:
        return LinearSVC()

    # --- new plug-and-play baselines (no pipeline changes needed) ---
    # Linear family
    if name in {"ridge", "ridge_classifier"}:
        return RidgeClassifier()

    if name in {"sgd_hinge"}:
        return SGDClassifier(loss="hinge", max_iter=2000, random_state=42)

    if name in {"sgd_log"}:
        return SGDClassifier(loss="log_loss", max_iter=2000, random_state=42)

    if name in {"sgd_huber", "sgd_modified_huber"}:
        return SGDClassifier(loss="modified_huber", max_iter=2000, random_state=42)

    if name in {"passive_aggressive", "pa"}:
        return PassiveAggressiveClassifier(max_iter=2000, random_state=42)

    # Naive Bayes variants
    if name in {"complement_nb", "cnb"}:
        return ComplementNB(alpha=0.1)

    if name in {"bernoulli_nb", "bnb"}:
        return BernoulliNB(alpha=0.1)

    # Tree-ish
    if name in {"decision_tree", "dt"}:
        return DecisionTreeClassifier(random_state=42)

    # Calibrated SVM to get probabilities
    if name in {"calibrated_linsvc", "calibrated_svc"}:
        base = LinearSVC()
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)

    # Sanity baselines
    if name in {"dummy_mf"}:
        return DummyClassifier(strategy="most_frequent")

    if name in {"dummy_uniform"}:
        return DummyClassifier(strategy="uniform", random_state=42)

    raise ValueError(f"Unknown model name: {name}")


# Canonical names to expose in argparse choices (aliases above still work)
AVAILABLE = [
    "naive_bayes",
    "logistic_regression",
    "random_forest",
    "svm",
    "ridge_classifier",
    "sgd_hinge",
    "sgd_log",
    "sgd_huber",
    "passive_aggressive",
    "complement_nb",
    "bernoulli_nb",
    "decision_tree",
    "calibrated_linsvc",
    "dummy_mf",
    "dummy_uniform",
]
