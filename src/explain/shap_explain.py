#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the AEUL – see LICENSE file.

from __future__ import annotations

from typing import Optional, Callable, Tuple
import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def split_pipeline(pipe: Pipeline):
    """Return (preprocessor, classifier) from a sklearn Pipeline."""
    if not isinstance(pipe, Pipeline):
        raise TypeError("Expected a sklearn Pipeline with preprocessor + classifier.")
    if len(pipe.steps) < 2:
        raise ValueError("Pipeline must have at least two steps (transformer, classifier).")
    preproc = pipe[:-1]
    clf = pipe.steps[-1][1]
    return preproc, clf


def transform(preproc: Pipeline, df):
    """Transform a (text, sender) dataframe into feature matrix."""
    X = preproc.transform(df)
    # Ensure float64 for SHAP
    if sparse.issparse(X):
        X = X.astype(np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
    return X


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else X


def proba_fn(clf) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function mapping X -> P(class=1). Works for classifiers without predict_proba."""
    if hasattr(clf, "predict_proba"):
        return lambda X: clf.predict_proba(X)[:, 1]
    # decision_function fallback -> sigmoid
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    if hasattr(clf, "decision_function"):
        return lambda X: sigmoid(np.ravel(clf.decision_function(X)))
    # worst-case: predict (0/1) as probability
    return lambda X: np.ravel(clf.predict(X)).astype(float)


def choose_explainer(clf, X_bg, use_dense_fallback=False):
    """
    Pick a SHAP explainer:
        - LinearExplainer for linear models
        - TreeExplainer for decision trees
        - PermutationExplainer fallback (works for NB, etc.)
    """
    # Linear family
    if isinstance(clf, (LogisticRegression, RidgeClassifier)) or (
        isinstance(clf, SGDClassifier) and getattr(clf, "loss", "") in {"log_loss", "modified_huber"}
    ):
        # LinearExplainer can take sparse matrices; keep X_bg as-is
        return shap.LinearExplainer(clf, X_bg, feature_pertubation="independent")

    # Trees
    if isinstance(clf, DecisionTreeClassifier):
        return shap.TreeExplainer(clf)

    # Fallback: PermutationExplainer needs a callable; dense is safer
    f = proba_fn(clf)
    X_bg_use = to_dense(X_bg) if use_dense_fallback else X_bg
    return shap.explainers.Permutation(f, X_bg_use)


def cap_rows(X, max_rows=200):
    if getattr(X, "shape", (0,))[0] <= max_rows:
        return X
    idx = np.random.RandomState(7).choice(X.shape[0], size=max_rows, replace=False)
    if sparse.issparse(X):
        return X[idx]
    return X[idx, :]


def shap_explain_global(pipe: Pipeline, df_eval, background=None, save_path: Optional[str] = None):
    """
    Create a SHAP global summary plot.
    - pipe: sklearn Pipeline(preproc, clf)
    - df_eval/background: pandas DataFrame with columns ["text","sender"]
    """
    preproc, clf = split_pipeline(pipe)

    # Transform data
    X_eval = transform(preproc, df_eval)
    X_bg = transform(preproc, background if background is not None else df_eval)

    # Keep it small & stable
    X_eval = cap_rows(X_eval, 200)
    X_bg = cap_rows(X_bg, 200)

    # Pick explainer
    # For NB / unknown models, we enable dense fallback (PermutationExplainer is safer dense)
    use_dense_fallback = True
    explainer = choose_explainer(clf, X_bg, use_dense_fallback=use_dense_fallback)

    # Compute SHAP values
    X_eval_for_explain = to_dense(X_eval) if isinstance(explainer, shap.explainers.Permutation) else X_eval
    shap_values = explainer(X_eval_for_explain)

    # Plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values.values if hasattr(shap_values, "values") else shap_values,
        X_eval_for_explain,
        show=False
    )
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def shap_explain_local(pipe: Pipeline, df_one_row, background=None, save_path: Optional[str] = None):
    """
    Save a local SHAP explanation (single row).
    """
    preproc, clf = split_pipeline(pipe)

    # Transform
    X_one = transform(preproc, df_one_row)
    X_bg = transform(preproc, background if background is not None else df_one_row)

    X_bg = cap_rows(X_bg, 200)

    use_dense_fallback = True
    explainer = choose_explainer(clf, X_bg, use_dense_fallback=use_dense_fallback)

    X_one_for_explain = to_dense(X_one) if isinstance(explainer, shap.explainers.Permutation) else X_one
    sv = explainer(X_one_for_explain)

    plt.figure(figsize=(7, 4.5))
    shap.plots.waterfall(sv[0], show=False)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
