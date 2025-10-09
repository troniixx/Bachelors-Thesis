#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import os, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from lime.lime_text import LimeTextExplainer
from sklearn.utils.validation import check_is_fitted

# ---- Auto add project root to pythonpath ----
THIS_FILE = Path(__file__).resolve()
for parent in THIS_FILE.parents:
    if (parent / "src").is_dir():
        REPO_ROOT = parent
        break
else:
    REPO_ROOT = THIS_FILE.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Custom LIME explainer import ----
try:
    from src.explain.lime_explain import explain_with_lime
except Exception:
    # Fallback to inline LIME
    print("[WARNING] Failed to import custom LIME explainer, falling back to inline LIME.")
    
# ---- Streamlit Configs ----
st.set_page_config(
    page_title="Explainable Phishing Detector (LIME DEMO)",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_DIR_DEFAULT = "models/tmp_sanity" # TODO: change to production model dir
MODEL_FILE_NAME = "pipeline.joblib" # TODO: change to production model name
FEEDBACK_PATH = Path("feedback/feedback.csv")
CLASS_NAMES = ["Benign", "Phishing"]

# ---- Helper functions ----
@st.cache_data(show_spinner="Loading model...")
def load_pipeline(model_dir: str):
    """
    Load a trained sklearn pipeline from the specified directory.
    """
    model_dir = Path(model_dir)
    p = model_dir / MODEL_FILE_NAME
    
    if not p.exists():
        alt = model_dir / "model.joblib"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
    pipe = load(p)
    
    try:
        check_is_fitted(pipe)
    except Exception:
        pass

    return pipe

def predict_proba_safe(pipeline, df_2col: pd.DataFrame):
    """
    Return probabilities for 2 class models; fallback to decision function or predict
    """
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df_2col)
        
        if proba.ndim == 1 or proba.shape[1] == 1:
            p1 = proba.ravel().astype(float)
            p0 = 1.0 - p1
            
            return np.vstack([p0, p1]).T
        
        return proba
    
    elif hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(df_2col), dtype=float)
        
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            p0 = 1.0 - p1

            return np.vstack([p0, p1]).T
        
        e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    preds = np.asarray(pipeline.poredict(df_2col), dtype=int)
    p1 = preds.astype(float)
    p0 = 1.0 - p1
    
    return np.vstack([p0, p1]).T

def lime_explain_for_single_text(pipeline, text: str, sender: str, num_features: int = 10):
    """
    Generate LIME explanation for one email
    """
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)
    
    def classifier_fn(text_list):
        df = pd.DataFrame({
            "text": text_list,
            "sender": [sender] * len(text_list)
        })
        return predict_proba_safe(pipeline, df)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=classifier_fn,
        num_features=num_features,
    )

    return exp