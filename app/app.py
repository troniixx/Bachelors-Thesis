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

from typing import Any

try:
    from src.fact_checker.checker import FactChecker, FactCheckResult, Evidence
    FACT_CHECKER_AVAILABLE = True
except Exception:
    FACT_CHECKER_AVAILABLE = False
    FactChecker = None  # type: ignore
    Evidence = Any      # type: ignore


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
    Accepts absolute or repo-root-relative paths.
    """
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute():
        model_dir_path = (REPO_ROOT / model_dir_path).resolve()

    p = model_dir_path / MODEL_FILE_NAME
    if not p.exists():
        alt = model_dir_path / "model.joblib"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(
                f"No model file found in {model_dir_path} "
                f"(looked for '{MODEL_FILE_NAME}' and 'model.joblib')."
            )

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
    
    preds = np.asarray(pipeline.predict(df_2col), dtype=int)
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

def save_feedback_row(row: dict):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    
    if FEEDBACK_PATH.exists():
        df.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)

@st.cache_resource
def get_fact_checker():
    if not FACT_CHECKER_AVAILABLE:
        return None
    try:
        return FactChecker()
    except Exception:
        return None
    
def level_to_label(level):
    if isinstance(level, (int, float, np.floating)):
        if level >= 0.8: return "Critical"
        if level >= 0.5: return "High"
        if level >= 0.2: return "Medium"
        return "Low"
    s = str(level).lower()
    if s in {"critical", "high", "medium", "low"}:
        return s.capitalize()
    return s

def severity_badge(label: str, level) -> str:
    lvl = level_to_label(level)
    color = { 
            "Critical": "#B71C1C",
            "High": "#E65100",
            "Medium": "#FBC02D",
            "Low": "#2E7D32"
    }.get(lvl, "#616161")
    
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:9999px;font-size:0.80rem;margin-right:6px">{label}: {lvl}</span>'

def render_factcheck_panel(res):
    badges = [
        severity_badge("Overall Risk", res.fact_risk),
        severity_badge("Brand Mismatch", res.components.get("brand_mismatch", 0)),
        severity_badge("TLD Risk", res.components.get("tld_risk", 0)),
        severity_badge("URL Obfuscation", res.components.get("url_obfuscation", 0)),
        severity_badge("Claim Risk", res.components.get("claim_risk", 0)),
    ]
    st.markdown(" ".join(badges), unsafe_allow_html=True)

    if getattr(res, "message", None):
        st.markdown("**Summary**")
        for m in res.message:
            st.markdown(f"- {m}")

    if getattr(res, "evidence", None):
        rows = []
        for ev in res.evidence:
            flat = {"type": getattr(ev, "label", "")}
            det = getattr(ev, "details", {})  # <- fixed name
            if isinstance(det, dict):
                for k, v in det.items():
                    flat[k] = v
            rows.append(flat)
        if rows:
            st.markdown("**Evidence Details**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---- Controls ----

st.sidebar.title("Settings")
model_dir = st.sidebar.text_input("Model Directory", value=MODEL_DIR_DEFAULT)
num_features = st.sidebar.slider("LIME: number of features", min_value=5, max_value=20, value=10, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("Leave sender empty if unknown.")

run_factcheck = st.sidebar.checkbox("Run fact-checker panel", value=True)
if not FACT_CHECKER_AVAILABLE:
    st.sidebar.caption("Fact-checker: not available (module import failed)")
else:
    st.sidebar.caption("Fact-checker: available")

pipe, load_error = None, None
try:
    pipe = load_pipeline(model_dir)
except Exception as e:
    load_error = str(e)
    
# ---- Main App ----

st.title("Explainable Phishing Detector - LIME Prototype")

if load_error:
    st.error(f"Failed to load model pipeline: {load_error}")
    st.stop()
    
with st.form("email_form"):
    sender = st.text_input("Sender (optional)", placeholder="e.g. support@example.com").strip().lower()
    email_text = st.text_area("Email Text", height=220, placeholder="Paste the email body here (subject + content)...")
    submitted = st.form_submit_button("Analyze Email")
    
if submitted:
    if not email_text.strip():
        st.warning("Please paste an email text to analyze.")
        st.stop()

    df_one = pd.DataFrame([{"text" : email_text, "sender": sender or ""}])
    proba = predict_proba_safe(pipe, df_one)
    p_legit, p_phish = float(proba[0,0]), float(proba[0,1])
    pred_label_idx = int(np.argmax(proba, axis=1)[0])
    pred_label = CLASS_NAMES[pred_label_idx]
    conf = p_phish if pred_label_idx == 1 else p_legit
    
    st.subheader("Prediction Result")
    st.markdown(f"**Predicted Class:** `{pred_label}`")
    st.markdown(f"**Confidence:** `{conf*100:.2f}%`")
    
    # --- Fact-Checker panel ---
    if run_factcheck:
        fc = get_fact_checker()
        st.subheader("Fact-Check Findings")
        if fc is None:
            st.info("Fact-checker not available or failed to initialize.")
        else:
            try:
                fc_res = fc.check(email_text, sender_email=(sender or None))
                render_factcheck_panel(fc_res)
            except Exception as e:
                st.warning(f"Fact-checking failed: {e}")

    
    with st.spinner("Generating LIME explanation..."):
        try:
            exp = lime_explain_for_single_text(pipe, email_text, sender or "", num_features)
            st.subheader("Top Contributing Features")
            as_list = exp.as_list()
            
            if as_list:
                df_feat = pd.DataFrame(as_list, columns=["token/feature", "weight"])
                st.dataframe(df_feat, use_container_width=True)
            try:
                import streamlit.components.v1 as components
                components.html(exp.as_html(), height=500, scrolling=True)
            except Exception:
                st.info("Failed to render full LIME HTML explanation.")
        except Exception as e:
            st.error(f"Failed to generate LIME explanation: {e}")
    
    # Feedback section
    st.subheader("Feedback (Optional)")
    st.caption("Do you agree with the prediction? Your feedback helps us improve the model.")
    agree = st.radio("I agree with the prediction:", options=["Yes", "No"], index=0, horizontal=True)
    user_label, user_comment = "", ""
    
    if agree == "No":
        user_label = st.selectbox("What do you think the correct label should be?", options=CLASS_NAMES, index=0)
        user_comment = st.text_area("Additional comments (optional):", height=80, placeholder="Your comments...")
        
    if st.button("Submit Feedback"):
        row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sender": sender or "",
            "text": email_text,
            "pred_label": pred_label,
            "p_phishing": round(p_phish, 6),
            "p_legitimate": round(p_legit, 6),
            "user_agrees": (agree == "Yes"),
            "user_label": user_label,
            "user_comment": user_comment,
            "model_dir": str(model_dir),
            "lime_num_features": num_features,
        }
        try:
            save_feedback_row(row)
            st.success(f"Feedback saved to {FEEDBACK_PATH}")
        except Exception as e:
            st.error(f"Could not save feedback: {e}")
            
# --- Footer ---
st.markdown("---")
st.caption(
    "© 2025 Mert Erol, University of Zurich. "
    "This is a prototype application for demonstration purposes only. "
    "Not intended for production use."
    "Privacy notice: inputs and feedback are stored locally (feedback/feedback.csv). "
    "No external transmission occurs. Avoid pasting sensitive personal data."
)