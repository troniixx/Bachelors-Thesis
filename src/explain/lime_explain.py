# Provides LIME-based local explanations for text classifiers (works with sklearn Pipelines)
from typing import List, Optional
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline

def wrap_for_pipeline(pipeline: Pipeline, sender_default: str = ""):
    """
    Returns a function f(texts: List[str]) -> proba ndarray[n,2] that:
    - wraps texts into a DataFrame with columns ['text','sender']
    - calls the pipeline's predict_proba / decision_function safely
    """
    clf = pipeline.named_steps.get("clf", None)

    def to_df(texts: List[str]) -> pd.DataFrame:
        return pd.DataFrame({"text": texts, "sender": [sender_default] * len(texts)})

    if hasattr(clf, "predict_proba"):
        return lambda texts: pipeline.predict_proba(to_df(texts))

    if hasattr(clf, "decision_function"):
        def _sigmoid(x):
            x = np.asarray(x, dtype=float)
            return 1.0 / (1.0 + np.exp(-x))
        
        def fn(texts):
            scores = pipeline.decision_function(to_df(texts))
            if np.ndim(scores) == 1:  # binary
                p1 = _sigmoid(scores)
                p0 = 1.0 - p1
                return np.vstack([p0, p1]).T
            # multiclass softmax
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            
            return e / e.sum(axis=1, keepdims=True)
        
        return fn

    # last resort: wrap hard labels into pseudo-probabilities
    return lambda texts: np.column_stack([
        1 - np.asarray(pipeline.predict(to_df(texts))),
        np.asarray(pipeline.predict(to_df(texts)))
    ]).astype(float)

def explain_with_lime(
    pipeline: Pipeline,
    text: str,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
    random_state: int = 42,
    sender_default: str = ""   # <- used for the DataFrame wrapper
):
    """
    Generate a LIME explanation object for a single text input.
    Returns the LIME Explanation; render via exp.as_list(), exp.as_html(), etc.
    """
    if class_names is None:
        class_names = ["ham", "spam"]

    explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)
    proba_fn = wrap_for_pipeline(pipeline, sender_default=sender_default)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=proba_fn,
        num_features=num_features
    )
    
    return exp

def save_lime_html(exp, path: str):
    html = exp.as_html()
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
