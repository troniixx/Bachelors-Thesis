# Provides LIME-based local explanations for text classifiers (works with sklearn Pipelines)
from typing import List, Optional, Union
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline

def predict_proba_safe(pipeline: Pipeline);
    """
    Returns a function f(texts) -> probas that works with LIME.
    Falls back to decision_function or predict if predict_proba is not available.
    """
    clf = pipeline.named_steps.get("clf", None)
    
    if hasattr(clf, "predict_proba"):
        return lambda texts: pipeline.predict_proba(texts)
    
    if hasattr(clf, "decision_function"):
        
        def sigmoid(x):
            x = np.asarray(x, dtype=float)

            return 1.0 / (1.0 + np.exp(-x))
        
        def fn(texts):
            scores = pipeline.decision_function(texts)
            
            if scores.ndim == 1:
                p1 = sigmoid(scores)
                p0 = 1.0 - p1
                
                return np.vstack([p0, p1]).T
            
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            
            return p

        return fn
    
    return lambda texts: np.column_stack([1 - pipeline.predict(texts), pipeline.predict(texts)])

def explain_with_lime(
    pipeline: Pipeline,
    text: str,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
    random_state: int = 42   
):
    
    """
    Generate a LIME explanation object for a single text input.
    Returns the LIME explanation: you can render via:
        - exp.as_list() # weights
        - exp.show_in_notebook(text=True) # HTML
        exp.as_html() # HTML string
    """
    if class_names is None:
        class_names = ["not spam", "spam"]
    
    explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)
    predict_fn = predict_proba_safe(pipeline)
    
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=num_features
    )
    
    return exp

def save_lime_html(exp, path: str, text: Optional[str] = None):
    """
    Save LIME explanation to an HTML file.
    """
    html = exp.as_html(text=text) if text is None else exp.as_html(present_data=True)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return path