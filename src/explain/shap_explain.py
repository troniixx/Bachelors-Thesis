# SHAP explanations (global + local) for sklearn Pipelines with ColumnTransformer features
from typing import Optional, Sequence, Tuple
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy import sparse

def feature_names(pipeline: Pipeline, X_df) -> Optional[Sequence[str]]:
    """
    Try to recover transformed feature names from the ColumnTransformer step.
    Returns None if not possible.
    """
    try:
        feats = pipeline.named_steps["features"]
        return feats.get_feature_names_out()
    except Exception:
        return None
    
def transform_X(pipeline: Pipeline, X_df):
    """
    Apply pipelines features step to raw dataframe X_df.
    Returns (X_trans, feature_names or None)
    """
    feats = pipeline.named_steps["features"]
    X_trans = feats.transform(X_df)
    names = feature_names(pipeline, X_df)

    if sparse.issparse(X_trans):
        X_dense = X_trans.toarray()
    else:
        X_dense = np.asarray(X_trans)
    
    return X_dense, names

def shap_explanation_global(
    pipeline: Pipeline,
    X_df_sample,
    max_display: int = 20,
    save_path: Optional[str] = None
):
    """
    Compute and (optionally) save a SHAP summary plot for a fitted pipeline.
    Works best for linear or tree models. For others, uses the generic Explainer.
    """
    clf = pipeline.named_steps.get["clf"]
    X_trans, names = transform_X(pipeline, X_df_sample)
    explainer = None
    
    try:
        if hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, X_trans, feature_dependence="independent")
        else:
            explainer = shap.Explainer(clf, X_trans)
            
    except Exception:
        explainer = shap.Explainer(clf, X_trans)

    shap_values = explainer(X_trans)
    
    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=names, max_display=max_display)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
    plt.close()
    
    return shap_values

def shap_explain_local(
    pipeline: Pipeline,
    X_one_row_df,
    save_path: Optional[str] = None
):
    """
    Local SHAP explanation for a single email (DataFrame with one row of ['text','sender']).
    Returns shap_values for that instance and optionally saves a waterfall plot if possible.
    """
    clf = pipeline.named_steps["clf"]
    X_trans, names = transform_X(pipeline, X_one_row_df)
    
    try:
        if hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, X_trans)
        else:
            explainer = shap.Explainer(clf, X_trans)
    except Exception:
        explainer = shap.Explainer(clf, X_trans)

    sv = explainer(X_trans)
    try:
        plt.figure()
        shap.plots.waterfall(sv[0], max_display=20, show=False)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=160)
        plt.close()
    except Exception:
        pass

    return sv