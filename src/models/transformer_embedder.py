#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# sklearn-compatible wrapper around Sentence-BERT to generate dense text embeddings

from typing import Iterable, List, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SBertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device=None,
    ):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    @staticmethod
    def coerce_str_list(seq: Iterable[Any]) -> List[str]:
        """Convert any iterable to a list of strings without NaNs/floats sneaking through."""
        out: List[str] = []
        for v in seq:
            if isinstance(v, str):
                out.append(v)
            elif v is None:
                out.append("")
            else:
                # Handle NaN floats or other odd types
                try:
                    # NaN != NaN -> True
                    if isinstance(v, float) and v != v:
                        out.append("")
                    else:
                        out.append(str(v))
                except Exception:
                    out.append("")
        return out

    def transform(self, X: Any) -> np.ndarray:
        # Accept DataFrame (prefer 'text' column), Series, or any iterable of texts
        if isinstance(X, pd.DataFrame):
            if "text" not in X.columns:
                raise ValueError("SBertEmbedder expects a DataFrame with a 'text' column.")
            texts = self.coerce_str_list(X["text"].tolist())
        elif isinstance(X, pd.Series):
            texts = self.coerce_str_list(X.tolist())
        elif isinstance(X, Iterable):
            texts = self.coerce_str_list(list(X))
        else:
            raise ValueError("Input should be a pandas DataFrame/Series or an iterable of texts.")

        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embs
