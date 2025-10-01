# sklearn-compatible wrapper around Sentence-BERT to generate dense text embeddings

from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

class SBertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 32, device = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
    def fit(self, X: pd.DataFrame, y = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            texts = X['text'].tolist()
        elif isinstance(X, Iterable):
            texts = list(X)
        else:
            raise ValueError("Input should be a pandas DataFrame or an iterable of texts.")
        
        embs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return embs