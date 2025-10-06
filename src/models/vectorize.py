# TF-IDF vectorizer

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .transformer_embedder import SBertEmbedder
from .fc_features import FactCheckerFeaturizer

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

def build_feature_space(use_tfidf=False, use_sbert=True, use_factchecker=True):
    """
    Build the combined feature space.
    - TF-IDF (optional, default off)
    - SBERT embeddings (default on)
    - FactChecker features (default on)
    """
    transformers = []

    if use_tfidf:
        transformers.append((
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                min_df=2,
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents="unicode"
            ),
            "text"
        ))

    if use_sbert:
        transformers.append(("sbert", SBertEmbedder(), "text"))

    if use_factchecker:
        transformers.append(("fc", FactCheckerFeaturizer(), ["text", "sender"]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )