# TF-IDF vectorizer

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .transformer_embedder import SBertEmbedder
from .fc_features import FactCheckerFeaturizer

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

