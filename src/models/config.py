# Constants (paths, seeds, columns)
from pathlib import Path

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Columns
TEXT_COLS = ["text", "subject", "body"]
LABEL_COL = "label"
SENDER_COL_CANDIDATES = ["from", "sender", "email", "reply-to", "fromn_address"]

# Reproducibility
RANDOM_STATE = 42
N_JOBS = -1  # use all available cores

# CV
N_FOLDS = 5

# TF-IDF
MAX_FEATURES = 50000
MIN_DF = 2
NGRAM_RANGE = (1, 2)