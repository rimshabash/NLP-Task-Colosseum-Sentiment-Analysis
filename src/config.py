# ==========================================================
# config.py
# Shared paths and constants used across all task modules
# ==========================================================

import os

# ----------------------------------------------------------
# Input / Output paths
# ----------------------------------------------------------
DATA_FILE = "rome_colosseum_visitor_reviews_final.csv"
CLEANED_FILE = os.path.join("outputs", "cleaned_reviews.csv")

IMAGES_DIR = "images"
OUTPUTS_DIR = "outputs"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ----------------------------------------------------------
# Shared constants
# ----------------------------------------------------------
LABELS = ["negative", "neutral", "positive"]

# Important dataset columns (used in Task 1 / Task 2)
REVIEW_COLUMN = "text"
RATING_COLUMN = "rating"
SENTIMENT_COLUMN = "sentiment_label"
