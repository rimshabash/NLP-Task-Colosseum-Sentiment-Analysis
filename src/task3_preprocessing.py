# ==========================================================
# Task 3 : Text Preprocessing
# ==========================================================

import os
import re
import string
import pandas as pd
import nltk

from src.config import DATA_FILE, OUTPUTS_DIR

# ----------------------------------------------------------
# Ensure Required NLTK Resources Are Available
# ----------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

# ----------------------------------------------------------
# Stopwords (normalized to match post-punctuation-stripped tokens)
# ----------------------------------------------------------
translator = str.maketrans("", "", string.punctuation)
stop_words = {w.translate(translator) for w in stopwords.words("english")}

# Negation words are intentionally kept in the cleaned text because they
# can flip the polarity of a sentence ("good" -> Positive, "not good" -> Negative)
negation_words = {
    "not", "no", "nor", "never",
    "dont", "didnt", "wont", "isnt", "wasnt", "arent",
    "couldnt", "wouldnt", "shouldnt",
}

stop_words = stop_words - negation_words


def preprocess_text(text):
    """Lowercase -> remove numbers -> remove punctuation -> tokenize ->
    remove stopwords (negation-aware) -> keep alphabetic -> lemmatize."""

    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def run(df=None):
    """Cleans the review text column and saves outputs/cleaned_reviews.csv.
    Accepts the DataFrame produced by Task 2; if None, loads DATA_FILE
    fresh so this module also works standalone.
    Returns df_clean (with the new 'cleaned_text' column) for Task 5."""

    if df is None:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)

    df = df.reset_index(drop=True)

    print("=" * 70)
    print("PREPROCESSING CHOICES")
    print("=" * 70)
    print("""
1. Lowercasing: normalizes case so 'Great'/'great'/'GREAT' are treated the same.
2. Number Removal: numbers rarely contribute to sentiment, so they are dropped.
3. Punctuation Removal: punctuation carries little sentiment signal on its own.
4. Tokenization: splits each review into individual words.
5. Stopword Removal: common low-signal words ('the', 'is', 'at') are removed.
   Stopwords are normalized to their punctuation-stripped form first, so
   contraction stopwords ('youre', 'ive', 'thatll') are removed correctly too.
6. Keeping Negation Words: 'not', 'never', 'dont', 'wasnt' etc. are kept
   because they can completely flip a sentence's sentiment.
       Example: "good" -> Positive | "not good" -> Negative
7. Lemmatization: reduces words to their base form
   ('walking'/'walked'/'walks' -> 'walk').
""")

    df_clean = df.copy()
    df_clean["cleaned_text"] = df_clean["text"].apply(preprocess_text)

    print("=" * 60)
    print("Text Preprocessing Completed")
    print("=" * 60)

    print("\nBefore and After Examples")
    print("=" * 60)
    for i in range(3):
        print(f"\nReview {i + 1}")
        print("-" * 60)
        print("Original Text:\n")
        print(df_clean.iloc[i]["text"])
        print("\nCleaned Text:\n")
        print(df_clean.iloc[i]["cleaned_text"])
        print("=" * 60)

    # Sanity check: negation preservation
    demo_pairs = [
        "This place is good.",
        "This place is not good.",
        "I would never come back here.",
    ]
    print("\nNegation Preservation Check")
    print("-" * 60)
    for sentence in demo_pairs:
        print(f"{sentence!r:38} -> {preprocess_text(sentence)!r}")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df_clean.to_csv(f"{OUTPUTS_DIR}/cleaned_reviews.csv", index=False)
    print(f"\nCleaned dataset saved as '{OUTPUTS_DIR}/cleaned_reviews.csv'")

    print("\n" + "=" * 70)
    print("TASK 3 SUMMARY")
    print("=" * 70)
    print(f"Total Reviews Processed : {len(df_clean)}")
    print("Original Text Column    : text")
    print("Cleaned Text Column     : cleaned_text")
    empty_after_cleaning = (df_clean["cleaned_text"].str.strip() == "").sum()
    print(f"Reviews Empty After Cleaning : {empty_after_cleaning}")

    print("\n✓ Task 3 Completed Successfully!")

    return df_clean
