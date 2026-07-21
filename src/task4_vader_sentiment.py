# ==========================================================
# Task 4 : Lexicon-Based Sentiment Analysis (VADER)
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import DATA_FILE, IMAGES_DIR, OUTPUTS_DIR, LABELS

analyzer = SentimentIntensityAnalyzer()


def _vader_sentiment(text):
    compound = analyzer.polarity_scores(str(text))["compound"]
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def run():
    """Applies VADER to the raw review text (NOT the heavily-cleaned
    text from Task 3, since VADER is designed to read punctuation,
    capitalization and intensifiers as sentiment signal) and evaluates
    it against the dataset's own sentiment_label.
    Returns the DataFrame with the 'vader_prediction' column added."""

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("TASK 4 : LEXICON-BASED SENTIMENT ANALYSIS (VADER)")
    print("=" * 70)

    required_columns = ["text", "sentiment_label"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    print("\n" + "=" * 70)
    print("MISSING VALUE HANDLING")
    print("=" * 70)

    rows_before = len(df)
    df = df.dropna(subset=["text", "sentiment_label"]).copy()
    rows_after = len(df)

    print(f"Rows Before Cleaning : {rows_before}")
    print(f"Rows After Cleaning  : {rows_after}")
    print(f"Rows Removed         : {rows_before - rows_after}")

    print("\nApplying VADER Sentiment Analysis...")
    df["vader_prediction"] = df["text"].apply(_vader_sentiment)
    print("Sentiment prediction completed successfully.")

    # Both labels are lowercased so the label schemes match exactly.
    true_labels = df["sentiment_label"].str.lower()
    predicted_labels = df["vader_prediction"].str.lower()

    accuracy = accuracy_score(true_labels, predicted_labels)

    precision_weighted = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    precision_macro = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    f1_macro = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Weighted Precision : {precision_weighted:.4f}   |   Macro Precision : {precision_macro:.4f}")
    print(f"Weighted Recall    : {recall_weighted:.4f}   |   Macro Recall    : {recall_macro:.4f}")
    print(f"Weighted F1-Score  : {f1_weighted:.4f}   |   Macro F1-Score  : {f1_macro:.4f}")

    print(f"""
Note: the large gap between Weighted F1 ({f1_weighted:.2f}) and Macro F1 ({f1_macro:.2f})
is expected on this ~90% positive dataset. Weighted metrics are dominated
by the large positive class and look strong almost regardless of what
happens on the minority classes. Macro F1 gives each class equal weight
and is the more honest measure of VADER's real performance here.
""")

    print("Actual Sentiment Distribution")
    print(true_labels.value_counts())
    print("\nPredicted Sentiment Distribution")
    print(predicted_labels.value_counts())

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    report_str = classification_report(true_labels, predicted_labels, labels=LABELS, zero_division=0)
    print(report_str)

    report_dict = classification_report(true_labels, predicted_labels, labels=LABELS,
                                         zero_division=0, output_dict=True)
    worst_class = min(LABELS, key=lambda c: report_dict[c]["f1-score"])
    print(f"Weakest class by F1-score: '{worst_class}' "
          f"(F1 = {report_dict[worst_class]['f1-score']:.2f}, "
          f"precision = {report_dict[worst_class]['precision']:.2f}, "
          f"recall = {report_dict[worst_class]['recall']:.2f})")

    cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title("Confusion Matrix - VADER", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/14_vader_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Confusion Matrix saved to {IMAGES_DIR}/14_vader_confusion_matrix.png")

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    print(df[["text", "sentiment_label", "vader_prediction"]].head(10))

    misclassified = df[true_labels != predicted_labels]
    print(f"\nMisclassified Reviews (sample of up to 5, out of {len(misclassified)} total)")
    print(misclassified[["text", "sentiment_label", "vader_prediction"]].head(5).to_string())

    df.to_csv(f"{OUTPUTS_DIR}/vader_predictions.csv", index=False)
    print("\nPrediction file saved successfully.")
    print(f"Location : {OUTPUTS_DIR}/vader_predictions.csv")

    print("\n" + "=" * 70)
    print("TASK 4 SUMMARY")
    print("=" * 70)
    print(f"Total Reviews Evaluated : {len(df)}")
    print(f"Accuracy                : {accuracy:.4f}")
    print(f"Weighted P / R / F1     : {precision_weighted:.4f} / {recall_weighted:.4f} / {f1_weighted:.4f}")
    print(f"Macro    P / R / F1     : {precision_macro:.4f} / {recall_macro:.4f} / {f1_macro:.4f}")
    print(f"Weakest Class           : {worst_class}")

    print("\nFiles Generated")
    print("------------------------------")
    print(f"1. {IMAGES_DIR}/14_vader_confusion_matrix.png")
    print(f"2. {OUTPUTS_DIR}/vader_predictions.csv")

    print("\n✓ Task 4 Completed Successfully!")
    print("=" * 70)

    return df
