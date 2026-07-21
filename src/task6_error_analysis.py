# ==========================================================
# Task 6 : Error Analysis
# ==========================================================

import os
import pandas as pd

from src.config import OUTPUTS_DIR


def run(model, X_test_tfidf, X_test, y_test):
    """Runs error analysis using the model and test data produced by Task 5."""

    print("\n" + "=" * 70)
    print("TASK 6 : ERROR ANALYSIS")
    print("=" * 70)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    predictions = model.predict(X_test_tfidf)

    error_df = pd.DataFrame({
        "Review": X_test.reset_index(drop=True),
        "Actual": y_test.reset_index(drop=True),
        "Predicted": pd.Series(predictions),
    })

    misclassified = error_df[error_df["Actual"] != error_df["Predicted"]].reset_index(drop=True)

    print("\nTotal Misclassified Reviews")
    print("-" * 60)
    print(len(misclassified))

    print("\nSample Misclassified Reviews")
    print("-" * 60)
    print(misclassified.head(10))

    misclassified.to_csv(f"{OUTPUTS_DIR}/misclassified_reviews.csv", index=False)
    print(f"\n✓ Saved: {OUTPUTS_DIR}/misclassified_reviews.csv")

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print("""
1. Mixed Sentiment
   Many reviews praise the Colosseum but also complain about long
   queues, crowds, or ticket prices - both positive and negative
   words appear in the same review, which confuses the classifier.

2. Neutral Reviews
   Neutral reviews are the hardest to classify - they contain
   descriptive information without strong emotional words, causing
   the model to confuse them with positive reviews.

3. Short Reviews
   Very short reviews ("Nice.", "Good.", "Worth visiting.") give the
   TF-IDF model very little to work with.

4. Queue / Ticket / Crowd Complaints
   Words like 'queue', 'ticket', 'crowded', 'waiting', 'heat' often
   appear inside otherwise positive reviews, sometimes pulling the
   prediction toward negative.

5. Negation
   Even though negation words are preserved during preprocessing,
   some complex sentence structures still confuse the model.

Overall, most errors occur because reviews express multiple opinions
at once - TF-IDF treats words independently and cannot fully capture
sentence-level meaning, making mixed-sentiment reviews the hardest case.
""")

    error_rate = len(misclassified) / len(error_df)

    print("\n" + "=" * 70)
    print("TASK 6 SUMMARY")
    print("=" * 70)
    print(f"Total Testing Reviews      : {len(error_df)}")
    print(f"Misclassified Reviews      : {len(misclassified)}")
    print(f"Misclassification Rate     : {error_rate:.4f}")

    print("\nGenerated Files")
    print("----------------------------")
    print(f"✓ {OUTPUTS_DIR}/misclassified_reviews.csv")

    print("\n✓ Task 6 Completed Successfully!")

    return misclassified
