# ==========================================================
# main.py
# Runs the complete Rome Colosseum review sentiment analysis
# pipeline, task by task, in order.
#
# Usage:
#     python main.py
# ==========================================================

from src import (
    task1_data_audit,
    task2_eda,
    task3_preprocessing,
    task4_vader_sentiment,
    task5_ml_classifier,
    task6_error_analysis,
)


def main():
    # Task 1 : Load the raw dataset and audit its schema/quality
    df = task1_data_audit.run()

    # Task 2 : EDA - missing values, ratings, sentiment, trip type,
    #          travel season, review length & outliers
    df = task2_eda.run(df)

    # Task 3 : Text preprocessing (cleaning, tokenizing, lemmatizing)
    df_clean = task3_preprocessing.run(df)

    # Task 4 : Lexicon-based sentiment analysis with VADER
    task4_vader_sentiment.run()

    # Task 5 : TF-IDF + ML classifiers (baselines, class weighting,
    #          oversampling, GridSearchCV) with full model comparison
    best_model, X_test_tfidf, X_test, y_test = task5_ml_classifier.run(df_clean)

    # Task 6 : Error analysis on the best model's misclassifications
    task6_error_analysis.run(best_model, X_test_tfidf, X_test, y_test)

    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
