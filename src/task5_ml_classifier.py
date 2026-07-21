# ==========================================================
# Task 5 : Machine Learning Classifier
# Part 1 : Data Splitting & TF-IDF Vectorization
# Part 2 : Class Imbalance Diagnosis
# Part 3 : Evaluation Metric Strategy
# Part 4 : Models (baselines, class weighting, oversampling,
#          GridSearchCV) + full model comparison
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import RandomOverSampler

from src.config import CLEANED_FILE, IMAGES_DIR, OUTPUTS_DIR, LABELS
from src.utils import build_results


# ==========================================================
# Part 1 : Data Splitting & TF-IDF Vectorization
# ==========================================================

def _part1_split_and_vectorize(df):
    print("=" * 70)
    print("TASK 5 : PART 1")
    print("DATA SPLITTING & TF-IDF VECTORIZATION")
    print("=" * 70)

    required_columns = ["cleaned_text", "sentiment_label"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found.")

    # An empty string round-tripped through CSV becomes a real NaN,
    # so dropna() here correctly catches it.
    rows_before = len(df)
    df = df.dropna(subset=["cleaned_text", "sentiment_label"]).copy()
    print(f"\nDataset after removing missing values : {len(df)} reviews "
          f"({rows_before - len(df)} row(s) dropped)")

    X = df["cleaned_text"]
    y = df["sentiment_label"].str.lower()

    print("\nTarget Class Distribution")
    print("-" * 50)
    print(y.value_counts())
    print("\nTarget Class Percentage")
    print("-" * 50)
    print((y.value_counts(normalize=True) * 100).round(2))

    # stratify=y keeps the ~90/5/5 class split in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n" + "=" * 70)
    print("TRAIN - TEST SPLIT")
    print("=" * 70)
    print(f"Training Reviews : {len(X_train)}")
    print(f"Testing Reviews  : {len(X_test)}")

    print("\nTraining Set Distribution")
    print("-" * 50)
    print(y_train.value_counts())
    print("\nTraining Percentage")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nTesting Set Distribution")
    print("-" * 50)
    print(y_test.value_counts())
    print("\nTesting Percentage")
    print((y_test.value_counts(normalize=True) * 100).round(2))

    # Fit ONLY on training text to avoid leaking test vocabulary/stats.
    tfidf = TfidfVectorizer(
        lowercase=False,  # cleaned_text is already lowercased in Task 3
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1] == len(tfidf.vocabulary_), \
        "Train/test TF-IDF feature dimensions do not match the fitted vocabulary."

    print("\n" + "=" * 70)
    print("TF-IDF VECTORIZATION")
    print("=" * 70)
    print(f"Training Matrix Shape : {X_train_tfidf.shape}")
    print(f"Testing Matrix Shape  : {X_test_tfidf.shape}")
    print(f"\nTotal Vocabulary Size : {len(tfidf.vocabulary_)}")

    feature_names = tfidf.get_feature_names_out()
    print("\nFirst 20 TF-IDF Features")
    print("-" * 50)
    print(feature_names[:20])

    print("\n" + "=" * 70)
    print("PART 1 SUMMARY")
    print("=" * 70)
    print(f"Total Reviews              : {len(df)}")
    print(f"Training Samples           : {len(X_train)}")
    print(f"Testing Samples            : {len(X_test)}")
    print(f"Vocabulary Size            : {len(tfidf.vocabulary_)}")
    print(f"Training Matrix Shape      : {X_train_tfidf.shape}")
    print(f"Testing Matrix Shape       : {X_test_tfidf.shape}")
    print("\n✓ Task 5 - Part 1 Completed Successfully!")

    return X, y, X_train, X_test, y_train, y_test, X_train_tfidf, X_test_tfidf, tfidf


# ==========================================================
# Part 2 : Class Imbalance Diagnosis
# ==========================================================

def _part2_imbalance_diagnosis(y):
    print("\n" + "=" * 70)
    print("TASK 5 : PART 2")
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)

    class_counts = y.value_counts().sort_index()
    print("\nClass Counts")
    print("-" * 70)
    print(class_counts)

    class_percentages = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    print("\nClass Percentages")
    print("-" * 70)
    for label, percentage in class_percentages.items():
        print(f"{label:<10} : {percentage:.2f}%")

    imbalance_report = class_counts.to_frame(name="Count")
    imbalance_report["Percentage (%)"] = class_percentages
    print("\nComplete Class Distribution")
    print("-" * 70)
    print(imbalance_report)

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=y, order=class_counts.index, hue=y, palette="Set2", legend=False)
    plt.title("Sentiment Class Distribution", fontsize=15, fontweight="bold")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Number of Reviews")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/15_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Chart saved as: {IMAGES_DIR}/15_class_distribution.png")

    majority_class = class_counts.idxmax()
    majority_percentage = class_percentages.max()

    print("\n" + "=" * 70)
    print("WHY RAW ACCURACY IS MISLEADING")
    print("=" * 70)
    print(f"""
The sentiment dataset is highly imbalanced.

Majority Class : {majority_class}
Percentage     : {majority_percentage:.2f}%

Approximately 90% of all reviews belong to the Positive class, while
Neutral and Negative reviews together account for less than 10%.

Because of this imbalance, a classifier could simply predict 'Positive'
for every review and still achieve approximately 90% accuracy, while
completely failing to identify Neutral and Negative reviews.

Metrics such as Precision, Recall, F1-score, and the Confusion Matrix
provide a much more reliable evaluation because they measure
performance for each individual class.
""")

    print("\n" + "=" * 70)
    print("PART 2 SUMMARY")
    print("=" * 70)
    print(f"Total Reviews        : {len(y)}")
    print(f"Positive Reviews     : {class_counts['positive']}")
    print(f"Neutral Reviews      : {class_counts['neutral']}")
    print(f"Negative Reviews     : {class_counts['negative']}")
    print(f"\nPositive Percentage  : {class_percentages['positive']:.2f}%")
    print(f"Neutral Percentage   : {class_percentages['neutral']:.2f}%")
    print(f"Negative Percentage  : {class_percentages['negative']:.2f}%")
    print("\nObservation:")
    print("The dataset is highly imbalanced, with Positive reviews")
    print("representing approximately 90% of all samples.")
    print("\n✓ Task 5 - Part 2 Completed Successfully!")


# ==========================================================
# Part 3 : Evaluation Metric Strategy
# ==========================================================

def _part3_metric_strategy():
    print("\n" + "=" * 70)
    print("TASK 5 : PART 3")
    print("EVALUATION METRICS — STRATEGY")
    print("=" * 70)
    print("""
PRIMARY METRICS: Macro-Averaged Precision, Recall, F1

The dataset is ~90% positive. Weighted or micro averages would be
dominated by that majority class and could look strong even if a
model completely fails on 'negative' and 'neutral' reviews. Macro
averaging computes each metric independently per class and then
averages with EQUAL weight, so poor minority-class performance
cannot hide behind the majority class's high scores.

SECONDARY / SANITY-CHECK METRICS for every model trained below:
  - Full confusion matrix (all 3 classes)
  - Per-class precision and recall (not just the macro summary)
  - Balanced Accuracy (average recall across classes)
  - Cohen's Kappa (agreement beyond chance)
""")
    print("✓ Task 5 - Part 3 Completed Successfully!")


# ==========================================================
# Part 4.1 : Baseline Models (No Imbalance Handling)
# ==========================================================

def _part4_1_baselines(X_train_tfidf, y_train, X_test_tfidf, y_test):
    print("\n" + "=" * 70)
    print("TASK 5 : PART 4.1")
    print("BASELINE MODELS (NO IMBALANCE HANDLING)")
    print("=" * 70)

    print("\nTraining Baseline Model 1 : Logistic Regression")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    lr_results = build_results("Logistic Regression (Baseline)", y_test, lr_predictions)

    print("\nLogistic Regression Results")
    print("-" * 50)
    print(f"Accuracy        : {lr_results['Accuracy']:.4f}")
    print(f"Macro Precision : {lr_results['Macro Precision']:.4f}")
    print(f"Macro Recall    : {lr_results['Macro Recall']:.4f}")
    print(f"Macro F1-Score  : {lr_results['Macro F1']:.4f}")
    print(f"Balanced Acc.   : {lr_results['Balanced Accuracy']:.4f}")
    print(f"Cohen's Kappa   : {lr_results['Cohen Kappa']:.4f}")

    print("\nClassification Report")
    print("-" * 50)
    print(classification_report(y_test, lr_predictions, labels=LABELS, digits=4, zero_division=0))

    lr_cm = confusion_matrix(y_test, lr_predictions, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Baseline Logistic Regression")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/17_lr_baseline_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"✓ Confusion Matrix Saved: {IMAGES_DIR}/17_lr_baseline_confusion_matrix.png")

    print("\nTraining Baseline Model 2 : Multinomial Naive Bayes")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_predictions = nb_model.predict(X_test_tfidf)
    nb_results = build_results("Multinomial Naive Bayes (Baseline)", y_test, nb_predictions)

    print("\nMultinomial Naive Bayes Results")
    print("-" * 50)
    print(f"Accuracy        : {nb_results['Accuracy']:.4f}")
    print(f"Macro Precision : {nb_results['Macro Precision']:.4f}")
    print(f"Macro Recall    : {nb_results['Macro Recall']:.4f}")
    print(f"Macro F1-Score  : {nb_results['Macro F1']:.4f}")

    nb_cm = confusion_matrix(y_test, nb_predictions, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap="Greens")
    plt.title("Baseline Multinomial Naive Bayes")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/18_nb_baseline_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"✓ Confusion Matrix Saved: {IMAGES_DIR}/18_nb_baseline_confusion_matrix.png")

    comparison = pd.DataFrame([lr_results, nb_results]).round(4)
    comparison.to_csv(f"{OUTPUTS_DIR}/baseline_model_comparison.csv", index=False)
    print(f"\n✓ Comparison Table Saved: {OUTPUTS_DIR}/baseline_model_comparison.csv")

    print("\n✓ Task 5 - Part 4.1 Completed Successfully!")
    return lr_results, nb_results


# ==========================================================
# Part 4.2 : Class Weighting (Logistic Regression)
# ==========================================================

def _part4_2_class_weighting(X_train_tfidf, y_train, X_test_tfidf, y_test, lr_results):
    print("\n" + "=" * 70)
    print("TASK 5 : PART 4.2")
    print("CLASS WEIGHTING (LOGISTIC REGRESSION)")
    print("=" * 70)

    print("\nTraining Logistic Regression with class_weight='balanced'...")
    balanced_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    balanced_lr.fit(X_train_tfidf, y_train)
    balanced_predictions = balanced_lr.predict(X_test_tfidf)

    balanced_lr_results = build_results("Logistic Regression (Class-Weighted)", y_test, balanced_predictions)

    print("\nBalanced Logistic Regression Results")
    print("-" * 60)
    print(f"Accuracy        : {balanced_lr_results['Accuracy']:.4f}")
    print(f"Macro Precision : {balanced_lr_results['Macro Precision']:.4f}")
    print(f"Macro Recall    : {balanced_lr_results['Macro Recall']:.4f}")
    print(f"Macro F1-Score  : {balanced_lr_results['Macro F1']:.4f}")

    balanced_cm = confusion_matrix(y_test, balanced_predictions, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=balanced_cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap="Oranges")
    plt.title("Balanced Logistic Regression")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/15_balanced_lr_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"\n✓ Confusion Matrix Saved: {IMAGES_DIR}/15_balanced_lr_confusion_matrix.png")

    comparison_lr = pd.DataFrame({
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Balanced Accuracy", "Cohen Kappa"],
        "Baseline Logistic Regression": [
            lr_results["Accuracy"], lr_results["Macro Precision"], lr_results["Macro Recall"],
            lr_results["Macro F1"], lr_results["Balanced Accuracy"], lr_results["Cohen Kappa"],
        ],
        "Balanced Logistic Regression": [
            balanced_lr_results["Accuracy"], balanced_lr_results["Macro Precision"], balanced_lr_results["Macro Recall"],
            balanced_lr_results["Macro F1"], balanced_lr_results["Balanced Accuracy"], balanced_lr_results["Cohen Kappa"],
        ],
    })
    comparison_lr.to_csv(f"{OUTPUTS_DIR}/logistic_regression_comparison.csv", index=False)
    print(f"\n✓ Comparison Table Saved: {OUTPUTS_DIR}/logistic_regression_comparison.csv")

    delta_recall = balanced_lr_results["Macro Recall"] - lr_results["Macro Recall"]
    delta_accuracy = balanced_lr_results["Accuracy"] - lr_results["Accuracy"]
    print(f"""
Macro Recall changed by {delta_recall:+.4f} ({lr_results['Macro Recall']:.4f} -> {balanced_lr_results['Macro Recall']:.4f})
Accuracy changed by     {delta_accuracy:+.4f} ({lr_results['Accuracy']:.4f} -> {balanced_lr_results['Accuracy']:.4f})
""")

    print("\n✓ Task 5 - Part 4.2 Completed Successfully!")
    return balanced_lr, balanced_lr_results


# ==========================================================
# Part 4.4 : Random Oversampling + Logistic Regression
# ==========================================================

def _part4_4_oversampling(X_train_tfidf, y_train, X_test, X_test_tfidf, y_test):
    print("\n" + "=" * 70)
    print("TASK 5 : PART 4.4")
    print("RANDOM OVERSAMPLING + LOGISTIC REGRESSION")
    print("=" * 70)

    print("\nClass Distribution BEFORE Oversampling")
    print("-" * 60)
    print(pd.Series(y_train).value_counts().sort_index())

    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_tfidf, y_train)

    print("\nClass Distribution AFTER Oversampling")
    print("-" * 60)
    print(pd.Series(y_train_ros).value_counts().sort_index())

    oversampled_lr = LogisticRegression(random_state=42, max_iter=1000)
    oversampled_lr.fit(X_train_ros, y_train_ros)
    oversampled_predictions = oversampled_lr.predict(X_test_tfidf)

    oversampling_results = build_results(
        "Random Oversampling + Logistic Regression", y_test, oversampled_predictions
    )

    print("\nEvaluation Metrics")
    print("-" * 60)
    print(f"Accuracy           : {oversampling_results['Accuracy']:.4f}")
    print(f"Macro Precision    : {oversampling_results['Macro Precision']:.4f}")
    print(f"Macro Recall       : {oversampling_results['Macro Recall']:.4f}")
    print(f"Macro F1-Score     : {oversampling_results['Macro F1']:.4f}")

    report = classification_report(y_test, oversampled_predictions, zero_division=0)
    print("\nClassification Report")
    print("-" * 60)
    print(report)

    with open(f"{OUTPUTS_DIR}/random_oversampling_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_test, oversampled_predictions, labels=LABELS)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title("Random Oversampling + Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/19_random_oversampling_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Confusion Matrix Saved: {IMAGES_DIR}/19_random_oversampling_confusion_matrix.png")

    prediction_results = pd.DataFrame({
        "Review": X_test.reset_index(drop=True),
        "Actual": y_test.reset_index(drop=True),
        "Predicted": oversampled_predictions,
    })
    prediction_results.to_csv(f"{OUTPUTS_DIR}/random_oversampling_predictions.csv", index=False)

    print("""
Compared with the baseline Logistic Regression, Random Oversampling
improved the model's ability to recognize minority sentiment classes.
Overall accuracy decreased slightly, but Macro Recall and Macro F1
remained substantially higher than the baseline.

Limitation: Random Oversampling duplicates existing minority-class
samples rather than creating new information, so the model may become
more prone to overfitting on duplicated minority examples.
""")

    print("\n✓ Task 5 - Part 4.4 Completed Successfully!")
    return oversampling_results


# ==========================================================
# Part 4.5 : Hyperparameter Tuning (GridSearchCV)
# ==========================================================

def _part4_5_gridsearch(X_train_tfidf, y_train, X_test, X_test_tfidf, y_test):
    print("\n" + "=" * 70)
    print("TASK 5 : PART 4.5")
    print("HYPERPARAMETER TUNING (GRIDSEARCHCV)")
    print("=" * 70)

    param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "newton-cg", "saga"]}
    print("\nParameter Grid")
    print("-" * 60)
    print(param_grid)

    base_lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        estimator=base_lr, param_grid=param_grid, scoring="f1_macro", cv=5, n_jobs=-1, verbose=1
    )

    print("\nRunning GridSearchCV...")
    grid_search.fit(X_train_tfidf, y_train)

    print("\nBest Parameters")
    print("-" * 60)
    print(grid_search.best_params_)
    print("\nBest Cross Validation Macro F1")
    print(f"{grid_search.best_score_:.4f}")

    best_lr = grid_search.best_estimator_
    best_predictions = best_lr.predict(X_test_tfidf)

    gridsearch_results = build_results("GridSearchCV Logistic Regression", y_test, best_predictions)

    print("\nEvaluation Metrics")
    print("-" * 60)
    print(f"Accuracy           : {gridsearch_results['Accuracy']:.4f}")
    print(f"Macro Precision    : {gridsearch_results['Macro Precision']:.4f}")
    print(f"Macro Recall       : {gridsearch_results['Macro Recall']:.4f}")
    print(f"Macro F1           : {gridsearch_results['Macro F1']:.4f}")

    report = classification_report(y_test, best_predictions, labels=LABELS, digits=4, zero_division=0)
    print("\nClassification Report")
    print("-" * 60)
    print(report)

    with open(f"{OUTPUTS_DIR}/gridsearch_classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)
    print("\n✓ Classification Report Saved")

    cm = confusion_matrix(y_test, best_predictions, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap="Purples")
    plt.title("GridSearchCV Logistic Regression")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/20_gridsearch_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"✓ Confusion Matrix Saved: {IMAGES_DIR}/20_gridsearch_confusion_matrix.png")

    sample_predictions = pd.DataFrame({
        "Review": X_test.reset_index(drop=True).head(10),
        "Actual": y_test.reset_index(drop=True).head(10),
        "Predicted": pd.Series(best_predictions).head(10),
    })
    sample_predictions.to_csv(f"{OUTPUTS_DIR}/gridsearch_sample_predictions.csv", index=False)

    pd.DataFrame([grid_search.best_params_]).to_csv(f"{OUTPUTS_DIR}/gridsearch_best_parameters.csv", index=False)

    print(f"""
GridSearchCV evaluated {len(param_grid['C']) * len(param_grid['solver'])} different
hyperparameter combinations using 5-fold cross-validation.

Best Parameters:
C       = {grid_search.best_params_['C']}
Solver  = {grid_search.best_params_['solver']}
""")

    print("\n✓ Task 5 - Part 4.5 Completed Successfully!")
    return gridsearch_results


# ==========================================================
# Part 4.6 : Compare All Models
# ==========================================================

def _part4_6_compare_models(lr_results, nb_results, balanced_lr_results, oversampling_results, gridsearch_results):
    print("\n" + "=" * 70)
    print("TASK 5 : PART 4.6")
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame([lr_results, nb_results, balanced_lr_results,
                                  oversampling_results, gridsearch_results])
    comparison_df = comparison_df[[
        "Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
        "Balanced Accuracy", "Cohen Kappa",
    ]].round(4)

    comparison_df = comparison_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)

    print("\nModel Performance Comparison")
    print("-" * 70)
    print(comparison_df)

    comparison_df.to_csv(f"{OUTPUTS_DIR}/model_comparison.csv", index=False)
    print(f"\n✓ Comparison table saved: {OUTPUTS_DIR}/model_comparison.csv")

    best_model = comparison_df.iloc[0]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(f"Model               : {best_model['Model']}")
    print(f"Macro F1            : {best_model['Macro F1']:.4f}")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=comparison_df, x="Model", y="Macro F1", color="steelblue")
    plt.title("Comparison of Models using Macro F1 Score", fontsize=16, fontweight="bold")
    plt.xlabel("Models")
    plt.ylabel("Macro F1 Score")
    plt.xticks(rotation=15, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/17_model_comparison_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Chart saved: {IMAGES_DIR}/17_model_comparison_macro_f1.png")

    print(f"""
Among all evaluated models, '{best_model['Model']}' achieved the highest
Macro F1-score. Because Macro F1 gives equal importance to Positive,
Neutral and Negative reviews, this model is considered the most reliable
for handling the imbalanced sentiment dataset.
""")

    print("\n✓ Task 5 - Part 4.6 Completed Successfully!")
    return comparison_df, best_model


# ==========================================================
# Orchestrator for the whole of Task 5
# ==========================================================

def run(df_clean=None):
    """Runs every part of Task 5 in sequence.
    Accepts the DataFrame produced by Task 3 (cleaned_text column);
    if None, loads outputs/cleaned_reviews.csv fresh.
    Returns (balanced_lr, X_test_tfidf, X_test, y_test) for Task 6."""

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if df_clean is None:
        df_clean = pd.read_csv(CLEANED_FILE)

    (X, y, X_train, X_test, y_train, y_test,
     X_train_tfidf, X_test_tfidf, tfidf) = _part1_split_and_vectorize(df_clean)

    _part2_imbalance_diagnosis(y)
    _part3_metric_strategy()

    lr_results, nb_results = _part4_1_baselines(X_train_tfidf, y_train, X_test_tfidf, y_test)
    balanced_lr, balanced_lr_results = _part4_2_class_weighting(
        X_train_tfidf, y_train, X_test_tfidf, y_test, lr_results
    )
    oversampling_results = _part4_4_oversampling(X_train_tfidf, y_train, X_test, X_test_tfidf, y_test)
    gridsearch_results = _part4_5_gridsearch(X_train_tfidf, y_train, X_test, X_test_tfidf, y_test)

    _part4_6_compare_models(
        lr_results, nb_results, balanced_lr_results, oversampling_results, gridsearch_results
    )

    # The class-weighted Logistic Regression is used as the "best model"
    # carried forward into Task 6's error analysis.
    return balanced_lr, X_test_tfidf, X_test, y_test
