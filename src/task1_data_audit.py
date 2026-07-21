# ==========================================================
# Task 1: Data Loading & Schema Audit
# ==========================================================

import os
import pandas as pd

from src.config import DATA_FILE, REVIEW_COLUMN, RATING_COLUMN, SENTIMENT_COLUMN


def run(file_path=DATA_FILE):
    """Loads the raw dataset and prints a full schema / quality audit.
    Returns the loaded (unmodified) DataFrame for use by Task 2."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("TASK 1 : DATA LOADING & SCHEMA AUDIT")
    print("=" * 70)

    print("\nDataset loaded successfully!")

    # ------------------------------------------------------
    # Dataset Shape
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. DATASET SHAPE")
    print("=" * 70)
    print(f"Number of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    # ------------------------------------------------------
    # Column Names
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. COLUMN NAMES")
    print("=" * 70)
    for index, column in enumerate(df.columns, start=1):
        print(f"{index}. {column}")

    # ------------------------------------------------------
    # Data Types
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DATA TYPES")
    print("=" * 70)
    print(df.dtypes)

    # ------------------------------------------------------
    # Dataset Information
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. DATASET INFORMATION")
    print("=" * 70)
    df.info()

    # ------------------------------------------------------
    # Sample Records
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. FIRST FIVE RECORDS")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("LAST FIVE RECORDS")
    print("=" * 70)
    print(df.tail())

    # ------------------------------------------------------
    # Numerical & Categorical Columns
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. FEATURE TYPES")
    print("=" * 70)

    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    print("\nNumerical Columns")
    print(numerical_columns)

    print("\nCategorical Columns")
    print(categorical_columns)

    # ------------------------------------------------------
    # Identify Important Columns
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. IMPORTANT COLUMNS")
    print("=" * 70)

    required_columns = [REVIEW_COLUMN, RATING_COLUMN, SENTIMENT_COLUMN]

    for column in required_columns:
        if column in df.columns:
            print(f"✓ '{column}' column found.")
        else:
            print(f"✗ '{column}' column NOT found.")

    print("\nReview Text Column      :", REVIEW_COLUMN)
    print("Rating Column           :", RATING_COLUMN)
    print("Sentiment Label Column  :", SENTIMENT_COLUMN)

    # ------------------------------------------------------
    # Missing Value Analysis
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_report = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": missing_values,
        "Missing Percentage (%)": missing_percentage.round(2),
    })

    print(missing_report)
    print("\nTotal Missing Values :", missing_values.sum())

    # ------------------------------------------------------
    # Duplicate Review Analysis
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. DUPLICATE REVIEW ANALYSIS")
    print("=" * 70)

    duplicate_reviews = df.duplicated(subset=[REVIEW_COLUMN]).sum()
    print(f"Duplicate Reviews : {duplicate_reviews}")

    if duplicate_reviews > 0:
        print("\nSample Duplicate Reviews\n")
        duplicate_examples = df[df.duplicated(subset=[REVIEW_COLUMN], keep=False)]
        print(duplicate_examples[[REVIEW_COLUMN, RATING_COLUMN, SENTIMENT_COLUMN]].head())
    else:
        print("No duplicate review texts found.")

    # ------------------------------------------------------
    # Duplicate Row Analysis
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("10. DUPLICATE ROW ANALYSIS")
    print("=" * 70)

    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate Rows : {duplicate_rows}")

    # ------------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("11. DATASET SUMMARY")
    print("=" * 70)

    print(f"Total Records             : {len(df)}")
    print(f"Total Features            : {len(df.columns)}")
    print(f"Numerical Features        : {len(numerical_columns)}")
    print(f"Categorical Features      : {len(categorical_columns)}")
    print(f"Review Text Column        : {REVIEW_COLUMN}")
    print(f"Rating Column             : {RATING_COLUMN}")
    print(f"Sentiment Label Column    : {SENTIMENT_COLUMN}")
    print(f"Total Missing Values      : {missing_values.sum()}")
    print(f"Duplicate Reviews         : {duplicate_reviews}")
    print(f"Duplicate Rows            : {duplicate_rows}")

    print("\n" + "=" * 70)
    print("Task 1 Completed Successfully")
    print("=" * 70)

    return df
