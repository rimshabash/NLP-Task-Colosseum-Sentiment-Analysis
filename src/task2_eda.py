# ==========================================================
# Task 2 : Exploratory Data Analysis (EDA)
# Part 1  : Missing values, rating/sentiment distributions
# Part 2  : Trip type & travel season analysis
# Part 3  : Review length analysis & outlier detection
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.ioff()

from src.config import DATA_FILE, IMAGES_DIR, OUTPUTS_DIR


def _rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


def run(df=None):
    """Runs the full Task 2 EDA pipeline (Parts 1-3).
    Accepts the DataFrame produced by Task 1; if None, loads DATA_FILE
    fresh so this module also works standalone.
    Returns the enriched DataFrame (derived_sentiment, word_count,
    character_count columns added) for use by Task 3."""

    os.makedirs(IMAGES_DIR, exist_ok=True)

    if df is None:
        df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("TASK 2 : EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ==================================================
    # PART 1 : Missing values, rating & sentiment charts
    # ==================================================

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing Values": missing_values,
        "Percentage (%)": missing_percentage.round(2),
    })
    print(missing_report)

    # Drop rows if missing percentage <= 0.5%, otherwise impute
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            percentage = (df[column].isnull().sum() / len(df)) * 100

            print(f"\nProcessing Column : {column}")
            print(f"Missing Percentage : {percentage:.2f}%")

            if percentage <= 0.5:
                rows_before = len(df)
                df = df.dropna(subset=[column])
                rows_removed = rows_before - len(df)
                print(f"Action Taken : Dropped {rows_removed} rows")
            else:
                if pd.api.types.is_numeric_dtype(df[column]):
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    print(f"Action Taken : Filled with Median ({median_value})")
                else:
                    mode_value = df[column].mode()[0]
                    df[column] = df[column].fillna(mode_value)
                    print(f"Action Taken : Filled with Mode ({mode_value})")

    print("\nRemaining Missing Values")
    print(df.isnull().sum())
    print("\nFinal Dataset Shape :", df.shape)

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13

    # ---- Star rating distribution ----
    print("\n" + "=" * 70)
    print("STAR RATING DISTRIBUTION")
    print("=" * 70)
    print(df["rating"].value_counts().sort_index())

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="rating", hue="rating", palette="viridis", legend=False)
    plt.title("Distribution of Star Ratings", fontweight="bold")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Reviews")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/01_rating_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {IMAGES_DIR}/01_rating_distribution.png")

    # ---- Sentiment label distribution ----
    print("\n" + "=" * 70)
    print("SENTIMENT LABEL DISTRIBUTION")
    print("=" * 70)
    print(df["sentiment_label"].value_counts())

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="sentiment_label", hue="sentiment_label", palette="Set2", legend=False)
    plt.title("Distribution of Sentiment Labels", fontweight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/02_sentiment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {IMAGES_DIR}/02_sentiment_distribution.png")

    # ---- Rating-derived sentiment ----
    df["derived_sentiment"] = df["rating"].apply(_rating_to_sentiment)

    print("\nDerived Sentiment Distribution")
    print(df["derived_sentiment"].value_counts())

    comparison = pd.crosstab(df["sentiment_label"], df["derived_sentiment"])
    print("\nComparison Table")
    print(comparison)

    plt.figure(figsize=(7, 5))
    sns.heatmap(comparison, annot=True, fmt="d", cmap="YlGnBu", linewidths=1, linecolor="white")
    plt.title("Dataset Sentiment vs Rating-Derived Sentiment", fontweight="bold")
    plt.xlabel("Rating-Derived Sentiment")
    plt.ylabel("Dataset Sentiment")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/03_sentiment_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {IMAGES_DIR}/03_sentiment_comparison.png")

    print("\n" + "=" * 70)
    print("PART 1 SUMMARY")
    print("=" * 70)
    print(f"Total Reviews               : {len(df)}")
    print(f"Positive Reviews            : {(df['derived_sentiment']=='Positive').sum()}")
    print(f"Neutral Reviews             : {(df['derived_sentiment']=='Neutral').sum()}")
    print(f"Negative Reviews            : {(df['derived_sentiment']=='Negative').sum()}")
    print("\n✓ Task 2 - Part 1 Completed Successfully!")

    # ==================================================
    # PART 2 : Trip Type & Travel Season Analysis
    # ==================================================

    print("\n" + "=" * 70)
    print("TRIP TYPE & TRAVEL SEASON ANALYSIS")
    print("=" * 70)

    required_columns = ["tripType", "travel_season", "rating"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    df_analysis = df.dropna(subset=required_columns)
    print(f"\nDataset used for analysis: {df_analysis.shape}")

    trip_counts = df_analysis["tripType"].value_counts().sort_values(ascending=False)
    print("\nReview Volume by Trip Type")
    print(trip_counts)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=trip_counts.index, y=trip_counts.values, hue=trip_counts.index, palette="tab10", legend=False)
    plt.title("Review Volume by Trip Type", fontsize=16, fontweight="bold")
    plt.xlabel("Trip Type")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=25)
    for i, value in enumerate(trip_counts.values):
        ax.text(i, value + value * 0.01, str(value), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/04_review_volume_tripType.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/04_review_volume_tripType.png")

    trip_rating = df_analysis.groupby("tripType")["rating"].mean().sort_values(ascending=False)
    print("\nAverage Rating by Trip Type")
    print(trip_rating.round(2))

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=trip_rating.index, y=trip_rating.values, hue=trip_rating.index, palette="Spectral", legend=False)
    plt.title("Average Rating by Trip Type", fontsize=16, fontweight="bold")
    plt.xlabel("Trip Type")
    plt.ylabel("Average Rating")
    plt.ylim(0, 5)
    plt.xticks(rotation=25)
    for i, value in enumerate(trip_rating.values):
        ax.text(i, value + 0.05, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/05_average_rating_tripType.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/05_average_rating_tripType.png")

    season_counts = df_analysis["travel_season"].value_counts()
    print("\nReview Volume by Travel Season")
    print(season_counts)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x=season_counts.index, y=season_counts.values, hue=season_counts.index, palette="coolwarm", legend=False)
    plt.title("Review Volume by Travel Season", fontsize=16, fontweight="bold")
    plt.xlabel("Travel Season")
    plt.ylabel("Number of Reviews")
    for i, value in enumerate(season_counts.values):
        ax.text(i, value + value * 0.01, str(value), ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/06_review_volume_travel_season.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/06_review_volume_travel_season.png")

    season_rating = df_analysis.groupby("travel_season")["rating"].mean().sort_values(ascending=False)
    print("\nAverage Rating by Travel Season")
    print(season_rating.round(2))

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x=season_rating.index, y=season_rating.values, hue=season_rating.index, palette="rocket", legend=False)
    plt.title("Average Rating by Travel Season", fontsize=16, fontweight="bold")
    plt.xlabel("Travel Season")
    plt.ylabel("Average Rating")
    plt.ylim(0, 5)
    for i, value in enumerate(season_rating.values):
        ax.text(i, value + 0.05, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/07_average_rating_travel_season.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/07_average_rating_travel_season.png")

    print("\n" + "=" * 70)
    print("SENTIMENT DISTRIBUTION BY TRIP TYPE")
    print("=" * 70)
    trip_sentiment = pd.crosstab(df_analysis["tripType"], df_analysis["derived_sentiment"])
    print(trip_sentiment)

    trip_sentiment.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.title("Sentiment Distribution by Trip Type", fontsize=16, fontweight="bold")
    plt.xlabel("Trip Type")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/08_sentiment_by_trip_type.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/08_sentiment_by_trip_type.png")

    print("\n" + "=" * 70)
    print("SENTIMENT DISTRIBUTION BY TRAVEL SEASON")
    print("=" * 70)
    season_sentiment = pd.crosstab(df_analysis["travel_season"], df_analysis["derived_sentiment"])
    print(season_sentiment)

    season_sentiment.plot(kind="bar", stacked=True, figsize=(9, 6), colormap="viridis")
    plt.title("Sentiment Distribution by Travel Season", fontsize=16, fontweight="bold")
    plt.xlabel("Travel Season")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/09_sentiment_by_travel_season.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/09_sentiment_by_travel_season.png")

    print("\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)
    print("\nTrip Type Statistics")
    print(df_analysis.groupby("tripType")["rating"].agg(
        Review_Count="count", Average_Rating="mean", Minimum_Rating="min", Maximum_Rating="max"
    ).round(2))

    print("\nTravel Season Statistics")
    print(df_analysis.groupby("travel_season")["rating"].agg(
        Review_Count="count", Average_Rating="mean", Minimum_Rating="min", Maximum_Rating="max"
    ).round(2))

    print("\n" + "=" * 70)
    print("PART 2 SUMMARY")
    print("=" * 70)
    print(f"Total Reviews Analyzed : {len(df_analysis)}")
    print(f"Trip Types             : {df_analysis['tripType'].nunique()}")
    print(f"Travel Seasons         : {df_analysis['travel_season'].nunique()}")
    print("\nHighest Rated Trip Type")
    print(trip_rating.idxmax(), "(", round(trip_rating.max(), 2), ")")
    print("\nLowest Rated Trip Type")
    print(trip_rating.idxmin(), "(", round(trip_rating.min(), 2), ")")
    print("\nHighest Rated Travel Season")
    print(season_rating.idxmax(), "(", round(season_rating.max(), 2), ")")
    print("\nLowest Rated Travel Season")
    print(season_rating.idxmin(), "(", round(season_rating.min(), 2), ")")

    df.to_csv(DATA_FILE, index=False)
    print("\n✓ Updated dataset saved successfully.")
    print("Location :", DATA_FILE)
    print("\n✓ Task 2 - Part 2 Completed Successfully!")

    # ==================================================
    # PART 3 : Review Length Analysis & Outlier Detection
    # ==================================================

    print("\n" + "=" * 70)
    print("REVIEW LENGTH ANALYSIS")
    print("=" * 70)

    if "word_count" not in df.columns:
        df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    if "character_count" not in df.columns:
        df["character_count"] = df["text"].astype(str).apply(len)

    print("\nReview Length Statistics (Word Count)")
    print(df["word_count"].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="word_count", bins=30, kde=True, color="royalblue")
    plt.title("Distribution of Review Length (Word Count)", fontsize=16, fontweight="bold")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/10_word_count_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/10_word_count_distribution.png")

    print("\nCharacter Count Statistics")
    print(df["character_count"].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="character_count", bins=30, kde=True, color="darkorange")
    plt.title("Distribution of Review Length (Character Count)", fontsize=16, fontweight="bold")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/11_character_count_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/11_character_count_distribution.png")

    plt.figure(figsize=(12, 4))
    sns.boxplot(x=df["word_count"], color="mediumseagreen")
    plt.title("Word Count Outliers", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/12_word_count_outliers.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart Saved: {IMAGES_DIR}/12_word_count_outliers.png")

    Q1 = df["word_count"].quantile(0.25)
    Q3 = df["word_count"].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    outliers = df[(df["word_count"] < lower_limit) | (df["word_count"] > upper_limit)]

    print("\nOutlier Detection")
    print("-" * 40)
    print("Q1 :", round(Q1, 2))
    print("Q3 :", round(Q3, 2))
    print("IQR :", round(IQR, 2))
    print("Lower Limit :", round(lower_limit, 2))
    print("Upper Limit :", round(upper_limit, 2))
    print("Number of Outliers :", len(outliers))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    outliers.to_csv(f"{OUTPUTS_DIR}/outliers_review.csv", index=False)
    print(f"✓ Outlier file saved: {OUTPUTS_DIR}/outliers_review.csv")

    print("\nSample Outlier Reviews")
    print(outliers[["title", "rating", "tripType", "travel_season", "word_count"]].head(10))

    print("\nAverage Word Count :", round(df["word_count"].mean(), 2))
    print("Average Character Count :", round(df["character_count"].mean(), 2))
    print("Longest Review :", df["word_count"].max(), "words")
    print("Shortest Review :", df["word_count"].min(), "words")

    print("\n" + "=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)
    print(f"""
Total Reviews              : {len(df)}
Average Rating             : {df['rating'].mean():.2f}
Average Word Count         : {df['word_count'].mean():.2f}
Average Character Count    : {df['character_count'].mean():.2f}
Detected Outliers          : {len(outliers)}
Trip Types                 : {df['tripType'].nunique()}
Travel Seasons              : {df['travel_season'].nunique()}
""")

    print("=" * 70)
    print("Task 2 Completed Successfully")
    print("=" * 70)
    print("\nCharts saved inside 'images' folder.")

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
1. Most visitors gave high ratings (4-5 stars), indicating an overall positive experience at the Colosseum.
2. The majority of reviews contain between 30 and 100 words, suggesting that visitors generally provide detailed feedback.
3. A small number of reviews were identified as outliers using the IQR method because they were significantly longer than typical reviews.
4. The average review length is approximately {:.2f} words and {:.2f} characters.
5. Review length varies considerably, but extremely long reviews represent only a small proportion of the dataset.
""".format(df["word_count"].mean(), df["character_count"].mean()))

    return df
