# Rome Colosseum Visitor Reviews — Sentiment Analysis

A complete, end-to-end NLP pipeline that analyzes **8,266 visitor reviews**
of the Colosseum (Rome) to understand sentiment patterns, compare a
lexicon-based approach (VADER) against multiple machine learning
classifiers, and identify where automated sentiment classification breaks
down on a highly imbalanced (~90% positive) dataset.

The project is organized into **6 tasks**, each implemented as an
independent, importable module, and orchestrated end-to-end by a single
`main.py`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Breakdown (Task 1–6)](#pipeline-breakdown-task-1-6)
- [Outputs Generated](#outputs-generated)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [Error Analysis Highlights](#error-analysis-highlights)
- [Requirements](#requirements)
- [Notes / Limitations](#notes--limitations)

---

## Project Overview

Visitor reviews of the Colosseum are collected and analyzed to answer:

- What is the overall sentiment of visitors (positive / neutral / negative)?
- Does sentiment vary by **trip type** (Couples, Family, Friends, Solo,
  Business) or **travel season**?
- How well does a simple lexicon-based sentiment tool (**VADER**) perform
  compared to the dataset's actual labels?
- Can a machine learning classifier (trained on TF-IDF features) beat
  VADER, and how do different **imbalance-handling techniques** (class
  weighting, oversampling, hyperparameter tuning) affect performance on a
  dataset that is ~90% positive?
- What kinds of reviews does the best model still get wrong, and why?

---

## Project Structure

```
sentiment_project/
├── main.py                       # Entry point — runs the full pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── src/
    ├── __init__.py
    ├── config.py                  # Shared paths & constants
    ├── utils.py                   # build_results() — shared metrics helper
    ├── task1_data_audit.py        # Task 1: Data loading & schema audit
    ├── task2_eda.py                # Task 2: Exploratory Data Analysis
    ├── task3_preprocessing.py     # Task 3: Text preprocessing
    ├── task4_vader_sentiment.py   # Task 4: Lexicon-based sentiment (VADER)
    ├── task5_ml_classifier.py     # Task 5: ML classifiers + imbalance handling
    └── task6_error_analysis.py   # Task 6: Error analysis on misclassifications
```

Running the pipeline creates two additional folders at runtime (not
committed to source control):

```
images/     # All saved chart PNGs (20+ visualizations)
outputs/    # All saved CSVs, prediction files, and text reports
```

---

## Dataset

The pipeline expects a CSV file named:

```
rome_colosseum_visitor_reviews_final.csv
```

placed in the **project root** (same folder as `main.py`).

**Dataset size used in this run:** 8,266 reviews × 16 columns, 0 missing
values, 0 duplicate rows.

Key columns used by the pipeline:

| Column               | Description                                              |
|-----------------------|-----------------------------------------------------------|
| `text`                | The raw review text                                       |
| `title`               | Review title (used in outlier sampling)                   |
| `rating`              | Star rating (1–5)                                          |
| `sentiment_label`     | Ground-truth sentiment label (Positive/Neutral/Negative)   |
| `tripType`            | COUPLES, FAMILY, FRIENDS, SOLO, BUSINESS, Unknown          |
| `travel_season`       | Spring, Summer, Autumn, Winter                             |
| `published_date`      | Date the review was published                              |
| `travel_month` / `published_month` / `published_year` | Date breakdown fields          |
| `published_platform`  | Platform the review was posted on                           |
| `helpful_votes`       | Number of helpful votes on the review                       |
| `word_count` / `review_length_tier` / `is_verified_travel` | Additional metadata columns present in the dataset |

> The dataset is not included in this repository. Place your own CSV with
> the columns above in the project root before running `main.py`.

---

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd sentiment_project
   ```

2. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   NLTK resources (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`)
   are downloaded automatically on first run by Task 3 if they're missing.

---

## Usage

Place `rome_colosseum_visitor_reviews_final.csv` in the project root, then
run:

```bash
python main.py
```

This runs Tasks 1 → 6 in sequence and prints a detailed, human-readable
report to the console while saving every chart and result file to disk.

### Running an individual task

```python
from src import task1_data_audit, task2_eda

df = task1_data_audit.run()
df = task2_eda.run(df)
```

---

## Pipeline Breakdown (Task 1–6)

### Task 1 — Data Loading & Schema Audit
Loaded 8,266 rows × 16 columns (5 numerical, 10 categorical). No missing
values and no duplicate reviews/rows were found in this dataset.

### Task 2 — Exploratory Data Analysis

**Part 1 — Ratings & sentiment**
- Star ratings are heavily skewed toward 5 stars: `1★:218, 2★:141, 3★:453,
  4★:1519, 5★:5935`.
- Sentiment labels: `Positive: 7454, Neutral: 453, Negative: 359` — the
  rating-derived sentiment matched the dataset's own labels exactly
  (perfect diagonal in the comparison table), confirming labels were
  generated directly from ratings.

**Part 2 — Trip type & travel season**
- Review volume by trip type: `COUPLES 3832, FAMILY 1825, Unknown 1228,
  FRIENDS 782, SOLO 559, BUSINESS 40`.
- Average rating by trip type: `BUSINESS 4.68` (highest) → `FRIENDS 4.50`
  (lowest).
- Review volume by season: `Summer 2773, Autumn 2597, Spring 1660, Winter
  1236`.
- Average rating by season: `Autumn 4.59` (highest) → `Summer 4.52`
  (lowest).

**Part 3 — Review length & outliers**
- Average review: **65.08 words / 354.83 characters**.
- Longest review: 1,028 words; shortest: 1 word.
- IQR bounds: Q1 = 29, Q3 = 77 → **604 reviews flagged as length
  outliers**, saved to `outputs/outliers_review.csv`.

### Task 3 — Text Preprocessing
Cleaned all 8,266 reviews (lowercasing, number/punctuation removal,
tokenization, negation-aware stopword removal, lemmatization). Only
**1 review** became empty after cleaning. Verified negation preservation
end-to-end, e.g.:

```
'This place is not good.'        -> 'place not good'
'I would never come back here.'  -> 'would never come back'
```

### Task 4 — Lexicon-Based Sentiment Analysis (VADER)
Applied VADER directly to the raw text and compared it against the true
labels:

| Metric | Score |
|---|---|
| Accuracy | 0.8462 |
| Weighted F1 | 0.8513 |
| **Macro F1** | **0.4505** |
| Weakest class | Neutral (F1 = 0.06) |

VADER's high accuracy is driven almost entirely by the Positive class; it
essentially cannot detect Neutral reviews (precision 0.09, recall 0.05).

### Task 5 — Machine Learning Classifier
- **Part 1:** Stratified 80/20 split (6,612 train / 1,654 test) + TF-IDF
  (5,000 features, unigrams+bigrams, `min_df=2`, `max_df=0.95`).
- **Part 2:** Confirmed imbalance — Positive 90.18%, Neutral 5.48%,
  Negative 4.34%.
- **Part 3:** Declared Macro F1 as the primary metric.
- **Part 4:** Trained and compared 5 models (see [Results](#results)).

### Task 6 — Error Analysis
Ran error analysis on the best model (**Class-Weighted Logistic
Regression**): **234 out of 1,654** test reviews were misclassified
(**14.15% error rate**), saved to `outputs/misclassified_reviews.csv`.

---

## Outputs Generated

**`images/`** — 20+ PNG charts: rating & sentiment distributions, trip
type/season breakdowns, review length histograms, outlier boxplot,
class distribution chart, confusion matrices for VADER + every ML model,
and the final model-comparison bar chart.

**`outputs/`**

| File | Description |
|---|---|
| `outliers_review.csv` | 604 reviews flagged as length outliers |
| `cleaned_reviews.csv` | 8,266 preprocessed reviews with `cleaned_text` |
| `vader_predictions.csv` | VADER predictions vs. true labels |
| `baseline_model_comparison.csv` | LR vs. NB baseline results |
| `logistic_regression_comparison.csv` | Baseline vs. class-weighted LR |
| `random_oversampling_report.txt` | Classification report after oversampling |
| `random_oversampling_predictions.csv` | Oversampled model's test predictions |
| `gridsearch_classification_report.txt` | Report for the tuned model |
| `gridsearch_best_parameters.csv` | Best hyperparameters (`C=10, solver=lbfgs`) |
| `model_comparison.csv` | Final ranked comparison of all 5 models |
| `misclassified_reviews.csv` | 234 test-set errors from the best model |

---

## Key Design Decisions

- **Macro F1 over Accuracy/Weighted metrics** — the dataset is ~90%
  positive, so a model predicting "Positive" for everything would still
  score ~90% accuracy while missing every Neutral/Negative review. Macro
  averaging weights every class equally, exposing this.
- **Negation-aware stopword removal** — `not`/`never`/`dont`/`wasnt` etc.
  are explicitly kept out of the stopword list so phrases like *"not
  good"* aren't destroyed during cleaning.
- **TF-IDF fit only on training data** (`fit_transform` on train,
  `transform` on test) — prevents test-vocabulary leakage.
- **VADER on raw text, ML models on cleaned text** — VADER is designed to
  read punctuation/capitalization/intensifiers as signal; TF-IDF benefits
  from the normalized, lemmatized version instead.
- **Oversampling applied only to the training set** — the 1,654-review
  test set stays untouched and representative of real-world proportions.

---

## Results

Final ranked model comparison (sorted by **Macro F1**, from this run):

| Rank | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Accuracy | Cohen's Kappa |
|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression (Class-Weighted)** | 0.8585 | 0.5420 | 0.6087 | **0.5650** | 0.6087 | 0.3811 |
| 2 | Random Oversampling + Logistic Regression | 0.8700 | 0.5470 | 0.5854 | 0.5624 | 0.5854 | 0.3784 |
| 3 | GridSearchCV Logistic Regression (`C=10, solver=lbfgs`) | 0.9057 | 0.6026 | 0.4540 | 0.4940 | 0.4540 | 0.2912 |
| 4 | Logistic Regression (Baseline) | 0.9093 | 0.7425 | 0.3916 | 0.4190 | 0.3916 | 0.1551 |
| 5 | Multinomial Naive Bayes (Baseline) | 0.9015 | 0.3005 | 0.3333 | 0.3161 | 0.3333 | 0.0000 |

**Best model: Logistic Regression (Class-Weighted)** — despite having
lower raw accuracy than the baselines, it achieves the highest Macro F1
and Balanced Accuracy, meaning it is by far the most reliable at actually
detecting Neutral and Negative reviews rather than defaulting to
Positive.

*For comparison, VADER (Task 4) scored Accuracy 0.8462 / Macro F1 0.4505
— close to the untuned baseline LR but well below the class-weighted
model.*

**Takeaway:** Class weighting gave the single biggest jump in minority-
class detection (Macro Recall **0.3916 → 0.6087**, a +0.2171 improvement)
at a modest cost to raw accuracy (−0.0508) — the expected and worthwhile
trade-off for an imbalanced dataset like this one.

---

## Error Analysis Highlights

Of 1,654 test reviews, **234 (14.15%)** were misclassified by the best
model. Common patterns observed:

1. **Mixed sentiment** — reviews praising the Colosseum while complaining
   about queues, crowds, or ticket prices (e.g. *"queue long hot sun
   could improve service..."* — true Positive, predicted Negative).
2. **Neutral reviews are hardest** — descriptive, low-emotion reviews are
   frequently confused with Positive or Negative.
3. **Short reviews** carry very little signal for TF-IDF.
4. **Queue / ticket / crowd complaint words** inside otherwise positive
   reviews sometimes pull predictions toward Negative.
5. **Negation edge cases** — even with negation words preserved, some
   complex sentence structures still trip up the model.

Overall, TF-IDF treats words independently and can't capture full
sentence-level meaning — mixed-sentiment reviews remain the hardest case
for every model tested.

---

## Requirements

See [`requirements.txt`](requirements.txt):

```
pandas
matplotlib
seaborn
nltk
scikit-learn
vaderSentiment
imbalanced-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Notes / Limitations

- TF-IDF treats words independently and cannot capture full sentence
  meaning — mixed-sentiment reviews (praise + complaint in one review)
  remain the hardest case for every model here.
- The dataset's raw CSV is not included; you must supply your own file
  named `rome_colosseum_visitor_reviews_final.csv` with the required
  columns.
- `images/` and `outputs/` are generated at runtime and are typically
  excluded from version control (see `.gitignore` suggestion below).
- Numbers reported above reflect one run on the full 8,266-review
  dataset; results will vary slightly if the dataset changes.

### Suggested `.gitignore`

```
images/
outputs/
__pycache__/
*.pyc
venv/
rome_colosseum_visitor_reviews_final.csv
```
