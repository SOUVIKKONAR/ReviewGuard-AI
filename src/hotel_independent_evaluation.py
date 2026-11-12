import pandas as pd
import numpy as np
import json

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/verified_dataset.csv"

OUTPUT_DIR = ROOT / "reports/hotel_independent"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load verified dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

X = df["review"].astype(str)
y = df["csv_label"].astype(int)
groups = df["hotel"].astype(str)

print("\n========== DATASET ==========")

print("Total reviews:", len(df))
print("Hotels:", groups.nunique())

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nHotel distribution:")
print(groups.value_counts())


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

models = {
    "KNN": KNeighborsClassifier(
        n_neighbors=9,
        weights="distance",
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        C=10,
        max_iter=10000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(
        alpha=0.01
    )
}


# --------------------------------------------------
# 4. Hotel-independent cross-validation
# --------------------------------------------------

logo = LeaveOneGroupOut()

results = []

all_predictions = []

for model_name, model in models.items():

    print(f"\n========== {model_name} ==========")

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X, y, groups), total=groups.nunique()),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = groups.iloc[train_idx]
        test_groups = groups.iloc[test_idx]

        # Fit feature extraction only on training data
        features = FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=30000,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode"
                )
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=20000,
                    min_df=2,
                    sublinear_tf=True
                )
            )
        ])

        X_train_vec = features.fit_transform(X_train)
        X_test_vec = features.transform(X_test)

        # Train model
        model.fit(X_train_vec, y_train)

        # Predict
        predictions = model.predict(X_test_vec)

        # Metrics
        metrics = {
            "model": model_name,
            "fold": fold,
            "hotel": test_groups.iloc[0],
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        fold_metrics.append(metrics)

        for index, prediction in zip(test_idx, predictions):

            all_predictions.append({
                "review_id": int(index),
                "original_filename": df.iloc[index]["original_filename"],
                "model": model_name,
                "hotel": groups.iloc[index],
                "actual": int(y.iloc[index]),
                "predicted": int(prediction)
            })

        print(
            f"Fold {fold:02d} | "
            f"Hotel: {test_groups.iloc[0]:15s} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

    # Aggregate metrics across all held-out predictions
    model_predictions = [
        item for item in all_predictions
        if item["model"] == model_name
    ]

    actual = [item["actual"] for item in model_predictions]
    predicted = [item["predicted"] for item in model_predictions]

    summary = {
        "model": model_name,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }

    results.append(summary)

    # Save fold-level results
    pd.DataFrame(fold_metrics).to_csv(
        OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_folds.csv",
        index=False
    )

    print("\nAggregated results:")
    print(classification_report(actual, predicted, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted))


# --------------------------------------------------
# 5. Save final results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "hotel_independent_comparison.csv",
    index=False
)

pd.DataFrame(all_predictions).to_csv(
    OUTPUT_DIR / "hotel_independent_predictions.csv",
    index=False
)

with open(OUTPUT_DIR / "hotel_independent_metrics.json", "w") as file:
    json.dump(results, file, indent=4)


# --------------------------------------------------
# 6. Display comparison
# --------------------------------------------------

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results_df.sort_values(
        by="f1",
        ascending=False
    ).to_string(index=False)
)

print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nHOTEL-INDEPENDENT EVALUATION COMPLETE!")