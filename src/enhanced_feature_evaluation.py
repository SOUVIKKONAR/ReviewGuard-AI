import json
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.linguistic_features import LinguisticFeatureExtractor


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/verified_dataset.csv"

OUTPUT_DIR = ROOT / "reports/enhanced_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

X = df["review"].astype(str)
y = df["csv_label"].astype(int)
groups = df["hotel"].astype(str)

print("\nDataset size:", len(df))
print("Hotels:", groups.nunique())


# --------------------------------------------------
# 3. Feature extraction configurations
# --------------------------------------------------

def create_features(feature_type):

    transformers = []

    if feature_type in ["tfidf", "combined"]:

        transformers.extend([

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

    if feature_type in ["linguistic", "combined"]:

        transformers.append(
            ("linguistic", LinguisticFeatureExtractor())
        )

    return FeatureUnion(transformers)


# --------------------------------------------------
# 4. Experiments
# --------------------------------------------------

experiments = {

    "E1_TFIDF_SVM": {
        "features": "tfidf",
        "model": LinearSVC(C=10, max_iter=10000, random_state=42)
    },

    "E2_LINGUISTIC_SVM": {
        "features": "linguistic",
        "model": LinearSVC(C=10, max_iter=10000, random_state=42)
    },

    "E3_COMBINED_SVM": {
        "features": "combined",
        "model": LinearSVC(C=10, max_iter=10000, random_state=42)
    },

    "E4_COMBINED_NAIVE_BAYES": {
        "features": "combined",
        "model": MultinomialNB(alpha=0.01)
    },

    "E5_COMBINED_KNN": {
        "features": "combined",
        "model": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            metric="cosine",
            algorithm="brute",
            n_jobs=-1
        )
    }
}


# --------------------------------------------------
# 5. Hotel-independent evaluation
# --------------------------------------------------

logo = LeaveOneGroupOut()

summary_results = []
all_predictions = []
all_fold_results = []

for experiment_name, config in experiments.items():

    print("\n" + "=" * 60)
    print(experiment_name)
    print("=" * 60)

    predictions = []
    actual_values = []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            logo.split(X, y, groups),
            total=groups.nunique()
        ),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Create a fresh feature extractor for every fold
        features = create_features(config["features"])

        X_train_features = features.fit_transform(X_train)
        X_test_features = features.transform(X_test)

        model = config["model"]

        model.fit(X_train_features, y_train)

        fold_predictions = model.predict(X_test_features)

        predictions.extend(fold_predictions)
        actual_values.extend(y_test.tolist())

        metrics = {
            "experiment": experiment_name,
            "fold": fold,
            "hotel": groups.iloc[test_idx[0]],
            "accuracy": accuracy_score(
                y_test, fold_predictions
            ),
            "precision": precision_score(
                y_test, fold_predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_test, fold_predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_test, fold_predictions,
                zero_division=0
            )
        }

        fold_results.append(metrics)
        all_fold_results.append(metrics)

        for index, prediction in zip(
            test_idx, fold_predictions
        ):

            all_predictions.append({
                "experiment": experiment_name,
                "hotel": groups.iloc[index],
                "actual": int(y.iloc[index]),
                "predicted": int(prediction)
            })

    # Aggregate across all 20 held-out hotel folds
    summary = {
        "experiment": experiment_name,
        "accuracy": accuracy_score(
            actual_values, predictions
        ),
        "precision": precision_score(
            actual_values, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            actual_values, predictions,
            zero_division=0
        ),
        "f1": f1_score(
            actual_values, predictions,
            zero_division=0
        )
    }

    summary_results.append(summary)

    print("\nAccuracy:", round(summary["accuracy"], 4))
    print("Precision:", round(summary["precision"], 4))
    print("Recall:", round(summary["recall"], 4))
    print("F1-score:", round(summary["f1"], 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(actual_values, predictions))


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)

summary_df = summary_df.sort_values(
    by="f1",
    ascending=False
)

summary_df.to_csv(
    OUTPUT_DIR / "enhanced_model_comparison.csv",
    index=False
)

pd.DataFrame(all_fold_results).to_csv(
    OUTPUT_DIR / "enhanced_fold_results.csv",
    index=False
)

pd.DataFrame(all_predictions).to_csv(
    OUTPUT_DIR / "enhanced_predictions.csv",
    index=False
)

with open(
    OUTPUT_DIR / "enhanced_metrics.json",
    "w"
) as file:

    json.dump(
        summary_results,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("FINAL ENHANCED MODEL COMPARISON")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nEXPERIMENT COMPLETE!")