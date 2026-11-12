import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.svm import LinearSVC

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

OUTPUT_DIR = ROOT / "reports/scaled_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

X = df["review"].astype(str)

y = df["csv_label"].astype(int)

groups = df["hotel"].astype(str)

print("\nDataset:", len(df))
print("Hotels:", groups.nunique())


# --------------------------------------------------
# 3. Feature extractor
# --------------------------------------------------

def create_features(linguistic_weight=None):

    transformers = [

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
    ]

    weights = None

    if linguistic_weight is not None:

        linguistic_pipeline = Pipeline([

            (
                "extractor",
                LinguisticFeatureExtractor()
            ),

            (
                "scaler",
                StandardScaler(with_mean=False)
            )

        ])

        transformers.append(
            ("linguistic", linguistic_pipeline)
        )

        weights = {
            "word": 1.0,
            "char": 1.0,
            "linguistic": linguistic_weight
        }

    return FeatureUnion(
        transformers,
        transformer_weights=weights
    )


# --------------------------------------------------
# 4. Experiments
# --------------------------------------------------

experiments = {

    "TFIDF_BASELINE": None,

    "LINGUISTIC_0.01": 0.01,

    "LINGUISTIC_0.05": 0.05,

    "LINGUISTIC_0.10": 0.10,

    "LINGUISTIC_0.25": 0.25,

    "LINGUISTIC_0.50": 0.50

}


# --------------------------------------------------
# 5. Hotel-independent evaluation
# --------------------------------------------------

logo = LeaveOneGroupOut()

summary_results = []

fold_results = []

prediction_results = []

for experiment_name, weight in experiments.items():

    print("\n" + "=" * 60)

    print(experiment_name)

    print("=" * 60)

    actual_values = []

    predicted_values = []

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

        features = create_features(weight)

        X_train_features = features.fit_transform(X_train)

        X_test_features = features.transform(X_test)

        model = LinearSVC(
            C=10,
            max_iter=50000,
            random_state=42
        )

        model.fit(X_train_features, y_train)

        predictions = model.predict(X_test_features)

        actual_values.extend(y_test.tolist())

        predicted_values.extend(predictions.tolist())

        fold_results.append({

            "experiment": experiment_name,

            "fold": fold,

            "hotel": groups.iloc[test_idx[0]],

            "accuracy": accuracy_score(
                y_test, predictions
            ),

            "f1": f1_score(
                y_test, predictions,
                zero_division=0
            )

        })

        for index, prediction in zip(
            test_idx, predictions
        ):

            prediction_results.append({

                "experiment": experiment_name,

                "hotel": groups.iloc[index],

                "actual": int(y.iloc[index]),

                "predicted": int(prediction)

            })

    # Overall metrics

    accuracy = accuracy_score(
        actual_values,
        predicted_values
    )

    precision = precision_score(
        actual_values,
        predicted_values,
        zero_division=0
    )

    recall = recall_score(
        actual_values,
        predicted_values,
        zero_division=0
    )

    f1 = f1_score(
        actual_values,
        predicted_values,
        zero_division=0
    )

    summary_results.append({

        "experiment": experiment_name,

        "linguistic_weight": weight,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1

    })

    print("\nAccuracy:", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall:", round(recall, 4))

    print("F1:", round(f1, 4))

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            actual_values,
            predicted_values
        )
    )


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)

summary_df = summary_df.sort_values(
    by="f1",
    ascending=False
)

summary_df.to_csv(
    OUTPUT_DIR / "scaled_feature_comparison.csv",
    index=False
)

pd.DataFrame(fold_results).to_csv(
    OUTPUT_DIR / "scaled_fold_results.csv",
    index=False
)

pd.DataFrame(prediction_results).to_csv(
    OUTPUT_DIR / "scaled_predictions.csv",
    index=False
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n" + "=" * 60)

print("FINAL SCALED FEATURE COMPARISON")

print("=" * 60)

print(summary_df.to_string(index=False))

print("\nResults saved to:")

print(OUTPUT_DIR)

print("\nSCALED FEATURE EXPERIMENT COMPLETE!")