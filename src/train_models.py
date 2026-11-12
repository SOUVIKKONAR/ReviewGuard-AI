import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.base import clone


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = (
    "DATASET/deceptive_opinion_spam/"
    "Chicago_Hotel_Reviews.csv"
)

MODEL_DIR = "models"
REPORT_DIR = "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =====================================================
# DATA LOADING
# =====================================================

def load_dataset():

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_PATH)

    df = df[["Label", "Ori_Review"]].copy()

    df.dropna(subset=["Label", "Ori_Review"], inplace=True)

    df["Ori_Review"] = df["Ori_Review"].astype(str).str.strip()

    df = df[df["Ori_Review"] != ""]

    df["Label"] = df["Label"].astype(int)

    print("Original reviews:", len(df))

    # Remove exact duplicate review text.
    duplicates = df.duplicated(
        subset=["Ori_Review"]
    ).sum()

    print("Exact duplicates removed:", duplicates)

    df.drop_duplicates(
        subset=["Ori_Review"],
        keep="first",
        inplace=True
    )

    print("Final dataset size:", len(df))

    print("\nClass distribution:")
    print(df["Label"].value_counts())

    return df


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def create_feature_extractor():

    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    char_features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=20000,
        min_df=2,
        sublinear_tf=True
    )

    features = FeatureUnion([
        ("word_tfidf", word_features),
        ("char_tfidf", char_features)
    ])

    return features


# =====================================================
# MODEL CONFIGURATION
# =====================================================

def get_models():

    models = {

        "KNN": {

            "pipeline": Pipeline([
                ("features", create_feature_extractor()),
                ("classifier", KNeighborsClassifier(
                    algorithm="brute",
                    metric="cosine"
                ))
            ]),

            "parameters": {

                "classifier__n_neighbors": [
                    3, 5, 7, 9, 11
                ],

                "classifier__weights": [
                    "uniform",
                    "distance"
                ]

            }
        },


        "SVM": {

            "pipeline": Pipeline([
                ("features", create_feature_extractor()),
                ("classifier", LinearSVC(
                    random_state=RANDOM_STATE,
                    max_iter=10000
                ))
            ]),

            "parameters": {

                "classifier__C": [
                    0.01, 0.1, 1, 10, 100
                ]

            }
        },


        "Naive Bayes": {

            "pipeline": Pipeline([
                ("features", create_feature_extractor()),
                ("classifier", MultinomialNB())
            ]),

            "parameters": {

                "classifier__alpha": [
                    0.01, 0.1, 0.5, 1.0, 2.0
                ]

            }
        }

    }

    return models


# =====================================================
# MODEL EVALUATION
# =====================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    print("\n" + "=" * 55)

    print(f"MODEL: {name}")

    print("=" * 55)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    print("\nConfusion Matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(matrix)

    return {

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Confusion_Matrix": matrix.tolist()

    }


# =====================================================
# TRAINING
# =====================================================

def train_all_models():

    df = load_dataset()

    X = df["Ori_Review"]
    y = df["Label"]

    # Independent training and testing sets.

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    cv = StratifiedKFold(

        n_splits=CV_FOLDS,

        shuffle=True,

        random_state=RANDOM_STATE

    )

    models = get_models()

    results = []

    best_models = {}

    for name, configuration in models.items():

        print("\n\n")
        print("#" * 65)
        print(f"STARTING TRAINING: {name}")
        print("#" * 65)

        grid = GridSearchCV(

            estimator=configuration["pipeline"],

            param_grid=configuration["parameters"],

            scoring="f1_macro",

            cv=cv,

            n_jobs=-1,

            verbose=2,

            refit=True,

            return_train_score=True

        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        print("\nBest Parameters:")
        print(grid.best_params_)

        print(
            "\nBest Cross-Validation F1:",
            round(grid.best_score_, 4)
        )

        # Evaluate only on the untouched test set.

        metrics = evaluate_model(
            name,
            best_model,
            X_test,
            y_test
        )

        metrics["Best_Parameters"] = {
            key: str(value)
            for key, value in grid.best_params_.items()
        }

        metrics["CV_F1_Macro"] = float(
            grid.best_score_
        )

        results.append(metrics)

        best_models[name] = best_model

        # Save the trained model.

        filename = (
            name.lower().replace(" ", "_") + ".pkl"
        )

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(
            best_model,
            model_path
        )

        print(f"\nModel saved: {model_path}")

        # Save cross-validation results.

        cv_results = pd.DataFrame(
            grid.cv_results_
        )

        cv_results.to_csv(
            os.path.join(
                REPORT_DIR,
                f"{name.lower().replace(' ', '_')}_cv_results.csv"
            ),
            index=False
        )

    # =================================================
    # FINAL COMPARISON
    # =================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            REPORT_DIR,
            "training_metrics.json"
        ),
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n\n")
    print("=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "CV_F1_Macro"
            ]
        ].to_string(index=False)
    )

    best_model_name = results_df.loc[
        results_df["F1_Score"].idxmax(),
        "Model"
    ]

    print("\nBest model based on test F1-score:")
    print(best_model_name)

    print("\nTraining completed successfully!")

    return best_models, results_df


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    train_all_models()