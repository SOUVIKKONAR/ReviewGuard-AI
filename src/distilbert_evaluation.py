import os
import random
import time
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from torch.amp import autocast, GradScaler
from tqdm import tqdm


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/processed/verified_dataset.csv"

MODEL_NAME = "distilbert/distilbert-base-uncased"

OUTPUT_DIR = "reports/distilbert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
N_SPLITS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# DATASET CLASS
# =====================================================

class ReviewDataset(Dataset):

    def __init__(self, reviews, labels, tokenizer):

        self.encodings = tokenizer(
            reviews,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


# =====================================================
# METRICS
# =====================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# =====================================================
# TRAIN DISTILBERT
# =====================================================

def train_distilbert(train_loader, test_loader, fold):

    print(f"\nInitializing DistilBERT for Fold {fold}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    start_time = time.time()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        progress = tqdm(
            train_loader,
            desc=f"Fold {fold} | Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch in progress:

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_loss = total_loss / len(train_loader)

        print(
            f"Fold {fold} | Epoch {epoch + 1} "
            f"| Average Loss: {average_loss:.4f}"
        )

    training_time = time.time() - start_time

    # ---------------------------------------------
    # EVALUATION
    # ---------------------------------------------

    model.eval()

    predictions = []
    actual_labels = []

    inference_start = time.time()

    with torch.no_grad():

        for batch in tqdm(
            test_loader,
            desc=f"Evaluating Fold {fold}"
        ):

            labels = batch["labels"].numpy()

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            with autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

            predicted = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            actual_labels.extend(labels)

    inference_time = time.time() - inference_start

    del model
    del optimizer
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        np.array(actual_labels),
        np.array(predictions),
        training_time,
        inference_time
    )


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("REVIEWGUARD - DISTILBERT EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print("GPU:", torch.cuda.get_device_name(0))

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    # ---------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "review",
        "truth_label",
        "hotel"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.dropna(
        subset=required_columns
    ).copy()

    df["review"] = df["review"].astype(str).str.strip()

    df = df[df["review"] != ""]

    # Remove exact duplicate reviews
    df = df.drop_duplicates(
        subset=["review"]
    ).reset_index(drop=True)

    # Label mapping:
    # 0 = Truthful
    # 1 = Deceptive

    df["truth_label"] = df["truth_label"].map({
        "truthful": 0,
        "deceptive": 1,
        "Truthful": 0,
        "Deceptive": 1
    }).fillna(df["truth_label"])

    df["truth_label"] = df["truth_label"].astype(int)

    if not set(df["truth_label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected labels. Expected 0 and 1.")

    reviews = df["review"].tolist()
    labels = df["truth_label"].to_numpy()
    groups = df["hotel"].astype(str).to_numpy()

    print("\nDataset:", len(df))
    print("\nLabel distribution:")
    print(df["truth_label"].value_counts())

    print("\nUnique hotels:", df["hotel"].nunique())

    # ---------------------------------------------
    # LOAD TOKENIZER
    # ---------------------------------------------

    print("\nLoading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------------------------
    # GROUP CROSS VALIDATION
    # ---------------------------------------------

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    all_results = []
    all_predictions = {}

    model_predictions = {
        "DistilBERT": [],
        "SVM": []
    }

    y_all = []

    for fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(reviews, labels, groups),
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("=" * 60)

        train_reviews = [reviews[i] for i in train_idx]
        test_reviews = [reviews[i] for i in test_idx]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        train_hotels = set(groups[train_idx])
        test_hotels = set(groups[test_idx])

        assert train_hotels.isdisjoint(test_hotels)

        print("Training reviews:", len(train_idx))
        print("Testing reviews:", len(test_idx))

        print("Training hotels:", len(train_hotels))
        print("Testing hotels:", len(test_hotels))

        print("Test hotels:", sorted(test_hotels))

        # -----------------------------------------
        # DISTILBERT DATA
        # -----------------------------------------

        train_dataset = ReviewDataset(
            train_reviews,
            y_train,
            tokenizer
        )

        test_dataset = ReviewDataset(
            test_reviews,
            y_test,
            tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda")
        )

        # -----------------------------------------
        # DISTILBERT TRAINING
        # -----------------------------------------

        (
            bert_actual,
            bert_pred,
            bert_train_time,
            bert_inference_time
        ) = train_distilbert(
            train_loader,
            test_loader,
            fold
        )

        bert_metrics = calculate_metrics(
            bert_actual,
            bert_pred
        )

        all_results.append({
            "fold": fold,
            "model": "DistilBERT",
            **bert_metrics,
            "training_time_seconds": bert_train_time,
            "inference_time_seconds": bert_inference_time
        })

        model_predictions["DistilBERT"].extend(
            bert_pred.tolist()
        )

        # -----------------------------------------
        # SVM BASELINE - SAME FOLD
        # -----------------------------------------

        print("\nTraining SVM baseline...")

        vectorizer = FeatureUnion([

            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=30000,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True
                )
            ),

            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=20000,
                    min_df=2,
                    sublinear_tf=True
                )
            )

        ])

        svm_start = time.time()

        X_train = vectorizer.fit_transform(
            train_reviews
        )

        X_test = vectorizer.transform(
            test_reviews
        )

        svm = LinearSVC(
            C=10,
            max_iter=10000,
            random_state=SEED
        )

        svm.fit(X_train, y_train)

        svm_pred = svm.predict(X_test)

        svm_train_time = time.time() - svm_start

        svm_metrics = calculate_metrics(
            y_test,
            svm_pred
        )

        all_results.append({
            "fold": fold,
            "model": "SVM",
            **svm_metrics,
            "training_time_seconds": svm_train_time,
            "inference_time_seconds": None
        })

        model_predictions["SVM"].extend(
            svm_pred.tolist()
        )

        y_all.extend(y_test.tolist())

        print("\nFold Results:")

        print("DistilBERT:", bert_metrics)
        print("SVM:", svm_metrics)

        del vectorizer, X_train, X_test, svm

    # ---------------------------------------------
    # SAVE FOLD RESULTS
    # ---------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        f"{OUTPUT_DIR}/fold_results.csv",
        index=False
    )

    # ---------------------------------------------
    # OVERALL OUT-OF-FOLD METRICS
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL FIVE-FOLD RESULTS")
    print("=" * 60)

    overall_results = []

    for model_name in ["DistilBERT", "SVM"]:

        predictions = np.array(
            model_predictions[model_name]
        )

        actual = np.array(y_all)

        metrics = calculate_metrics(
            actual,
            predictions
        )

        cm = confusion_matrix(
            actual,
            predictions
        )

        print(f"\n{model_name}")

        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Precision:", round(metrics["precision"], 4))
        print("Recall:", round(metrics["recall"], 4))
        print("F1 Score:", round(metrics["f1"], 4))

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(
            classification_report(
                actual,
                predictions,
                target_names=[
                    "Truthful",
                    "Deceptive"
                ],
                zero_division=0
            )
        )

        overall_results.append({
            "model": model_name,
            **metrics,
            "confusion_matrix": cm.tolist()
        })

        pd.DataFrame({
            "actual": actual,
            "predicted": predictions
        }).to_csv(
            f"{OUTPUT_DIR}/{model_name.lower()}_predictions.csv",
            index=False
        )

    with open(
        f"{OUTPUT_DIR}/overall_metrics.json",
        "w"
    ) as file:

        json.dump(
            overall_results,
            file,
            indent=4
        )

    print("\nAll results saved in:", OUTPUT_DIR)

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()