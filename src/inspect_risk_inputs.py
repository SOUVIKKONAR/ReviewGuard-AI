import pandas as pd
from pathlib import Path

dataset_path = Path("data/processed/verified_dataset.csv")
predictions_path = Path(
    "reports/hotel_independent/hotel_independent_predictions.csv"
)

dataset = pd.read_csv(dataset_path)
predictions = pd.read_csv(predictions_path)

# Match the dataset preprocessing used in previous experiments
dataset = dataset.drop_duplicates(subset=["review"]).reset_index(drop=True)

print("\n========== DATASET ==========")
print("Total reviews:", len(dataset))
print("\nReviews per hotel:")
print(dataset["hotel"].value_counts().sort_index())

print("\n========== PREDICTIONS ==========")

for model in predictions["model"].unique():

    model_df = predictions[
        predictions["model"] == model
    ].reset_index(drop=True)

    print(f"\nModel: {model}")
    print("Prediction rows:", len(model_df))

    print("\nPredictions per hotel:")
    print(model_df["hotel"].value_counts().sort_index())

    if model == "SVM":

        print("\nChecking actual-label alignment...")

        if len(model_df) != len(dataset):
            print("WARNING: Row counts do not match.")
            continue

        same_hotel = (
            model_df["hotel"].astype(str).values
            == dataset["hotel"].astype(str).values
        )

        same_label = (
            model_df["actual"].astype(int).values
            == dataset["truth_label"]
            .map({"truthful": 0, "deceptive": 1})
            .astype(int)
            .values
        )

        print("Hotel alignment:", same_hotel.all())
        print("Actual label alignment:", same_label.all())

        print(
            "Hotel mismatches:",
            (~same_hotel).sum()
        )

        print(
            "Label mismatches:",
            (~same_label).sum()
        )

print("\n========== SIMILARITY FILE ==========")

pairs = pd.read_csv(
    "reports/semantic_similarity/suspicious_pairs.csv"
)

print("Suspicious pairs:", len(pairs))
print("Minimum review ID:", min(
    pairs["review_id_1"].min(),
    pairs["review_id_2"].min()
))
print("Maximum review ID:", max(
    pairs["review_id_1"].max(),
    pairs["review_id_2"].max()
))

print("\nInput inspection completed.")