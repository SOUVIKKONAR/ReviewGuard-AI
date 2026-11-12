import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/verified_dataset.csv"

SIMILARITY_PATH = (
    ROOT / "reports/semantic_similarity/similarity_matrix.npy"
)

GROUP_PATH = (
    ROOT / "reports/group_risk/threshold_0.85/"
    "review_group_membership.csv"
)

OUTPUT_DIR = ROOT / "reports/group_quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.85


def main():

    df = pd.read_csv(DATA_PATH)

    df = (
        df.drop_duplicates(subset=["review"])
        .reset_index(drop=True)
    )

    similarity = np.load(SIMILARITY_PATH)

    membership = pd.read_csv(GROUP_PATH)

    results = []

    print("\n========== GROUP QUALITY ANALYSIS ==========")

    for group_id, group in membership.groupby("group_id"):

        members = group["review_id"].to_numpy()

        size = len(members)

        submatrix = similarity[
            np.ix_(members, members)
        ]

        upper = submatrix[
            np.triu_indices(size, k=1)
        ]

        hotels = df.iloc[members]["hotel"]

        labels = df.iloc[members]["csv_label"]

        svm_predictions = group["svm_prediction"]

        deceptive_count = int(
            (labels == 1).sum()
        )

        predicted_deceptive_count = int(
            (svm_predictions == 1).sum()
        )

        results.append({

            "group_id": group_id,

            "group_size": size,

            "average_pairwise_similarity": float(
                upper.mean()
            ),

            "minimum_pairwise_similarity": float(
                upper.min()
            ),

            "maximum_pairwise_similarity": float(
                upper.max()
            ),

            "pairwise_density": float(
                np.mean(upper >= THRESHOLD)
            ),

            "hotel_count": hotels.nunique(),

            "deceptive_count": deceptive_count,

            "actual_deceptive_rate": float(
                labels.mean()
            ),

            "svm_deceptive_count": predicted_deceptive_count,

            "svm_deceptive_rate": float(
                svm_predictions.mean()
            )

        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "group_size",
        ascending=False
    )

    result_df.to_csv(
        OUTPUT_DIR / "group_quality_analysis.csv",
        index=False
    )

    print("\nLargest groups:")

    print(
        result_df.head(15).to_string(index=False)
    )

    print("\nOverall group statistics:")

    print(
        result_df[
            [
                "group_size",
                "average_pairwise_similarity",
                "minimum_pairwise_similarity",
                "pairwise_density",
                "hotel_count"
            ]
        ].describe().round(4)
    )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
