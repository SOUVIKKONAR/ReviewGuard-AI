import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/verified_dataset.csv"

PREDICTION_PATH = (
    ROOT / "reports/hotel_independent/"
    "hotel_independent_predictions.csv"
)

SIMILARITY_PATH = (
    ROOT / "reports/semantic_similarity/"
    "similarity_matrix.npy"
)

OUTPUT_DIR = ROOT / "reports/group_risk"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.75, 0.80, 0.85, 0.90]

SELECTED_MODEL = "SVM"

MIN_GROUP_SIZE = 2


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

def load_inputs():

    print("\nLoading project data...")

    df = pd.read_csv(DATA_PATH)

    df = (
        df.drop_duplicates(subset=["review"])
        .reset_index(drop=True)
    )

    predictions = pd.read_csv(PREDICTION_PATH)

    predictions = predictions[
        predictions["model"] == SELECTED_MODEL
    ].copy()

    predictions = predictions.sort_values(
        "review_id"
    ).reset_index(drop=True)

    similarity = np.load(SIMILARITY_PATH)

    # Validate dataset and prediction alignment

    if len(df) != len(predictions):
        raise ValueError(
            "Dataset and prediction counts do not match."
        )

    if not np.array_equal(
        predictions["review_id"].to_numpy(),
        df.index.to_numpy()
    ):
        raise ValueError("Review ID alignment failed.")

    if not np.array_equal(
        predictions["hotel"].to_numpy(),
        df["hotel"].to_numpy()
    ):
        raise ValueError("Hotel alignment failed.")

    if not np.array_equal(
        predictions["actual"].to_numpy(),
        df["csv_label"].to_numpy()
    ):
        raise ValueError("Actual label alignment failed.")

    if similarity.shape != (len(df), len(df)):
        raise ValueError(
            "Similarity matrix dimensions do not match dataset."
        )

    if not np.isfinite(similarity).all():
        raise ValueError("Similarity matrix contains invalid values.")

    print("Dataset:", len(df))
    print("SVM predictions:", len(predictions))
    print("Similarity matrix:", similarity.shape)
    print("All alignment checks passed.")

    df["svm_prediction"] = predictions["predicted"].to_numpy()

    return df, similarity


# ==================================================
# 3. BUILD REVIEW GROUPS
# ==================================================

def build_review_groups(df, similarity, threshold):

    print(f"\nBuilding groups at threshold {threshold}...")

    # Only create edges for sufficiently similar pairs.
    # Each undirected pair is represented once in the graph.

    adjacency = csr_matrix(
        (
            similarity >= threshold
        ).astype(np.int8)
    )

    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Connected components identify groups of linked reviews.

    number_of_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True
    )

    group_data = []

    group_ids = {}

    for component_id in range(number_of_components):

        members = np.where(labels == component_id)[0]

        if len(members) < MIN_GROUP_SIZE:
            continue

        group_id = f"G_{threshold}_{len(group_data) + 1:04d}"

        group_ids[component_id] = group_id

        group_data.append({
            "group_id": group_id,
            "members": members
        })

    return group_data, labels, group_ids


# ==================================================
# 4. CALCULATE GROUP INDICATORS
# ==================================================

def calculate_group_metrics(
    df,
    similarity,
    group_data,
    threshold
):

    group_results = []

    review_results = []

    for group in group_data:

        members = group["members"]

        group_id = group["group_id"]

        group_size = len(members)

        # Similarity submatrix for this group

        submatrix = similarity[
            np.ix_(members, members)
        ]

        upper_indices = np.triu_indices(
            group_size,
            k=1
        )

        pair_scores = submatrix[upper_indices]

        # Group similarity metrics

        average_similarity = float(
            np.mean(pair_scores)
        )

        maximum_similarity = float(
            np.max(pair_scores)
        )

        edge_count = int(
            np.sum(pair_scores >= threshold)
        )

        possible_edges = (
            group_size * (group_size - 1) / 2
        )

        similarity_density = (
            edge_count / possible_edges
        )

        # SVM predictions

        predictions = df.iloc[
            members
        ]["svm_prediction"].to_numpy()

        deceptive_rate = float(
            np.mean(predictions == 1)
        )

        # Actual labels are used for research evaluation,
        # NOT for operational risk calculation.

        actual_labels = df.iloc[
            members
        ]["csv_label"].to_numpy()

        actual_deceptive_rate = float(
            np.mean(actual_labels == 1)
        )

        hotels = sorted(
            df.iloc[members]["hotel"].unique().tolist()
        )

        group_results.append({

            "group_id": group_id,

            "threshold": threshold,

            "group_size": group_size,

            "average_similarity": round(
                average_similarity, 4
            ),

            "maximum_similarity": round(
                maximum_similarity, 4
            ),

            "similarity_density": round(
                similarity_density, 4
            ),

            "similarity_edges": edge_count,

            "possible_edges": int(
                possible_edges
            ),

            "svm_deceptive_rate": round(
                deceptive_rate, 4
            ),

            "actual_deceptive_rate": round(
                actual_deceptive_rate, 4
            ),

            "hotel_count": len(hotels),

            "hotels": ", ".join(hotels)

        })

        # Individual review membership

        for review_id in members:

            review_results.append({

                "review_id": int(review_id),

                "group_id": group_id,

                "hotel": df.iloc[
                    review_id
                ]["hotel"],

                "svm_prediction": int(
                    df.iloc[
                        review_id
                    ]["svm_prediction"]
                ),

                "actual_label": int(
                    df.iloc[
                        review_id
                    ]["csv_label"]
                ),

                "similarity_threshold": threshold

            })

    return group_results, review_results


# ==================================================
# 5. RUN THRESHOLD EXPERIMENTS
# ==================================================

def run_experiments(df, similarity):

    comparison_results = []

    for threshold in THRESHOLDS:

        group_data, labels, group_ids = build_review_groups(
            df,
            similarity,
            threshold
        )

        groups, reviews = calculate_group_metrics(
            df,
            similarity,
            group_data,
            threshold
        )

        groups_df = pd.DataFrame(groups)

        reviews_df = pd.DataFrame(reviews)

        threshold_dir = (
            OUTPUT_DIR / f"threshold_{threshold}"
        )

        threshold_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        groups_df.to_csv(
            threshold_dir / "group_metrics.csv",
            index=False
        )

        reviews_df.to_csv(
            threshold_dir / "review_group_membership.csv",
            index=False
        )

        grouped_reviews = len(reviews_df)

        summary = {

            "threshold": threshold,

            "total_groups": len(groups_df),

            "grouped_reviews": grouped_reviews,

            "ungrouped_reviews": (
                len(df) - grouped_reviews
            ),

            "largest_group": (
                int(groups_df["group_size"].max())
                if not groups_df.empty else 0
            ),

            "average_group_size": (
                round(
                    groups_df["group_size"].mean(), 2
                )
                if not groups_df.empty else 0
            )

        }

        comparison_results.append(summary)

        print("\nThreshold:", threshold)

        print("Groups:", summary["total_groups"])

        print("Grouped reviews:", grouped_reviews)

        print(
            "Ungrouped reviews:",
            summary["ungrouped_reviews"]
        )

        print(
            "Largest group:",
            summary["largest_group"]
        )

    comparison_df = pd.DataFrame(
        comparison_results
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "threshold_comparison.csv",
        index=False
    )

    return comparison_df


# ==================================================
# 6. MAIN
# ==================================================

def main():

    print("\n======================================")
    print(" REVIEWGUARD GROUP RISK ENGINE")
    print("======================================")

    df, similarity = load_inputs()

    comparison = run_experiments(
        df,
        similarity
    )

    print("\n========== THRESHOLD COMPARISON ==========")

    print(
        comparison.to_string(index=False)
    )

    print("\nReports saved to:")

    print(OUTPUT_DIR)

    print("\nGROUP RISK ENGINE COMPLETED!")


if __name__ == "__main__":

    main()