import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/verified_dataset.csv"

SIMILARITY_PATH = (
    ROOT / "reports/semantic_similarity/similarity_matrix.npy"
)

OUTPUT_DIR = ROOT / "reports/group_algorithm_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.75, 0.80, 0.85, 0.90]

K_VALUES = [3, 5, 7]

MIN_GROUP_SIZE = 2


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df = (
        df.drop_duplicates(subset=["review"])
        .reset_index(drop=True)
    )

    similarity = np.load(SIMILARITY_PATH)

    if similarity.shape != (len(df), len(df)):
        raise ValueError(
            "Similarity matrix dimensions do not match dataset."
        )

    if not np.isfinite(similarity).all():
        raise ValueError("Similarity matrix contains invalid values.")

    print("\nDataset reviews:", len(df))
    print("Hotels:", df["hotel"].nunique())
    print("Similarity matrix:", similarity.shape)

    return df, similarity


# ==================================================
# 3. GROUP METRICS
# ==================================================

def evaluate_groups(
    df,
    similarity,
    labels,
    algorithm,
    threshold,
    parameter,
    scope
):

    results = []

    unique_groups = np.unique(labels)

    valid_groups = [
        group_id
        for group_id in unique_groups
        if group_id != -1
    ]

    grouped_reviews = 0

    for group_number, group_id in enumerate(valid_groups, start=1):

        members = np.where(labels == group_id)[0]

        if len(members) < MIN_GROUP_SIZE:
            continue

        grouped_reviews += len(members)

        submatrix = similarity[
            np.ix_(members, members)
        ]

        upper = submatrix[
            np.triu_indices(len(members), k=1)
        ]

        actual_edges = np.sum(upper >= threshold)

        possible_edges = len(upper)

        results.append({

            "algorithm": algorithm,

            "scope": scope,

            "threshold": threshold,

            "parameter": parameter,

            "group_id": f"G_{group_number:04d}",

            "group_size": len(members),

            "average_similarity": float(
                np.mean(upper)
            ),

            "minimum_similarity": float(
                np.min(upper)
            ),

            "maximum_similarity": float(
                np.max(upper)
            ),

            "pairwise_density": float(
                actual_edges / possible_edges
            ),

            "hotel_count": int(
                df.iloc[members]["hotel"].nunique()
            )

        })

    group_df = pd.DataFrame(results)

    summary = {

        "algorithm": algorithm,

        "scope": scope,

        "threshold": threshold,

        "parameter": parameter,

        "total_groups": len(group_df),

        "grouped_reviews": grouped_reviews,

        "ungrouped_reviews": (
            len(df) - grouped_reviews
        ),

        "largest_group": (
            int(group_df["group_size"].max())
            if not group_df.empty else 0
        ),

        "average_group_size": (
            round(group_df["group_size"].mean(), 2)
            if not group_df.empty else 0
        ),

        "average_pairwise_similarity": (
            round(group_df["average_similarity"].mean(), 4)
            if not group_df.empty else 0
        ),

        "average_pairwise_density": (
            round(group_df["pairwise_density"].mean(), 4)
            if not group_df.empty else 0
        )

    }

    return group_df, summary


# ==================================================
# 4. CONNECTED COMPONENTS
# ==================================================

def connected_components_groups(similarity, threshold):

    adjacency = csr_matrix(
        (similarity >= threshold).astype(np.int8)
    )

    adjacency.setdiag(0)

    adjacency.eliminate_zeros()

    _, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True
    )

    return labels


# ==================================================
# 5. DBSCAN
# ==================================================

def dbscan_groups(similarity, threshold):

    distance = np.clip(
        1.0 - similarity,
        0.0,
        2.0
    )

    np.fill_diagonal(distance, 0.0)

    model = DBSCAN(
        eps=1.0 - threshold,
        min_samples=2,
        metric="precomputed"
    )

    return model.fit_predict(distance)


# ==================================================
# 6. MUTUAL K-NEAREST NEIGHBOURS
# ==================================================

def mutual_knn_groups(similarity, threshold, k):

    n = similarity.shape[0]

    # Exclude self-similarity

    scores = similarity.copy()

    np.fill_diagonal(scores, -np.inf)

    # Identify top-k neighbours for each review

    top_k = np.argpartition(
        scores,
        -k,
        axis=1
    )[:, -k:]

    neighbour_sets = [
        set(row)
        for row in top_k
    ]

    rows = []
    cols = []

    # Create an edge only when:
    # 1. Reviews are mutual neighbours.
    # 2. Similarity reaches the threshold.

    for i in range(n):

        for j in neighbour_sets[i]:

            if i >= j:
                continue

            mutual = i in neighbour_sets[j]

            if mutual and similarity[i, j] >= threshold:

                rows.extend([i, j])
                cols.extend([j, i])

    adjacency = csr_matrix(
        (
            np.ones(len(rows), dtype=np.int8),
            (rows, cols)
        ),
        shape=(n, n)
    )

    _, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True
    )

    return labels


# ==================================================
# 7. SAME-HOTEL AND CROSS-HOTEL ANALYSIS
# ==================================================

def run_scope_experiment(
    df,
    similarity,
    algorithm,
    threshold,
    parameter=None
):

    n = len(df)

    if algorithm == "Connected Components":

        labels = connected_components_groups(
            similarity,
            threshold
        )

    elif algorithm == "DBSCAN":

        labels = dbscan_groups(
            similarity,
            threshold
        )

    elif algorithm == "Mutual KNN":

        labels = mutual_knn_groups(
            similarity,
            threshold,
            parameter
        )

    else:

        raise ValueError("Unknown algorithm.")

    # Same-hotel scope:
    # Retain edges only between reviews of the same hotel.

    hotel_values = df["hotel"].to_numpy()

    same_hotel_similarity = similarity.copy()

    same_hotel_mask = (
        hotel_values[:, None] == hotel_values[None, :]
    )

    same_hotel_similarity[
        ~same_hotel_mask
    ] = -1.0

    # Cross-hotel scope:
    # Retain edges only between reviews of different hotels.

    cross_hotel_similarity = similarity.copy()

    cross_hotel_mask = (
        hotel_values[:, None] != hotel_values[None, :]
    )

    cross_hotel_similarity[
        ~cross_hotel_mask
    ] = -1.0

    scope_results = []

    for scope, scoped_similarity in [
        ("all_hotels", similarity),
        ("same_hotel", same_hotel_similarity),
        ("cross_hotel", cross_hotel_similarity)
    ]:

        if algorithm == "Connected Components":

            scope_labels = connected_components_groups(
                scoped_similarity,
                threshold
            )

        elif algorithm == "DBSCAN":

            scope_labels = dbscan_groups(
                scoped_similarity,
                threshold
            )

        else:

            scope_labels = mutual_knn_groups(
                scoped_similarity,
                threshold,
                parameter
            )

        group_df, summary = evaluate_groups(
            df,
            similarity,
            scope_labels,
            algorithm,
            threshold,
            parameter,
            scope
        )

        scope_results.append((group_df, summary))

    return scope_results


# ==================================================
# 8. MAIN EXPERIMENT
# ==================================================

def main():

    df, similarity = load_data()

    all_group_results = []

    all_summaries = []

    for threshold in THRESHOLDS:

        print(
            f"\n========== THRESHOLD {threshold} =========="
        )

        algorithms = [
            ("Connected Components", [None]),
            ("DBSCAN", [None]),
            ("Mutual KNN", K_VALUES)
        ]

        for algorithm, parameters in algorithms:

            for parameter in parameters:

                print(
                    f"\nRunning {algorithm} | "
                    f"Parameter: {parameter}"
                )

                scope_results = run_scope_experiment(
                    df,
                    similarity,
                    algorithm,
                    threshold,
                    parameter
                )

                for group_df, summary in scope_results:

                    all_group_results.append(group_df)

                    all_summaries.append(summary)

                    print(
                        f"{summary['scope']}: "
                        f"Groups={summary['total_groups']}, "
                        f"Grouped reviews={summary['grouped_reviews']}, "
                        f"Largest={summary['largest_group']}"
                    )

    final_groups = pd.concat(
        all_group_results,
        ignore_index=True
    )

    final_summary = pd.DataFrame(
        all_summaries
    )

    final_groups.to_csv(
        OUTPUT_DIR / "all_group_details.csv",
        index=False
    )

    final_summary.to_csv(
        OUTPUT_DIR / "algorithm_comparison.csv",
        index=False
    )

    print("\n========== FINAL COMPARISON ==========")

    print(
        final_summary.to_string(index=False)
    )

    print("\nReports saved to:")

    print(OUTPUT_DIR)

    print("\nGROUP ALGORITHM COMPARISON COMPLETED!")


if __name__ == "__main__":

    main()
