from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse.csgraph import connected_components


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "verified_dataset.csv"
)

PREDICTION_PATH = (
    ROOT_DIR
    / "reports"
    / "hotel_independent"
    / "hotel_independent_predictions.csv"
)

EMBEDDING_PATH = (
    ROOT_DIR
    / "reports"
    / "semantic_similarity"
    / "review_embeddings.npy"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "reports"
    / "mutual_knn_validation"
)

K_VALUES = [3, 5, 7]

THRESHOLDS = [0.80, 0.85, 0.90]

SCOPES = [
    "all_hotels",
    "same_hotel",
    "cross_hotel"
]

MIN_GROUP_SIZE = 2

TEXT_PREVIEW_LENGTH = 250


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_labels(labels):
    """
    Convert truthful/deceptive labels to numeric labels.

    0 = Truthful
    1 = Deceptive
    """

    mapping = {
        "truthful": 0,
        "deceptive": 1
    }

    if pd.api.types.is_numeric_dtype(labels):
        converted = pd.to_numeric(
            labels,
            errors="coerce"
        )
    else:
        converted = (
            labels.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    if converted.isna().any():
        raise ValueError(
            "Unknown or missing labels encountered."
        )

    if not converted.isin([0, 1]).all():
        raise ValueError(
            "Labels must contain only 0 and 1."
        )

    return converted.astype(int).to_numpy()


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():

    print("\nLoading dataset...")

    dataset = pd.read_csv(DATASET_PATH)

    dataset = (
        dataset
        .drop_duplicates(subset=["review"])
        .reset_index(drop=True)
    )

    required_columns = [
        "review",
        "truth_label",
        "hotel",
        "original_filename"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset missing columns: {missing_columns}"
        )

    dataset["review_id"] = np.arange(len(dataset))

    print(f"Dataset rows: {len(dataset)}")

    print("\nLoading SVM predictions...")

    predictions = pd.read_csv(PREDICTION_PATH)

    required_prediction_columns = [
        "review_id",
        "model",
        "hotel",
        "actual",
        "predicted",
        "original_filename"
    ]

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_prediction_columns:
        raise ValueError(
            "Prediction CSV missing columns: "
            f"{missing_prediction_columns}"
        )

    # Select SVM predictions only.
    predictions = predictions[
        predictions["model"]
        .astype(str)
        .str.strip()
        .str.lower() == "svm"
    ].copy()

    predictions["review_id"] = pd.to_numeric(
        predictions["review_id"],
        errors="coerce"
    )

    if predictions["review_id"].isna().any():
        raise ValueError(
            "Invalid review_id values in predictions."
        )

    predictions["review_id"] = (
        predictions["review_id"].astype(int)
    )

    predictions = (
        predictions
        .sort_values("review_id")
        .reset_index(drop=True)
    )

    # Check row counts.
    if len(dataset) != len(predictions):
        raise ValueError(
            f"Dataset rows ({len(dataset)}) and "
            f"SVM prediction rows ({len(predictions)}) "
            "do not match."
        )

    # Check review ID alignment.
    if not np.array_equal(
        dataset["review_id"].to_numpy(),
        predictions["review_id"].to_numpy()
    ):
        raise ValueError(
            "Review ID alignment failed."
        )

    # Check hotel alignment.
    if not np.array_equal(
        dataset["hotel"].astype(str).to_numpy(),
        predictions["hotel"].astype(str).to_numpy()
    ):
        raise ValueError(
            "Hotel alignment failed."
        )

    # Check original filename alignment.
    if not np.array_equal(
        dataset["original_filename"].astype(str).to_numpy(),
        predictions["original_filename"].astype(str).to_numpy()
    ):
        raise ValueError(
            "Original filename alignment failed."
        )

    # Convert labels to numeric representation.
    dataset_labels = convert_labels(
        dataset["truth_label"]
    )

    prediction_actual_labels = convert_labels(
        predictions["actual"]
    )

    prediction_labels = convert_labels(
        predictions["predicted"]
    )

    # Validate actual labels.
    if not np.array_equal(
        dataset_labels,
        prediction_actual_labels
    ):
        raise ValueError(
            "Ground-truth label alignment failed."
        )

    print("Review ID alignment verified.")
    print("Hotel alignment verified.")
    print("Filename alignment verified.")
    print("Ground-truth label alignment verified.")

    dataset["actual_label"] = dataset_labels

    dataset["svm_prediction"] = prediction_labels

    print("\nGround-truth distribution:")

    print(
        dataset["actual_label"]
        .value_counts()
        .sort_index()
        .rename(index={
            0: "Truthful",
            1: "Deceptive"
        })
    )

    print("\nLoading semantic embeddings...")

    embeddings = np.load(EMBEDDING_PATH)

    if embeddings.shape[0] != len(dataset):
        raise ValueError(
            f"Embedding count mismatch: "
            f"{embeddings.shape[0]} vs {len(dataset)}"
        )

    print("Embedding shape:", embeddings.shape)

    # Normalize embeddings.
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embeddings = embeddings / norms

    # Compute cosine similarity.
    similarity_matrix = embeddings @ embeddings.T

    np.fill_diagonal(
        similarity_matrix,
        -1
    )

    print("Similarity matrix generated.")

    return dataset, similarity_matrix


# ============================================================
# BUILD SCOPE CANDIDATES
# ============================================================

def build_scope_candidates(dataset, scope):

    hotels = (
        dataset["hotel"]
        .astype(str)
        .to_numpy()
    )

    n = len(dataset)

    if scope == "all_hotels":

        candidate_mask = np.ones(
            (n, n),
            dtype=bool
        )

    elif scope == "same_hotel":

        candidate_mask = (
            hotels[:, None] == hotels[None, :]
        )

    elif scope == "cross_hotel":

        candidate_mask = (
            hotels[:, None] != hotels[None, :]
        )

    else:

        raise ValueError(
            f"Unknown scope: {scope}"
        )

    np.fill_diagonal(
        candidate_mask,
        False
    )

    return candidate_mask


# ============================================================
# MUTUAL KNN GRAPH
# ============================================================

def build_mutual_knn_graph(
    similarity_matrix,
    candidate_mask,
    k,
    threshold
):

    n = similarity_matrix.shape[0]

    top_k_neighbors = []

    # Each review independently selects its top-k
    # eligible neighbours within the selected scope.
    for i in range(n):

        candidates = np.flatnonzero(
            candidate_mask[i]
        )

        if len(candidates) == 0:

            top_k_neighbors.append(set())

            continue

        scores = similarity_matrix[
            i,
            candidates
        ]

        order = np.argsort(
            scores
        )[::-1]

        selected = candidates[
            order[:min(k, len(order))]
        ]

        # Apply similarity threshold.
        selected = selected[
            similarity_matrix[i, selected] >= threshold
        ]

        top_k_neighbors.append(
            set(selected.tolist())
        )

    # Create mutual neighbour graph.
    adjacency = np.zeros(
        (n, n),
        dtype=np.uint8
    )

    for i in range(n):

        for j in top_k_neighbors[i]:

            if i in top_k_neighbors[j]:

                adjacency[i, j] = 1
                adjacency[j, i] = 1

    return adjacency


# ============================================================
# EXTRACT CONNECTED GROUPS
# ============================================================

def extract_groups(adjacency):

    graph = adjacency.astype(bool)

    n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True
    )

    groups = []

    for component_id in range(n_components):

        members = np.flatnonzero(
            labels == component_id
        )

        if len(members) >= MIN_GROUP_SIZE:

            groups.append(members)

    # Largest groups first.
    groups.sort(
        key=len,
        reverse=True
    )

    return groups


# ============================================================
# CALCULATE GROUP METRICS
# ============================================================

def calculate_group_metrics(
    group_id,
    members,
    dataset,
    similarity_matrix,
    scope,
    k,
    threshold,
    adjacency
):

    group_similarities = similarity_matrix[
        np.ix_(members, members)
    ]

    upper_indices = np.triu_indices(
        len(members),
        k=1
    )

    pair_similarities = group_similarities[
        upper_indices
    ]

    group_adjacency = adjacency[
        np.ix_(members, members)
    ]

    edge_count = int(
        np.sum(group_adjacency) // 2
    )

    possible_edges = (
        len(members) * (len(members) - 1) // 2
    )

    density = (
        edge_count / possible_edges
        if possible_edges > 0
        else 0
    )

    group_data = dataset.iloc[members]

    actual_deceptive = int(
        (group_data["actual_label"] == 1).sum()
    )

    actual_truthful = int(
        (group_data["actual_label"] == 0).sum()
    )

    svm_deceptive = int(
        (group_data["svm_prediction"] == 1).sum()
    )

    hotel_names = sorted(
        group_data["hotel"]
        .astype(str)
        .unique()
    )

    return {

        "group_id": group_id,

        "scope": scope,

        "k": k,

        "threshold": threshold,

        "group_size": len(members),

        "edge_count": edge_count,

        "possible_edges": possible_edges,

        "density": density,

        "average_pair_similarity": float(
            np.mean(pair_similarities)
        ),

        "minimum_pair_similarity": float(
            np.min(pair_similarities)
        ),

        "maximum_pair_similarity": float(
            np.max(pair_similarities)
        ),

        "actual_deceptive_count": actual_deceptive,

        "actual_truthful_count": actual_truthful,

        "actual_deceptive_rate": (
            actual_deceptive / len(members)
        ),

        "svm_deceptive_count": svm_deceptive,

        "svm_deceptive_rate": (
            svm_deceptive / len(members)
        ),

        "hotel_count": len(hotel_names),

        "hotels": ", ".join(hotel_names)

    }


# ============================================================
# EXPORT GROUP MEMBERS
# ============================================================

def save_group_members(
    group_id,
    members,
    dataset,
    scope,
    k,
    threshold
):

    rows = []

    for member in members:

        row = dataset.iloc[member]

        rows.append({

            "group_id": group_id,

            "scope": scope,

            "k": k,

            "threshold": threshold,

            "review_id": int(member),

            "hotel": row["hotel"],

            "actual_label": int(
                row["actual_label"]
            ),

            "svm_prediction": int(
                row["svm_prediction"]
            ),

            "original_filename": row["original_filename"],

            "review_text": str(
                row["review"]
            )[:TEXT_PREVIEW_LENGTH]

        })

    return rows


# ============================================================
# EXPORT REVIEW PAIRS
# ============================================================

def save_group_pairs(
    group_id,
    members,
    dataset,
    similarity_matrix,
    scope,
    k,
    threshold,
    adjacency
):

    rows = []

    for position, i in enumerate(members):

        for j in members[position + 1:]:

            similarity = float(
                similarity_matrix[i, j]
            )

            rows.append({

                "group_id": group_id,

                "scope": scope,

                "k": k,

                "threshold": threshold,

                "review_id_1": int(i),

                "review_id_2": int(j),

                "hotel_1": dataset.iloc[i]["hotel"],

                "hotel_2": dataset.iloc[j]["hotel"],

                "similarity": similarity,

                "is_direct_mutual_knn_edge": int(
                    adjacency[i, j]
                ),

                "actual_label_1": int(
                    dataset.iloc[i]["actual_label"]
                ),

                "actual_label_2": int(
                    dataset.iloc[j]["actual_label"]
                ),

                "review_1": str(
                    dataset.iloc[i]["review"]
                )[:TEXT_PREVIEW_LENGTH],

                "review_2": str(
                    dataset.iloc[j]["review"]
                )[:TEXT_PREVIEW_LENGTH]

            })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset, similarity_matrix = load_inputs()

    summary_rows = []

    group_rows = []

    member_rows = []

    pair_rows = []

    print("\nStarting Mutual KNN validation...")

    for scope in SCOPES:

        print("\n" + "=" * 65)

        print(f"SCOPE: {scope}")

        print("=" * 65)

        candidate_mask = build_scope_candidates(
            dataset,
            scope
        )

        for k in K_VALUES:

            for threshold in THRESHOLDS:

                print(
                    f"\nRunning {scope} | "
                    f"k={k} | threshold={threshold}"
                )

                adjacency = build_mutual_knn_graph(
                    similarity_matrix,
                    candidate_mask,
                    k,
                    threshold
                )

                groups = extract_groups(
                    adjacency
                )

                grouped_reviews = sum(
                    len(group)
                    for group in groups
                )

                largest_group = max(
                    (len(group) for group in groups),
                    default=0
                )

                total_edges = int(
                    np.sum(adjacency) // 2
                )

                summary_rows.append({

                    "scope": scope,

                    "k": k,

                    "threshold": threshold,

                    "group_count": len(groups),

                    "grouped_reviews": grouped_reviews,

                    "ungrouped_reviews": (
                        len(dataset) - grouped_reviews
                    ),

                    "largest_group": largest_group,

                    "total_edges": total_edges

                })

                for group_number, members in enumerate(
                    groups,
                    start=1
                ):

                    group_id = (
                        f"{scope}_K{k}_T{threshold}_"
                        f"G{group_number:04d}"
                    )

                    metrics = calculate_group_metrics(
                        group_id,
                        members,
                        dataset,
                        similarity_matrix,
                        scope,
                        k,
                        threshold,
                        adjacency
                    )

                    group_rows.append(metrics)

                    member_rows.extend(
                        save_group_members(
                            group_id,
                            members,
                            dataset,
                            scope,
                            k,
                            threshold
                        )
                    )

                    pair_rows.extend(
                        save_group_pairs(
                            group_id,
                            members,
                            dataset,
                            similarity_matrix,
                            scope,
                            k,
                            threshold,
                            adjacency
                        )
                    )

                print(
                    f"Groups: {len(groups)} | "
                    f"Grouped reviews: {grouped_reviews} | "
                    f"Largest group: {largest_group} | "
                    f"Edges: {total_edges}"
                )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "validation_summary.csv",
        index=False
    )

    pd.DataFrame(group_rows).to_csv(
        OUTPUT_DIR / "group_details.csv",
        index=False
    )

    pd.DataFrame(member_rows).to_csv(
        OUTPUT_DIR / "group_members.csv",
        index=False
    )

    pd.DataFrame(pair_rows).to_csv(
        OUTPUT_DIR / "group_pairs.csv",
        index=False
    )

    print("\n" + "=" * 65)

    print("MUTUAL KNN VALIDATION COMPLETED")

    print("=" * 65)

    print("\nOutput directory:")

    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("1. validation_summary.csv")

    print("2. group_details.csv")

    print("3. group_members.csv")

    print("4. group_pairs.csv")

    print("\nValidation safeguards:")

    print("- Dataset and predictions were alignment-checked.")

    print("- Actual labels were not used for group construction.")

    print("- SVM predictions were used only for group evaluation.")

    print("- Similarity thresholds are experimental, not validated fraud cutoffs.")

    print(
        "\nReminder: similarity indicates textual resemblance, "
        "not proof of coordinated or fake reviews."
    )


if __name__ == "__main__":

    main()
