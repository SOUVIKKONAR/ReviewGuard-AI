import os
import json
import time

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/processed/verified_dataset.csv"

EMBEDDING_PATH = (
    "reports/semantic_similarity/review_embeddings.npy"
)

OUTPUT_DIR = "reports/pca_experiment"

DIMENSIONS = [384, 128, 64, 32, 16]

K_VALUES = range(2, 16)

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    print("\nLoading dataset and embeddings...")

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=["review", "truth_label", "hotel"]
    ).copy()

    df["review"] = df["review"].astype(str).str.strip()

    df = df[df["review"] != ""]

    df = df.drop_duplicates(
        subset=["review"]
    ).reset_index(drop=True)

    embeddings = np.load(EMBEDDING_PATH)

    if len(df) != len(embeddings):
        raise ValueError(
            "Dataset and embedding counts do not match."
        )

    # Convert labels only for retrospective evaluation.
    label_mapping = {
        "truthful": 0,
        "deceptive": 1,
        "Truthful": 0,
        "Deceptive": 1
    }

    labels = df["truth_label"].replace(
        label_mapping
    ).astype(int).to_numpy()

    print("Reviews:", len(df))
    print("Original embedding dimensions:", embeddings.shape[1])

    return df, embeddings, labels


# =====================================================
# REDUCE DIMENSIONS
# =====================================================

def reduce_dimensions(embeddings, dimensions):

    if dimensions == embeddings.shape[1]:

        return embeddings.copy(), 1.0

    print(f"\nApplying PCA: {dimensions} dimensions")

    pca = PCA(
        n_components=dimensions,
        random_state=RANDOM_STATE
    )

    reduced = pca.fit_transform(embeddings)

    explained_variance = float(
        np.sum(pca.explained_variance_ratio_)
    )

    # Normalize for cosine-based clustering evaluation.
    reduced = normalize(reduced)

    print(
        "Explained variance:",
        round(explained_variance * 100, 2),
        "%"
    )

    return reduced, explained_variance


# =====================================================
# FIND BEST K
# =====================================================

def evaluate_k_values(embeddings):

    results = []

    best_score = -1
    best_k = None
    best_labels = None

    for k in K_VALUES:

        print(f"Testing K={k}")

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=300
        )

        cluster_labels = model.fit_predict(
            embeddings
        )

        score = silhouette_score(
            embeddings,
            cluster_labels,
            metric="cosine"
        )

        results.append({

            "k": k,

            "silhouette_score": float(score),

            "inertia": float(model.inertia_)

        })

        print(
            f"K={k} | Silhouette={score:.4f}"
        )

        if score > best_score:

            best_score = score
            best_k = k
            best_labels = cluster_labels.copy()

    return best_k, best_score, best_labels, results


# =====================================================
# CLUSTER EVALUATION
# =====================================================

def evaluate_clusters(
    df,
    cluster_labels,
    true_labels
):

    temp = df.copy()

    temp["cluster"] = cluster_labels

    temp["evaluation_label"] = true_labels

    summary = []

    for cluster_id, group in temp.groupby("cluster"):

        total = len(group)

        deceptive = (
            group["evaluation_label"] == 1
        ).sum()

        truthful = (
            group["evaluation_label"] == 0
        ).sum()

        summary.append({

            "cluster": int(cluster_id),

            "total_reviews": int(total),

            "truthful_reviews": int(truthful),

            "deceptive_reviews": int(deceptive),

            "deceptive_percentage": round(
                deceptive / total * 100, 2
            ),

            "unique_hotels": int(
                group["hotel"].nunique()
            )

        })

    return pd.DataFrame(summary)


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def main():

    print("=" * 65)
    print("REVIEWGUARD - PCA CLUSTERING EXPERIMENT")
    print("=" * 65)

    df, original_embeddings, true_labels = load_data()

    experiment_results = []

    for dimensions in DIMENSIONS:

        print("\n" + "=" * 65)

        print(
            f"EXPERIMENT: {dimensions} DIMENSIONS"
        )

        print("=" * 65)

        start_time = time.time()

        reduced_embeddings, explained_variance = (
            reduce_dimensions(
                original_embeddings,
                dimensions
            )
        )

        best_k, best_score, cluster_labels, k_results = (
            evaluate_k_values(
                reduced_embeddings
            )
        )

        # -----------------------------------------
        # CLUSTER EVALUATION
        # -----------------------------------------

        cluster_summary = evaluate_clusters(
            df,
            cluster_labels,
            true_labels
        )

        # External evaluation only.
        ari = adjusted_rand_score(
            true_labels,
            cluster_labels
        )

        nmi = normalized_mutual_info_score(
            true_labels,
            cluster_labels
        )

        duration = time.time() - start_time

        print("\nBest K:", best_k)

        print(
            "Best silhouette:",
            round(best_score, 4)
        )

        print(
            "ARI:",
            round(ari, 4)
        )

        print(
            "NMI:",
            round(nmi, 4)
        )

        print("\nCluster distribution:")

        print(cluster_summary.to_string(index=False))

        # -----------------------------------------
        # SAVE EXPERIMENT RESULTS
        # -----------------------------------------

        experiment_dir = os.path.join(
            OUTPUT_DIR,
            f"dimensions_{dimensions}"
        )

        os.makedirs(
            experiment_dir,
            exist_ok=True
        )

        pd.DataFrame(k_results).to_csv(
            os.path.join(
                experiment_dir,
                "k_selection_results.csv"
            ),
            index=False
        )

        cluster_summary.to_csv(
            os.path.join(
                experiment_dir,
                "cluster_summary.csv"
            ),
            index=False
        )

        pd.DataFrame({

            "review": df["review"],

            "hotel": df["hotel"],

            "actual_label": true_labels,

            "cluster": cluster_labels

        }).to_csv(
            os.path.join(
                experiment_dir,
                "clustered_reviews.csv"
            ),
            index=False
        )

        experiment_results.append({

            "dimensions": dimensions,

            "explained_variance_percentage": round(
                explained_variance * 100, 2
            ),

            "best_k": best_k,

            "silhouette_score": float(best_score),

            "adjusted_rand_index": float(ari),

            "normalized_mutual_information": float(nmi),

            "execution_time_seconds": round(
                duration, 2
            )

        })

    # ---------------------------------------------
    # FINAL COMPARISON
    # ---------------------------------------------

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df.to_csv(
        f"{OUTPUT_DIR}/pca_comparison.csv",
        index=False
    )

    with open(
        f"{OUTPUT_DIR}/pca_comparison.json",
        "w"
    ) as file:

        json.dump(
            experiment_results,
            file,
            indent=4
        )

    print("\n" + "=" * 65)

    print("FINAL PCA COMPARISON")

    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nPCA experiment completed!")

    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()