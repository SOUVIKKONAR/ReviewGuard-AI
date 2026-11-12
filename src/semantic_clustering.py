import os
import json

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/processed/verified_dataset.csv"

EMBEDDING_PATH = (
    "reports/semantic_similarity/review_embeddings.npy"
)

OUTPUT_DIR = "reports/semantic_clustering"

K_VALUES = range(2, 16)

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    print("\nLoading verified dataset...")

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

    print("Reviews:", len(df))
    print("Embedding shape:", embeddings.shape)

    return df, embeddings


# =====================================================
# FIND OPTIMAL K
# =====================================================

def find_optimal_k(embeddings):

    print("\nEvaluating different K values...")

    results = []

    for k in K_VALUES:

        print(f"Testing K = {k}")

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=300
        )

        cluster_labels = model.fit_predict(
            embeddings
        )

        silhouette = silhouette_score(
            embeddings,
            cluster_labels,
            metric="cosine"
        )

        results.append({

            "k": k,

            "inertia": float(model.inertia_),

            "silhouette_score": float(silhouette)

        })

        print(
            f"K={k} | "
            f"Silhouette={silhouette:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"{OUTPUT_DIR}/k_selection_results.csv",
        index=False
    )

    # Select K with the highest silhouette score
    best_k = int(
        results_df.loc[
            results_df["silhouette_score"].idxmax(),
            "k"
        ]
    )

    print("\nBest K:", best_k)

    return best_k, results_df


# =====================================================
# PERFORM K-MEANS CLUSTERING
# =====================================================

def perform_clustering(df, embeddings, k):

    print("\nPerforming K-Means clustering...")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=300
    )

    clusters = model.fit_predict(
        embeddings
    )

    df = df.copy()

    df["cluster"] = clusters

    print("Clustering completed.")

    return df, model


# =====================================================
# CLUSTER ANALYSIS
# =====================================================

def analyze_clusters(df):

    print("\nAnalysing clusters...")

    summary = []

    for cluster_id, group in df.groupby("cluster"):

        total = len(group)

        deceptive = (
            group["truth_label"] == "deceptive"
        ).sum()

        truthful = (
            group["truth_label"] == "truthful"
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
            ),

            "hotel_distribution": json.dumps(
                group["hotel"].value_counts().to_dict()
            )

        })

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        "deceptive_percentage",
        ascending=False
    )

    summary_df.to_csv(
        f"{OUTPUT_DIR}/cluster_summary.csv",
        index=False
    )

    print("\nCluster Summary:")

    print(
        summary_df[
            [
                "cluster",
                "total_reviews",
                "truthful_reviews",
                "deceptive_reviews",
                "deceptive_percentage",
                "unique_hotels"
            ]
        ].to_string(index=False)
    )

    return summary_df


# =====================================================
# REPRESENTATIVE REVIEWS
# =====================================================

def save_representative_reviews(df, embeddings, model):

    print("\nExtracting representative reviews...")

    records = []

    for cluster_id in sorted(df["cluster"].unique()):

        indices = np.where(
            df["cluster"].values == cluster_id
        )[0]

        cluster_embeddings = embeddings[indices]

        centroid = model.cluster_centers_[cluster_id]

        distances = np.linalg.norm(
            cluster_embeddings - centroid,
            axis=1
        )

        nearest_positions = np.argsort(
            distances
        )[:5]

        for position in nearest_positions:

            original_index = indices[position]

            row = df.iloc[original_index]

            records.append({

                "cluster": int(cluster_id),

                "review": row["review"],

                "truth_label": row["truth_label"],

                "hotel": row["hotel"],

                "distance_from_centroid": float(
                    distances[position]
                )

            })

    representatives = pd.DataFrame(records)

    representatives.to_csv(
        f"{OUTPUT_DIR}/representative_reviews.csv",
        index=False
    )

    return representatives


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 60)
    print("REVIEWGUARD - SEMANTIC K-MEANS CLUSTERING")
    print("=" * 60)

    df, embeddings = load_data()

    best_k, k_results = find_optimal_k(
        embeddings
    )

    clustered_df, model = perform_clustering(
        df,
        embeddings,
        best_k
    )

    summary_df = analyze_clusters(
        clustered_df
    )

    save_representative_reviews(
        clustered_df,
        embeddings,
        model
    )

    clustered_df.to_csv(
        f"{OUTPUT_DIR}/clustered_reviews.csv",
        index=False
    )

    results = {

        "selected_k": best_k,

        "total_clusters": int(
            clustered_df["cluster"].nunique()
        ),

        "silhouette_score": float(
            k_results.loc[
                k_results["k"] == best_k,
                "silhouette_score"
            ].iloc[0]
        ),

        "total_reviews": len(clustered_df)

    }

    with open(
        f"{OUTPUT_DIR}/clustering_statistics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING COMPLETED")
    print("=" * 60)

    print("Selected K:", best_k)

    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()