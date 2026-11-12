import os
import json
import time

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/processed/verified_dataset.csv"

OUTPUT_DIR = "reports/semantic_similarity"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
TOP_K = 10

# Initial candidate threshold, to be calibrated experimentally.
SIMILARITY_THRESHOLD = 0.80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    print("\nLoading verified dataset...")

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "review",
        "truth_label",
        "hotel",
        "sentiment",
        "original_filename"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.dropna(
        subset=["review", "truth_label", "hotel"]
    ).copy()

    df["review"] = df["review"].astype(str).str.strip()

    df = df[df["review"] != ""]

    # Remove exact duplicate reviews
    df = df.drop_duplicates(
        subset=["review"]
    ).reset_index(drop=True)

    print("Total reviews:", len(df))

    print("\nLabel distribution:")
    print(df["truth_label"].value_counts())

    print("\nUnique hotels:", df["hotel"].nunique())

    return df


# =====================================================
# GENERATE SEMANTIC EMBEDDINGS
# =====================================================

def generate_embeddings(df):

    print("\nLoading Sentence Transformer...")
    print("Device:", DEVICE)

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    reviews = df["review"].tolist()

    print("\nGenerating embeddings...")

    start_time = time.time()

    embeddings = model.encode(
        reviews,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    generation_time = time.time() - start_time

    embeddings = embeddings.cpu().numpy()

    print("\nEmbedding generation completed.")

    print("Embedding shape:", embeddings.shape)

    print(
        "Generation time:",
        round(generation_time, 2),
        "seconds"
    )

    np.save(
        f"{OUTPUT_DIR}/review_embeddings.npy",
        embeddings
    )

    return embeddings, generation_time


# =====================================================
# CALCULATE COSINE SIMILARITY
# =====================================================

def calculate_similarity(embeddings):

    print("\nCalculating cosine similarity...")

    # Normalized embeddings:
    # Cosine similarity = dot product

    embedding_tensor = torch.tensor(
        embeddings,
        dtype=torch.float32,
        device=DEVICE
    )

    similarity_matrix = torch.matmul(
        embedding_tensor,
        embedding_tensor.T
    )

    similarity_matrix = similarity_matrix.cpu().numpy()

    # Remove self-similarity
    np.fill_diagonal(similarity_matrix, -1.0)

    np.save(
        f"{OUTPUT_DIR}/similarity_matrix.npy",
        similarity_matrix
    )

    print("Similarity matrix shape:", similarity_matrix.shape)

    return similarity_matrix


# =====================================================
# FIND TOP-K SIMILAR REVIEWS
# =====================================================

def find_top_similar_reviews(df, similarity_matrix):

    print("\nFinding top similar reviews...")

    records = []

    for i in range(len(df)):

        top_indices = np.argsort(
            similarity_matrix[i]
        )[-TOP_K:][::-1]

        for rank, j in enumerate(top_indices, start=1):

            records.append({

                "review_id": i,
                "similar_review_id": int(j),

                "similarity_rank": rank,

                "similarity_score": float(
                    similarity_matrix[i][j]
                ),

                "review": df.iloc[i]["review"],

                "similar_review": df.iloc[j]["review"],

                "review_label": df.iloc[i]["truth_label"],

                "similar_review_label": df.iloc[j]["truth_label"],

                "review_hotel": df.iloc[i]["hotel"],

                "similar_review_hotel": df.iloc[j]["hotel"]

            })

    results_df = pd.DataFrame(records)

    results_df.to_csv(
        f"{OUTPUT_DIR}/similar_reviews.csv",
        index=False
    )

    print("Top similar reviews saved.")

    return results_df


# =====================================================
# IDENTIFY HIGH-SIMILARITY REVIEW PAIRS
# =====================================================

def identify_suspicious_pairs(df, similarity_matrix):

    print("\nIdentifying high-similarity review pairs...")

    records = []

    n = len(df)

    for i in range(n):

        for j in range(i + 1, n):

            score = similarity_matrix[i][j]

            if score >= SIMILARITY_THRESHOLD:

                records.append({

                    "review_id_1": i,
                    "review_id_2": j,

                    "review_1": df.iloc[i]["review"],
                    "review_2": df.iloc[j]["review"],

                    "label_1": df.iloc[i]["truth_label"],
                    "label_2": df.iloc[j]["truth_label"],

                    "hotel_1": df.iloc[i]["hotel"],
                    "hotel_2": df.iloc[j]["hotel"],

                    "sentiment_1": df.iloc[i]["sentiment"],
                    "sentiment_2": df.iloc[j]["sentiment"],

                    "similarity_score": float(score),

                    "same_hotel": (
                        df.iloc[i]["hotel"]
                        == df.iloc[j]["hotel"]
                    ),

                    "same_label": (
                        df.iloc[i]["truth_label"]
                        == df.iloc[j]["truth_label"]
                    )

                })

    pairs_df = pd.DataFrame(records)

    if not pairs_df.empty:

        pairs_df = pairs_df.sort_values(
            by="similarity_score",
            ascending=False
        )

    pairs_df.to_csv(
        f"{OUTPUT_DIR}/suspicious_pairs.csv",
        index=False
    )

    print("High-similarity pairs:", len(pairs_df))

    return pairs_df


# =====================================================
# SIMILARITY STATISTICS
# =====================================================

def generate_statistics(
    df,
    similarity_matrix,
    suspicious_pairs,
    generation_time
):

    print("\nGenerating similarity statistics...")

    n = len(df)

    # Extract upper triangle to avoid duplicate pairs
    upper_triangle = similarity_matrix[
        np.triu_indices(n, k=1)
    ]

    statistics = {

        "total_reviews": int(n),

        "embedding_dimension": int(
            np.load(
                f"{OUTPUT_DIR}/review_embeddings.npy",
                mmap_mode="r"
            ).shape[1]
        ),

        "device": DEVICE,

        "embedding_generation_time_seconds": round(
            generation_time, 2
        ),

        "similarity_threshold": SIMILARITY_THRESHOLD,

        "total_review_pairs": int(len(upper_triangle)),

        "average_similarity": float(
            np.mean(upper_triangle)
        ),

        "maximum_similarity": float(
            np.max(upper_triangle)
        ),

        "minimum_similarity": float(
            np.min(upper_triangle)
        ),

        "high_similarity_pairs": int(
            len(suspicious_pairs)
        )

    }

    with open(
        f"{OUTPUT_DIR}/similarity_statistics.json",
        "w"
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4
        )

    print("\nSimilarity Statistics:")

    for key, value in statistics.items():

        print(f"{key}: {value}")


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 60)
    print("REVIEWGUARD - SEMANTIC SIMILARITY DETECTION")
    print("=" * 60)

    if DEVICE == "cuda":

        print("GPU:", torch.cuda.get_device_name(0))

    df = load_dataset()

    embeddings, generation_time = generate_embeddings(df)

    similarity_matrix = calculate_similarity(
        embeddings
    )

    find_top_similar_reviews(
        df,
        similarity_matrix
    )

    suspicious_pairs = identify_suspicious_pairs(
        df,
        similarity_matrix
    )

    generate_statistics(
        df,
        similarity_matrix,
        suspicious_pairs,
        generation_time
    )

    print("\n" + "=" * 60)
    print("SEMANTIC SIMILARITY COMPLETED")
    print("=" * 60)

    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()