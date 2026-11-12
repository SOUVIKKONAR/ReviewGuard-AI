import pandas as pd
import numpy as np
import json
import os


INPUT_PATH = "reports/semantic_similarity/suspicious_pairs.csv"

OUTPUT_DIR = "reports/semantic_similarity"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    print("=" * 60)
    print("REVIEWGUARD - SIMILARITY PATTERN ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("No high-similarity pairs found.")
        return

    # ---------------------------------------
    # LABEL COMBINATION ANALYSIS
    # ---------------------------------------

    def classify_pair(row):

        labels = {
            str(row["label_1"]).lower(),
            str(row["label_2"]).lower()
        }

        if labels == {"deceptive"}:
            return "Deceptive-Deceptive"

        elif labels == {"truthful"}:
            return "Truthful-Truthful"

        return "Truthful-Deceptive"

    df["pair_category"] = df.apply(
        classify_pair,
        axis=1
    )

    print("\n1. LABEL COMBINATION DISTRIBUTION")

    label_distribution = df[
        "pair_category"
    ].value_counts()

    print(label_distribution)

    # ---------------------------------------
    # SAME / DIFFERENT HOTEL ANALYSIS
    # ---------------------------------------

    print("\n2. HOTEL DISTRIBUTION")

    hotel_distribution = df[
        "same_hotel"
    ].value_counts()

    print(hotel_distribution)

    # ---------------------------------------
    # SAME HOTEL + LABEL COMBINATION
    # ---------------------------------------

    print("\n3. CROSS ANALYSIS")

    cross_analysis = pd.crosstab(
        df["pair_category"],
        df["same_hotel"]
    )

    print(cross_analysis)

    # ---------------------------------------
    # SIMILARITY STATISTICS BY CATEGORY
    # ---------------------------------------

    print("\n4. CATEGORY-WISE SIMILARITY")

    similarity_analysis = df.groupby(
        "pair_category"
    )["similarity_score"].agg([
        "count",
        "mean",
        "median",
        "max",
        "min"
    ])

    print(similarity_analysis)

    # ---------------------------------------
    # MOST SIMILAR REVIEW PAIRS
    # ---------------------------------------

    print("\n5. TOP 20 MOST SIMILAR PAIRS")

    top_pairs = df.nlargest(
        20,
        "similarity_score"
    )

    for _, row in top_pairs.iterrows():

        print("\nSimilarity:", round(
            row["similarity_score"], 4
        ))

        print("Category:", row["pair_category"])

        print("Hotel 1:", row["hotel_1"])

        print("Hotel 2:", row["hotel_2"])

        print("Review 1:", row["review_1"][:250])

        print("Review 2:", row["review_2"][:250])

        print("-" * 50)

    # ---------------------------------------
    # SAVE RESULTS
    # ---------------------------------------

    label_distribution.to_csv(
        f"{OUTPUT_DIR}/label_pair_distribution.csv"
    )

    hotel_distribution.to_csv(
        f"{OUTPUT_DIR}/hotel_pair_distribution.csv"
    )

    cross_analysis.to_csv(
        f"{OUTPUT_DIR}/hotel_label_cross_analysis.csv"
    )

    similarity_analysis.to_csv(
        f"{OUTPUT_DIR}/similarity_category_statistics.csv"
    )

    top_pairs.to_csv(
        f"{OUTPUT_DIR}/top_20_similar_pairs.csv",
        index=False
    )

    # ---------------------------------------
    # SUMMARY JSON
    # ---------------------------------------

    summary = {

        "total_high_similarity_pairs": int(len(df)),

        "pair_categories": {
            str(key): int(value)
            for key, value in label_distribution.items()
        },

        "same_hotel_pairs": int(
            df["same_hotel"].sum()
        ),

        "different_hotel_pairs": int(
            (~df["same_hotel"]).sum()
        ),

        "average_similarity": float(
            df["similarity_score"].mean()
        )

    }

    with open(
        f"{OUTPUT_DIR}/similarity_analysis_summary.json",
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nAnalysis completed successfully!")

    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()