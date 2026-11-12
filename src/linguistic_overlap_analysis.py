from pathlib import Path
from difflib import SequenceMatcher
import json
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)


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

GROUP_PAIRS_PATH = (
    ROOT_DIR
    / "reports"
    / "mutual_knn_validation"
    / "group_pairs.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "reports"
    / "linguistic_overlap"
)

MIN_COMMON_PHRASE_TOKENS = 4

TOP_SHARED_PHRASES = 10

STOP_WORDS = set(ENGLISH_STOP_WORDS)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def tokenize(text):

    text = str(text).lower()

    return re.findall(
        r"\b[a-z]+\b",
        text
    )


def remove_stopwords(tokens):

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def create_ngrams(tokens, n):

    if len(tokens) < n:
        return []

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def jaccard_similarity(set_a, set_b):

    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b

    if not union:
        return 0.0

    intersection = set_a & set_b

    return len(intersection) / len(union)


# ============================================================
# LONGEST COMMON PHRASE
# ============================================================

def longest_common_phrase(tokens_a, tokens_b):

    matcher = SequenceMatcher(
        None,
        tokens_a,
        tokens_b,
        autojunk=False
    )

    match = matcher.find_longest_match(
        0,
        len(tokens_a),
        0,
        len(tokens_b)
    )

    phrase_tokens = tokens_a[
        match.a:match.a + match.size
    ]

    return {
        "length": match.size,
        "phrase": " ".join(phrase_tokens)
    }


# ============================================================
# TF-IDF DISTINCTIVE PHRASE EXTRACTION
# ============================================================

def build_tfidf_matrix(reviews):

    print("\nBuilding TF-IDF phrase representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(2, 3),
        min_df=2,
        max_df=0.80,
        sublinear_tf=True,
        max_features=50000
    )

    matrix = vectorizer.fit_transform(reviews)

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    print(
        "TF-IDF matrix shape:",
        matrix.shape
    )

    return matrix, feature_names


def extract_shared_tfidf_phrases(
    matrix,
    feature_names,
    index_a,
    index_b
):

    row_a = matrix[index_a]

    row_b = matrix[index_b]

    common_indices = np.intersect1d(
        row_a.indices,
        row_b.indices,
        assume_unique=True
    )

    if len(common_indices) == 0:
        return [], 0.0

    values_a = dict(
        zip(row_a.indices, row_a.data)
    )

    values_b = dict(
        zip(row_b.indices, row_b.data)
    )

    shared_phrases = []

    for feature_index in common_indices:

        score_a = values_a[feature_index]

        score_b = values_b[feature_index]

        combined_score = min(
            score_a,
            score_b
        )

        shared_phrases.append({

            "phrase": str(
                feature_names[feature_index]
            ),

            "score": float(
                combined_score
            )

        })

    shared_phrases.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    total_score = sum(
        item["score"]
        for item in shared_phrases
    )

    return (
        shared_phrases[:TOP_SHARED_PHRASES],
        float(total_score)
    )


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_inputs():

    print("\nLoading verified dataset...")

    dataset = pd.read_csv(
        DATASET_PATH
    )

    dataset = (
        dataset
        .drop_duplicates(subset=["review"])
        .reset_index(drop=True)
    )

    required_dataset_columns = [
        "review",
        "hotel",
        "truth_label",
        "original_filename"
    ]

    for column in required_dataset_columns:

        if column not in dataset.columns:

            raise ValueError(
                f"Dataset missing column: {column}"
            )

    dataset["review_id"] = np.arange(
        len(dataset)
    )

    print(
        "Dataset reviews:",
        len(dataset)
    )

    print("\nLoading Mutual KNN group pairs...")

    pairs = pd.read_csv(
        GROUP_PAIRS_PATH
    )

    required_pair_columns = [
        "group_id",
        "scope",
        "k",
        "threshold",
        "review_id_1",
        "review_id_2",
        "similarity",
        "is_direct_mutual_knn_edge"
    ]

    for column in required_pair_columns:

        if column not in pairs.columns:

            raise ValueError(
                f"Group pairs file missing column: {column}"
            )

    # Keep direct Mutual KNN edges only.
    pairs = pairs[
        pairs["is_direct_mutual_knn_edge"] == 1
    ].copy()

    pairs["review_id_1"] = pd.to_numeric(
        pairs["review_id_1"],
        errors="raise"
    ).astype(int)

    pairs["review_id_2"] = pd.to_numeric(
        pairs["review_id_2"],
        errors="raise"
    ).astype(int)

    if pairs.empty:

        raise ValueError(
            "No direct Mutual KNN edges found."
        )

    # Validate review IDs.
    if (
        pairs["review_id_1"].min() < 0
        or pairs["review_id_2"].min() < 0
        or pairs["review_id_1"].max() >= len(dataset)
        or pairs["review_id_2"].max() >= len(dataset)
    ):

        raise ValueError(
            "Invalid review IDs found in group pairs."
        )

    print(
        "Direct Mutual KNN edges:",
        len(pairs)
    )

    return dataset, pairs


# ============================================================
# ANALYSE ONE REVIEW PAIR
# ============================================================

def analyse_pair(
    pair,
    dataset,
    tfidf_matrix,
    feature_names
):

    index_a = int(
        pair["review_id_1"]
    )

    index_b = int(
        pair["review_id_2"]
    )

    review_a = str(
        dataset.iloc[index_a]["review"]
    )

    review_b = str(
        dataset.iloc[index_b]["review"]
    )

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    tokens_a = tokenize(review_a)

    tokens_b = tokenize(review_b)

    filtered_a = remove_stopwords(
        tokens_a
    )

    filtered_b = remove_stopwords(
        tokens_b
    )

    # --------------------------------------------------------
    # WORD-LEVEL OVERLAP
    # --------------------------------------------------------

    words_a = set(filtered_a)

    words_b = set(filtered_b)

    word_jaccard = jaccard_similarity(
        words_a,
        words_b
    )

    # --------------------------------------------------------
    # BIGRAM OVERLAP
    # --------------------------------------------------------

    bigrams_a = set(
        create_ngrams(tokens_a, 2)
    )

    bigrams_b = set(
        create_ngrams(tokens_b, 2)
    )

    bigram_jaccard = jaccard_similarity(
        bigrams_a,
        bigrams_b
    )

    common_bigrams = (
        bigrams_a & bigrams_b
    )

    # --------------------------------------------------------
    # TRIGRAM OVERLAP
    # --------------------------------------------------------

    trigrams_a = set(
        create_ngrams(tokens_a, 3)
    )

    trigrams_b = set(
        create_ngrams(tokens_b, 3)
    )

    trigram_jaccard = jaccard_similarity(
        trigrams_a,
        trigrams_b
    )

    common_trigrams = (
        trigrams_a & trigrams_b
    )

    # --------------------------------------------------------
    # LONGEST COMMON PHRASE
    # --------------------------------------------------------

    longest_phrase = longest_common_phrase(
        tokens_a,
        tokens_b
    )

    # --------------------------------------------------------
    # TF-IDF SHARED PHRASES
    # --------------------------------------------------------

    shared_phrases, shared_tfidf_score = (
        extract_shared_tfidf_phrases(
            tfidf_matrix,
            feature_names,
            index_a,
            index_b
        )
    )

    # --------------------------------------------------------
    # HOTEL RELATIONSHIP
    # --------------------------------------------------------

    hotel_a = str(
        dataset.iloc[index_a]["hotel"]
    )

    hotel_b = str(
        dataset.iloc[index_b]["hotel"]
    )

    hotel_relationship = (
        "same_hotel"
        if hotel_a == hotel_b
        else "cross_hotel"
    )

    # --------------------------------------------------------
    # GROUND-TRUTH LABELS
    # --------------------------------------------------------

    label_a = str(
        dataset.iloc[index_a]["truth_label"]
    )

    label_b = str(
        dataset.iloc[index_b]["truth_label"]
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "group_id": pair["group_id"],

        "scope": pair["scope"],

        "k": int(pair["k"]),

        "threshold": float(pair["threshold"]),

        "review_id_1": index_a,

        "review_id_2": index_b,

        "hotel_1": hotel_a,

        "hotel_2": hotel_b,

        "hotel_relationship": hotel_relationship,

        "semantic_similarity": float(
            pair["similarity"]
        ),

        "actual_label_1": label_a,

        "actual_label_2": label_b,

        "review_length_1": len(tokens_a),

        "review_length_2": len(tokens_b),

        "shared_unique_words": len(
            words_a & words_b
        ),

        "word_jaccard": word_jaccard,

        "shared_bigrams": len(
            common_bigrams
        ),

        "bigram_jaccard": bigram_jaccard,

        "shared_trigrams": len(
            common_trigrams
        ),

        "trigram_jaccard": trigram_jaccard,

        "longest_common_phrase_length": (
            longest_phrase["length"]
        ),

        "longest_common_phrase": (
            longest_phrase["phrase"]
        ),

        "shared_tfidf_phrase_count": len(
            shared_phrases
        ),

        "shared_tfidf_score": shared_tfidf_score,

        "top_shared_phrases": " | ".join(
            item["phrase"]
            for item in shared_phrases
        ),

        "review_1": review_a,

        "review_2": review_b

    }


# ============================================================
# GROUP-LEVEL LINGUISTIC SUMMARY
# ============================================================

def create_group_summary(edge_data):

    df = pd.DataFrame(
        edge_data
    )

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(
        [
            "group_id",
            "scope",
            "k",
            "threshold"
        ],
        dropna=False
    )

    summary = grouped.agg(

        direct_edge_count=(
            "semantic_similarity",
            "count"
        ),

        average_semantic_similarity=(
            "semantic_similarity",
            "mean"
        ),

        average_word_jaccard=(
            "word_jaccard",
            "mean"
        ),

        average_bigram_jaccard=(
            "bigram_jaccard",
            "mean"
        ),

        average_trigram_jaccard=(
            "trigram_jaccard",
            "mean"
        ),

        average_shared_bigrams=(
            "shared_bigrams",
            "mean"
        ),

        average_shared_trigrams=(
            "shared_trigrams",
            "mean"
        ),

        average_longest_phrase_length=(
            "longest_common_phrase_length",
            "mean"
        ),

        average_shared_tfidf_score=(
            "shared_tfidf_score",
            "mean"
        )

    ).reset_index()

    return summary


# ============================================================
# IDENTIFY PHRASE-OVERLAP CANDIDATES
# ============================================================

def identify_phrase_overlap_candidates(edge_data):

    df = pd.DataFrame(
        edge_data
    )

    if df.empty:
        return df

    # Exploratory filters for manual inspection.
    candidates = df[
        (
            df["shared_trigrams"] >= 1
        )
        |
        (
            df["shared_bigrams"] >= 2
        )
        |
        (
            df["longest_common_phrase_length"]
            >= MIN_COMMON_PHRASE_TOKENS
        )
    ].copy()

    candidates = candidates.sort_values(
        [
            "shared_trigrams",
            "shared_bigrams",
            "longest_common_phrase_length",
            "semantic_similarity"
        ],
        ascending=False
    )

    return candidates


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def generate_summary(edge_data):

    df = pd.DataFrame(
        edge_data
    )

    if df.empty:

        raise ValueError(
            "No edge analysis results available."
        )

    # --------------------------------------------------------
    # FIXED UNIQUE REVIEW-PAIR COUNTING
    # --------------------------------------------------------

    review_pairs = df[
        [
            "review_id_1",
            "review_id_2"
        ]
    ].to_numpy()

    # Normalize pair order so (1, 2) and (2, 1)
    # are treated as the same review pair.
    normalized_pairs = np.sort(
        review_pairs,
        axis=1
    )

    unique_review_pairs = len(
        set(
            map(
                tuple,
                normalized_pairs
            )
        )
    )

    # --------------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------------

    summary = {

        "total_direct_edges_analysed": int(
            len(df)
        ),

        "unique_review_pairs": int(
            unique_review_pairs
        ),

        "same_hotel_edges": int(
            (
                df["hotel_relationship"]
                == "same_hotel"
            ).sum()
        ),

        "cross_hotel_edges": int(
            (
                df["hotel_relationship"]
                == "cross_hotel"
            ).sum()
        ),

        "average_semantic_similarity": float(
            df["semantic_similarity"].mean()
        ),

        "average_word_jaccard": float(
            df["word_jaccard"].mean()
        ),

        "average_bigram_jaccard": float(
            df["bigram_jaccard"].mean()
        ),

        "average_trigram_jaccard": float(
            df["trigram_jaccard"].mean()
        ),

        "edges_with_shared_trigrams": int(
            (
                df["shared_trigrams"] > 0
            ).sum()
        ),

        "edges_with_long_common_phrase": int(
            (
                df["longest_common_phrase_length"]
                >= MIN_COMMON_PHRASE_TOKENS
            ).sum()
        )

    }

    # --------------------------------------------------------
    # SAME-HOTEL VS CROSS-HOTEL COMPARISON
    # --------------------------------------------------------

    relationship_summary = (
        df.groupby("hotel_relationship")
        .agg(

            edge_count=(
                "semantic_similarity",
                "count"
            ),

            avg_semantic_similarity=(
                "semantic_similarity",
                "mean"
            ),

            avg_word_jaccard=(
                "word_jaccard",
                "mean"
            ),

            avg_bigram_jaccard=(
                "bigram_jaccard",
                "mean"
            ),

            avg_trigram_jaccard=(
                "trigram_jaccard",
                "mean"
            ),

            avg_shared_trigrams=(
                "shared_trigrams",
                "mean"
            )

        )
        .reset_index()
        .to_dict(orient="records")
    )

    summary["hotel_relationship_comparison"] = (
        relationship_summary
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD INPUTS
    # --------------------------------------------------------

    dataset, pairs = load_inputs()

    reviews = (
        dataset["review"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # BUILD TF-IDF MATRIX
    # --------------------------------------------------------

    tfidf_matrix, feature_names = (
        build_tfidf_matrix(reviews)
    )

    print(
        "\nStarting linguistic overlap analysis..."
    )

    edge_results = []

    total = len(pairs)

    # --------------------------------------------------------
    # PROCESS ALL DIRECT EDGES
    # --------------------------------------------------------

    for position, (_, pair) in enumerate(
        pairs.iterrows(),
        start=1
    ):

        result = analyse_pair(
            pair,
            dataset,
            tfidf_matrix,
            feature_names
        )

        edge_results.append(
            result
        )

        if (
            position % 100 == 0
            or position == total
        ):

            print(
                f"Processed {position}/{total} edges"
            )

    # --------------------------------------------------------
    # EDGE-LEVEL RESULTS
    # --------------------------------------------------------

    edge_df = pd.DataFrame(
        edge_results
    )

    edge_df.to_csv(
        OUTPUT_DIR / "edge_linguistic_features.csv",
        index=False
    )

    # --------------------------------------------------------
    # GROUP-LEVEL RESULTS
    # --------------------------------------------------------

    group_summary = create_group_summary(
        edge_results
    )

    group_summary.to_csv(
        OUTPUT_DIR / "group_linguistic_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # PHRASE-OVERLAP CANDIDATES
    # --------------------------------------------------------

    phrase_candidates = (
        identify_phrase_overlap_candidates(
            edge_results
        )
    )

    phrase_candidates.to_csv(
        OUTPUT_DIR / "suspicious_phrase_pairs.csv",
        index=False
    )

    # --------------------------------------------------------
    # JSON SUMMARY
    # --------------------------------------------------------

    summary = generate_summary(
        edge_results
    )

    with open(
        OUTPUT_DIR / "linguistic_analysis_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "LINGUISTIC OVERLAP ANALYSIS COMPLETED"
    )

    print("=" * 65)

    print(
        "\nTotal direct edges analysed:",
        summary["total_direct_edges_analysed"]
    )

    print(
        "\nUnique review pairs:",
        summary["unique_review_pairs"]
    )

    print(
        "\nSame-hotel edges:",
        summary["same_hotel_edges"]
    )

    print(
        "\nCross-hotel edges:",
        summary["cross_hotel_edges"]
    )

    print(
        "\nAverage semantic similarity:",
        round(
            summary["average_semantic_similarity"],
            4
        )
    )

    print(
        "\nAverage word Jaccard:",
        round(
            summary["average_word_jaccard"],
            4
        )
    )

    print(
        "\nAverage bigram Jaccard:",
        round(
            summary["average_bigram_jaccard"],
            4
        )
    )

    print(
        "\nAverage trigram Jaccard:",
        round(
            summary["average_trigram_jaccard"],
            4
        )
    )

    print(
        "\nEdges with shared trigrams:",
        summary["edges_with_shared_trigrams"]
    )

    print(
        "\nEdges with long common phrases:",
        summary["edges_with_long_common_phrase"]
    )

    print("\nOutput directory:")

    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("1. edge_linguistic_features.csv")

    print("2. group_linguistic_summary.csv")

    print("3. suspicious_phrase_pairs.csv")

    print("4. linguistic_analysis_summary.json")

    print(
        "\nIMPORTANT: Phrase-overlap candidates are "
        "for manual inspection, not confirmed fake reviews."
    )


if __name__ == "__main__":

    main()