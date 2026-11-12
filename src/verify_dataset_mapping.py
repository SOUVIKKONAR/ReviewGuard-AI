import pandas as pd
import re
from pathlib import Path
from collections import Counter

# Project paths
ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "DATASET/deceptive_opinion_spam/Chicago_Hotel_Reviews.csv"

CORPUS_PATH = (
    ROOT
    / "DATASET/original_corpus/op_spam_v1.4"
)


def normalize_text(text):
    """Normalize text for reliable matching."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# --------------------------------------------------
# 1. Load CSV dataset
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("\n========== CSV DATASET ==========")
print("Total records:", len(df))
print("\nOriginal label distribution:")
print(df["Label"].value_counts().sort_index())


# --------------------------------------------------
# 2. Read original corpus
# --------------------------------------------------

original_reviews = {}

for file_path in CORPUS_PATH.rglob("*.txt"):

    filename = file_path.stem

    parts = filename.split("_")

    if len(parts) != 3:
        continue

    original_class = parts[0]
    hotel = parts[1]
    review_id = parts[2]

    if original_class not in ["t", "d"]:
        continue

    truth_label = (
        "truthful" if original_class == "t"
        else "deceptive"
    )

    sentiment = file_path.parts[-5]

    review_text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    normalized = normalize_text(review_text)

    original_reviews[normalized] = {
        "truth_label": truth_label,
        "hotel": hotel,
        "review_id": review_id,
        "sentiment": sentiment,
        "filename": filename + ".txt"
    }


print("\n========== ORIGINAL CORPUS ==========")
print("Total original reviews:", len(original_reviews))

print(
    "Original class distribution:",
    Counter(
        item["truth_label"]
        for item in original_reviews.values()
    )
)


# --------------------------------------------------
# 3. Match CSV reviews against original corpus
# --------------------------------------------------

matched_records = []
unmatched_records = []

for _, row in df.iterrows():

    normalized = normalize_text(row["Ori_Review"])

    if normalized in original_reviews:

        metadata = original_reviews[normalized]

        matched_records.append({
            "review": row["Ori_Review"],
            "csv_label": row["Label"],
            "truth_label": metadata["truth_label"],
            "hotel": metadata["hotel"],
            "sentiment": metadata["sentiment"],
            "original_filename": metadata["filename"]
        })

    else:
        unmatched_records.append(row["Ori_Review"])


matched_df = pd.DataFrame(matched_records)


# --------------------------------------------------
# 4. Display matching results
# --------------------------------------------------

print("\n========== MATCHING RESULTS ==========")

print("CSV records:", len(df))
print("Matched records:", len(matched_df))
print("Unmatched records:", len(unmatched_records))

if len(matched_df) > 0:

    print("\nCSV Label vs Original Truth Label:")
    print(
        pd.crosstab(
            matched_df["csv_label"],
            matched_df["truth_label"]
        )
    )

    print("\nCSV Label vs Original Sentiment:")
    print(
        pd.crosstab(
            matched_df["csv_label"],
            matched_df["sentiment"]
        )
    )

    print("\nHotel distribution:")
    print(matched_df["hotel"].value_counts())

    # Save recovered metadata
    output_path = (
        ROOT / "data/processed/verified_dataset.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    matched_df.to_csv(output_path, index=False)

    print("\nVerified dataset saved to:")
    print(output_path)


# --------------------------------------------------
# 5. Show unmatched examples
# --------------------------------------------------

if unmatched_records:

    print("\n========== UNMATCHED EXAMPLES ==========")

    for review in unmatched_records[:5]:
        print("\n", str(review)[:250])


print("\n========== VERIFICATION COMPLETE ==========")