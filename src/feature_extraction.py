import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


PROMOTIONAL_WORDS = [
    "amazing",
    "excellent",
    "perfect",
    "best",
    "awesome",
    "wonderful",
    "must buy"
]


def extract_basic_features(texts):

    features = []

    for text in texts:

        text = str(text).lower()

        words = text.split()

        word_count = len(words)

        promotional_count = sum(
            text.count(word)
            for word in PROMOTIONAL_WORDS
        )

        features.append([
            word_count,
            len(text),
            promotional_count
        ])

    return np.array(features)


def create_tfidf_features(texts):

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer