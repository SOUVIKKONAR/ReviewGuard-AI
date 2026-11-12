import re
import numpy as np

from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = [
            "word_count",
            "character_count",
            "average_word_length",
            "unique_word_ratio",
            "sentence_count",
            "average_sentence_length",
            "first_person_ratio",
            "second_person_ratio",
            "exclamation_count",
            "question_count",
            "uppercase_ratio",
            "punctuation_ratio",
            "promotional_word_ratio",
            "repeated_word_ratio",
            "digit_ratio"
        ]

        self.first_person = {
            "i", "me", "my", "mine",
            "we", "us", "our", "ours"
        }

        self.second_person = {
            "you", "your", "yours"
        }

        self.promotional_words = {
            "amazing", "excellent", "wonderful",
            "fantastic", "perfect", "best",
            "outstanding", "incredible", "awesome",
            "superb", "exceptional", "beautiful",
            "highly", "recommend", "unforgettable",
            "luxurious", "spectacular"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        results = []

        for text in X:

            text = str(text)

            words = re.findall(r"\b[\w']+\b", text.lower())

            word_count = len(words)
            character_count = len(text)

            unique_words = set(words)

            average_word_length = (
                np.mean([len(word) for word in words])
                if words else 0
            )

            unique_word_ratio = (
                len(unique_words) / word_count
                if word_count else 0
            )

            sentences = [
                sentence.strip()
                for sentence in re.split(r"[.!?]+", text)
                if sentence.strip()
            ]

            sentence_count = len(sentences)

            average_sentence_length = (
                word_count / sentence_count
                if sentence_count else 0
            )

            first_person_count = sum(
                word in self.first_person for word in words
            )

            second_person_count = sum(
                word in self.second_person for word in words
            )

            first_person_ratio = (
                first_person_count / word_count
                if word_count else 0
            )

            second_person_ratio = (
                second_person_count / word_count
                if word_count else 0
            )

            exclamation_count = text.count("!")
            question_count = text.count("?")

            uppercase_count = sum(
                char.isupper() for char in text
            )

            uppercase_ratio = (
                uppercase_count / character_count
                if character_count else 0
            )

            punctuation_count = sum(
                not char.isalnum() and not char.isspace()
                for char in text
            )

            punctuation_ratio = (
                punctuation_count / character_count
                if character_count else 0
            )

            promotional_count = sum(
                word in self.promotional_words
                for word in words
            )

            promotional_word_ratio = (
                promotional_count / word_count
                if word_count else 0
            )

            repeated_word_count = (
                word_count - len(unique_words)
            )

            repeated_word_ratio = (
                repeated_word_count / word_count
                if word_count else 0
            )

            digit_count = sum(
                char.isdigit() for char in text
            )

            digit_ratio = (
                digit_count / character_count
                if character_count else 0
            )

            features = [
                word_count,
                character_count,
                average_word_length,
                unique_word_ratio,
                sentence_count,
                average_sentence_length,
                first_person_ratio,
                second_person_ratio,
                exclamation_count,
                question_count,
                uppercase_ratio,
                punctuation_ratio,
                promotional_word_ratio,
                repeated_word_ratio,
                digit_ratio
            ]

            results.append(features)

        return csr_matrix(
            np.asarray(results, dtype=np.float64)
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names)