import pandas as pd

AMAZON_TRAIN_PATH = (
    "DATASET/amazon_review_polarity_csv/train.csv"
)

AMAZON_TEST_PATH = (
    "DATASET/amazon_review_polarity_csv/test.csv"
)


def load_amazon_dataset(sample_size=None):

    columns = ["label", "title", "review"]

    train_df = pd.read_csv(
        AMAZON_TRAIN_PATH,
        header=None,
        names=columns
    )

    test_df = pd.read_csv(
        AMAZON_TEST_PATH,
        header=None,
        names=columns
    )

    if sample_size:
        train_df = train_df.sample(
            n=min(sample_size, len(train_df)),
            random_state=42
        )

    return train_df, test_df


if __name__ == "__main__":

    train, test = load_amazon_dataset(1000)

    print("Training Dataset:")
    print(train.head())

    print("\nDataset Shape:")
    print(train.shape)

    print("\nLabel Distribution:")
    print(train["label"].value_counts())