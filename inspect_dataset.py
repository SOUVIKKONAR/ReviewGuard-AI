import pandas as pd

path = "DATASET/deceptive_opinion_spam/Chicago_Hotel_Reviews.csv"

df = pd.read_csv(path)

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)