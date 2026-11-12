import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
path = "DATASET/deceptive_opinion_spam/Chicago_Hotel_Reviews.csv"

df = pd.read_csv(path)

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("\n========== DATASET OVERVIEW ==========")

print("Total Reviews:", len(df))

print("\nLabel Distribution:")
print(df["Label"].value_counts())

print("\nLabel Percentages:")
print(df["Label"].value_counts(normalize=True) * 100)

print("\nRating Distribution:")
print(df["Rating"].value_counts().sort_index())

print("\nLabel vs Rating:")
print(pd.crosstab(df["Label"], df["Rating"]))

# Review length
df["review_length"] = df["Ori_Review"].astype(str).apply(
    lambda x: len(x.split())
)

print("\nReview Length Statistics:")
print(df["review_length"].describe())

# Duplicate reviews
print(
    "\nExact Duplicate Reviews:",
    df["Ori_Review"].duplicated().sum()
)

# Average review length by label
print("\nAverage Review Length by Label:")
print(df.groupby("Label")["review_length"].mean())

# Visualizations

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Label"
)

plt.title("Review Label Distribution")
plt.savefig("reports/label_distribution.png")
plt.close()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Rating",
    hue="Label"
)

plt.title("Rating Distribution by Label")
plt.savefig("reports/rating_distribution.png")
plt.close()


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Label",
    y="review_length"
)

plt.title("Review Length by Label")
plt.savefig("reports/review_length.png")
plt.close()


print("\nAnalysis completed!")
print("Graphs saved inside reports/")