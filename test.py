import pandas as pd
import re

def remove_reuters(text):
    return re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)


true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

# print(df.shape)
# print(df.isnull().sum())
# print(df["subject"].value_counts())
# print(df["label"].value_counts())


# print(df.groupby("subject")["label"].value_counts())
# print(df[df["label"]==1]["text"].iloc[0][:200])
# print(df[df["label"]==0]["text"].iloc[0][:200])

df["has_reuters"] = df["text"].str.contains("Reuters", case=False)
print(df.groupby("label")["has_reuters"].mean())

df["text"] = df["text"].apply(remove_reuters)
df["has_reuters"] = df["text"].str.contains("Reuters", case=False)
print(df.groupby("label")["has_reuters"].mean())