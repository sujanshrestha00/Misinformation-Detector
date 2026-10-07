import pandas as pd
from preprocess import preprocess
 
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")
 
true_df["label"] = 1
fake_df["label"] = 0
 
df = pd.concat([true_df, fake_df], ignore_index=True)
 
# --- Remove empty and duplicate articles (ITEM 1) ---
# Same article in both train and test would inflate accuracy.
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]
n_before = len(df)
df = df.drop_duplicates(subset="text")
print(f"Removed {n_before - len(df)} duplicate articles")
 
# Keep the original text too (needed for embeddings and the ablation)
df["raw_text"] = df["text"]
df["text"] = df["raw_text"].apply(preprocess)
 
# Cleaning can make two different articles identical, so dedupe again
df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset="text")
 
df[["raw_text", "text", "label"]].to_csv("data/cleaned.csv", index=False)
 
print("Done. Saved to data/cleaned.csv")
print(df.shape)
print(df["label"].value_counts())