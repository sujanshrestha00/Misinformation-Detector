"""ITEM 7: does preprocessing matter?  Same model, different text preparation."""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import preprocess
from data_utils import load_data, get_split

df = load_data()
train_df, test_df = get_split(df)

outside = None
if os.path.exists("data/outside.csv"):
    outside = pd.read_csv("data/outside.csv").dropna(subset=["text"])

# name -> (training column, vectorizer settings, function applied to outside text)
configs = {
    "raw text":          ("raw_text", dict(stop_words=None),      lambda t: t),
    "stopwords removed": ("raw_text", dict(stop_words="english"), lambda t: t),
    "full preprocess":   ("text",     dict(stop_words=None),      preprocess),
}

rows = []
for name, (col, kwargs, prep_outside) in configs.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), **kwargs)),
        ("clf", LogisticRegression(max_iter=1000, C=0.3)),
    ])
    pipe.fit(train_df[col], train_df["label"])
    row = {"setup": name, "in-dataset acc %": round(pipe.score(test_df[col], test_df["label"]) * 100, 2)}
    if outside is not None:
        preds = pipe.predict(outside["text"].apply(prep_outside))
        row["outside acc %"] = round((preds == outside["label"]).mean() * 100, 1)
    rows.append(row)
    print("done:", name)

print()
print(pd.DataFrame(rows).to_string(index=False))