"""ITEM 5: test BOTH saved models on articles from OUTSIDE the dataset.

Create data/outside.csv with two columns:
    text,label
where label is 1 for real, 0 for fake. Paste FULL articles (not headlines).
"""
import json
import os
import pandas as pd
import joblib
from preprocess import preprocess

UNCERTAIN_BELOW = 0.70

models = {
    "Logistic Regression": joblib.load("model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
}
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("data/outside.csv").dropna(subset=["text"])
X = vectorizer.transform(df["text"].apply(preprocess))

metrics = json.load(open("metrics.json")) if os.path.exists("metrics.json") else {}

for name, model in models.items():
    classes = list(model.classes_)
    p_real = model.predict_proba(X)[:, classes.index(1)]
    pred = (p_real >= 0.5).astype(int)
    conf = [max(p, 1 - p) for p in p_real]
    uncertain = [c < UNCERTAIN_BELOW for c in conf]
    correct = pred == df["label"].values

    print(f"\n=========== {name} ===========")
    for i in range(len(df)):
        verdict = "UNSURE" if uncertain[i] else ("REAL" if pred[i] == 1 else "FAKE")
        truth = "real" if df["label"].iloc[i] == 1 else "fake"
        print(f"{i:3d} | truth={truth} | model={verdict:6s} ({conf[i]*100:.0f}%) | {df['text'].iloc[i][:50]!r}")

    acc = correct.mean() * 100
    n_unsure = sum(uncertain)
    conf_idx = [i for i in range(len(df)) if not uncertain[i]]
    wrong = sum(1 for i in conf_idx if not correct[i])
    print(f"\nArticles: {len(df)}")
    print(f"Accuracy (forced guess on every article): {acc:.1f}%")
    print(f"Said 'unsure': {n_unsure}")
    print(f"Confident predictions: {len(conf_idx)} | CONFIDENTLY WRONG: {wrong}")

    metrics.setdefault(name, {})["outside"] = round(acc, 1)

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved outside accuracy to metrics.json (shown in the app's Model info).")