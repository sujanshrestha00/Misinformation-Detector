"""ITEM 5: test the saved model on articles from OUTSIDE the dataset.
 
Create data/outside.csv with two columns:
    text,label
where label is 1 for real, 0 for fake. Paste FULL articles (not headlines).
"""
import pandas as pd
import joblib
from preprocess import preprocess
 
UNCERTAIN_BELOW = 0.70
 
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("data/outside.csv").dropna(subset=["text"])
 
classes = list(model.classes_)
proba = model.predict_proba(vectorizer.transform(df["text"].apply(preprocess)))
p_real = proba[:, classes.index(1)]
 
df["pred"] = (p_real >= 0.5).astype(int)
df["confidence"] = [max(p, 1 - p) for p in p_real]
df["uncertain"] = df["confidence"] < UNCERTAIN_BELOW
df["correct"] = df["pred"] == df["label"]
 
print("\n--- Per article ---")
for i, r in df.iterrows():
    verdict = "UNSURE" if r["uncertain"] else ("REAL" if r["pred"] == 1 else "FAKE")
    truth = "real" if r["label"] == 1 else "fake"
    print(f"{i:3d} | truth={truth} | model={verdict:6s} ({r['confidence']*100:.0f}%) | {r['text'][:50]!r}")
 
print("\n--- Summary ---")
print("Articles:", len(df))
print("Accuracy (forcing a guess on every article):", round(df["correct"].mean() * 100, 1), "%")
print("Said 'unsure':", int(df["uncertain"].sum()))
confident = df[~df["uncertain"]]
if len(confident):
    print("Confident predictions:", len(confident),
          "| accuracy on those:", round(confident["correct"].mean() * 100, 1), "%")
    wrong = confident[~confident["correct"]]
    print("CONFIDENTLY WRONG:", len(wrong))