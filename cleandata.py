import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_reuters(text):
    return re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_and_lemmatize(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

df["text"] = df["text"].apply(remove_reuters)
df["text"] = df["text"].str.replace(r"(?i)\breuters\b", "", regex=True)
df["text"] = df["text"].apply(clean_text)
df["text"] = df["text"].apply(remove_stopwords_and_lemmatize)

df.to_csv("data/cleaned.csv", index=False)

print("Done. Saved to data/cleaned.csv")
print(df.shape)
print(df["text"].iloc[0][:300])