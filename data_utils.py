import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/cleaned.csv")
    return df.dropna(subset=["raw_text", "text"])

def get_split(df):
    """Same split everywhere, so all experiments are comparable."""
    return train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )