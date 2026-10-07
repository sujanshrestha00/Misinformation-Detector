import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import load_data, get_split

df = load_data()
train_df, test_df = get_split(df)
X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000, C=0.3)
lr_model.fit(X_train_tfidf, y_train)

print("=== Logistic Regression ===")
print("Train accuracy:", lr_model.score(X_train_tfidf, y_train))
print("Test accuracy:", lr_model.score(X_test_tfidf, y_test))
y_pred_lr = lr_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_lr, target_names=["Fake", "Real"]))
print(confusion_matrix(y_test, y_pred_lr))

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

print("\n=== Random Forest ===")
print("Train accuracy:", rf_model.score(X_train_tfidf, y_train))
print("Test accuracy:", rf_model.score(X_test_tfidf, y_test))
y_pred_rf = rf_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_rf, target_names=["Fake", "Real"]))
print(confusion_matrix(y_test, y_pred_rf))

# --- ITEM 2: top words per class ---
# Logistic Regression = one weight per word.
# Positive weight pushes toward Real (label 1), negative toward Fake (label 0).
names = np.array(vectorizer.get_feature_names_out())
coefs = lr_model.coef_[0]
order = np.argsort(coefs)
print("\n=== Top 25 words/phrases pushing toward FAKE ===")
print(list(names[order[:25]]))
print("\n=== Top 25 words/phrases pushing toward REAL ===")
print(list(names[order[-25:][::-1]]))
print("\nIf these look like source names/styles (e.g. 'said', 'washington',")
print("'featured image') instead of claims, the model learned SOURCE STYLE.")

# Save in-dataset accuracy so the app's "Model info" button can show it
import json
metrics = {
    "Logistic Regression": {"in_dataset": round(lr_model.score(X_test_tfidf, y_test) * 100, 1)},
    "Random Forest": {"in_dataset": round(rf_model.score(X_test_tfidf, y_test) * 100, 1)},
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(lr_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(rf_model, "rf_model.pkl")
print("\nModels and vectorizer saved.")