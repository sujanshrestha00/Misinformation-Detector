import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/cleaned.csv")
df = df.dropna(subset=["text"])  # just in case any row became empty after cleaning

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)



model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

train_accuracy = model.score(X_train_tfidf, y_train)
test_accuracy = model.score(X_test_tfidf, y_test)

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)


y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print(confusion_matrix(y_test, y_pred))


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

rf_train_acc = rf_model.score(X_train_tfidf, y_train)
rf_test_acc = rf_model.score(X_test_tfidf, y_test)

print("RF Train accuracy:", rf_train_acc)
print("RF Test accuracy:", rf_test_acc)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved.")