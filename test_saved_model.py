import joblib

loaded_model = joblib.load("model.pkl")
loaded_vectorizer = joblib.load("vectorizer.pkl")

sample_text = ["government announces new tax policy today"]
sample_vector = loaded_vectorizer.transform(sample_text)
prediction = loaded_model.predict(sample_vector)

print("Prediction (0=Fake, 1=Real):", prediction)