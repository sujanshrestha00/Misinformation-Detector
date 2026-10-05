import tkinter as tk
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_and_lemmatize(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords_and_lemmatize(text)
    return text

def check_article():
    raw_text = text_box.get("1.0", tk.END).strip()
    if not raw_text:
        result_label.config(text="Please paste some text first.", fg="black")
        return

    cleaned = preprocess(raw_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    confidence = probability[prediction] * 100

    if prediction == 1:
        result_label.config(text=f"✅ Likely REAL — {confidence:.1f}% confidence", fg="green")
    else:
        result_label.config(text=f"⚠️ Likely FAKE — {confidence:.1f}% confidence", fg="red")

window = tk.Tk()
window.title("Misinformation Detector")
window.geometry("500x400")

label = tk.Label(window, text="Paste a news article or headline below:")
label.pack(pady=10)

text_box = tk.Text(window, height=10, width=55)
text_box.pack(pady=5)

check_button = tk.Button(window, text="Check Article", command=check_article)
check_button.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

window.mainloop()