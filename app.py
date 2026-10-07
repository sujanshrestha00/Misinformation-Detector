import json
import os
import tkinter as tk
from tkinter import messagebox
import joblib
from preprocess import preprocess

models = {
    "Logistic Regression": joblib.load("model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
}
vectorizer = joblib.load("vectorizer.pkl")

MIN_WORDS = 50          # shorter text gives meaningless results
UNCERTAIN_BELOW = 70.0  # below this confidence we say "not sure"

def judge(model, vector):
    """Return (verdict, confidence%) for one model."""
    probability = model.predict_proba(vector)[0]
    classes = list(model.classes_)          # never assume the order
    p_real = probability[classes.index(1)]
    p_fake = probability[classes.index(0)]
    confidence = max(p_real, p_fake) * 100
    if confidence < UNCERTAIN_BELOW:
        return "unsure", confidence
    return ("real" if p_real > p_fake else "fake"), confidence

TEXT = {
    "real": ("Style resembles RELIABLE news", "green"),
    "fake": ("Style resembles FAKE-news sites", "red"),
    "unsure": ("Not sure (doesn't resemble training data)", "orange"),
}

def check_article():
    raw_text = text_box.get("1.0", tk.END).strip()
    for lbl in result_labels.values():
        lbl.config(text="")
    agree_label.config(text="")

    if not raw_text:
        agree_label.config(text="Please paste some text first.", fg="black")
        return
    if len(raw_text.split()) < MIN_WORDS:
        agree_label.config(
            text=f"Too short. Paste at least {MIN_WORDS} words (a few paragraphs).", fg="black")
        return

    try:
        vector = vectorizer.transform([preprocess(raw_text)])
        results = {name: judge(m, vector) for name, m in models.items()}
    except Exception as e:
        agree_label.config(text=f"Error: {e}", fg="black")
        return

    for name, (verdict, conf) in results.items():
        msg, color = TEXT[verdict]
        result_labels[name].config(text=f"{name}: {msg} ({conf:.1f}%)", fg=color)

    verdicts = {v for v, _ in results.values()}
    if len(verdicts) == 1:
        agree_label.config(text="Both models agree.", fg="black")
    else:
        agree_label.config(text="The models disagree. Treat the result with extra caution.", fg="black")

def show_info():
    if not os.path.exists("metrics.json"):
        messagebox.showinfo("Model info", "Run train_model.py first.")
        return
    metrics = json.load(open("metrics.json"))
    lines = []
    for name, m in metrics.items():
        inside = m.get("in_dataset", "n/a")
        outside = m.get("outside", "not tested yet")
        lines.append(f"{name}\n  Same-source test accuracy: {inside}%\n  Outside-news accuracy: {outside}%\n")
    lines.append("The same-source number is high because the test articles come from the\n"
                 "same sources as the training data. The outside number is the more honest one.\n"
                 "Neither number is the accuracy for the article you pasted.")
    messagebox.showinfo("Model info", "\n".join(lines))

def clear_all():
    text_box.delete("1.0", tk.END)
    for lbl in result_labels.values():
        lbl.config(text="")
    agree_label.config(text="")

window = tk.Tk()
window.title("Misinformation Detector")
window.geometry("580x600")

tk.Label(window, text="Paste a news article below:").pack(pady=10)

frame = tk.Frame(window)
frame.pack(pady=5)
scroll = tk.Scrollbar(frame)
scroll.pack(side=tk.RIGHT, fill=tk.Y)
text_box = tk.Text(frame, height=12, width=62, wrap=tk.WORD, yscrollcommand=scroll.set)
text_box.pack(side=tk.LEFT)
scroll.config(command=text_box.yview)

buttons = tk.Frame(window)
buttons.pack(pady=10)
tk.Button(buttons, text="Check Article", command=check_article).pack(side=tk.LEFT, padx=5)
tk.Button(buttons, text="Clear", command=clear_all).pack(side=tk.LEFT, padx=5)
tk.Button(buttons, text="Model info", command=show_info).pack(side=tk.LEFT, padx=5)

result_labels = {}
for name in models:
    lbl = tk.Label(window, text="", font=("Arial", 11, "bold"), wraplength=540)
    lbl.pack(pady=4)
    result_labels[name] = lbl

agree_label = tk.Label(window, text="", font=("Arial", 10), wraplength=540)
agree_label.pack(pady=6)

tk.Label(
    window,
    text="Note: trained on 2016-2017 political news. It detects writing style,\nnot truth, and may not generalize to other sources, topics or years.",
    font=("Arial", 8), fg="gray",
).pack(pady=10)

window.mainloop()