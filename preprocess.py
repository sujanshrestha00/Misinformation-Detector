import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if missing (so the project runs on a fresh machine)
for pkg in ("stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()

_negation_words = {"not", "no", "nor", "never", "cannot", "neither", "nothing", "none"}
_stop_words = set(stopwords.words('english')) - _negation_words

def remove_reuters(text):
    text = re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"(?i)\breuters\b", "", text)
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_and_lemmatize(text):
    words = text.split()
    words = [_lemmatizer.lemmatize(w) for w in words if w not in _stop_words]
    return " ".join(words)

def preprocess(text):
    text = remove_reuters(text)
    text = clean_text(text)
    text = remove_stopwords_and_lemmatize(text)
    return text