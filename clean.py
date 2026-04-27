import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


CUSTOM_STOPWORDS = {
    "reuters",
    "getty",
    "breitbart",
    "ap",
    "com",
    "said",
    "image",
    "pic",
    "mr",
    "sen",
}


def _ensure_nltk_resources() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_resources()
_STOPWORDS = set(stopwords.words("english")) | CUSTOM_STOPWORDS
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in _STOPWORDS]
    tokens = [_LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(tokens)