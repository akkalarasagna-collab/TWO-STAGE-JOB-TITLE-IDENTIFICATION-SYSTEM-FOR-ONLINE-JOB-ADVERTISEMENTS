# ml/preprocess.py — Text preprocessing pipeline

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources if not already present
for resource in ('stopwords', 'wordnet', 'omw-1.4'):
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialise once — reused for every call
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """
    Clean and normalise a raw job description string.

    Steps:
      1. Lowercase
      2. Remove special characters and digits (keep spaces)
      3. Tokenise on whitespace
      4. Remove English stopwords
      5. Lemmatize each remaining token
      6. Rejoin into a single string

    Returns the cleaned string.
    """
    if not text:
        return ''

    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove special characters and numbers (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Step 3 — Tokenise
    tokens = text.split()

    # Step 4 & 5 — Remove stopwords and lemmatize
    cleaned_tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    ]

    return ' '.join(cleaned_tokens)
