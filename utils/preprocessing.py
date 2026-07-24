import re

def clean_text_for_tfidf(text):
    """
    Aggressive cleaning strictly for TF-IDF (lowercasing, removing punctuation).
    Used in the baseline Model 1 implementation.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()
