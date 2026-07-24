import re

# A standard list of English stopwords that confuse lexical TF-IDF matching
STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
    'when', 'where', 'how', 'which', 'who', 'whom', 'this', 'that', 'these', 
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
    'has', 'had', 'do', 'does', 'did', 'to', 'from', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
    'can', 'will', 'just', 'don', 'should', 'now', 'of', 'for', 'with', 'about', 
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
    'below', 'up', 'down', 'by'
])

def clean_text_for_tfidf(text):
    """
    Aggressive cleaning strictly for TF-IDF (lowercasing, removing punctuation, stopwords).
    Used in the baseline Model 1 implementation.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # FAILED EXPERIMENT: Removing stopwords dropped MAP@3 from 0.3117 to ~0.27!
    # It turns out that in MCQ matching, words like "not" or "is" hold critical intent signals.
    # We are keeping this commented out to maintain the higher 0.3117 baseline.
    # 
    # words = text.split()
    # words = [w for w in words if w not in STOPWORDS]
    # return " ".join(words).strip()
    
    return text.strip()
