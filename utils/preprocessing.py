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
    Aggressive cleaning strictly for TF-IDF (lowercasing, removing punctuation).
    Used in the baseline Model 1 implementation.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words).strip()

def get_options_list(row):
    """Extracts the 5 options from a dataframe row into a clean list."""
    return [str(row[opt]) for opt in ['A', 'B', 'C', 'D', 'E']]

def format_multiple_choice(prompt, option):
    """Concatenates a prompt and an option safely."""
    return f"{str(prompt).strip()} {str(option).strip()}"