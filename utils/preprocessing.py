import re

def clean_text_for_tfidf(text):
    """
    Aggressive cleaning strictly for TF-IDF (lowercasing, removing punctuation).
    Used in the baseline Model 1 implementation.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def get_options_list(row):
    """Extracts the 5 options from a dataframe row into a clean list."""
    return [str(row[opt]) for opt in ['A', 'B', 'C', 'D', 'E']]

def format_multiple_choice(prompt, option):
    """Concatenates a prompt and an option safely."""
    return f"{str(prompt).strip()} {str(option).strip()}"

def format_transformer_option(option_text, option_letter):
    """
    Formats options specifically for semantic transformer models (MiniLM/DeBERTa).
    Unlike TF-IDF, transformers need sentence structure and punctuation intact.
    We append the option letter to provide explicit structural cues.
    """
    return f"({option_letter}) {str(option_text).strip()}"