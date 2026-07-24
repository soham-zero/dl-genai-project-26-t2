import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_tfidf_baseline(val_df, clean_func):
    """
    Runs the TF-IDF naive baseline. 
    Fits the vectorizer on prompts and options, then computes cosine similarity 
    between the prompt and each option to rank the top 3 choices.
    """
    predictions = []
    
    # Simple naive loop (mirrors the notebook implementation)
    for idx, row in val_df.iterrows():
        prompt = clean_func(row['prompt'])
        opts = [clean_func(row[opt]) for opt in ['A', 'B', 'C', 'D', 'E']]
        
        # Fit a local vectorizer just for this question's vocabulary
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([prompt] + opts)
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Rank top 3
            top_indices = np.argsort(sim_scores)[::-1][:3]
            pred_str = " ".join([['A', 'B', 'C', 'D', 'E'][i] for i in top_indices])
        except ValueError:
            # Fallback if text is completely empty and vectorizer fails
            pred_str = "A B C" 
            
        predictions.append(pred_str)
        
    return predictions
