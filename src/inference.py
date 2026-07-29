import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os

def run_inference(test_csv_path='test.csv', 
                  deberta_model_path='./models/finetuned_microsoft_deberta-v3-small', 
                  roberta_model_path='./models/finetuned_roberta-base',
                  output_csv='submission.csv',
                  batch_size=8,
                  deberta_weight=0.70,
                  roberta_weight=0.30):
    
    print("Loading models and tokenizers...")
    deb_tok = AutoTokenizer.from_pretrained(deberta_model_path)
    deb_model = AutoModelForMultipleChoice.from_pretrained(deberta_model_path)
    
    rob_tok = AutoTokenizer.from_pretrained(roberta_model_path)
    rob_model = AutoModelForMultipleChoice.from_pretrained(roberta_model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    deb_model.eval()
    rob_model.eval()
    deb_model.to(device)
    rob_model.to(device)
    
    test_df = pd.read_csv(test_csv_path)
    
    letters = ['A', 'B', 'C', 'D', 'E']
    submission_preds = []
    
    print(f"Running inference on {len(test_df)} rows...")
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        
        prompts = sum([[context] * 5 for context in batch_df['prompt']], [])
        options = sum([[f"(A) {row['A']}", f"(B) {row['B']}", f"(C) {row['C']}", f"(D) {row['D']}", f"(E) {row['E']}"] for _, row in batch_df.iterrows()], [])
        
        d_inputs = deb_tok(prompts, options, truncation=True, padding=True, return_tensors="pt")
        d_inputs = {k: v.view(len(batch_df), 5, -1).to(device) for k, v in d_inputs.items()}
        
        r_inputs = rob_tok(prompts, options, truncation=True, padding=True, return_tensors="pt")
        r_inputs = {k: v.view(len(batch_df), 5, -1).to(device) for k, v in r_inputs.items()}
        
        with torch.no_grad():
            d_probs = F.softmax(deb_model(**d_inputs).logits, dim=-1).cpu().numpy()
            r_probs = F.softmax(rob_model(**r_inputs).logits, dim=-1).cpu().numpy()
            
        ensembled_probs = (d_probs * deberta_weight) + (r_probs * roberta_weight)
        
        for p in ensembled_probs:
            top_indices = np.argsort(p)[::-1][:3]
            pred_str = " ".join([letters[idx] for idx in top_indices])
            submission_preds.append(pred_str)
            
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': submission_preds
    })
    
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

if __name__ == '__main__':
    run_inference()
