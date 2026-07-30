import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os

def run_inference(test_csv_path='test.csv', 
                  model_path='./models/finetuned_deberta', 
                  output_csv='submission.csv',
                  batch_size=8):
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMultipleChoice.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.eval()
    model.to(device)
    
    test_df = pd.read_csv(test_csv_path)
    
    letters = ['A', 'B', 'C', 'D', 'E']
    submission_preds = []
    
    print(f"Running inference on {len(test_df)} rows...")
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i+batch_size]
        
        prompts = sum([[context] * 5 for context in batch_df['prompt']], [])
        options = sum([[f"(A) {row['A']}", f"(B) {row['B']}", f"(C) {row['C']}", f"(D) {row['D']}", f"(E) {row['E']}"] for _, row in batch_df.iterrows()], [])
        
        inputs = tokenizer(prompts, options, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.view(len(batch_df), 5, -1).to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
            
        for p in probs:
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
