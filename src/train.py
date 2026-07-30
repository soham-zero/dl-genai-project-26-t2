import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice, 
    TrainingArguments, 
    Trainer
)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from sklearn.model_selection import train_test_split
import os

# Helper for preprocessing
def preprocess_function(examples, tokenizer):
    prompts = [[context] * 5 for context in examples['prompt']]
    options = [
        [
            f"(A) {examples['A'][i]}", 
            f"(B) {examples['B'][i]}", 
            f"(C) {examples['C'][i]}", 
            f"(D) {examples['D'][i]}", 
            f"(E) {examples['E'][i]}"
        ] for i in range(len(examples['prompt']))
    ]
    
    # Flatten everything
    prompts = sum(prompts, [])
    options = sum(options, [])
    
    tokenized_examples = tokenizer(prompts, options, truncation=True, padding=False)
    
    # Unflatten
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def train_model(train_csv_path='train.csv', model_name="microsoft/deberta-v3-small", output_dir="./models/finetuned_deberta"):
    print(f"Loading data from {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    df.drop_duplicates(subset=['prompt'], inplace=True)
    
    # Create strict 80/20 split to avoid data leakage
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Train Split: {len(train_df)} rows")
    print(f"Validation Split: {len(val_df)} rows")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    # Map letters to integer labels
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # Using .copy() to avoid SettingWithCopyWarning
    train_df_80 = train_df.copy()
    val_df_20 = val_df.copy()
    
    train_df_80['label'] = train_df_80['answer'].map(label_map)
    val_df_20['label'] = val_df_20['answer'].map(label_map)
    
    train_ds = Dataset.from_pandas(train_df_80[['prompt', 'A', 'B', 'C', 'D', 'E', 'label']])
    val_ds = Dataset.from_pandas(val_df_20[['prompt', 'A', 'B', 'C', 'D', 'E', 'label']])
    
    tokenized_train = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E'])
    tokenized_val = val_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E'])
    
    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.replace('/', '_')}",
        eval_strategy="epoch",
        learning_rate=1e-5,
        warmup_steps=100,
        adam_epsilon=1e-6,
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        fp16=False, # GPU Acceleration
        report_to="none" # Disabled W&B until Session 15
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )
    
    trainer.train()
    
    # Save the fine-tuned model
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model()
