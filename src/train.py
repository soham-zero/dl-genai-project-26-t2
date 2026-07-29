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

def train_model(model_name="microsoft/deberta-v3-small", train_df=None, val_df=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    # Map letters to integer labels
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    train_df['label'] = train_df['answer'].map(label_map)
    val_df['label'] = val_df['answer'].map(label_map)
    
    train_ds = Dataset.from_pandas(train_df[['prompt', 'A', 'B', 'C', 'D', 'E', 'label']])
    val_ds = Dataset.from_pandas(val_df[['prompt', 'A', 'B', 'C', 'D', 'E', 'label']])
    
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
        report_to="none" # Disabled W&B for Session 14
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
    save_path = f"./models/finetuned_{model_name.replace('/', '_')}"
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    print("This script is meant to be imported or run directly on Kaggle with prepared data.")
