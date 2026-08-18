# Project Report for Smart MCQ Solver Challenge (T22026)

## 1. Abstract / Executive Summary
This project tackles the Smart MCQ Solver Challenge, requiring the development of an intelligent pipeline to rank multiple-choice question answers based on complex contextual prompts. The primary objective was to maximize the Mean Average Precision at 3 (MAP@3) metric. We engineered and evaluated three distinct architectural paradigms:
- **A custom Bidirectional LSTM** trained entirely from scratch.
- **A zero-shot semantic embedder** (MiniLM).
- **A state-of-the-art fine-tuned Transformer** with Disentangled Attention (DeBERTa-v3).

Our final fine-tuned DeBERTa model successfully learned the underlying syntactic logic of the dataset, achieving a near-perfect validation score of **0.9975**, proving the superiority of task-specific fine-tuning over both smaller from-scratch architectures and zero-shot semantic matching.

## 2. Introduction
### Problem Statement
The challenge requires analyzing a contextual prompt and five potential string options (A, B, C, D, E). The task is to logically deduce the correct answer and output a ranked list of the top 3 most likely options.
### Project Objective
The goal was to build a machine learning pipeline capable of leveraging advanced Generative AI architectures to solve Reading Comprehension questions. Furthermore, we aimed to implement robust MLOps practices by tracking hyperparameter convergence and custom classification metrics (F1 Macro, Accuracy) using Weights & Biases (WandB).
### Report Structure
This report details our dataset preprocessing strategy (Section 3), tokenization implementation (Section 4), architectural experimentation and tuning (Section 5), comparative analysis of model performance (Section 6), error analysis of failures (Section 7), and final conclusions (Section 8).

## 3. Dataset & Preprocessing
### Dataset Description
The provided training dataset contains textual prompts, 5 answer options, and a target label. 
### Exploratory Data Analysis (EDA)
Our EDA confirmed zero missing values and zero exact duplicates. We observed that prompt lengths were relatively uniform. A critical tokenization check confirmed that 100% of the prompts easily fit within a 256-token context window, meaning no text truncation would occur for our transformer models. The correct answer distribution (A-E) was perfectly balanced, mitigating the need for class weighting.
### Data Preprocessing & Leakage Prevention
Because the data was completely clean, no extreme transformations were necessary. Crucially, a strict 80/20 `train_test_split` was enforced *before* any tokenization or preprocessing. This guaranteed zero data leakage, ensuring our validation metrics accurately reflected generalization capability.

## 4. Tokenization Strategy
### Primary Tokenizer Selection (DeBERTa-v3)
For our best-performing model (DeBERTa-v3), we utilized the `DebertaV2TokenizerFast` (SentencePiece). 
### Justification
Unlike standard WordPiece tokenizers, SentencePiece builds vocabulary directly from raw text without requiring pre-tokenization (like space separation), making it highly resilient to varying textual formats and multi-lingual noise.
### Implementation Details
Using the Hugging Face `AutoTokenizer`, we concatenated the prompt with each of the 5 options, creating 5 distinct sequence pairs per question. Because our EDA proved no prompts exceeded 256 tokens, we were able to safely rely on dynamic padding without fear of silently truncating critical context.

## 5. Modeling, Architecture & Experimentation

### 5.1 Model 1: Neural Network Built From Scratch (Bi-LSTM)
- **Architecture Flow**: `Raw Text -> Custom Vocabulary (Token IDs) -> Embedding Layer (dim=32) -> 1-Layer Bidirectional LSTM (hidden=16) -> Concat(Forward, Backward) -> Linear Classifier -> Softmax (Top 3)`
- **Salient Points**: We built this fundamental model to establish a "from scratch" deep learning baseline. However, we discovered a crucial Deep Learning lesson: training from scratch on a small dataset (~1,600 rows) is incredibly difficult. Because the dataset was so small, a large neural network would simply memorize the data.

### 5.2 Model 2: Zero-Shot Semantic Transformer (MiniLM)
- **Architecture Flow**: `Raw Text -> WordPiece Tokenizer -> MiniLM-L6-v2 Encoder -> Mean Pooling -> Dense Vector (dim=384) -> Cosine Similarity -> Rank Options`
- **Salient Points**: This model was selected to evaluate if pre-trained semantic understanding was sufficient to solve the challenge without fine-tuning. Options were ranked based on cosine similarity to the prompt vector.

### 5.3 Model 3: Fine-Tuned Transformer Model (DeBERTa-v3)
- **Architecture Flow**: `Prompt + Options -> SentencePiece Tokenizer -> DeBERTa-v3-small Encoder (Disentangled Attention) -> Pooler Output -> Linear Classification Head -> Softmax -> Rank Options`
- **Salient Points**: DeBERTa improves upon standard BERT by introducing a Disentangled Attention mechanism, where each word is represented using two vectors that encode its content and position separately. This is highly advantageous for reading comprehension MCQs where spatial positioning of subjects is critical.
- **Architecture Choice**: We opted for a pure DeBERTa-v3 model. While ensembles (e.g., blending with RoBERTa) are common, preliminary research suggested that simpler, high-capacity models often generalize better on small datasets without suffering from "Probability Dilution."

### 5.4 Hyperparameter Tuning & Logs
- **Bi-LSTM Tuning**: Initial runs with hidden dimensions of 128 and 64 massively overfit the training data, scoring high training accuracy but failing on the validation set. By tracking loss curves on WandB, we iteratively *reduced* the model capacity to `embed_dim=32` and `hidden_dim=16` with a high `weight_decay=1e-2`. This "nerfing" forced the model to generalize better, stabilizing the validation MAP@3.
- **DeBERTa-v3 Tuning**: We optimized using AdamW. Initial learning rates of `3e-5` proved too aggressive, causing catastrophic forgetting of pre-trained weights. We tracked the trials in WandB and found that a lower `learning_rate=1e-5` with a 100-step linear warmup provided smooth convergence. We utilized `gradient_accumulation_steps=2` with a batch size of 8 to simulate a larger effective batch size of 16 on limited hardware. The model was trained for 5 epochs without freezing any layers, allowing full adaptation to the MCQ format.

## 6. Model Evaluation & Results
All models were evaluated on an isolated 20% validation split. 

| Model Architecture | MAP@3 | Accuracy | F1 Macro |
| :--- | :--- | :--- | :--- |
| Model 1 (Bi-LSTM Scratch) | 0.5554 | 0.3825 | 0.3856 |
| Model 2 (MiniLM Semantic) | 0.3950 | 0.2375 | 0.2350 |
| Model 3 (Fine-Tuned DeBERTa) | 0.9975 | 0.9975 | 0.9974 |

## 7. Error Analysis & Model Failures
To better understand the limitations of our pipeline, we performed error analysis on instances where the models failed.

- **Bi-LSTM Failures**: Failed on prompts requiring complex logic or external knowledge. Its vocabulary was strictly limited to the training set, severely capping its semantic depth.
- **MiniLM Failures**: Suffered heavily from distractor options. It ranked incorrect answers highly if they shared similar words with the prompt, failing to logically reason.
- **DeBERTa-v3 Failures**: Achieved near-perfect accuracy (0.9975). The single failure occurred on an extremely ambiguous prompt where two options were syntactically identical but semantically opposite.

## 8. Final Conclusion
Our pipeline evolution yielded several profound insights across our three architectures:

- **Bi-LSTM (From Scratch)**: Proved that small datasets suffer from data starvation, requiring models that already "know" how to read.
- **MiniLM (Zero-Shot)**: Showed that dense embeddings are a step in the right direction, but ultimately capped our potential by failing on distractor options.
- **DeBERTa-v3 (Fine-Tuned)**: Was the absolute breakthrough—its massive pre-trained knowledge base combined with our precise MCQ fine-tuning and strict hyperparameter tuning (learning rate `1e-5`, batch size `16`) allowed it to achieve a near-perfect validation score of **0.9975**.