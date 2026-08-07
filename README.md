# Smart MCQ Solver — IITM DL & GenAI Project

This repository contains my end-to-end implementation for the Deep Learning and Generative AI term project. The goal of this project was to build a machine learning pipeline capable of understanding complex contextual prompts and ranking multiple-choice answers logically.

## Quick Links

- 🚀 **[Live Deployment (Hugging Face Spaces)](https://huggingface.co/spaces/soham-zero/Smart-MCQ-Solver)** — Test the final model yourself through a web UI.
- 📄 **[Full Project Report](Project_Report.md)** — Detailed analysis of the dataset, tokenization strategy, model architectures, and final metrics.

## Final Results

After evaluating a TF-IDF baseline and a zero-shot MiniLM model, I achieved my best results by fine-tuning a **DeBERTa-v3-small** model specifically for the Multiple Choice QA format.

* **Kaggle Leaderboard Score (MAP@3):** `0.75353`

## Repository Structure

The `notebooks/` folder contains milestone notebooks (milestone-1 through milestone-5) tracking incremental progress across the term, along with `final_notebook.ipynb` which is the main implementation.

### final_notebook.ipynb

This notebook is the core of the project and covers three distinct approaches to the MCQ problem:

1. **Baseline (TF-IDF + Cosine Similarity):** A classical retrieval approach using TF-IDF vectors to rank answer options by their cosine similarity to the question prompt. Served as the initial benchmark.

2. **Pretrained Model (zero-shot MiniLM):** Zero-shot classification using `all-MiniLM-L6-v2` from Sentence Transformers, with no task-specific fine-tuning. Showed strong out-of-the-box performance compared to the baseline.

3. **Fine-Tuned Model (DeBERTa-v3-small):** The final and best-performing approach. Each (question, option) pair was formatted as a multiple choice input and passed through a fine-tuned `microsoft/deberta-v3-small` model. Training used AdamW with a learning rate of 1e-5 over 5 epochs, tracked via Weights & Biases.

The notebook also includes the final Kaggle submission generation and exports the trained model weights for deployment.

## Deployment

The trained model is deployed as a Gradio web app on Hugging Face Spaces: [Smart-MCQ-Solver](https://huggingface.co/spaces/soham-zero/Smart-MCQ-Solver). The deployment code lives directly inside that Space's repository.

