import torch
from transformers import RobertaTokenizer, RobertaConfig, TrainingArguments, Trainer
from modeling_roberta import RobertaForSequenceClassification
from datasets import Dataset
import pandas as pd
import ast
import numpy as np
from logging_config import setup_logging
import multiprocessing

# Fix for multiprocessing issues on MacOS
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Set up logging configuration
setup_logging()

import logging

# Initialize the logger
logger = logging.getLogger(__name__)

logger.info("Logging setup complete.")

# Check if MPS is available (Apple Silicon)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.info(f"Using device: {device}")

# Disable wandb if not needed for local training
# import wandb

# Load the datasets from the saved directories
from datasets import load_from_disk
train_dataset = load_from_disk("data/sample_dataset/train")
val_dataset = load_from_disk("data/sample_dataset/val")
test_dataset = load_from_disk("data/sample_dataset/test") if "data/sample_dataset/test" else None

logger.info("Loaded datasets. Preprocessing...")

# Define the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large", cache_dir=".cache")

# Preprocessing function to format the data correctly
def preprocess_dataset(examples):
    # Tokenize the text
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    
    # Add the personality traits as labels
    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels_openness": examples["openness"],
        "labels_conscientiousness": examples["conscientiousness"],
        "labels_extraversion": examples["extraversion"],
        "labels_agreeableness": examples["agreeableness"],
        "labels_neuroticism": examples["neuroticism"],
    }
    
    return result

# Apply preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_dataset, batched=True)
val_dataset = val_dataset.map(preprocess_dataset, batched=True)
if test_dataset:
    test_dataset = test_dataset.map(preprocess_dataset, batched=True)

# Remove the original columns that are no longer needed
columns_to_remove = ['text', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
train_dataset = train_dataset.remove_columns(columns_to_remove)
val_dataset = val_dataset.remove_columns(columns_to_remove)
if test_dataset:
    test_dataset = test_dataset.remove_columns(columns_to_remove)

logger.info("Preprocessing complete. Dataset ready for training.")

# if using MSE loss, then we should set num_labels to 1; if using Cross Entropy loss, then we should set num_labels to 3
num_labels = 1 # we have already defined the sub-loss for each personality dimension in modeling_roberta.py
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=num_labels, cache_dir=".cache")

training_args = TrainingArguments(
    output_dir="checkpoint/",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=1e-5,
    per_device_train_batch_size=4,          # Reduced batch size for MPS
    per_device_eval_batch_size=4,           # Reduced batch size for MPS
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,                     # Only save the last 2 checkpoints
    load_best_model_at_end=True,            # Load the best model at the end
    metric_for_best_model="eval_loss",      # Metric for determining best model
    greater_is_better=False,
    report_to="none",                       # Disable wandb for local training
    logging_dir='./logs',
    logging_steps=10,
    log_level='info',
    dataloader_num_workers=0,               # Disable multi-processing to avoid issues on MacOS
    gradient_accumulation_steps=16,
    fp16=False,                             # Disable fp16 for MPS
    remove_unused_columns=True,             # Make sure this is True to remove unused columns
)

logger.info(f"Training dataset columns: {train_dataset.column_names}")
logger.info(f"Validation dataset columns: {val_dataset.column_names}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

if __name__ == "__main__":
    trainer.train()

# O, C, E, A, N
# [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# (p'(O), y_O), (p'(C), y_C), (p'(E), y_E), (p'(A), y_A), (p'(N), y_N)