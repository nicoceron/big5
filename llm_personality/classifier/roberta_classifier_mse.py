import torch
from transformers import RobertaTokenizer, RobertaConfig, TrainingArguments, Trainer
from modeling_roberta import RobertaForSequenceClassification
from datasets import Dataset
import pandas as pd
import ast
import numpy as np
from logging_config import setup_logging

# Set up logging configuration
setup_logging()

import logging

# Initialize the logger
logger = logging.getLogger(__name__)

logger.info("Logging setup complete.")

import wandb

# Load the datasets from the saved directories
from datasets import load_from_disk
train_dataset = None
val_dataset = None
test_dataset = None

# if using MSE loss, then we should set num_labels to 1; if using Cross Entropy loss, then we should set num_labels to 3
num_labels = 1 # we have already defined the sub-loss for each personality dimension in modeling_roberta.py
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=num_labels, cache_dir=".cache")

training_args = TrainingArguments(
    output_dir="checkpoint/",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,                     # Only save the last 2 checkpoints
    load_best_model_at_end=True,            # Load the best model at the end
    metric_for_best_model="eval_loss",      # Metric for determining best model
    greater_is_better=False,
    report_to="wandb",
    logging_dir='./logs',
    logging_steps=10,
    log_level='info',
    dataloader_num_workers=4,
    gradient_accumulation_steps=16,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# O, C, E, A, N
# [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# (p'(O), y_O), (p'(C), y_C), (p'(E), y_E), (p'(A), y_A), (p'(N), y_N)