#!/usr/bin/env python3
import argparse
import os
import json
import logging
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_personality_data(data_file):
    """Load personality training data from JSONL file."""
    logger.info(f"Loading personality data from {data_file}")
    
    data = []
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found")
        return []
    
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse line: {line[:100]}... Error: {e}")
    
    logger.info(f"Loaded {len(data)} examples")
    return data

def prepare_training_data(personality_data, trait, tokenizer, max_length=512):
    """Prepare the data for fine-tuning."""
    logger.info(f"Preparing training data for trait: {trait}")
    
    texts = []
    
    for item in personality_data:
        # Extract the relevant information
        trait_level = item.get('level_P2', 'unknown')
        prompt = item.get('prompt_P2', '')
        response = item.get('response_P2', '')
        
        if prompt and response:
            # Format as a conversation
            conversation = f"<|im_start|>system\nYou are a helpful assistant with {trait_level} {trait}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            texts.append(conversation)
    
    # If we don't have enough data, generate synthetic examples
    if len(texts) < 100:
        logger.warning(f"Only {len(texts)} examples found, generating synthetic data...")
        synthetic_texts = generate_synthetic_training_data(trait, 500 - len(texts))
        texts.extend(synthetic_texts)
    
    logger.info(f"Prepared {len(texts)} training examples")
    
    # Tokenize the texts
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return tokenized_dataset

def generate_synthetic_training_data(trait, num_examples):
    """Generate synthetic training data for a specific trait."""
    trait_descriptions = {
        'o': {
            'high': 'high openness to experience - creative, imaginative, and intellectually curious',
            'low': 'low openness to experience - practical, conventional, and prefers familiar experiences'
        },
        'c': {
            'high': 'high conscientiousness - organized, disciplined, and goal-oriented',
            'low': 'low conscientiousness - spontaneous, flexible, and adaptable'
        },
        'e': {
            'high': 'high extraversion - outgoing, energetic, and sociable',
            'low': 'low extraversion - reserved, thoughtful, and prefers quiet activities'
        },
        'a': {
            'high': 'high agreeableness - cooperative, trusting, and considerate',
            'low': 'low agreeableness - direct, questioning, and assertive'
        },
        'n': {
            'high': 'high neuroticism - emotionally sensitive and reactive to stress',
            'low': 'low neuroticism - emotionally stable and resilient'
        }
    }
    
    topics = [
        "What are your thoughts on innovation and new technologies?",
        "How do you approach problem-solving?",
        "What motivates you in your daily life?",
        "How do you handle stressful situations?",
        "What's your perspective on working with others?",
        "How do you make important decisions?",
        "What role does creativity play in your life?",
        "How do you prefer to spend your free time?",
        "What's your approach to learning new things?",
        "How do you handle criticism or feedback?"
    ]
    
    synthetic_texts = []
    
    for i in range(num_examples):
        level = 'high' if i % 2 == 0 else 'low'
        topic = topics[i % len(topics)]
        trait_desc = trait_descriptions[trait][level]
        
        # Generate a simple response based on the trait and level
        if trait == 'o' and level == 'high':
            response = f"I find this topic fascinating! There are so many creative possibilities to explore here. I love thinking about innovative approaches and seeing connections that others might miss."
        elif trait == 'o' and level == 'low':
            response = f"I prefer to stick with tried-and-true methods. What's worked in the past is likely to work again, and I value stability over constant change."
        elif trait == 'c' and level == 'high':
            response = f"I approach this systematically. I like to plan things out, set clear goals, and work methodically toward achieving them."
        elif trait == 'c' and level == 'low':
            response = f"I tend to be more flexible and go with the flow. I adapt as situations change rather than sticking rigidly to a plan."
        elif trait == 'e' and level == 'high':
            response = f"I love discussing this with others! I get energized by bouncing ideas around and hearing different perspectives from lots of people."
        elif trait == 'e' and level == 'low':
            response = f"I prefer to think about this quietly first. I like to process information internally before sharing my thoughts with others."
        elif trait == 'a' and level == 'high':
            response = f"I try to consider how this affects everyone involved. Cooperation and finding common ground are really important to me."
        elif trait == 'a' and level == 'low':
            response = f"I focus on what makes logical sense, even if it might not be what everyone wants to hear. Sometimes you have to be direct."
        elif trait == 'n' and level == 'high':
            response = f"This kind of situation can be quite stressful for me. I tend to worry about potential problems and feel things quite intensely."
        else:  # trait == 'n' and level == 'low'
            response = f"I generally stay pretty calm about these things. I don't let stress get to me easily and prefer to focus on solutions."
        
        conversation = f"<|im_start|>system\nYou are a helpful assistant with {trait_desc}.<|im_end|>\n<|im_start|>user\n{topic}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        synthetic_texts.append(conversation)
    
    return synthetic_texts

def fine_tune_model(args):
    """Fine-tune a model for a specific personality trait."""
    logger.info(f"Fine-tuning model for trait: {args.trait}")
    
    # Load tokenizer and model - use a valid HuggingFace model
    valid_model_name = "microsoft/DialoGPT-medium"
    if args.base_model and args.base_model != "gemma3":
        model_name = args.base_model
    else:
        # If base_model is gemma3 or not specified, use DialoGPT
        model_name = valid_model_name
    
    logger.info(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load and prepare training data
    personality_data = load_personality_data(args.train_data)
    if not personality_data:
        logger.error("No training data found")
        return False
    
    # Prepare dataset
    train_dataset = prepare_training_data(personality_data, args.trait, tokenizer)
    
    # Split into train and eval
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_pin_memory=False,  # Helps with MacOS issues
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Model saved to {args.output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fine-tune personality-specific models")
    parser.add_argument("--trait", required=True, choices=['o', 'c', 'e', 'a', 'n'],
                        help="Personality trait to fine-tune for")
    parser.add_argument("--train-data", required=True,
                        help="Path to training data JSONL file")
    parser.add_argument("--base-model", default="microsoft/DialoGPT-medium",
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune the model
    success = fine_tune_model(args)
    
    if success:
        logger.info("Fine-tuning completed successfully!")
    else:
        logger.error("Fine-tuning failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 