import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse

# --- Configuration ---
# Model
BASE_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct" 

# Dataset

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1 
GRADIENT_ACCUMULATION_STEPS = 8 
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
MAX_SEQ_LENGTH = 512

# --- Helper Function to Format Dataset ---
def format_alpaca_prompt(example):
    if example.get("input"):
        formatted_string = f"Instruction: {example['instruction']}\\nInput: {example['input']}\\nOutput: {example['output']}"
    else:
        formatted_string = f"Instruction: {example['instruction']}\\nOutput: {example['output']}"
    return formatted_string

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Llama-3 model using SFT and LoRA.")
    parser.add_argument("--dataset_path", type=str, default="data/sft_alpaca/sft_data.jsonl", help="Path to the SFT JSONL dataset.")
    parser.add_argument("--output_dir", type=str, default="./results_sft", help="Base directory to save training results and LoRA adapters.")
    parser.add_argument("--new_model_name", type=str, default="big5-llama-3-8b-sft-adapters", help="Subdirectory name for LoRA adapters within output_dir.")
    # Add other hyperparameter args if needed in the future
    args = parser.parse_args()

    # --- Device Setup & Precision ---
    mps_available = torch.backends.mps.is_available()
    use_bf16 = mps_available 
    
    if mps_available:
        print("MPS is available! Using MPS.")
        device = torch.device("mps")
        torch_dtype = torch.bfloat16 
    else:
        print("MPS not available. Using CUDA if available, else CPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 

    print(f"Using device: {device}, dtype: {torch_dtype}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer.pad_token to tokenizer.eos_token")
    
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    print(f"Set tokenizer.model_max_length to: {tokenizer.model_max_length}")

    # --- Load Base Model ---
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # --- Prepare model for LoRA training ---
    print("Preparing model for LoRA training...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Load and Prepare Dataset ---
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # --- Training Arguments ---
    print("Setting up training arguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adamw_torch", 
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not use_bf16 and not mps_available,  
        bf16=use_bf16,                           
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="tensorboard",
        logging_steps=10, 
        save_strategy="epoch", 
    )

    # --- Initialize SFTTrainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_alpaca_prompt, 
        args=training_arguments,
    )
    
    print(f"Starting fine-tuning with standard Hugging Face Trainer on device: {device}...")
    trainer.train()

    # Define the final path for saving adapters
    final_adapter_path = os.path.join(args.output_dir, args.new_model_name)
    os.makedirs(final_adapter_path, exist_ok=True) # Ensure directory exists

    print(f"Saving LoRA adapters to: {final_adapter_path}")
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    print("Fine-tuning complete!")
    print(f"LoRA adapters saved in {final_adapter_path}")

if __name__ == "__main__":
    main() 