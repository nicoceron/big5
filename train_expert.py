from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments
import json

def train_expert_model(trait, level, base_model_name, dataset_path, output_dir):
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    train_data = []
    for entry in dataset[trait][level]:
        train_data.append({
            "instruction": entry["train_data"]["instruction"],
            "input": entry["train_data"]["input"],
            "output": entry["train_data"]["output"]
        })
    
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/tmp",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=1,
        logging_steps=100,
        save_strategy="epoch",
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=None,
    )
    
    trainer.train()
    
    model.save_pretrained(f"{output_dir}/{trait}_{level}")
    tokenizer.save_pretrained(f"{output_dir}/{trait}_{level}")
    
    return f"{output_dir}/{trait}_{level}"
