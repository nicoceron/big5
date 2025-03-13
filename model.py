import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import random
import pandas as pd
import json

class PSYCHSTEERFramework:
    def __init__(self, base_model_name="meta-llama/Llama-3-8b", expert_models_path="./expert_models/"):
        """
        Initialize the PSYCHSTEER framework.
        
        Args:
            base_model_name: The base model to use for generation
            expert_models_path: Path to expert models fine-tuned for personality traits
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        
        # Load expert models for each trait (high and low)
        self.expert_models = {}
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            for level in ["high", "low"]:
                model_path = f"{expert_models_path}{trait}_{level}"
                try:
                    self.expert_models[f"{trait}_{level}"] = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
                except:
                    print(f"Expert model for {trait}_{level} not found. Will use base model with prompting instead.")
    
    def generate_response(self, context, trait, level, max_length=50):
        """
        Generate a response based on personality trait.
        
        Args:
            context: The conversation context (Speaker X's utterance)
            trait: One of the Big Five traits
            level: "high" or "low"
            max_length: Maximum length of generated response
            
        Returns:
            Generated response with specified personality trait
        """
        # Create prompt with personality instruction
        prompt = f"You are a helpful assistant with the following Big Five personality traits: {trait.capitalize()} - {level}\n\n{context}"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Check if we have an expert model for this trait
        if f"{trait}_{level}" in self.expert_models:
            # Use the expert model (DExperts approach)
            outputs = self.expert_models[f"{trait}_{level}"].generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        else:
            # Fallback to base model with instruction prompting
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part, not the instruction
        response = response.split(context)[-1].strip()
        
        return response
    
    
    def _create_prompt(self, scenario, trait, level):
        """Create a detailed prompt for generation"""
        person_y = scenario.get("PersonY", "Person Y")
        person_x = scenario.get("PersonX", "Person X")
        narrative = scenario.get("narrative", "")
        
        trait_descriptions = {
            "openness_high": "You are an open person with a vivid imagination and a passion for the arts. You are emotionally expressive and have a strong sense of adventure. Your intellect is sharp and your views are liberal. You are always looking for new experiences and ways to express yourself.",
            "openness_low": "You prefer tradition and routine over novelty. You are practical and conventional in your thinking. You prefer facts over abstract ideas and tend to be cautious about new experiences. You generally stick with what you know.",
            "conscientiousness_high": "You are organized, disciplined, and detail-oriented. You strive for achievement and are dependable. You think before acting and prefer planned activities. You have high standards and work diligently to meet your goals.",
            "conscientiousness_low": "You are spontaneous and flexible, preferring not to plan things in advance. You can be disorganized and may procrastinate. You are relaxed about deadlines and rules, and tend to act on impulse rather than deliberation.",
            "extraversion_high": "You are outgoing, energetic, and enthusiastic. You enjoy social interactions and being the center of attention. You are talkative, assertive, and draw energy from being around people. You seek excitement and stimulation.",
            "extraversion_low": "You are reserved and introspective, preferring quiet environments. You value solitude and tend to think before speaking. You find social interactions draining and need time alone to recharge. You are more thoughtful than talkative.",
            "agreeableness_high": "You are compassionate, cooperative, and considerate of others' feelings. You give people the benefit of the doubt and avoid conflict. You're helpful, forgiving, and generally trusting of others. You prioritize harmony in relationships.",
            "agreeableness_low": "You are direct and straightforward in communication. You can be skeptical of others' motives and don't hesitate to challenge opinions. You prioritize logic over feelings and may come across as tough-minded. You're competitive rather than cooperative.",
            "neuroticism_high": "You experience emotions intensely and are sensitive to stress. You worry about things going wrong and can be self-critical. You notice threats easily and experience anxiety, anger, or sadness more readily than others. You're emotionally reactive.",
            "neuroticism_low": "You are emotionally stable and resilient. You remain calm under pressure and rarely feel sad or depressed. You're not easily bothered by stressful situations and recover quickly from setbacks. You have a general sense of well-being."
        }
        
        prompt = f"Imagine you are {person_y}, your task is to act/speak as {person_y} would.\n"
        prompt += f"You should try your best to infer and achieve {person_y}'s goal in a single turn that align with their character traits.\n"
        prompt += f"Additionally, maintaining the conversation's naturalness and realism is essential.\n"
        prompt += f"Here is the context of this interaction:\n```\n"
        prompt += f"Scenario: {narrative}\n"
        prompt += f"Participants: {person_x} and {person_y}\n"
        prompt += f"{person_y}'s big five personality description: The person has {level} {trait}.\n"
        prompt += f"{trait_descriptions[f'{trait}_{level}']}\n```\n.\n\n"
        prompt += f"Conversation starts:\n"
        prompt += f"Turn #0: {person_x} said: \"{scenario.get('input', '')}\"\n\n"
        prompt += f"You are at Turn #1.\n\n"
        prompt += f"Please generate your argument directly and concisely within 50 words:"
        
        return prompt
    

class PersonalityEvaluator:
    """Evaluate the quality of generated personality-steered responses"""
    
    def __init__(self, classifier_path):
        """
        Initialize the personality evaluator with a trained classifier
        
        Args:
            classifier_path: Path to the trained RoBERTa classifier
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the trained classifier
        self.model = torch.load(classifier_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer for RoBERTa
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        
    def evaluate_dataset(self, dataset_path):
        """
        Evaluate a dataset of generated responses
        
        Args:
            dataset_path: Path to the BIG5-CHAT dataset
            
        Returns:
            Evaluation metrics for each trait
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        results = {}
        
        for trait in dataset.keys():
            trait_accuracy = {"high": 0, "low": 0}
            
            for level in ["high", "low"]:
                correct = 0
                total = len(dataset[trait][level])
                
                for entry in dataset[trait][level]:
                    response = entry["response"]
                    
                    # Predict trait scores for the response
                    inputs = self.tokenizer(response, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Get predicted trait scores (5 dimensions: O, C, E, A, N)
                    scores = outputs.squeeze().cpu().numpy()
                    
                    # Map trait to index (0=O, 1=C, 2=E, 3=A, 4=N)
                    trait_indices = {
                        "openness": 0,
                        "conscientiousness": 1,
                        "extraversion": 2,
                        "agreeableness": 3,
                        "neuroticism": 4
                    }
                    
                    # Check if prediction matches desired level
                    trait_score = scores[trait_indices[trait]]
                    predicted_level = "high" if trait_score > 0.5 else "low"
                    
                    if predicted_level == level:
                        correct += 1
                
                trait_accuracy[level] = (correct / total) * 100
            
            results[trait] = trait_accuracy
            
        # Calculate average accuracy across all traits
        avg_accuracy = 0
        count = 0
        
        for trait, accuracies in results.items():
            for level, accuracy in accuracies.items():
                avg_accuracy += accuracy
                count += 1
                
        results["average"] = avg_accuracy / count
        
        return results
    
def train_expert_model(trait, level, base_model_name, dataset_path, output_dir):
    """
    Train an expert model for a specific personality trait
    
    Args:
        trait: One of the Big Five traits
        level: "high" or "low"
        base_model_name: Base model to fine-tune
        dataset_path: Path to training data
        output_dir: Where to save the trained model
    """
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    train_data = []
    for entry in dataset[trait][level]:
        train_data.append({
            "instruction": entry["train_data"]["instruction"],
            "input": entry["train_data"]["input"],
            "output": entry["train_data"]["output"]
        })
    
    # Create training arguments
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=None,  # Will be handled by the trainer
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    model.save_pretrained(f"{output_dir}/{trait}_{level}")
    tokenizer.save_pretrained(f"{output_dir}/{trait}_{level}")
    
    return f"{output_dir}/{trait}_{level}"



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BIG5-CHAT dataset and train expert models")
    parser.add_argument("--expert_models_path", type=str, default="./expert_models/", help="Path to expert models")
    parser.add_argument("--train_experts", action="store_true", help="Whether to train expert models")
    parser.add_argument("--evaluate", action="store_true", help="Whether to evaluate generated dataset")
    parser.add_argument("--classifier_path", type=str, default="./classifier.pt", help="Path to personality classifier")
    parser.add_argument("--output_path", type=str, default="../big5_chat_dataset.json", help="Output path for BIG5-CHAT dataset")
    
    args = parser.parse_args()
    
    # Train expert models if requested
    if args.train_experts:
        print("Training expert models...")
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            for level in ["high", "low"]:
                print(f"Training {trait}_{level} expert model...")
                train_expert_model(trait, level, base_model_name="meta-llama/Llama-3-8b", dataset_path=args.output_path, output_dir=args.expert_models_path)
    
    # Evaluate dataset if requested
    if args.evaluate:
        print("Evaluating generated dataset...")
        evaluator = PersonalityEvaluator(args.classifier_path)
        results = evaluator.evaluate_dataset(args.output_path)
        
        print("\nEvaluation Results:")
        for trait, accuracies in results.items():
            if trait == "average":
                print(f"Average accuracy: {accuracies:.2f}%")
            else:
                print(f"{trait.capitalize()}: High={accuracies['high']:.2f}%, Low={accuracies['low']:.2f}%")

if __name__ == "__main__":
    main()
