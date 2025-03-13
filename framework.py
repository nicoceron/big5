import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json

class Framework:
    def __init__(self, base_model_name="meta-llama/Llama-3-8b", expert_models_path="./expert_models/"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        
        self.expert_models = {}
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            for level in ["high", "low"]:
                model_path = f"{expert_models_path}{trait}_{level}"
                try:
                    self.expert_models[f"{trait}_{level}"] = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
                except:
                    print(f"Expert model for {trait}_{level} not found. Will use base model with prompting instead.")
    
    def generate_response(self, context, trait, level, max_length=50):
        prompt = f"You are a helpful assistant with the following Big Five personality traits: {trait.capitalize()} - {level}\n\n{context}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if f"{trait}_{level}" in self.expert_models:
            outputs = self.expert_models[f"{trait}_{level}"].generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        else:
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split(context)[-1].strip()
        
        return response
    
    def _create_prompt(self, scenario, trait, level):
        person_y = scenario.get("PersonY", "Person Y")
        person_x = scenario.get("PersonX", "Person X")
        narrative = scenario.get("narrative", "")
        
        trait_descriptions = {
            "openness_high": "You are an open person with a vivid imagination...",
            "openness_low": "You prefer tradition and routine...",
            "conscientiousness_high": "You are organized, disciplined, and detail-oriented...",
            "conscientiousness_low": "You are spontaneous and flexible...",
            "extraversion_high": "You are outgoing, energetic, and enthusiastic...",
            "extraversion_low": "You are reserved and introspective...",
            "agreeableness_high": "You are compassionate, cooperative, and considerate of others...",
            "agreeableness_low": "You are direct and straightforward...",
            "neuroticism_high": "You experience emotions intensely and are sensitive to stress...",
            "neuroticism_low": "You are emotionally stable and resilient..."
        }
        
        prompt = f"Imagine you are {person_y}, your task is to act/speak as {person_y} would.\n"
        prompt += f"Here is the context of this interaction:\n```\n"
        prompt += f"Scenario: {narrative}\n"
        prompt += f"{person_y}'s big five personality description: The person has {level} {trait}.\n"
        prompt += f"{trait_descriptions[f'{trait}_{level}']}\n```\n.\n\n"
        prompt += f"Conversation starts:\n"
        prompt += f"Turn #0: {person_x} said: \"{scenario.get('input', '')}\"\n\n"
        prompt += f"You are at Turn #1.\n\n"
        prompt += f"Please generate your argument directly and concisely within 50 words:"
        
        return prompt
