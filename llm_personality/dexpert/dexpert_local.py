import torch
from ollama_adapter import OllamaTokenizer, OllamaForCausalLM
import ollama
import logging
import numpy as np

# Initialize the logger
logger = logging.getLogger(__name__)

class DExpertGenerator():
    def __init__(self, args, args_expert=None, args_antiexpert=None):
        """
        Initialize the DExpert generator with Ollama models.
        
        Args:
            args: Arguments for the base model
            args_expert: Arguments for the expert model
            args_antiexpert: Arguments for the anti-expert model
        """
        self.args = args
        self.args_expert = args_expert
        self.args_antiexpert = args_antiexpert
        
        # Initialize tokenizer (we'll use the same tokenizer for all models)
        self.tokenizer = OllamaTokenizer()
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.pad_token_id = torch.tensor(self.tokenizer.eos_token_id)
        
        # Check available models
        try:
            models_info = ollama.list()
            available_models = []
            
            if 'models' in models_info:
                # New format
                for model in models_info['models']:
                    if 'name' in model:
                        available_models.append(model['name'])
            else:
                # Alternative format
                import subprocess
                result = subprocess.run(['ollama', 'list'], 
                                        capture_output=True, 
                                        text=True, 
                                        check=True)
                
                # Parse output to get model names
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header line
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            available_models.append(parts[0])
            
            logger.info(f"Available Ollama models: {available_models}")
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            available_models = []
        
        # Use gemma3:12b as the base model if available, otherwise use the first available model
        base_model = args.model_id if hasattr(args, 'model_id') and args.model_id else "gemma3:12b"
        if base_model not in available_models:
            base_model = available_models[0] if available_models else "gemma3"
            
        logger.info(f"Using {base_model} as base model")
        
        # Initialize the base model
        self.model = OllamaForCausalLM(base_model)
        
        # Initialize expert model if specified
        if args_expert is not None and args_expert.model_id is not None:
            expert_model = args_expert.model_id
            if expert_model in available_models:
                logger.info(f"Using {expert_model} as expert model")
                self.model.expert = OllamaForCausalLM(expert_model)
            else:
                logger.warning(f"Expert model {expert_model} not found, using base model")
                self.model.expert = None
        else:
            self.model.expert = None
            
        # Initialize anti-expert model if specified
        if args_antiexpert is not None and args_antiexpert.model_id is not None:
            antiexpert_model = args_antiexpert.model_id
            if antiexpert_model in available_models:
                logger.info(f"Using {antiexpert_model} as anti-expert model")
                self.model.antiexpert = OllamaForCausalLM(antiexpert_model)
            else:
                logger.warning(f"Anti-expert model {antiexpert_model} not found, using base model")
                self.model.antiexpert = None
        else:
            self.model.antiexpert = None
    
    def generate(self, messages, messages_expert=None, messages_antiexpert=None, alpha=None):
        """
        Generate text using the DExpert system by combining outputs from base, expert, and anti-expert models.
        
        Args:
            messages: List of message dictionaries for the base model
            messages_expert: List of message dictionaries for the expert model
            messages_antiexpert: List of message dictionaries for the anti-expert model
            alpha: Control parameter for expert influence (0-1)
            
        Returns:
            Generated text
        """
        # Format messages for Ollama
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Generate response using the base model
        base_response = ollama.chat(
            model=self.model.model_name,
            messages=formatted_messages
        )
        
        # If we don't have expert/anti-expert models or alpha is not set, return base response
        if (not self.model.expert and not self.model.antiexpert) or alpha is None:
            return base_response["message"]["content"]
        
        # Generate expert response if available
        expert_response = None
        if self.model.expert and messages_expert:
            expert_messages = []
            for msg in messages_expert:
                expert_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            try:
                expert_response = ollama.chat(
                    model=self.model.expert.model_name,
                    messages=expert_messages
                )
            except Exception as e:
                logger.error(f"Error generating expert response: {e}")
        
        # Generate anti-expert response if available
        antiexpert_response = None
        if self.model.antiexpert and messages_antiexpert:
            antiexpert_messages = []
            for msg in messages_antiexpert:
                antiexpert_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
            try:
                antiexpert_response = ollama.chat(
                    model=self.model.antiexpert.model_name,
                    messages=antiexpert_messages
                )
            except Exception as e:
                logger.error(f"Error generating anti-expert response: {e}")
        
        # Combine the responses
        if expert_response and not antiexpert_response:
            # Expert only mode: Interpolate between base and expert
            logger.info(f"Using expert model with alpha={alpha}")
            combined_response = self._combine_responses(
                base_response["message"]["content"],
                expert_response["message"]["content"],
                alpha
            )
            return combined_response
        
        elif antiexpert_response and not expert_response:
            # Anti-expert only mode: Move away from anti-expert
            logger.info(f"Using anti-expert model with alpha={alpha}")
            # Invert alpha since we want to move away from anti-expert
            combined_response = self._combine_responses(
                base_response["message"]["content"],
                antiexpert_response["message"]["content"],
                1.0 - alpha
            )
            return combined_response
            
        elif expert_response and antiexpert_response:
            # Full DExpert mode: Attract to expert, repel from anti-expert
            logger.info(f"Using both expert and anti-expert models with alpha={alpha}")
            # Simple implementation: 
            # 1. Move towards expert with weight alpha/2
            # 2. Move away from anti-expert with weight alpha/2
            expert_weight = alpha / 2.0
            antiexpert_weight = 1.0 - (alpha / 2.0)
            
            # First combine base with expert
            partial_response = self._combine_responses(
                base_response["message"]["content"],
                expert_response["message"]["content"],
                expert_weight
            )
            
            # Then adjust away from anti-expert
            final_response = self._combine_responses(
                partial_response,
                antiexpert_response["message"]["content"],
                antiexpert_weight
            )
            
            return final_response
        
        # Fallback to base response
        return base_response["message"]["content"]
    
    def _combine_responses(self, response1, response2, weight):
        """
        Simple text combination method - for demonstration.
        In a real implementation, you would combine the model logits.
        
        This function just picks more text from one response based on weight.
        
        Args:
            response1: First response text
            response2: Second response text
            weight: Weight for response1 (0-1)
            
        Returns:
            Combined response text
        """
        # Ensure responses are not too long
        max_length = 100  # tokens/words approximately
        r1_words = response1.split()[:max_length]
        r2_words = response2.split()[:max_length]
        
        # Determine how many words to take from each response
        total_words = min(len(r1_words) + len(r2_words), max_length)
        r1_count = int(total_words * weight)
        r2_count = total_words - r1_count
        
        # Create combined response
        combined_words = r1_words[:r1_count] + r2_words[:r2_count]
        np.random.shuffle(combined_words)  # Shuffle to simulate mixing
        
        logger.warning("This is a simplistic response combination method for demonstration only.")
        logger.warning("A real DExpert implementation would combine model logits, not final text.")
        
        # Instead of this naive implementation, let's just return response1 with weight-based probability
        if np.random.random() < weight:
            return response2
        else:
            return response1 