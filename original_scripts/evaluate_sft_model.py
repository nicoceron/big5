import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import re
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Log file can be added here if needed
    ]
)
logger = logging.getLogger(__name__)

BFI_ITEMS = [
    # Item_ID, Text, Trait, Reverse_Scored (True/False)
    (1, "Is talkative", "E", False),
    (2, "Tends to find fault with others", "A", True),
    (3, "Does a thorough job", "C", False),
    (4, "Is depressed, blue", "N", False),
    (5, "Is original, comes up with new ideas", "O", False),
    (6, "Is reserved", "E", True),
    (7, "Is helpful and unselfish with others", "A", False),
    (8, "Can be somewhat careless", "C", True),
    (9, "Is relaxed, handles stress well", "N", True),
    (10, "Is curious about many different things", "O", False),
    (11, "Is full of energy", "E", False),
    (12, "Starts quarrels with others", "A", True),
    (13, "Is a reliable worker", "C", False),
    (14, "Can be tense", "N", False),
    (15, "Is ingenious, a deep thinker", "O", False),
    (16, "Generates a lot of enthusiasm", "E", False),
    (17, "Has a forgiving nature", "A", False),
    (18, "Tends to be disorganized", "C", True),
    (19, "Worries a lot", "N", False),
    (20, "Has an active imagination", "O", False),
    (21, "Tends to be quiet", "E", True),
    (22, "Is generally trusting", "A", False),
    (23, "Tends to be lazy", "C", True),
    (24, "Is emotionally stable, not easily upset", "N", True),
    (25, "Is inventive", "O", False),
    (26, "Has an assertive personality", "E", False),
    (27, "Can be cold and aloof", "A", True),
    (28, "Perseveres until the task is finished", "C", False),
    (29, "Can be moody", "N", False),
    (30, "Values artistic, aesthetic experiences", "O", False),
    (31, "Is sometimes shy, inhibited", "E", True),
    (32, "Is considerate, kind to almost everyone", "A", False),
    (33, "Does things efficiently", "C", False),
    (34, "Remains calm in tense situations", "N", True),
    (35, "Prefers work that is routine", "O", True),
    (36, "Is outgoing, sociable", "E", False),
    (37, "Is sometimes rude to others", "A", True),
    (38, "Makes plans and follows through on them", "C", False),
    (39, "Gets nervous easily", "N", False),
    (40, "Likes to reflect, play with ideas", "O", False),
    (41, "Has few artistic interests", "O", True),
    (42, "Likes to cooperate with others", "A", False),
    (43, "Is easily distracted", "C", True),
    (44, "Is sophisticated in art, music, or literature", "O", False),
]

BFI_SCORING_KEY = {
    "E": {"name": "Extraversion", "items": [1, 6, 11, 16, 21, 26, 31, 36]},
    "A": {"name": "Agreeableness", "items": [2, 7, 12, 17, 22, 27, 32, 37, 42]}, # Item 42 added as per phenx source
    "C": {"name": "Conscientiousness", "items": [3, 8, 13, 18, 23, 28, 33, 38, 43]},
    "N": {"name": "Neuroticism", "items": [4, 9, 14, 19, 24, 29, 34, 39]},
    "O": {"name": "Openness", "items": [5, 10, 15, 20, 25, 30, 35, 40, 41, 44]},
}
# Note: The phenx source for Agreeableness included 9 items (2R, 7, 12R, 17, 22, 27R, 32, 37R, 42), 
# while the Wisconsin source had 8 (omitting 42). I'm using the PhenX version with 9 items for A.

def load_model_and_tokenizer(base_model_name, peft_model_path=None, device="mps"):
    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16, 
        device_map=device, 
    )
    
    if peft_model_path:
        logger.info(f"Loading PEFT model from: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
        model = model.merge_and_unload() 
        logger.info(f"Loading tokenizer from PEFT path: {peft_model_path}") 
        tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    else:
        logger.info(f"Using base model directly. Loading tokenizer from base model path: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        logger.debug("Tokenizer pad_token is None, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_response(model, tokenizer, user_prompt, system_prompt=None, device="mps", max_new_tokens=10):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    inputs_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    
    logger.debug(f"Tokenized input shape: {inputs_tokenized.shape}")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs_tokenized,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    
    # Log raw output tokens
    generated_token_ids = outputs[0][inputs_tokenized.shape[1]:]
    logger.debug(f"Generated token IDs: {generated_token_ids}")

    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    logger.debug(f"Decoded response: '{response_text}'")
    return response_text.strip()

def parse_likert_response(text_response):
    # Try to find a single digit from 1 to 5, possibly surrounded by simple non-numeric characters
    # e.g., "is a 3", "response: 4.", "My answer is 2"
    match = re.search(r"[^0-9]*([1-5])[^0-9]*", text_response) # More lenient regex
    if match:
        return int(match.group(1))
    
    # Fallback: look for spelled out numbers (simple cases)
    text_response_lower = text_response.lower()
    if "one" in text_response_lower or "1" in text_response_lower: return 1
    if "two" in text_response_lower or "2" in text_response_lower: return 2
    if "three" in text_response_lower or "3" in text_response_lower: return 3
    if "four" in text_response_lower or "4" in text_response_lower: return 4
    if "five" in text_response_lower or "5" in text_response_lower: return 5
        
    logger.warning(f"Could not parse Likert score from response: '{text_response}'")
    return None

def evaluate_personality(model, tokenizer, personality_prompt=None, device="mps"):
    responses = {}
    logger.info("Starting BFI-44 evaluation...")
    if personality_prompt:
        logger.info(f"Using personality conditioning prompt: '{personality_prompt}'")

    for item_id, item_text, trait_abbr, is_reversed in BFI_ITEMS:
        bfi_question_prompt = (
            f"Please rate your agreement with the following statement on a scale of 1 (Disagree a lot / Strongly disagree) "
            f"to 5 (Agree a lot / Strongly agree):\\n\\n"
            f"Statement: I see myself as someone who... {item_text}\\n\\n"
            f"Your answer MUST be a single number from 1 to 5. Do not write any other words or sentences."
        )
        
        logger.info(f"Presenting item {item_id}: '{item_text}'")
        # Pass the overall personality_prompt as system_prompt, and bfi_question_prompt as user_prompt
        text_response = generate_response(model, tokenizer, user_prompt=bfi_question_prompt, system_prompt=personality_prompt, device=device)
        score = parse_likert_response(text_response)
        
        if score is not None:
            logger.info(f"Item {item_id} - Model response: '{text_response}', Parsed score: {score}")
            # Reverse score if needed
            actual_score = (6 - score) if is_reversed else score
            responses[item_id] = {"text": item_text, "raw_score": score, "scored_value": actual_score, "trait_abbr": trait_abbr, "is_reversed": is_reversed}
        else:
            logger.warning(f"Item {item_id} - Failed to parse score from: '{text_response}'")
            responses[item_id] = {"text": item_text, "raw_score": None, "scored_value": None, "trait_abbr": trait_abbr, "is_reversed": is_reversed}

    # Calculate trait scores
    trait_scores = {}
    for trait_abbr, data in BFI_SCORING_KEY.items():
        trait_name = data["name"]
        item_ids_for_trait = data["items"]
        
        scores_for_this_trait = []
        for item_id in item_ids_for_trait:
            if item_id in responses and responses[item_id]["scored_value"] is not None:
                scores_for_this_trait.append(responses[item_id]["scored_value"])
        
        if scores_for_this_trait:
            average_score = sum(scores_for_this_trait) / len(scores_for_this_trait)
            trait_scores[trait_name] = average_score
            logger.info(f"Trait: {trait_name}, Average Score: {average_score:.2f} (from {len(scores_for_this_trait)} items)")
        else:
            trait_scores[trait_name] = None
            logger.warning(f"Trait: {trait_name}, Could not calculate score (no valid item responses)")
            
    return {"bfi_responses": responses, "trait_scores": trait_scores}

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the BFI-44 personality test.")
    parser.add_argument("--base_model_name", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct", help="Base model name from Hugging Face.")
    parser.add_argument("--peft_model_path", type=str, default=None, help="Path to the PEFT LoRA adapter model directory. If None, uses base model only.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the evaluation results (JSON).")
    parser.add_argument("--device", type=str, default="mps", help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps').")
    parser.add_argument("--personality_prompt", type=str, default=None, help="System prompt to condition the base model's personality. Only used if peft_model_path is None.")

    args = parser.parse_args()

    logger.info(f"Using device: {args.device}")
    
    model, tokenizer = load_model_and_tokenizer(args.base_model_name, args.peft_model_path, device=args.device)
    
    # Determine if we are using a PEFT model or a base model with prompting
    eval_personality_prompt = None
    if args.peft_model_path:
        logger.info("Evaluating fine-tuned PEFT model.")
        # Add a default system prompt for PEFT models to guide BFI responses
        eval_personality_prompt = (
            "You are a language model. Please rate your agreement with the following statements "
            "based on your typical ways of thinking and behaving, using a scale of 1 (Disagree a lot) "
            "to 5 (Agree a lot). Your answer MUST be a single number from 1 to 5."
        )
        # If a specific personality_prompt is also provided for a PEFT model (e.g. for testing), let it override the default.
        if args.personality_prompt:
            logger.info(f"Overriding default PEFT BFI prompt with user-provided prompt: {args.personality_prompt}")
            eval_personality_prompt = args.personality_prompt
            
    elif not args.peft_model_path and args.personality_prompt:
        eval_personality_prompt = args.personality_prompt
        logger.info("Evaluating base model with personality prompt.")
    else:
        logger.info("Evaluating base model without specific personality prompt (will use its default behavior).")

    evaluation_results = evaluate_personality(model, tokenizer, personality_prompt=eval_personality_prompt, device=args.device)

    logger.info("\\n--- Final Trait Scores ---")
    for trait, score in evaluation_results["trait_scores"].items():
        if score is not None:
            logger.info(f"{trait}: {score:.2f}")
        else:
            logger.info(f"{trait}: Not calculable")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        # Add script arguments to the output for traceability
        evaluation_results["script_args"] = vars(args)
        evaluation_results["timestamp"] = datetime.now().isoformat()
        with open(args.output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Evaluation results saved to {args.output_file}")

if __name__ == "__main__":
    main() 