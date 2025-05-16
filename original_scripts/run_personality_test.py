#!/usr/bin/env python3
import argparse
import os
import logging
import sys
import ollama
import pandas as pd
import json
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/personality_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def get_model_name(personality_trait, level):
    """Get the appropriate model name for a personality trait and level."""
    if personality_trait == 'o':
        trait_name = "openness"
    elif personality_trait == 'c':
        trait_name = "conscientiousness"
    elif personality_trait == 'e':
        trait_name = "extraversion"
    elif personality_trait == 'a':
        trait_name = "agreeableness"
    elif personality_trait == 'n':
        trait_name = "neuroticism"
    else:
        raise ValueError(f"Unknown personality trait: {personality_trait}")
    
    full_model_name = f"gemma3-{trait_name}:latest"
    return full_model_name

def generate_response(model_name, prompt):
    """Generate a response using the specified model."""
    try:
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error generating response with model {model_name}: {e}")
        return f"Error: {str(e)}"

def load_data(input_file):
    """Load data from a CSV or JSONL file."""
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
            data = df.to_dict(orient='records')
        elif input_file.endswith('.jsonl'):
            data = []
            with open(input_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            logger.error(f"Unsupported file format: {input_file}")
            return None
        
        logger.info(f"Loaded {len(data)} examples from {input_file}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {input_file}: {e}")
        return None

def run_personality_test(args):
    """Run the personality test with the specified trait and prompt."""
    logger.info(f"Running personality test for trait: {args.trait}")
    
    # Get the model name for the specified trait
    model_name = get_model_name(args.trait, "high")
    
    # Determine the prompt to use
    if args.input_file:
        # Load data from the input file
        data = load_data(args.input_file)
        if not data:
            return False
        
        # Use either the specified example index or select a random example
        if args.example_index is not None and args.example_index < len(data):
            example = data[args.example_index]
        else:
            example = random.choice(data)
        
        # Use the provided prompt field if available, otherwise construct from topic
        if args.prompt:
            prompt = args.prompt
        elif "prompt" in example:
            prompt = example["prompt"]
        else:
            topic = example.get("topic", "")
            persona = example.get("persona", "")
            if persona:
                prompt = f"You are {persona}. What are your thoughts on {topic}?"
            else:
                prompt = f"What are your thoughts on {topic}?"
        
        # Print information about the selected example
        logger.info(f"Using example with topic: {example.get('topic', 'None')}")
        if 'persona' in example:
            logger.info(f"Persona: {example.get('persona', 'None')}")
        if 'dialogue' in example and example['dialogue']:
            logger.info(f"Contains dialogue with {len(example['dialogue'])} turns")
    else:
        # Use the provided prompt or a default one
        if args.prompt:
            prompt = args.prompt
        else:
            # Default prompts for each trait
            if args.trait == 'o':
                prompt = "Describe how you would approach learning a completely new skill that is outside your comfort zone."
            elif args.trait == 'c':
                prompt = "Describe how you would organize a complex project with multiple deadlines."
            elif args.trait == 'e':
                prompt = "Describe how you would behave at a social gathering where you don't know many people."
            elif args.trait == 'a':
                prompt = "Describe how you would handle a situation where someone disagrees strongly with your opinion."
            elif args.trait == 'n':
                prompt = "Describe how you would respond to receiving unexpected criticism about your work."
    
    logger.info(f"Prompt: {prompt}")
    
    # Generate responses
    logger.info(f"Generating response with {model_name}")
    response = generate_response(model_name, prompt)
    
    print(f"\n=== Response using {model_name} ===")
    print(response)
    print("===================================\n")

    # Save the response to file if output path is provided
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        result = {
            "trait": args.trait,
            "model": model_name,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add example details if using input file
        if args.input_file and 'example' in locals():
            result["example"] = example
            
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved response to {args.output_file}")

    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test personality-based text generation")
    parser.add_argument("--trait", "-t", required=True, choices=['o', 'c', 'e', 'a', 'n'],
                       help="Personality trait to test (o=Openness, c=Conscientiousness, e=Extraversion, a=Agreeableness, n=Neuroticism)")
    parser.add_argument("--prompt", "-p", type=str,
                       help="Custom prompt to use (if not specified, a default prompt will be used)")
    parser.add_argument("--input-file", "-i", type=str,
                       help="Input file (CSV or JSONL) containing topics and prompts")
    parser.add_argument("--example-index", type=int,
                       help="Index of the example to use from the input file (if not specified, a random example will be chosen)")
    parser.add_argument("--output-file", "-o", type=str,
                       help="Output file to save the response as JSON")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    
    # Run the personality test
    success = run_personality_test(args)
    
    if success:
        logger.info("Personality test completed successfully")
        return 0
    else:
        logger.error("Personality test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 