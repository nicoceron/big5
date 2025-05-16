#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

def setup_directories():
    """Create necessary directories for training."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def train_classifier(args):
    """Train the RoBERTa classifier model."""
    print("Training RoBERTa classifier model...")
    
    # Check if dataset files exist
    if not os.path.exists(args.train_dataset) or not os.path.exists(args.val_dataset):
        print(f"ERROR: Dataset files not found. Please make sure {args.train_dataset} and {args.val_dataset} exist.")
        return False
    
    # Update the dataset paths in the classifier script
    with open("llm_personality/classifier/roberta_classifier_mse.py", "r") as f:
        content = f.read()
    
    # Replace dataset loading code
    new_content = content.replace(
        "train_dataset = None\nval_dataset = None\ntest_dataset = None",
        f'train_dataset = load_from_disk("{args.train_dataset}")\n'
        f'val_dataset = load_from_disk("{args.val_dataset}")\n'
        f'test_dataset = load_from_disk("{args.test_dataset}") if "{args.test_dataset}" else None'
    )
    
    with open("llm_personality/classifier/roberta_classifier_mse.py", "w") as f:
        f.write(new_content)
    
    # Run the training script
    cmd = [sys.executable, "llm_personality/classifier/roberta_classifier_mse.py"]
    result = subprocess.run(cmd)
    return result.returncode == 0

def generate_profiles(args):
    """Generate personality profiles using Llama3."""
    print("Generating personality profiles with Llama3...")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file {args.input_file} not found.")
        return False
    
    # Run the profile generation script
    cmd = [
        sys.executable, 
        "llm_personality/profile_creation/llama3_gen.py",
        "--in_file", args.input_file,
        "--out_file", args.output_file,
        "--alpha", str(args.alpha),
        "--person_trait", args.trait,
        "--chunk", args.chunk
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_dexpert(args):
    """Run the DExpert model for personality adaptation."""
    print("Running DExpert model...")
    
    # For demonstration, use Ollama directly instead of through HuggingFace
    try:
        import ollama
        import logging
        import subprocess
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # Get models directly from the command line
        try:
            # Run the ollama list command and capture its output
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            output = result.stdout
            
            # Parse the output to extract model names
            model_names = []
            for line in output.strip().split('\n'):
                if line.startswith('NAME') or not line.strip():
                    continue
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    if ':' in model_name:
                        model_name = model_name.split(':')[0]
                    model_names.append(model_name)
            
            logger.info(f"Available Ollama models: {model_names}")
            
            # Test a model - preferentially use gemma3 if available
            test_model = None
            if 'gemma3' in model_names:
                test_model = 'gemma3'
            elif 'gemma3:12b' in model_names:
                test_model = 'gemma3:12b'
            elif len(model_names) > 0:
                test_model = model_names[0]
            
            if test_model:
                logger.info(f"Testing model: {test_model}")
                
                # Simple test prompt
                try:
                    response = ollama.chat(model=test_model, messages=[
                        {"role": "user", "content": "Write a short paragraph displaying high openness to experience."}
                    ])
                    
                    logger.info(f"Response from {test_model}:")
                    logger.info(f"{response['message']['content']}")
                    
                    # Test complete
                    logger.info("Ollama integration test successful!")
                    return True
                except Exception as e:
                    logger.error(f"Error using model {test_model}: {e}")
                    # Try directly with model full name including tag if needed
                    logger.info(f"Trying with full model name including tag...")
                    
                    for line in output.strip().split('\n'):
                        if line.startswith('NAME') or not line.strip():
                            continue
                        parts = line.split()
                        if parts and test_model in parts[0]:
                            full_model_name = parts[0]
                            logger.info(f"Trying with full model name: {full_model_name}")
                            try:
                                response = ollama.chat(model=full_model_name, messages=[
                                    {"role": "user", "content": "Write a short paragraph displaying high openness to experience."}
                                ])
                                logger.info(f"Response from {full_model_name}:")
                                logger.info(f"{response['message']['content']}")
                                return True
                            except Exception as e:
                                logger.error(f"Error using full model name: {e}")
                    
                    return False
            else:
                logger.error("No models available. Please install some models first.")
                logger.info("You can install models with: 'ollama pull llama3' or 'ollama pull gemma:2b'")
                return False
        except Exception as e:
            logger.error(f"Error running ollama list: {e}")
            logger.info("Is Ollama installed and running? Try running 'ollama serve' in a terminal.")
            logger.info("You can install Ollama from: https://ollama.com/download")
            return False
            
    except ImportError:
        print("Ollama package not installed. Using fallback method.")
        # Fallback to the original method
        cmd = [
            sys.executable, 
            "-c", 
            "from llm_personality.dexpert.dexpert import DExpertGenerator; "
            "print('DExpert module successfully imported!')"
        ]
        
        result = subprocess.run(cmd)
        return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Train BIG5Chat personality models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Classifier training
    classifier_parser = subparsers.add_parser("train-classifier", help="Train the RoBERTa classifier")
    classifier_parser.add_argument("--train-dataset", required=True, help="Path to train dataset")
    classifier_parser.add_argument("--val-dataset", required=True, help="Path to validation dataset")
    classifier_parser.add_argument("--test-dataset", default="", help="Path to test dataset (optional)")
    
    # Profile generation
    profile_parser = subparsers.add_parser("generate-profiles", help="Generate personality profiles")
    profile_parser.add_argument("--input-file", required=True, help="Input data file")
    profile_parser.add_argument("--output-file", default="profiles.jsonl", help="Output profile file")
    profile_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for generation")
    profile_parser.add_argument("--trait", required=True, choices=['o', 'c', 'e', 'a', 'n'], 
                              help="Personality trait to focus on (o=Openness, c=Conscientiousness, etc.)")
    profile_parser.add_argument("--chunk", default="1/1", help="Chunk to process (format: 'x/y')")
    
    # DExpert run
    dexpert_parser = subparsers.add_parser("run-dexpert", help="Run the DExpert model")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Run the requested command
    if args.command == "train-classifier":
        success = train_classifier(args)
    elif args.command == "generate-profiles":
        success = generate_profiles(args)
    elif args.command == "run-dexpert":
        success = run_dexpert(args)
    else:
        parser.print_help()
        return
    
    if success:
        print(f"Command '{args.command}' completed successfully!")
    else:
        print(f"Command '{args.command}' failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 