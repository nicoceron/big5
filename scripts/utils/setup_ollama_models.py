#!/usr/bin/env python3
import argparse
import subprocess
import logging
import sys
import json
import time
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model files for each personality trait
MODEL_FILES = {
    'o': {
        'high': 'modelfiles/openness-high.modelfile',
        'low': 'modelfiles/openness-low.modelfile',
    },
    'c': {
        'high': 'modelfiles/conscientiousness-high.modelfile',
        'low': 'modelfiles/conscientiousness-low.modelfile',
    },
    'e': {
        'high': 'modelfiles/extraversion-high.modelfile',
        'low': 'modelfiles/extraversion-low.modelfile',
    },
    'a': {
        'high': 'modelfiles/agreeableness-high.modelfile',
        'low': 'modelfiles/agreeableness-low.modelfile',
    },
    'n': {
        'high': 'modelfiles/neuroticism-high.modelfile',
        'low': 'modelfiles/neuroticism-low.modelfile',
    }
}

# Define personality trait descriptions for each modelfile
PERSONALITY_DESCRIPTIONS = {
    'o': {
        'high': "You have high openness to experience. You are creative, imaginative, curious, and open to new ideas and experiences. You enjoy art, literature, and philosophical discussions. You tend to think in abstract and complex ways.",
        'low': "You have low openness to experience. You are practical, conventional, and prefer routine and familiar experiences. You focus on concrete facts rather than theoretical possibilities. You prefer straightforward and direct communication."
    },
    'c': {
        'high': "You have high conscientiousness. You are organized, methodical, and diligent. You plan ahead, follow rules, and maintain high standards for yourself. You are reliable, disciplined, and careful in your work.",
        'low': "You have low conscientiousness. You tend to be spontaneous, flexible, and less concerned with schedules and organization. You prefer to go with the flow rather than making detailed plans. You may occasionally miss deadlines or lose track of details."
    },
    'e': {
        'high': "You have high extraversion. You are outgoing, energetic, and enjoy social interactions. You are talkative, assertive, and gain energy from being around others. You prefer group activities and tend to think out loud.",
        'low': "You have low extraversion (introversion). You are reserved, quiet, and prefer one-on-one interactions or solitude. You think before you speak and need time alone to recharge. You may find large social gatherings draining."
    },
    'a': {
        'high': "You have high agreeableness. You are kind, cooperative, and empathetic. You value harmony and avoid conflicts. You give people the benefit of the doubt and are willing to compromise for the sake of the group.",
        'low': "You have low agreeableness. You tend to be direct, straightforward, and skeptical. You prioritize truth over tact and are willing to challenge others. You stand firm in your positions and don't hesitate to engage in healthy debate."
    },
    'n': {
        'high': "You have high neuroticism. You experience emotions intensely and may be prone to stress, anxiety, or mood swings. You are sensitive to negative events and may worry about potential problems. You are in touch with your emotions and notice subtle environmental changes.",
        'low': "You have low neuroticism (emotional stability). You are calm, even-tempered, and resilient under stress. You recover quickly from setbacks and rarely dwell on negative emotions. You maintain a positive outlook and can stay composed in difficult situations."
    }
}

# Define a base Modelfile template
MODELFILE_TEMPLATE = """
FROM {base_model}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50

SYSTEM """

def create_modelfile_directory():
    """Create directory for modelfiles if it doesn't exist."""
    os.makedirs('modelfiles', exist_ok=True)

def create_modelfiles(base_model="gemma3"):
    """Create modelfiles for each personality trait level."""
    create_modelfile_directory()
    
    for trait, levels in MODEL_FILES.items():
        trait_name = get_trait_name(trait)
        
        for level, modelfile_path in levels.items():
            # Create the system prompt for this personality trait
            system_prompt = PERSONALITY_DESCRIPTIONS[trait][level]
            
            # Create the modelfile content
            modelfile_content = MODELFILE_TEMPLATE.format(base_model=base_model)
            modelfile_content += f"{system_prompt}\n"
            
            # Write the modelfile
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            logger.info(f"Created modelfile for {trait_name} ({level}) at {modelfile_path}")

def get_available_ollama_models():
    """Get the list of available Ollama models."""
    try:
        # First try the Ollama Python API
        try:
            import ollama
            models_info = ollama.list()
            
            # Check if models are in the expected format
            if 'models' in models_info:
                # Extract model names from the JSON response
                model_names = []
                for model in models_info['models']:
                    if 'name' in model:
                        model_names.append(model['name'])
                return model_names
        except (ImportError, KeyError, Exception) as e:
            logger.warning(f"Error using Ollama Python API: {e}")
        
        # Fallback to CLI command
        result = subprocess.run(['ollama', 'list'], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        
        # Parse output to get model names
        lines = result.stdout.strip().split('\n')
        models = []
        
        for line in lines[1:]:  # Skip header line
            if line.strip():
                parts = line.strip().split()
                if parts:
                    models.append(parts[0])
        
        return models
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'ollama list': {e}")
        logger.error(f"Output: {e.stdout}")
        return []
    except Exception as e:
        logger.error(f"Error getting available Ollama models: {e}")
        return []

def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        subprocess.run(['ollama', 'list'], 
                      capture_output=True, 
                      text=True, 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        logger.error("Ollama command not found. Please install Ollama from https://ollama.com")
        return False

def create_ollama_model(model_name, modelfile_path):
    """Create an Ollama model from a modelfile."""
    try:
        logger.info(f"Creating model {model_name} from {modelfile_path}...")
        
        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', modelfile_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Successfully created model {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating model {model_name}: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

def get_trait_name(trait_code):
    """Convert trait code to full name."""
    names = {
        'o': 'Openness',
        'c': 'Conscientiousness',
        'e': 'Extraversion',
        'a': 'Agreeableness',
        'n': 'Neuroticism'
    }
    return names.get(trait_code, trait_code)

def setup_personality_models(base_model="gemma3", force_recreate=False):
    """Set up all the personality models needed for the Big Five traits."""
    # Check if Ollama is running
    if not check_ollama_running():
        logger.error("Ollama is not running. Please start Ollama and try again.")
        return False
    
    # Get available models
    available_models = get_available_ollama_models()
    logger.info(f"Available Ollama models: {available_models}")
    
    # Check if base model is available
    if base_model not in available_models:
        logger.warning(f"Base model '{base_model}' not found. Attempting to pull it...")
        try:
            subprocess.run(['ollama', 'pull', base_model], check=True)
            logger.info(f"Successfully pulled {base_model}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull {base_model}: {e}")
            return False
    
    # Create modelfiles
    create_modelfiles(base_model)
    
    # Create or update personality models
    models_created = 0
    models_to_create = []
    
    for trait, levels in MODEL_FILES.items():
        trait_name = get_trait_name(trait)
        
        for level_name, modelfile_path in levels.items():
            model_name = f"gemma3-{trait_name.lower()}"
            if level_name == "high":
                # For high traits, use the trait name directly
                pass
            else:
                # For low traits, add a suffix
                model_name += f"-{level_name}"
            
            if model_name in available_models and not force_recreate:
                logger.info(f"Model {model_name} already exists, skipping...")
            else:
                models_to_create.append((model_name, modelfile_path))
    
    # Create models
    for model_name, modelfile_path in models_to_create:
        if create_ollama_model(model_name, modelfile_path):
            models_created += 1
    
    logger.info(f"Created {models_created} models out of {len(models_to_create)} required.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up Ollama models for personality generation")
    parser.add_argument("--base-model", default="gemma3", help="Base model to use for personality models")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreation of models even if they exist")
    parser.add_argument("--list-available", action="store_true", help="List available Ollama models and exit")
    
    args = parser.parse_args()
    
    if args.list_available:
        if check_ollama_running():
            models = get_available_ollama_models()
            print("Available Ollama models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("Ollama is not running. Please start Ollama and try again.")
        return 0
    
    if setup_personality_models(args.base_model, args.force_recreate):
        logger.info("Successfully set up personality models!")
        return 0
    else:
        logger.error("Failed to set up personality models.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 