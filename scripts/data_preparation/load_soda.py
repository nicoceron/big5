#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import logging
from datasets import load_dataset
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_soda_dataset(args):
    """Load the AllenAI SODA dataset."""
    logger.info("Loading the SODA dataset...")
    
    try:
        dataset = load_dataset("allenai/soda")
        logger.info(f"Successfully loaded SODA dataset with splits: {dataset.keys()}")
        
        # Get the split specified by the user
        split_data = dataset[args.split]
        logger.info(f"Using '{args.split}' split with {len(split_data)} examples")
        
        return split_data
    except Exception as e:
        logger.error(f"Error loading SODA dataset: {e}")
        return None

def prepare_data_for_personality_testing(dataset, args):
    """Convert the SODA dataset into a format suitable for our personality testing."""
    logger.info("Preparing SODA data for personality testing...")
    
    # Create a directory to store the prepared data
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    csv_path = os.path.join(args.output_dir, f"soda_{args.split}.csv")
    jsonl_path = os.path.join(args.output_dir, f"soda_{args.split}.jsonl")
    
    # Check if both files already exist and have content (and skip-existing is enabled)
    if args.skip_existing and os.path.exists(csv_path) and os.path.exists(jsonl_path):
        try:
            with open(csv_path, 'r') as f:
                csv_content = f.read().strip()
            
            with open(jsonl_path, 'r') as f:
                jsonl_content = f.read().strip()
                
            if csv_content and jsonl_content:
                logger.info(f"Found existing data files. Using them instead of recreating.")
                # Try to read a few examples to validate the content
                df = pd.read_csv(csv_path)
                logger.info(f"Found {len(df)} examples in existing CSV file.")
                
                if not df.empty:
                    logger.info("\nExample data entries:")
                    for i, example in enumerate(df.head(3).to_dict('records')):
                        logger.info(f"Example {i+1}:")
                        for key, value in example.items():
                            logger.info(f"  {key}: {value}")
                        logger.info("")
                
                return csv_path, jsonl_path
        except Exception as e:
            logger.warning(f"Error reading existing files: {e}. Will recreate them.")
    
    # Convert the dataset into a list of dictionaries
    data_list = []
    examples_processed = 0
    examples_with_topics = 0
    examples_with_personas = 0
    examples_with_dialogues = 0
    
    # Process the dataset with a larger scan to find good examples
    scan_limit = min(args.scan_size if hasattr(args, 'scan_size') else len(dataset), len(dataset))
    
    # First pass: scan through dataset to understand the structure
    logger.info(f"Scanning first 100 examples to understand data structure...")
    sample_items = list(dataset.take(100)) if hasattr(dataset, 'take') else list(dataset)[:100]
    
    for i, item in enumerate(sample_items[:10]):
        logger.info(f"Sample item {i+1} keys: {list(item.keys())}")
        if i < 3:
            for key, value in item.items():
                if isinstance(value, (str, int, float)):
                    logger.info(f"  {key}: {str(value)[:100]}...")
                elif isinstance(value, list):
                    logger.info(f"  {key}: [list with {len(value)} items]")
                else:
                    logger.info(f"  {key}: {type(value)}")
    
    # Process the dataset
    for item in tqdm(dataset.take(scan_limit) if hasattr(dataset, 'take') else dataset[:scan_limit], desc="Processing SODA dataset"):
        # Check if we have enough examples
        if len(data_list) >= args.max_examples:
            break
            
        # Extract data with better fallbacks
        topic = item.get("topic", "").strip() if item.get("topic") else ""
        persona = item.get("persona", "").strip() if item.get("persona") else ""
        dialogue = item.get("dialogue", []) if item.get("dialogue") else []
        narrative = item.get("narrative", "").strip() if item.get("narrative") else ""
        
        # Get speaker names with fallbacks
        person_x = item.get("PersonX", item.get("personX", "PersonX")).strip() if item.get("PersonX") or item.get("personX") else "PersonX"
        person_y = item.get("PersonY", item.get("personY", "PersonY")).strip() if item.get("PersonY") or item.get("personY") else "PersonY"
        
        # More flexible filtering - include examples that have at least some useful content
        has_topic = bool(topic)
        has_persona = bool(persona)
        has_dialogue = bool(dialogue and len(dialogue) > 0)
        has_narrative = bool(narrative)
        
        # Skip only if completely empty
        if not (has_topic or has_persona or has_dialogue or has_narrative):
            continue
        
        # Apply optional filters if specified
        if hasattr(args, 'require_topic') and args.require_topic and not has_topic:
            continue
        if hasattr(args, 'require_persona') and args.require_persona and not has_persona:
            continue
        if hasattr(args, 'require_dialogue') and args.require_dialogue and not has_dialogue:
            continue
            
        examples_processed += 1
        
        # Count different types of content
        if has_topic:
            examples_with_topics += 1
        if has_persona:
            examples_with_personas += 1
        if has_dialogue:
            examples_with_dialogues += 1
        
        # Generate better fallback content
        if not topic and persona:
            topic = f"Discussion with {persona}"
        elif not topic and narrative:
            topic = "General conversation"
        elif not topic:
            topic = "Conversation"
            
        if not narrative:
            if topic and persona:
                narrative = f"A conversation between {person_x} and {person_y} about {topic}, where {person_y} is {persona}"
            elif topic:
                narrative = f"A conversation about {topic}"
            else:
                narrative = "A conversation between two people"
        
        # Format the data for our use case
        entry = {
            "topic": topic,
            "persona": persona,
            "dialogue": json.dumps(dialogue) if isinstance(dialogue, list) else str(dialogue),
            "narrative": narrative,
            "PersonX": person_x,
            "PersonY": person_y,
            # Generate a prompt using the topic and persona
            "prompt": generate_prompt(topic, persona)
        }
        
        data_list.append(entry)
    
    logger.info(f"Processed {examples_processed} examples")
    logger.info(f"Found {examples_with_topics} with topics, {examples_with_personas} with personas, {examples_with_dialogues} with dialogues")
    logger.info(f"Selected {len(data_list)} examples for the final dataset")
    
    # If we still don't have enough data, generate synthetic examples
    if len(data_list) < args.max_examples // 10:  # Less than 10% of requested
        logger.warning(f"Only found {len(data_list)} examples, generating additional synthetic examples")
        synthetic_data = generate_synthetic_examples(args.max_examples - len(data_list))
        data_list.extend(synthetic_data)
        logger.info(f"Added {len(synthetic_data)} synthetic examples, total: {len(data_list)}")
    
    # Create a DataFrame
    df = pd.DataFrame(data_list)
    
    # Save the prepared data
    if not df.empty:
        # Save as CSV for easy viewing/editing
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV data to {csv_path}")
        
        # Save as JSONL for processing
        with open(jsonl_path, 'w') as f:
            for entry in data_list:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Saved JSONL data to {jsonl_path}")
    
    # Print some examples
    logger.info("\nExample data entries:")
    for i, example in enumerate(data_list[:3]):
        logger.info(f"Example {i+1}:")
        for key, value in example.items():
            if key == 'dialogue':
                try:
                    dialogue_data = json.loads(value) if isinstance(value, str) else value
                    logger.info(f"  {key}: [list with {len(dialogue_data)} turns]")
                except:
                    logger.info(f"  {key}: {str(value)[:100]}...")
            else:
                logger.info(f"  {key}: {str(value)[:100]}...")
        logger.info("")
    
    return csv_path, jsonl_path

def generate_synthetic_examples(num_examples):
    """Generate synthetic conversation examples for training."""
    logger.info(f"Generating {num_examples} synthetic examples")
    
    topics = [
        "Environmental activism", "Technology and society", "Education reform", "Healthcare access",
        "Work-life balance", "Climate change", "Social media impact", "Mental health awareness",
        "Economic inequality", "Cultural diversity", "Political engagement", "Scientific research",
        "Art and creativity", "Sports and competition", "Travel experiences", "Food and nutrition",
        "Personal growth", "Relationships", "Career development", "Community involvement"
    ]
    
    personas = [
        "a thoughtful teacher", "an enthusiastic entrepreneur", "a curious student",
        "a dedicated healthcare worker", "an experienced manager", "a creative artist",
        "a community volunteer", "a tech professional", "a environmental advocate",
        "a concerned parent", "a recent graduate", "a retired professional",
        "a small business owner", "a social worker", "a researcher", "a journalist"
    ]
    
    synthetic_data = []
    for i in range(num_examples):
        topic = topics[i % len(topics)]
        persona = personas[i % len(personas)]
        
        # Generate simple dialogue
        dialogue = [
            {"speaker": "PersonX", "text": f"What do you think about {topic}?"},
            {"speaker": "PersonY", "text": f"As {persona}, I find this topic quite important to discuss."}
        ]
        
        entry = {
            "topic": topic,
            "persona": persona,
            "dialogue": json.dumps(dialogue),
            "narrative": f"A conversation between PersonX and PersonY about {topic}, where PersonY is {persona}",
            "PersonX": "PersonX",
            "PersonY": "PersonY",
            "prompt": generate_prompt(topic, persona)
        }
        synthetic_data.append(entry)
    
    return synthetic_data

def generate_prompt(topic, persona):
    """Generate a prompt based on the topic and persona."""
    if topic and persona:
        return f"As {persona}, what are your thoughts on {topic}?"
    elif topic:
        return f"Let's discuss the topic of {topic}. What are your thoughts on this subject?"
    elif persona:
        return f"You are {persona}. How would you introduce yourself and what matters to you?"
    else:
        return "Let's have a conversation. What would you like to discuss?"

def main():
    parser = argparse.ArgumentParser(description="Load and prepare the SODA dataset")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--output-dir", default="data/soda",
                        help="Directory to save the processed data")
    parser.add_argument("--max-examples", type=int, default=100,
                        help="Maximum number of examples to process")
    parser.add_argument("--require-topic", action="store_true",
                        help="Only include examples with topics")
    parser.add_argument("--require-persona", action="store_true",
                        help="Only include examples with personas")
    parser.add_argument("--require-dialogue", action="store_true",
                        help="Only include examples with dialogues")
    parser.add_argument("--scan-size", type=int, default=10000,
                        help="Number of examples to scan for filtering (set higher to find more matches)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip processing if output files already exist")
    
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_soda_dataset(args)
    if dataset is None:
        return 1
    
    # Prepare the data for personality testing
    csv_path, jsonl_path = prepare_data_for_personality_testing(dataset, args)
    
    logger.info(f"\nSuccessfully prepared SODA data!")
    logger.info(f"Use this data with the personality test script:")
    logger.info(f"python run_personality_test.py --trait o --input-file {csv_path}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 