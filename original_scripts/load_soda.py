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
    
    # Check if both files already exist and have content
    if os.path.exists(csv_path) and os.path.exists(jsonl_path):
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
    
    # Process the dataset
    for item in tqdm(dataset, desc="Processing SODA dataset"):
        # Check if we have enough examples
        if len(data_list) >= args.max_examples:
            break
            
        # Filter for examples with topics and personas if requested
        topic = item.get("topic", "").strip()
        persona = item.get("persona", "").strip()
        dialogue = item.get("dialogue", [])
        narrative = item.get("narrative", "").strip()
        # Get speaker names if available
        person_x = item.get("PersonX", "PersonX").strip()
        person_y = item.get("PersonY", "PersonY").strip()
        
        # Skip examples without topics if filtering is enabled
        if args.require_topic and not topic:
            continue
            
        # Skip examples without personas if filtering is enabled
        if args.require_persona and not persona:
            continue
            
        # Skip examples without dialogues if filtering is enabled
        if args.require_dialogue and not dialogue:
            continue
            
        examples_processed += 1
        
        if topic:
            examples_with_topics += 1
        
        # Format the data for our use case
        entry = {
            "topic": topic,
            "persona": persona,
            "dialogue": dialogue,
            "narrative": narrative if narrative else f"A conversation about {topic}",
            "PersonX": person_x,
            "PersonY": person_y,
            # Generate a prompt using the topic and persona
            "prompt": generate_prompt(topic, persona)
        }
        
        data_list.append(entry)
    
    logger.info(f"Processed {examples_processed} examples, found {examples_with_topics} with topics")
    logger.info(f"Selected {len(data_list)} examples for the final dataset")
    
    # Create a DataFrame
    df = pd.DataFrame(data_list)
    
    # Save the prepared data if we have data
    if not df.empty:
        # Save as CSV for easy viewing/editing
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV data to {csv_path}")
        
        # Save as JSONL for processing
        with open(jsonl_path, 'w') as f:
            for entry in data_list:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Saved JSONL data to {jsonl_path}")
    else:
        # Use our pre-made files if we don't have data from the API
        logger.warning("No examples found in SODA dataset. Using pre-made examples.")
        # Don't overwrite existing files if they exist
        
    # Print some examples
    logger.info("\nExample data entries:")
    for i, example in enumerate(data_list[:3]):
        logger.info(f"Example {i+1}:")
        for key, value in example.items():
            if key == 'dialogue':
                logger.info(f"  {key}: [{len(value)} turns]")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("")
    
    return csv_path, jsonl_path

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