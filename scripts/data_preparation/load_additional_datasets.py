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

def load_persona_chat():
    """Load the PersonaChat dataset."""
    try:
        logger.info("Loading PersonaChat dataset...")
        dataset = load_dataset("bavard/personachat_truecased", trust_remote_code=True)
        return dataset
    except Exception as e:
        logger.warning(f"Could not load PersonaChat: {e}")
        return None

def load_empathetic_dialogues():
    """Load the Empathetic Dialogues dataset."""
    try:
        logger.info("Loading Empathetic Dialogues dataset...")
        dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
        return dataset
    except Exception as e:
        logger.warning(f"Could not load Empathetic Dialogues: {e}")
        return None

def load_daily_dialog():
    """Load the DailyDialog dataset."""
    try:
        logger.info("Loading DailyDialog dataset...")
        dataset = load_dataset("daily_dialog", trust_remote_code=True)
        return dataset
    except Exception as e:
        logger.warning(f"Could not load DailyDialog: {e}")
        return None

def load_personality_detection():
    """Load personality detection datasets."""
    datasets = []
    
    # Try to load various personality-related datasets
    dataset_names = [
        "essay",  # For personality detection
        "personality-detection",
        "big5-personality"
    ]
    
    for name in dataset_names:
        try:
            logger.info(f"Attempting to load {name} dataset...")
            dataset = load_dataset(name)
            datasets.append((name, dataset))
            logger.info(f"Successfully loaded {name}")
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")
    
    return datasets

def process_persona_chat(dataset, output_dir, max_examples=10000):
    """Process PersonaChat into our format."""
    logger.info("Processing PersonaChat dataset...")
    
    data_list = []
    split = "train"
    
    if split not in dataset:
        logger.warning(f"Split {split} not found in PersonaChat")
        return []
    
    for i, item in enumerate(tqdm(dataset[split], desc="Processing PersonaChat")):
        if len(data_list) >= max_examples:
            break
            
        # Extract personality descriptions and history
        personality = item.get("personality", [])
        history = item.get("history", [])
        candidates = item.get("candidates", [])
        
        if personality and history:
            persona_str = " ".join(personality)
            conversation = history + [candidates[0]] if candidates else history
            
            entry = {
                "topic": "Personal conversation",
                "persona": persona_str,
                "dialogue": json.dumps([{"speaker": f"Person{i%2+1}", "text": turn} for i, turn in enumerate(conversation)]),
                "narrative": f"A conversation with someone who describes themselves as: {persona_str}",
                "PersonX": "Person1",
                "PersonY": "Person2",
                "prompt": f"You are someone who {persona_str}. Let's have a conversation."
            }
            data_list.append(entry)
    
    return data_list

def process_empathetic_dialogues(dataset, output_dir, max_examples=10000):
    """Process Empathetic Dialogues into our format."""
    logger.info("Processing Empathetic Dialogues dataset...")
    
    data_list = []
    split = "train"
    
    if split not in dataset:
        logger.warning(f"Split {split} not found in Empathetic Dialogues")
        return []
    
    for i, item in enumerate(tqdm(dataset[split], desc="Processing Empathetic Dialogues")):
        if len(data_list) >= max_examples:
            break
            
        # Extract emotional context and conversation
        emotion = item.get("context", "")
        utterance = item.get("utterance", "")
        speaker_idx = item.get("speaker_idx", 0)
        
        if emotion and utterance:
            entry = {
                "topic": f"Emotional conversation about {emotion}",
                "persona": f"someone experiencing {emotion}",
                "dialogue": json.dumps([{"speaker": f"Person{speaker_idx+1}", "text": utterance}]),
                "narrative": f"A conversation about {emotion}",
                "PersonX": "Person1",
                "PersonY": "Person2",
                "prompt": f"Let's discuss feelings related to {emotion}."
            }
            data_list.append(entry)
    
    return data_list

def process_daily_dialog(dataset, output_dir, max_examples=10000):
    """Process DailyDialog into our format."""
    logger.info("Processing DailyDialog dataset...")
    
    data_list = []
    split = "train"
    
    if split not in dataset:
        logger.warning(f"Split {split} not found in DailyDialog")
        return []
    
    for i, item in enumerate(tqdm(dataset[split], desc="Processing DailyDialog")):
        if len(data_list) >= max_examples:
            break
            
        # Extract dialog and topics
        dialog = item.get("dialog", [])
        act = item.get("act", [])
        emotion = item.get("emotion", [])
        
        if dialog and len(dialog) > 1:
            dialogue_formatted = [{"speaker": f"Person{i%2+1}", "text": turn} for i, turn in enumerate(dialog)]
            
            # Determine topic from acts or emotions
            topic = "Daily conversation"
            if act and len(act) > 0:
                topic = f"Conversation involving {', '.join(set(str(a) for a in act[:3]))}"
            
            entry = {
                "topic": topic,
                "persona": "a conversational participant",
                "dialogue": json.dumps(dialogue_formatted),
                "narrative": f"A daily conversation between two people",
                "PersonX": "Person1",
                "PersonY": "Person2",
                "prompt": "Let's have a conversation about everyday topics."
            }
            data_list.append(entry)
    
    return data_list

def generate_additional_synthetic_data(num_examples=5000):
    """Generate additional synthetic personality data."""
    logger.info(f"Generating {num_examples} additional synthetic examples...")
    
    # Big Five personality traits with descriptions
    personality_templates = {
        "high_openness": [
            "someone who loves exploring new ideas and experiences",
            "a person who is very creative and imaginative",
            "someone who enjoys abstract thinking and innovation",
            "a person who is curious about the world around them"
        ],
        "low_openness": [
            "someone who prefers familiar and traditional approaches",
            "a person who likes routine and predictable experiences",
            "someone who focuses on practical rather than abstract matters",
            "a person who values conventional wisdom"
        ],
        "high_conscientiousness": [
            "someone who is very organized and detail-oriented",
            "a person who plans ahead and meets deadlines",
            "someone who is disciplined and self-controlled",
            "a person who takes responsibilities seriously"
        ],
        "low_conscientiousness": [
            "someone who is spontaneous and flexible",
            "a person who prefers to go with the flow",
            "someone who is more relaxed about deadlines",
            "a person who adapts quickly to changing situations"
        ],
        "high_extraversion": [
            "someone who is outgoing and energetic",
            "a person who enjoys social gatherings and meeting new people",
            "someone who feels comfortable being the center of attention",
            "a person who gains energy from social interactions"
        ],
        "low_extraversion": [
            "someone who is thoughtful and reserved",
            "a person who prefers quiet activities and small groups",
            "someone who enjoys solitary pursuits",
            "a person who thinks before speaking"
        ],
        "high_agreeableness": [
            "someone who is kind and considerate of others",
            "a person who avoids conflict and seeks harmony",
            "someone who is trusting and cooperative",
            "a person who puts others' needs before their own"
        ],
        "low_agreeableness": [
            "someone who is direct and honest in their opinions",
            "a person who questions others' motives",
            "someone who prioritizes logic over emotions",
            "a person who stands firm in their beliefs"
        ],
        "high_neuroticism": [
            "someone who is sensitive and emotionally reactive",
            "a person who worries about potential problems",
            "someone who experiences stress intensely",
            "a person who is deeply affected by criticism"
        ],
        "low_neuroticism": [
            "someone who is calm and emotionally stable",
            "a person who handles stress well",
            "someone who remains composed under pressure",
            "a person who is resilient in difficult situations"
        ]
    }
    
    conversation_topics = [
        "work-life balance", "career goals", "relationships", "health and wellness",
        "technology trends", "environmental issues", "education", "travel experiences",
        "hobbies and interests", "current events", "personal growth", "family matters",
        "financial planning", "creative projects", "social issues", "entertainment",
        "food and cooking", "exercise and fitness", "learning new skills", "volunteering"
    ]
    
    synthetic_data = []
    
    for i in range(num_examples):
        # Choose personality type and template
        personality_type = list(personality_templates.keys())[i % len(personality_templates)]
        persona_template = personality_templates[personality_type][i % len(personality_templates[personality_type])]
        topic = conversation_topics[i % len(conversation_topics)]
        
        # Generate dialogue
        dialogue = [
            {"speaker": "PersonX", "text": f"I'd like to discuss {topic} with you. What are your thoughts?"},
            {"speaker": "PersonY", "text": f"As {persona_template}, I have some interesting perspectives on {topic}."}
        ]
        
        entry = {
            "topic": topic,
            "persona": persona_template,
            "dialogue": json.dumps(dialogue),
            "narrative": f"A conversation about {topic} with {persona_template}",
            "PersonX": "PersonX",
            "PersonY": "PersonY",
            "prompt": f"You are {persona_template}. Let's discuss {topic}.",
            "personality_type": personality_type
        }
        synthetic_data.append(entry)
    
    return synthetic_data

def main():
    parser = argparse.ArgumentParser(description="Load additional personality datasets")
    parser.add_argument("--output-dir", default="data/additional",
                        help="Directory to save the processed data")
    parser.add_argument("--max-examples", type=int, default=10000,
                        help="Maximum number of examples per dataset")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_data = []
    
    # Load and process PersonaChat
    persona_chat = load_persona_chat()
    if persona_chat:
        persona_data = process_persona_chat(persona_chat, args.output_dir, args.max_examples)
        all_data.extend(persona_data)
        logger.info(f"Added {len(persona_data)} examples from PersonaChat")
    
    # Load and process Empathetic Dialogues
    empathetic = load_empathetic_dialogues()
    if empathetic:
        empathetic_data = process_empathetic_dialogues(empathetic, args.output_dir, args.max_examples)
        all_data.extend(empathetic_data)
        logger.info(f"Added {len(empathetic_data)} examples from Empathetic Dialogues")
    
    # Load and process DailyDialog
    daily_dialog = load_daily_dialog()
    if daily_dialog:
        daily_data = process_daily_dialog(daily_dialog, args.output_dir, args.max_examples)
        all_data.extend(daily_data)
        logger.info(f"Added {len(daily_data)} examples from DailyDialog")
    
    # Generate additional synthetic data
    synthetic_data = generate_additional_synthetic_data(5000)
    all_data.extend(synthetic_data)
    logger.info(f"Added {len(synthetic_data)} synthetic examples")
    
    # Save combined data
    if all_data:
        df = pd.DataFrame(all_data)
        
        csv_path = os.path.join(args.output_dir, "additional_personality_data.csv")
        jsonl_path = os.path.join(args.output_dir, "additional_personality_data.jsonl")
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV data to {csv_path}")
        
        with open(jsonl_path, 'w') as f:
            for entry in all_data:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Saved JSONL data to {jsonl_path}")
        
        logger.info(f"Total examples collected: {len(all_data)}")
    else:
        logger.warning("No data was collected from any source")
    
    return 0

if __name__ == "__main__":
    exit(main()) 