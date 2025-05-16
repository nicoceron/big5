import json
import sys
import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import re
from pathlib import Path
np.random.seed(42)

from prompts import generate_prompt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import DExpertGenerator from correct location
# Try to import from local implementation first
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dexpert'))
    from dexpert_local import DExpertGenerator
    logger.info("Using local Ollama-based implementation of DExpertGenerator")
except ImportError:
    try:
        from llm_personality.dexpert.dexpert_local import DExpertGenerator
        logger.info("Using package-based local Ollama implementation of DExpertGenerator")
    except ImportError:
        try:
            from llm_personality.dexpert.dexpert import DExpertGenerator
            logger.info("Using original DExpertGenerator implementation")
        except ImportError:
            logger.error("Could not import DExpertGenerator from any location")
            sys.exit(1)

def remove_double_quotes(s):
    # Remove double quotes from the beginning and end
    return re.sub(r'^"+|"+$', '', s)


class CO3Sotopia():
    def __init__(self, args):
        self.args = args
        
        class Args:
            # For local implementation, use the Gemma3 model
            model_id = "gemma3:12b"
            cache_dir = None

        class ArgsExpert:
            # For local implementation, use personality-specific variants if specified
            model_id = None
            cache_dir = None
            lora = False
            
            # Map personality traits to Ollama models
            trait_map = {
                'o': 'gemma3-openness',
                'c': 'gemma3-conscientiousness',
                'e': 'gemma3-extraversion',
                'a': 'gemma3-agreeableness',
                'n': 'gemma3-neuroticism'
            }
            
            def __init__(self, trait=None):
                if trait and trait in self.trait_map:
                    self.model_id = self.trait_map[trait]
                    logger.info(f"Using {self.model_id} as expert model for trait {trait}")

        # Set up the expert model based on the trait
        args_expert = ArgsExpert(args.person_trait)
        
        logger.info(f"Initializing DExpertGenerator with base model {Args.model_id}")
        self.model = DExpertGenerator(args=Args, args_expert=args_expert)
        self.data = pd.read_csv(args.in_file).to_dict(orient='records')
        curr_idx, total_idx = args.chunk.split("/")
        curr_idx = int(curr_idx)
        total_idx = int(total_idx)
        parts = len(self.data) // total_idx if len(self.data) >= total_idx else 1
        self.data = self.data[(curr_idx-1)*parts:curr_idx*parts]
        
    def process_response(self, response):
        try:
            # split by \n\n and get the middle one?
            if ":\n\n" in response:
                response = response.split(":\n\n")
                response = response[-1]
            # remove quotation marks at the beginning and the end
            response = remove_double_quotes(response)
            return response
        except Exception as e:
            logger.error(f"Error during processing response: {e}")
            logger.error(f"Response: {response}")
            return response
        
    def generate_messages(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        return messages
    
    def generate_expert_messages(self, big_five_level):
        level_lst = ['high', 'low']
        if big_five_level[0] != -1:
            prompt_person_str = f"Openness - {level_lst[big_five_level[0]]}"
        elif big_five_level[1] != -1:
            prompt_person_str = f"Conscientiousness - {level_lst[big_five_level[1]]}"
        elif big_five_level[2] != -1:
            prompt_person_str = f"Extraversion - {level_lst[big_five_level[2]]}"
        elif big_five_level[3] != -1:
            prompt_person_str = f"Agreeableness - {level_lst[big_five_level[3]]}"
        elif big_five_level[4] != -1:
            prompt_person_str = f"Neuroticism - {level_lst[big_five_level[4]]}"
        
        prompt = f"Help me complete the sentence with certain Big Five Personality: {prompt_person_str}\n"
        messages = [
            {"role": "user", "content": prompt}
        ]
        return messages
    
    def generate_dialogue_turn1(self, idx, env_info, p2_big_five, out_f=None, speaker_x_utterance=None):
        # generate prompt for turn 1
        prompt_turn_1 = generate_prompt(
            env_info, # SODA row, contains topic/persona for P1 if needed by generate_prompt
            current_turn_index=1,
            # p1_personality_and_values=p1_big_five, # Not used for P1
            p2_personality_and_values=p2_big_five, # P2 (Speaker Y) personality
            p1_argument = speaker_x_utterance, # Speaker X's utterance from SODA
        )
        
        is_high = p2_big_five[0] == 0 or p2_big_five[1] == 0 or p2_big_five[2] == 0 or p2_big_five[3] == 0 or p2_big_five[4] == 0
        trait_level = "high" if is_high else "low"
        logger.info(f"Generating turn 1 for environment {idx} with {trait_level} trait level")
        
        response_turn_1 = self.process_response(self.model.generate(
            messages = self.generate_messages(prompt_turn_1),
            messages_expert = self.generate_expert_messages(p2_big_five),
            alpha = self.args.alpha,
        ))
        
        result_info = {
            "env_idx": idx,
            "env_info": { # Log relevant parts of env_info
                "topic": env_info.get("topic"),
                "persona_P1": env_info.get("persona"), # Persona of Speaker X from SODA
                "utterance_P1": speaker_x_utterance
            },
            "personality_P2": " ".join(map(str, p2_big_five)), # Speaker Y's target personality
            "trait_P2": self.args.person_trait, # o, c, e, a, n
            "level_P2": "high" if p2_big_five[p2_big_five_abbr[self.args.person_trait]] == 0 else "low",
            "turn": 1, # Signifies Speaker Y's response (the generated one)
            "prompt_P2": prompt_turn_1, # Prompt given to LLM for generating Speaker Y
            "response_P2": response_turn_1, # Generated response of Speaker Y
        }
        if out_f is not None:
            json.dump(result_info, out_f)
            out_f.write("\n")
            out_f.flush()
    
    def run(self):
        Path(self.args.out_file).touch(exist_ok=True)
        
        # Load existing content if any
        try:
            with open(self.args.out_file, 'r') as f:
                content = f.read().strip()
                out_f_content = [json.loads(i) for i in content.split('\n') if i.strip()] if content else []
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not parse existing content in {self.args.out_file}, starting fresh")
            out_f_content = []
        
        last_env_idx = None
        if len(out_f_content) > 0:
            last_env_idx = out_f_content[-1].get('env_idx')
            # turn = out_f_content[-1].get('turn') # Turn is always 1 now for the actual output record
            # personality_low = "1" in out_f_content[-1].get('personality', "").split(" ") if turn == 1 else None
            # Simplified resume: determine if the last record for last_env_idx was high or low
            last_level_P2 = out_f_content[-1].get('level_P2')
        
        out_f = open(self.args.out_file, 'a')
        p2_big_five_ref = [-1, -1, -1, -1, -1]
        p2_big_five_abbr = {'o': 0, 'c': 1, 'e': 2, 'a': 3, 'n': 4}
        
        p2_big_five_high, p2_big_five_low = p2_big_five_ref.copy(), p2_big_five_ref.copy()
        p2_big_five_high[p2_big_five_abbr[self.args.person_trait]] = 0 # high
        p2_big_five_low[p2_big_five_abbr[self.args.person_trait]] = 1 # low
        
        # Determine start_idx and if the first item to process needs only low or both high and low.
        start_idx = 0
        process_only_low_for_first_item = False

        if last_env_idx is not None:
            # If the last processed item was for this trait and was 'high',
            # we need to process 'low' for the same env_idx.
            if out_f_content[-1].get('trait_P2') == self.args.person_trait and last_level_P2 == 'high':
                start_idx = last_env_idx
                process_only_low_for_first_item = True
            else: # Last item was 'low' or a different trait, so move to the next env_idx
                start_idx = last_env_idx + 1

        # Process the data (self.data is already chunked)
        for idx, env_info in tqdm(enumerate(self.data), total=len(self.data)):
            current_data_original_idx = (self.args.chunk.split("/")[0]-1)*(len(self.data) // self.args.chunk.split("/")[1] ) + idx # Map current loop idx to original SODA idx if chunking
            actual_env_idx_for_output = idx # Use loop index for records within this chunk

            if actual_env_idx_for_output < start_idx:
                continue

            try:
                dialogue_json_str = env_info.get('dialogue', '[]')
                if isinstance(dialogue_json_str, str):
                    dialogue_turns = json.loads(dialogue_json_str)
                else:
                    dialogue_turns = dialogue_json_str

                if not dialogue_turns or not isinstance(dialogue_turns, list) or not isinstance(dialogue_turns[0], dict) or 'text' not in dialogue_turns[0]:
                    logger.warning(f"Skipping env_idx {actual_env_idx_for_output} (original: {env_info.get('orig_idx', 'N/A')}) due to missing/invalid Speaker X utterance: {dialogue_turns}")
                    continue
                speaker_x_utterance = dialogue_turns[0]['text']

                if actual_env_idx_for_output == start_idx and process_only_low_for_first_item:
                    # Only process 'low' for this item as 'high' was done previously
                    logger.info(f"Resuming: Processing LOW for env_idx {actual_env_idx_for_output} (trait {self.args.person_trait})")
                    self.generate_dialogue_turn1(actual_env_idx_for_output, env_info, p2_big_five_low, out_f, speaker_x_utterance)
                else:
                    # Process both 'high' and 'low' for this item
                    logger.info(f"Processing HIGH for env_idx {actual_env_idx_for_output} (trait {self.args.person_trait})")
                    self.generate_dialogue_turn1(actual_env_idx_for_output, env_info, p2_big_five_high, out_f, speaker_x_utterance)
                    logger.info(f"Processing LOW for env_idx {actual_env_idx_for_output} (trait {self.args.person_trait})")
                    self.generate_dialogue_turn1(actual_env_idx_for_output, env_info, p2_big_five_low, out_f, speaker_x_utterance)
            
            except Exception as e:
                logger.error(f"Error processing env_idx {actual_env_idx_for_output} for trait {self.args.person_trait}: {e}")
                logger.error(f"Problematic env_info content: {env_info}")
                continue
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating dialogues')
    parser.add_argument("--in_file", type=str, default="", help="The file of the sampled soda training data")
    parser.add_argument("--out_file", type=str, default="env_profiles.jsonl")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--person_trait", type=str, choices=['o', 'c', 'e', 'a', 'n'])
    parser.add_argument("--chunk", type=str, default="1/2")
    args = parser.parse_args()
    soda_maker = CO3Sotopia(args)
    soda_maker.run()