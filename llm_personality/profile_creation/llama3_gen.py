import json
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import re
from pathlib import Path
np.random.seed(42)

from prompts import generate_prompt
from dexpert import DExpertGenerator

def remove_double_quotes(s):
    # Remove double quotes from the beginning and end
    return re.sub(r'^"+|"+$', '', s)


class CO3Sotopia():
    def __init__(self, args):
        self.args = args
        
        class Args:
            model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
            cache_dir = None

        class ArgsExpert:
            model_id = None
            cache_dir = None
            lora = False
        self.model = DExpertGenerator(args=Args, args_expert=ArgsExpert)
        self.data = pd.read_csv(args.in_file).to_dict(orient='records')
        curr_idx, total_idx = args.chunk.split("/")
        curr_idx = int(curr_idx)
        total_idx = int(total_idx)
        parts = len(self.data) // total_idx
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
            print("Error during processing response", response)
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
    
    def generate_dialogue_turn0(self, idx, env_info, out_f=None):
        # generate prompt for turn 0
        self.prompt_turn_0 = generate_prompt(
            env_info,
            current_turn_index=0,
            # p1_personality_and_values=p1_big_five,
            # p2_personality_and_values=p2_big_five,
        )
        
        self.response_turn_0 = self.process_response(self.model.generate(
            messages = self.generate_messages(self.prompt_turn_0),
            # messages_expert = self.generate_expert_messages(p1_big_five),
            alpha = self.args.alpha,
        ))
        
        result_info = {
            "env_idx": idx,
            "env_info": env_info,
            "turn": 0,
            "prompt": self.prompt_turn_0,
            "response": self.response_turn_0,
        }
        if out_f is not None:
            json.dump(result_info, out_f)
            out_f.write("\n")
            out_f.flush()
    
    def generate_dialogue_turn1(self, idx, env_info, p2_big_five, out_f=None):
        
        
        # generate prompt for turn 1
        prompt_turn_1 = generate_prompt(
            env_info,
            current_turn_index=1,
            # p1_personality_and_values=p1_big_five,
            p2_personality_and_values=p2_big_five,
            p1_argument = self.response_turn_0,
        )
        
        response_turn_1 = self.process_response(self.model.generate(
            messages = self.generate_messages(prompt_turn_1),
            messages_expert = self.generate_expert_messages(p2_big_five),
            alpha = self.args.alpha,
        ))
        
        result_info = {
            "env_idx": idx,
            "env_info": env_info,
            "personality": " ".join(map(str, p2_big_five)),
            "turn": 1,
            "prompt": prompt_turn_1,
            "response": response_turn_1,
        }
        if out_f is not None:
            json.dump(result_info, out_f)
            out_f.write("\n")
            out_f.flush()
    
    def run(self):
        Path(self.args.out_file).touch(exist_ok=True)
        
        out_f_content = [json.loads(i) for i in open(self.args.out_file, 'r').readlines()]
        
        last_env_idx, turn, personality_low = None, None, None
        if len(out_f_content) > 0:
            last_env_idx = out_f_content[-1]['env_idx']
            turn = out_f_content[-1]['turn']
            personality_low = "0" in out_f_content[-1]['personality'].split(" ") if turn == 1 else None
        
        out_f = open(self.args.out_file, 'a')
        p2_big_five_ref = [-1, -1, -1, -1, -1]
        p2_big_five_abbr = {'o': 0, 'c': 1, 'e': 2, 'a': 3, 'n': 4}
        
        p2_big_five_high, p2_big_five_low = p2_big_five_ref.copy(), p2_big_five_ref.copy()
        p2_big_five_high[p2_big_five_abbr[self.args.person_trait]] = 0 # high
        p2_big_five_low[p2_big_five_abbr[self.args.person_trait]] = 1 # low
        
        if last_env_idx is None:
            for idx, env_info in tqdm(enumerate(self.data)):
                self.generate_dialogue_turn0(idx, env_info, out_f)
                self.generate_dialogue_turn1(idx, env_info, p2_big_five_high, out_f)
                self.generate_dialogue_turn1(idx, env_info, p2_big_five_low, out_f)
        else:
            if turn == 1 and not personality_low:
                is_new_turn = True
            else:
                is_new_turn = False
                
            if not is_new_turn:
                if turn == 0:
                    self.generate_dialogue_turn0(last_env_idx, self.data[last_env_idx], None)
                    self.generate_dialogue_turn1(last_env_idx, self.data[last_env_idx], p2_big_five_high, out_f)
                    self.generate_dialogue_turn1(last_env_idx, self.data[last_env_idx], p2_big_five_low, out_f)
                else:
                    self.generate_dialogue_turn0(last_env_idx, self.data[last_env_idx], None)
                    self.generate_dialogue_turn1(last_env_idx, self.data[last_env_idx], p2_big_five_high, None)
                    self.generate_dialogue_turn1(last_env_idx, self.data[last_env_idx], p2_big_five_low, out_f)
            
            offset = last_env_idx+1
            self.data = self.data[offset:]
            for idx, env_info in tqdm(enumerate(self.data)):
                self.generate_dialogue_turn0(idx + offset, env_info, out_f)
                self.generate_dialogue_turn1(idx + offset, env_info, p2_big_five_high, out_f)
                self.generate_dialogue_turn1(idx + offset, env_info, p2_big_five_low, out_f)
            
    
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