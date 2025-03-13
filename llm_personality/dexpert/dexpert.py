import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_llama import LlamaForCausalLM

from dexpert_sample import evolve_dexpert_sampling

evolve_dexpert_sampling()



class DExpertGenerator():
    def __init__(self, args, args_expert=None, args_antiexpert=None):
        
        self.args = args
        self.args_expert = args_expert
        self.args_antiexpert = args_antiexpert
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.pad_token_id = torch.tensor(self.tokenizer.eos_token_id)
        
        self.model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        
        if args_expert is not None:
            self.model.expert = LlamaForCausalLM.from_pretrained(
                args_expert.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=args_expert.cache_dir
            )
        else:
            self.model.expert = None
            
        if args_antiexpert is not None:
            self.model.antiexpert = LlamaForCausalLM.from_pretrained(
                args_antiexpert.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=args_antiexpert.cache_dir
            )
        else:
            self.model.antiexpert = None
    
    def generate(self, messages, messages_expert=None, messages_antiexpert=None, alpha=None):

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        input_ids_expert = self.tokenizer.apply_chat_template(
            messages_expert,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device) if messages_expert is not None else None
        
        input_ids_antiexpert = self.tokenizer.apply_chat_template(
            messages_antiexpert,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device) if messages_antiexpert is not None else None
        outputs = self.model.generate(
            input_ids,
            input_ids_expert=input_ids_expert,              # dexpert
            input_ids_antiexpert=input_ids_antiexpert,      # dexpert
            alpha=alpha,                                    # dexpert
            max_new_tokens=1024,
            eos_token_id=self.terminators,
            pad_token_id=self.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        return result
