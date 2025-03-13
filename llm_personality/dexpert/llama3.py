from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


class LLAMA3():
    def __init__(self, llama_version):
        if llama_version == "llama3-8b":
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            cache_dir = None
        elif llama_version == "llama3-70b":
            model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
            cache_dir = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def generate(self, prompts):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompts}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        return result
