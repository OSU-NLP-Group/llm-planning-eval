from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import torch

class HFGenerator():

    def __init__(self, base_model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            trust_remote_code=True,
            use_auth_token=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=True,
            device_map="auto"
        )
        self.model.eval()

        self.base_model_name = base_model_name
        self.device = device
    

    def generate(self, prompt, generation_config):
        batch = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            completion = self.model.generate(
                inputs=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=300,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=GenerationConfig(**generation_config)
            )

        return [self.tokenizer.decode(c, skip_special_tokens=True) for c in completion]


    def ppl(self, prompt, completion):
        batch = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        input_length = len(batch["input_ids"][0])

        batch = self.tokenizer(
            completion, 
            return_tensors="pt", 
            add_special_tokens=False
        )

        input_ids = batch["input_ids"].to(self.device)
        labels = torch.zeros_like(input_ids) - 100
        labels[:, input_length:] = input_ids[:, input_length:]

        with torch.no_grad():
            loss = self.model(
                input_ids=input_ids,
                labels=labels
            ).loss.item()

        return loss * (len(input_ids[0]) - input_length)


class HFLoraGenerator(HFGenerator):

    def __init__(self, base_model_name, peft_model_dir, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            trust_remote_code=True,
            use_auth_token=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            trust_remote_code=True,
            use_auth_token=True,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model_dir
        )
        self.model.eval()

        self.device = device
