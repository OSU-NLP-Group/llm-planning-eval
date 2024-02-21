from utils.constants import TEMPLATE_GSM8K_EVAL, TEMPLATE_GSM8K_EVAL_RES
from utils.exec_py import exec_py

from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch


class CodeLlamaEvaluatorPy():

    def __init__(self, model_name_or_dir, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_dir,
            use_fast=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        self.device = device


    def score(self, question, candidates, evaluation_config):        
        scores = []
        for cand_py in candidates:

            try:
                answer = exec_py(cand_py)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    answer = "ERROR"

            if evaluation_config["use_exec_res"]:
                batch = self.tokenizer(
                    TEMPLATE_GSM8K_EVAL_RES.format(
                        question, cand_py, str(answer)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                batch = self.tokenizer(
                    TEMPLATE_GSM8K_EVAL.format(
                        question, cand_py
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
                
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
            
                # CodeLlama-Instruct: "No" 1939 "Yes" 3869
                scores.append(
                    - torch.nn.functional.softmax(
                        outputs["logits"][:, -1, :], dim=-1
                    ).flatten()[3869].item()
                )

        return scores