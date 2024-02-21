from utils.constants import TEMPLATE_EVAL, TEMPLATE_EVAL_RES, INST_CODELLAMA_EVAL, INST_CODELLAMA_EVAL_RES, INST_CODELLAMA_EVAL_DB

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch
import sqlite3


class CodeLlamaEvaluator():

    def __init__(self, model_name_or_dir, db_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_dir,
            use_fast=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )#.to(device)
        self.model.eval()

        self.db_path = db_path
        self.device = device


    def score(self, db_id, question, candidates, evaluation_config):
        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            result = ""

            try:
                cursor.execute(cand_sql)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    result = "ERROR"

            if evaluation_config["use_exec_res"]:
                if result != "ERROR":
                    result = [(c[0], []) for c in cursor.description]
                    rows = []
                    for i in range(5):
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)

                    if i == 0:
                        result = "None"
                    else:
                        for values in rows:
                            for c, v in zip(result, values):
                                c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

                        result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])

                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL_RES.format(
                        TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL.format(
                        TEMPLATE_EVAL.format(question, cand_sql)
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


    def score_fewshot(self, db_id, question, candidates, retriever, evaluation_config):
        demos = retriever.retrieve(question)

        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            result = ""

            try:
                cursor.execute(cand_sql)
            except:
                if evaluation_config["check_exec"]:
                    scores.append(1)
                    continue
                else:
                    result = "ERROR"

            if evaluation_config["use_exec_res"]:
                if result != "ERROR":
                    result = [(c[0], []) for c in cursor.description]
                    rows = []
                    for i in range(5):
                        row = cursor.fetchone()
                        if row is None:
                            break
                        rows.append(row)

                    if i == 0:
                        result = "None"
                    else:
                        for values in rows:
                            for c, v in zip(result, values):
                                c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

                        result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
                
                prompt_strs = []
                for d in demos:
                    prompt_strs.append(
                        TEMPLATE_EVAL_RES.format(d["question"], d["sql"], d["exec_res"]) + "\n-- Answer: Yes"
                    )
                    prompt_strs.append(
                        TEMPLATE_EVAL_RES.format(d["question"], d["neg_sql"], d["neg_exec_res"]) + "\n-- Answer: No"
                    )
                prompt_strs.append(
                    TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                )
                
                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL_RES.format(
                        "\n\n".join(prompt_strs)
                    ),
                    return_tensors="pt", 
                    add_special_tokens=False
                )
            else:
                prompt_strs = []
                for d in demos:
                    prompt_strs.append(
                        TEMPLATE_EVAL.format(d["question"], d["sql"]) + "\n-- Answer: Yes"
                    )
                    prompt_strs.append(
                        TEMPLATE_EVAL.format(d["question"], d["neg_sql"]) + "\n-- Answer: No"
                    )
                prompt_strs.append(
                    TEMPLATE_EVAL.format(question, cand_sql)
                )

                batch = self.tokenizer(
                    INST_CODELLAMA_EVAL.format(
                        "\n\n".join(prompt_strs)
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


class CodeLlamaLoraEvaluator(CodeLlamaEvaluator):

    def __init__(self, model_name_or_dir, peft_model_dir, db_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_dir,
            use_fast=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            device_map="auto"
        )#.to(device)
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model_dir
        )
        self.model.eval()

        self.db_path = db_path
        self.device = device