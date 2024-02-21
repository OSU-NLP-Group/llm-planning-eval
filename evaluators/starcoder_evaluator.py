from utils.constants import TEMPLATE_EVAL

from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch
import sqlite3


# DB_PATH = "../spider/database"


class StarcoderEvaluator():

    def __init__(self, model_name_or_dir, db_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_dir,
            use_fast=False,
            trust_remote_code=True,
            use_auth_token=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=True
        ).to(device)
        self.model.eval()

        self.db_path = db_path
        self.device = device


    def score(self, db_id, question, candidates):
        conn = sqlite3.connect(f'{self.db_path}/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()

        scores = []
        for cand_sql in candidates:
            try:
                cursor.execute(cand_sql)
                # if cursor.fetchone() is None:
                #     raise Exception("Empty result")
            except:
                scores.append(1)
                continue

            result = [(c[0], []) for c in cursor.description]
            rows = []
            for i in range(5):
                row = cursor.fetchone()
                if row is None:
                    break
                rows.append(row)

            if i == 0:
                scores.append(1)
                continue

            for values in rows:
                for c, v in zip(result, values):
                    c[1].append((v[:128] + "..." if type(v) == str and len(v) > 128 else str(v)))

            result = "-- " + "\n-- ".join([c[0].lower() + ": " + ", ".join(c[1]) for c in result])
            
            batch = self.tokenizer(
                TEMPLATE_EVAL.format(question, cand_sql, result),
                return_tensors="pt", 
                add_special_tokens=False
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
            
                scores.append(
                    - torch.sigmoid(
                        outputs["logits"][:, -1, 10922]
                    ).flatten().item()
                )

        return scores
