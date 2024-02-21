from utils.constants import TEMPLATE_EVAL, TEMPLATE_EVAL_RES, INST_CODELLAMA_EVAL, INST_CODELLAMA_EVAL_RES

from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

import backoff
import numpy as np
import openai #1.8.0
import sqlite3
import torch


client = openai.OpenAI(max_retries=5, timeout=60.0)


@backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError))
def openai_chat_engine(engine, msg, stop_tok="\n\n"):
    response = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "user", "content": msg}
        ],
        logprobs=True,
        top_logprobs=5,
        stop=stop_tok,
        max_tokens=1
    )

    return response


class OpenaiEvaluator():

    def __init__(self, base_model_name, db_path):
        self.engine = base_model_name
        self.db_path = db_path

    
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

                msg = INST_CODELLAMA_EVAL_RES.format(
                        TEMPLATE_EVAL_RES.format(question, cand_sql, result)
                    )
            else:
                msg = INST_CODELLAMA_EVAL.format(
                        TEMPLATE_EVAL.format(question, cand_sql)
                    )

            # Remove [INST] and [/INST] from CodeLlama prompt
            msg = msg[7:-8]

            responses = openai_chat_engine(self.engine, msg)
            top_logprobs = responses.choices[0].logprobs.content[0].top_logprobs
            token_score = dict([(t.token, np.exp(t.logprob)) for t in top_logprobs])
            
            s = 0
            if "Yes" in token_score:
                s = token_score["Yes"]
            elif "No" in token_score:
                s = 1 - token_score["No"]
            scores.append(-s)

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

                msg = INST_CODELLAMA_EVAL_RES.format(
                    "\n\n".join(prompt_strs)
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

                msg = INST_CODELLAMA_EVAL.format(
                    "\n\n".join(prompt_strs)
                )

            # Remove [INST] and [/INST] from CodeLlama prompt
            msg = msg[7:-8]

            responses = openai_chat_engine(self.engine, msg)
            top_logprobs = responses.choices[0].logprobs.content[0].top_logprobs
            token_score = dict([(t.token, np.exp(t.logprob)) for t in top_logprobs])

            s = 0
            if "Yes" in token_score:
                s = token_score["Yes"]
            elif "No" in token_score:
                s = 1 - token_score["No"]
            scores.append(-s)

        return scores
