from utils.constants import TEMPLATE_GSM8K_EVAL, TEMPLATE_GSM8K_EVAL_RES
from utils.exec_py import exec_py

from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

import backoff
import numpy as np
import openai #1.8.0
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


class OpenaiEvaluatorPy():

    def __init__(self, base_model_name):
        self.engine = base_model_name

    
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
                msg = TEMPLATE_GSM8K_EVAL_RES.format(
                        question, cand_py, str(answer)
                    )
            else:
                msg = TEMPLATE_GSM8K_EVAL.format(
                        question, cand_py
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