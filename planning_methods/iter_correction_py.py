from utils.constants import TEMPLATE_GSM8K, GSM8K_ITER_CORR

from copy import deepcopy
from transformers import GenerationConfig

import numpy as np
import json
import torch


def iter_correction_py(example, generator, evaluator, args):
    config = json.load(open(args.generation_config))
    evaluation_config = json.load(open(args.evaluation_config))

    # Step 1: Prompt the generator and sample initial plans.
    model_inp = TEMPLATE_GSM8K.format(example["question"])
    prompt = model_inp
    responses = generator.generate(prompt, config)
    py_completions = list(set([r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] for r in responses if r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] != ""]))
    
    # Planning iteration setup.
    current_score = 18
    patience = 0
    candidates_scores = {}
    answer_py = ""

    for t in range(10):
        # Step 2: Score the current batch of plans.
        if args.evaluator_name == "oracle":
            scores = evaluator.score(example["question"], py_completions, example["answer"])
        else:
            scores = evaluator.score(example["question"], py_completions, evaluation_config)

        # Step 3: Find the plan with highest score. Scores are negated for min heap implementation in tree search.
        best_score = min(scores)

        # Step 4: Check termination conditions and replace the old plan with the currently best one, if any.
        if best_score < -0.99:
            answer_py = py_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_py] = best_score
            break
        elif best_score >= current_score:
            patience += 1
            if patience >= 3:
                break
        else:
            answer_py = py_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_py] = best_score
            patience = 0

        # Step 5: Prompt the generator for 0-shot correction. Sample a new batch of plans.
        prompt = GSM8K_ITER_CORR.format(example["question"], answer_py)
        responses = generator.generate(prompt, config)
        py_completions = list(set([r.split(" [/INST] ## Fixed Python Program:\n")[-1].split("\n\n## ")[0] for r in responses if r.split(" [/INST] ## Fixed Python Program:\n")[-1].split("\n\n## ")[0] != ""]))

    answer = answer_py

    example_log = deepcopy(example)
    example_log["candidates"] = candidates_scores

    return answer, example_log