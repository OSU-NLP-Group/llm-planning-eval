from utils.constants import TEMPLATE_GSM8K

from copy import deepcopy
from transformers import GenerationConfig

import heapq
import numpy as np
import json
import torch


def mc_tot_py(example, generator, evaluator, args):
    config = json.load(open(args.generation_config))
    evaluation_config = json.load(open(args.evaluation_config))

    model_inp = TEMPLATE_GSM8K.format(example["question"])
    prompt = model_inp
    
    partial_py = ""
    current_score = 18
    heap = []

    for t in range(50):
        # Step 1: Prompt the generator and sample initial steps.
        responses = generator.generate(prompt, config)

        py_completions = list(set([r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] for r in responses if r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] != ""]))

        # Segment generator completions to find the first new step.
        steps = set([
            (
                py[len(partial_py):].lstrip().split("\n")[0].rstrip()
                if len(py) > len(partial_py)
                else py
            )
            for py in py_completions
        ])

        # Step 2: For each new step, score it with the best the Monte-Carlo rollout.
        step_score = {}
        for s in steps:
            # Sample Monte-Carlo rollouts.
            mc_prompt = model_inp + partial_py + "\n" + s
            mc_rollouts = generator.generate(mc_prompt, config)
            mc_py_completions = list(set([r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] for r in mc_rollouts if r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] != ""]))

            # Evaluate Monte-Carlo rollouts.
            if args.evaluator_name == "oracle":
                scores = evaluator.score(example["question"], mc_py_completions, example["answer"])
            else:
                scores = evaluator.score(example["question"], mc_py_completions, evaluation_config)

            # Find the plan with highest score. Scores are negated for min heap implementation.
            step_score[s] = min(scores)

        # Step 3: Update the heap memory with new steps and scores.
        for k, v in step_score.items():
            if v <= 0: # prune if all rollouts have execution errors
                heapq.heappush(heap, (v, partial_py + "\n" + k))

        # If the heap is empty before finding the first complete program (e.g. all of them are non-executable), return the first initial plan.
        if len(heap) == 0:
            if not "answer =" in partial_py:
                partial_py = py_completions[0]
            break

        # Step 4: Pop out the current best (partial) plan in heap. If complete, return it.
        current_score, partial_py = heapq.heappop(heap)
        if "answer =" in partial_py:
            break
        else:
            prompt = model_inp + partial_py
    
        # For debugging
        # print(current_score, partial_py)
        # print(heap)

    answer = partial_py
    heapq.heappush(heap, (current_score, partial_py))

    example_log = deepcopy(example)
    example_log["heap"] = dict([(t[1], t[0]) for t in heap])

    return answer, example_log