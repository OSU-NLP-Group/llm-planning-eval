from utils.constants import TEMPLATE_GSM8K

from copy import deepcopy
from transformers import GenerationConfig

import numpy as np
import json
import torch


def greedy_py(example, generator, evaluator, args):
    config = json.load(open(args.generation_config))

    model_inp = TEMPLATE_GSM8K.format(example["question"])
    prompt = model_inp

    responses = generator.generate(prompt, config)

    py_completions = list(set([r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] for r in responses if r.split(" [/INST] ## Python Program:\n")[-1].split("\n\n## ")[0] != ""]))

    example_log = deepcopy(example)
    example_log["top_n"] = py_completions

    return py_completions[0], example_log