from utils.constants import TEMPLATE, INST_CODELLAMA_GEN
from utils.normalize_sql import normalize_sql

from copy import deepcopy
from transformers import GenerationConfig

import numpy as np
import json
import torch


def greedy(example, generator, evaluator, retriever_gen, retriever_eval, args, log):
    config = json.load(open(args.generation_config))

    if retriever_gen is None:
        model_inp = TEMPLATE.format(example["db_id"], example["schema"], example["question"])
        prompt = model_inp
    else:
        demos = retriever_gen.retrieve(example["question"])
        model_inp = "\n\n".join([TEMPLATE.format(ex["db_id"], ex["schema"], ex["question"]) + ex["sql"] for ex in demos])
        prompt = model_inp + "\n\n" + TEMPLATE.format(example["db_id"], example["schema"], example["question"])

    if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
        prompt = INST_CODELLAMA_GEN.format(prompt) + " SELECT"

    responses = generator.generate(prompt, config)

    if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
        sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in responses if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
    else:
        sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in responses if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))

    example_log = deepcopy(example)
    example_log["top_n"] = sql_completions
    log.append(example_log)

    return sql_completions[0].replace("\n", " ")