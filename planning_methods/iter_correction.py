from utils.constants import TEMPLATE, TEMPLATE_CORR, SQL_KEYWORDS, INST_CODELLAMA_GEN, INST_CODELLAMA_ITER_CORR
from utils.normalize_sql import normalize_sql

from copy import deepcopy
from func_timeout import func_timeout
from transformers import GenerationConfig

import numpy as np
import json
import torch


def iter_correction(example, generator, evaluator, retriever_gen, retriever_eval, args, log):
    config = json.load(open(args.generation_config))
    evaluation_config = json.load(open(args.evaluation_config))

    # Step 1: Prompt the generator and sample initial plans.
    if retriever_gen is None:
        model_inp = TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    else:
        demos = retriever_gen.retrieve(example["question"])
        model_inp = "\n\n".join([TEMPLATE.format(ex["db_id"], ex["schema"], ex["question"]) + ex["sql"] for ex in demos])
        model_inp = model_inp + "\n\n" + TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    
    prompt = model_inp
    if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
        prompt = INST_CODELLAMA_GEN.format(prompt) + " SELECT"

    responses = generator.generate(prompt, config)

    if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
        sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in responses if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
    else:
        sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in responses if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))
    
    # Planning iteration setup.
    current_score = 18 #0
    patience = 0
    candidates_scores = {}
    answer_sql = ""

    for t in range(10):
        # Step 2: Score the current batch of plans.
        if args.evaluator_name == "oracle":
            try:
                scores = func_timeout(300.0, evaluator.score, args=(example["db_id"], example["question"], sql_completions, example["sql"]))
            except:
                scores = [0]
        elif retriever_eval is None:
            scores = evaluator.score(example["db_id"], example["question"], sql_completions, evaluation_config)
        else:
            scores = evaluator.score_fewshot(example["db_id"], example["question"], sql_completions, retriever_eval, evaluation_config)

        # Step 3: Find the plan with highest score. Scores are negated for min heap implementation in tree search.
        best_score = min(scores)

        # Step 4: Check termination conditions and replace the old plan with the currently best one, if any.
        if best_score < -0.99:
            answer_sql = sql_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_sql] = best_score
            break
        elif best_score >= current_score:
            patience += 1
            if patience >= 3:
                break
        else:
            answer_sql = sql_completions[np.argmin(scores)]
            current_score = best_score
            candidates_scores[answer_sql] = best_score
            patience = 0


        # Step 5: Prompt the generator for 0-shot correction. Sample a new batch of plans.
        prompt = TEMPLATE_CORR.format(example["db_id"], example["schema"], example["question"], answer_sql)
        if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
            prompt = INST_CODELLAMA_ITER_CORR.format(prompt) + " SELECT"

        responses = generator.generate(prompt, config)

        if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
            sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in responses if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
        else:
            sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in responses if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))

    answer = answer_sql.replace("\n", " ")

    example_log = deepcopy(example)
    example_log["candidates"] = candidates_scores
    log.append(example_log)

    return answer