from utils.exec_py import exec_py

from accelerate.utils import set_seed
from tqdm import tqdm

import argparse
import json
import numpy as np
import openai
import random
import torch


def select_models(args):
    generator = None
    evaluator = None

    if args.generator_name.startswith("gpt"):
        from generators.openai_generator import OpenaiGenerator
        generator = OpenaiGenerator(args.generator_name)
    else:
        from generators.hf_generator import HFGenerator
        generator = HFGenerator(args.generator_name, device="cuda")

    if args.evaluator_name == "oracle":
        from evaluators.oracle_evaluator_py import OracleEvaluatorPy
        evaluator = OracleEvaluatorPy(args.oracle_prob)
    elif args.evaluator_name.startswith("codellama"):
        from evaluators.codellama_evaluator_py import CodeLlamaEvaluatorPy
        evaluator = CodeLlamaEvaluatorPy(args.evaluator_name, device="cuda")
    elif args.evaluator_name.startswith("gpt"):
        from evaluators.openai_evaluator_py import OpenaiEvaluatorPy
        evaluator = OpenaiEvaluatorPy(args.evaluator_name)

    return generator, evaluator


def select_method(method_name):
    if method_name == "mctot":
        from planning_methods.mc_tot_py import mc_tot_py
        return mc_tot_py
    elif method_name == "greedy":
        from planning_methods.greedy_py import greedy_py
        return greedy_py
    elif method_name == "rerank":
        from planning_methods.rerank_py import rerank_py
        return rerank_py
    elif method_name == "iter_corr":
        from planning_methods.iter_correction_py import iter_correction_py
        return iter_correction_py
    else:
        raise Exception("Invalid method.")


def inference(generator, evaluator, args):
    test_data = json.load(open(args.test_fname))
    method = select_method(args.method_name)

    results = []
    log = []

    #global answer

    for ex in tqdm(test_data):
        #answer = None
        solution, example_log = method(ex, generator, evaluator, args)

        example_log["solution"] = solution

        try:
            answer = exec_py(solution)
            #exec(solution, globals())
        except:
            results.append(0)
            example_log["pred_answer"] = "ERROR"
            example_log["acc"] = 0
            log.append(example_log)
            continue

        if answer is not None and answer == ex["answer"]:
            results.append(1)
            example_log["pred_answer"] = str(answer)
            example_log["acc"] = 1
        else:
            results.append(0)
            example_log["pred_answer"] = str(answer)
            example_log["acc"] = 0

        log.append(example_log)

    out = open("log/" + args.log_fname, "w+", encoding="utf-8")
    json.dump(log, out, indent=2)
    out.close()

    print("Accuracy: {:<20.4f}".format(sum(results) / len(results)))


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--test_fname', type=str, default='data/spider_dev.json')
    args_parser.add_argument('--log_fname', type=str, default='spider_dev.json')

    args_parser.add_argument('--method_name', type=str, default='mctot')
    args_parser.add_argument('--generator_name', type=str, default='')
    args_parser.add_argument('--evaluator_name', type=str, default='') #codellama/CodeLlama-13b-Instruct-hf
    args_parser.add_argument('--oracle_prob', type=float, default=1.0)

    args_parser.add_argument('--seed', type=int, default=42)
    args_parser.add_argument('--generation_config', type=str, default='generation_configs/temp_sampling.json')
    args_parser.add_argument('--evaluation_config', type=str, default='')

    args = args_parser.parse_args()
    set_seed_all(args.seed)

    generator, evaluator = select_models(args)

    inference(generator, evaluator, args)