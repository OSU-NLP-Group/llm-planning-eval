from accelerate.utils import set_seed
from copy import deepcopy
from tqdm import tqdm

import argparse
import evaluate
import json
import numpy as np
import openai
import random
import torch

# evaluation_config = {
#     "check_exec": False,
#     "use_exec_res": False
# }

def select_models(args):
    evaluator = None
    retriever_eval = None

    if args.evaluator_name == "oracle":
        from evaluators.oracle_evaluator import OracleEvaluator
        evaluator = OracleEvaluator(args.db_path, args.oracle_prob)
    elif args.evaluator_name.startswith("codellama"):
        if args.evaluator_peft_dir == "":
            from evaluators.codellama_evaluator import CodeLlamaEvaluator
            evaluator = CodeLlamaEvaluator(args.evaluator_name, args.db_path, device="cuda")
        else:
            from evaluators.codellama_evaluator import CodeLlamaLoraEvaluator
            evaluator = CodeLlamaLoraEvaluator(args.evaluator_name, args.evaluator_peft_dir, args.db_path, device="cuda")      
    elif args.evaluator_name.startswith("gpt"):
        from evaluators.openai_evaluator import OpenaiEvaluator
        evaluator = OpenaiEvaluator(args.evaluator_name, args.db_path)

    if args.retriever_name == "bm25":
        from retrievers.bm25 import BM25Retriever
        if args.retriever_corpus_eval:
            retriever_eval = BM25Retriever(args.retriever_name, args.retriever_corpus_eval, args.retrieve_k)

    return evaluator, retriever_eval


def intrinsic_eval(evaluator, retriever_eval, args):
    test_data = json.load(open(args.test_fname))
    evaluation_config = json.load(open(args.evaluation_config))

    results = []
    labels = []
    log = []

    # Pairwise selection accuracy
    pairs_count = 0
    pws_acc = 0

    # Example-level metrics
    hit = 0
    mrr = 0

    for ex in tqdm(test_data):
        sql_completions = ex["top_n"]

        if args.evaluator_name == "oracle":
            scores = evaluator.score(ex["db_id"], ex["question"], sql_completions, ex["sql"])
        elif retriever_eval is None:
            scores = evaluator.score(ex["db_id"], ex["question"], sql_completions, evaluation_config)
        else:
            scores = evaluator.score_fewshot(ex["db_id"], ex["question"], sql_completions, retriever_eval, evaluation_config)

        scores = [-s for s in scores]
        for a in range(len(sql_completions)):
            for b in range(a + 1, len(sql_completions)):
                if ex["top_n_label"][a] != ex["top_n_label"][b]:
                    pairs_count += 1
                    if (
                        (ex["top_n_label"][a] == 1 and scores[a] > scores[b]) or
                        (ex["top_n_label"][b] == 1 and scores[b] > scores[a])
                    ):
                        pws_acc += 1


        if args.evaluator_name == "oracle":
            cls_res = [(1 if s > 0.99 else 0) for s in scores]
        else:
            cls_res = [(1 if s > 0.5 else 0) for s in scores]

        ex_log = deepcopy(ex)
        ex_log["pred_scores"] = scores
        ex_log["pred_labels"] = cls_res
        log.append(ex_log)

        results += cls_res
        labels += ex["top_n_label"]

        scores_labels = [(s, g) for s, g in zip(scores, ex["top_n_label"])]
        scores_labels.sort(key=lambda x: x[0], reverse=True)
        reranked_labels = [tu[1] for tu in scores_labels]

        if reranked_labels[0] == 1:
            hit += 1
        for idx, l in enumerate(reranked_labels):
            if l == 1:
                mrr += (1 / (idx + 1))
                break

    # acc_metric = evaluate.load("accuracy")
    # acc = acc_metric.compute(predictions=results, references=labels)["accuracy"]

    f1_metric = evaluate.load("f1")
    pos_f1 = f1_metric.compute(predictions=results, references=labels, pos_label=1)["f1"]
    neg_f1 = f1_metric.compute(predictions=results, references=labels, pos_label=0)["f1"]
    macro_f1 = (pos_f1 + neg_f1) / 2

    print(
        "Pair Count: {}\nPWS Acc: {:<20.4f}\nSQL Count: {}\nPos F1: {:<20.4f}\nNeg F1: {:<20.4f}\nMacro F1: {:<20.4f}\nHit @ 1: {:<20.4f}\nMRR: {:<20.4f}\n".format(
            pairs_count, pws_acc / pairs_count, len(results), pos_f1, neg_f1, macro_f1, hit / len(test_data), mrr / len(test_data)
        )
    )

    if args.log_fname != "":
        out = open("log/" + args.log_fname, "w+", encoding="utf-8")
        json.dump(log, out, indent=2)
        out.close()


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
    args_parser.add_argument('--log_fname', type=str, default='')
    args_parser.add_argument('--dataset_name', type=str, default='spider')
    args_parser.add_argument('--db_path', type=str, default='../spider/database')

    args_parser.add_argument('--evaluator_name', type=str, default='') #codellama/CodeLlama-13b-Instruct-hf
    args_parser.add_argument('--evaluator_peft_dir', type=str, default='') 
    # /research/nfs_sun_397/chen.8336/codellama/spider-evaluator-rr-42
    args_parser.add_argument('--oracle_prob', type=float, default=1.0)
    args_parser.add_argument('--retriever_name', type=str, default='')
    args_parser.add_argument('--retriever_corpus_eval', type=str, default='')
    args_parser.add_argument('--retrieve_k', type=int, default=1)
    args_parser.add_argument('--evaluation_config', type=str, default='')

    args_parser.add_argument('--seed', type=int, default=42)

    args = args_parser.parse_args()
    set_seed_all(args.seed)

    evaluator, retriever_eval = select_models(args)

    intrinsic_eval(evaluator, retriever_eval, args)