from utils.constants import TEMPLATE_EVAL, TEMPLATE_EVAL_RES, INST_CODELLAMA_EVAL, INST_CODELLAMA_EVAL_RES
from utils.exec_eval import eval_exec_match
from utils.normalize_sql import normalize_sql

from tqdm import tqdm

import argparse
import json
import sqlite3


def preproc_ex(example, use_exec_res=False):
    exs = []

    for i in range(len(example["top_n"])):
        if use_exec_res:
            src = INST_CODELLAMA_EVAL_RES.format(
                    TEMPLATE_EVAL_RES.format(example["question"], example["top_n"][i], example["top_n_exec_res"][i])
                )
        else:
            src = INST_CODELLAMA_EVAL.format(
                    TEMPLATE_EVAL.format(example["question"], example["top_n"][i])
                )

        tgt = "Yes" if example["top_n_label"][i] == 1 else "No"

        exs.append({"src": src, "tgt": tgt})

    if example["sql"] not in example["top_n"]:
        if use_exec_res:
            src = INST_CODELLAMA_EVAL_RES.format(
                    TEMPLATE_EVAL_RES.format(example["question"], example["sql"], example["exec_res"])
                )
        else:
            src = INST_CODELLAMA_EVAL.format(
                    TEMPLATE_EVAL.format(example["question"], example["sql"])
                )

        exs.append({"src": src, "tgt": "Yes"})

    return exs


def preprocess(args):
    data = []
    for example in tqdm(json.load(open(args.data_fname))):
        data += preproc_ex(example, args.use_exec_res) 

    print(len(data))

    out = open(args.out_fname, "w+")
    json.dump(data, out, indent=2)
    out.close()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_fname', type=str, default='')
    args_parser.add_argument('--out_fname', type=str, default='')
    args_parser.add_argument('--use_exec_res', action="store_true")
    args = args_parser.parse_args()

    preprocess(args)